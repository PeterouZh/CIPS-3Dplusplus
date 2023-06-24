from collections import defaultdict
import argparse
import math
import random
import os
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.distributed as dist
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import moxing_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.pytorch import torch_utils
from tl2 import tl2_utils
from tl2.proj.logger.textlogger import summary_defaultdict2txtfig, global_textlogger
from tl2.proj.fvcore import build_model

from losses import *
from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
# from options import BaseOptions
# from model import Generator, VolumeRenderDiscriminator
# from dataset import MultiResolutionDataset
from exp.stylesdf.options import BaseOptions
from exp.stylesdf.models.model import Generator, VolumeRenderDiscriminator
from exp.stylesdf.dataset import MultiResolutionDataset
from exp.stylesdf.utils import (data_sampler,
                                requires_grad,
                                accumulate,
                                sample_data,
                                make_noise,
                                mixing_noise,
                                generate_camera_params)


try:
  import wandb
except ImportError:
  wandb = None


def sphere_init_func(opt,
                     generator,
                     g_module,
                     g_optim,
                     g_ema,
                     d_module,
                     device):
  init_pbar = range(10000)
  if get_rank() == 0:
    init_pbar = tqdm(init_pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

  generator.zero_grad()
  for idx in init_pbar:
    noise = mixing_noise(3, opt.style_dim, opt.mixing, device)
    cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(opt.renderer_output_size, device,
                                                                             batch=3,
                                                                             uniform=opt.camera.uniform,
                                                                             azim_range=opt.camera.azim,
                                                                             elev_range=opt.camera.elev,
                                                                             fov_ang=opt.camera.fov,
                                                                             dist_radius=opt.camera.dist_radius)
    sdf, target_values = g_module.init_forward(noise, cam_extrinsics, focal, near, far)
    loss = F.l1_loss(sdf, target_values)
    loss.backward()
    g_optim.step()
    generator.zero_grad()
    if get_rank() == 0:
      init_pbar.set_description((f"MLP init to sphere procedure - Loss: {loss.item():.4f}"))

  accumulate(g_ema, g_module, 0)

  saved_path = os.path.join("cache_pretrained", f"sdf_init_models.pt")
  torch.save(
    {
      "g": g_module.state_dict(),
      "d": d_module.state_dict(),
      "g_ema": g_ema.state_dict(),
    },
    saved_path,
  )
  print(f'Successfully saved checkpoint for SDF initialized MLP in {saved_path}.')


def update_D(opt,
             generator,
             discriminator,
             d_optim,
             loader,
             device,
             loss_dict,
             **kwargs):
  # update discriminator
  requires_grad(generator, False)
  requires_grad(discriminator, True)
  discriminator.zero_grad()
  _, real_imgs = next(loader)
  real_imgs = real_imgs.to(device)
  noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)
  cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(
    opt.renderer_output_size, device,
    batch=opt.batch,
    uniform=opt.camera.uniform,
    azim_range=opt.camera.azim,
    elev_range=opt.camera.elev,
    fov_ang=opt.camera.fov,
    dist_radius=opt.camera.dist_radius)
  gen_imgs = []
  for j in range(0, opt.batch, opt.chunk):
    curr_noise = [n[j:j + opt.chunk] for n in noise]
    _, fake_img = generator(curr_noise,
                            cam_extrinsics[j:j + opt.chunk],
                            focal[j:j + opt.chunk],
                            near[j:j + opt.chunk],
                            far[j:j + opt.chunk])

    gen_imgs += [fake_img]

  gen_imgs = torch.cat(gen_imgs, 0)
  fake_pred, fake_viewpoint_pred = discriminator(gen_imgs.detach())
  if global_cfg.lambda_pose > 0:
    d_view_loss = global_cfg.lambda_pose * viewpoints_loss(fake_viewpoint_pred, gt_viewpoints)
  else:
    d_view_loss = 0.

  real_imgs.requires_grad = True
  real_pred, _ = discriminator(real_imgs)
  # GAN loss
  d_gan_loss = d_logistic_loss(real_pred, fake_pred)
  # gp loss
  grad_penalty = d_r1_loss(real_pred, real_imgs)
  r1_loss = global_cfg.lambda_gp * 0.5 * grad_penalty

  # total loss
  d_loss = d_gan_loss + r1_loss + d_view_loss
  d_loss.backward()
  d_optim.step()

  loss_dict["d_loss_gan"]['d_loss_gan'] = d_gan_loss.item()
  loss_dict["d_loss_r1"]['d_loss_r1'] = r1_loss.item()
  loss_dict["d_loss_pose"]['d_loss_pose'] = d_view_loss.item() if torch.is_tensor(d_view_loss) else d_view_loss
  loss_dict["d_loss_total"]['d_loss_total'] = d_loss.item()
  loss_dict['d_logits']["d_logits_real"] = real_pred.detach().mean().item()
  loss_dict['d_logits']["d_logits_fake"] = fake_pred.detach().mean().item()
  pass

def update_G(opt,
             generator,
             discriminator,
             g_optim,
             device,
             loss_dict):
  # update generator

  requires_grad(generator, True)
  requires_grad(discriminator, False)
  for j in range(0, opt.batch, opt.chunk):
    noise = mixing_noise(opt.chunk, opt.style_dim, opt.mixing, device)
    cam_extrinsics, focal, near, far, curr_gt_viewpoints = generate_camera_params(
      opt.renderer_output_size, device,
      batch=opt.chunk,
      uniform=opt.camera.uniform,
      azim_range=opt.camera.azim,
      elev_range=opt.camera.elev,
      fov_ang=opt.camera.fov,
      dist_radius=opt.camera.dist_radius)

    return_sdf = global_cfg.lambda_min_surf > 0
    return_eikonal = global_cfg.lambda_eikonal > 0

    out = generator(noise, cam_extrinsics, focal, near, far,
                    return_sdf=return_sdf,
                    return_eikonal=return_eikonal)

    fake_img = out[1]
    if return_sdf:
      sdf = out[2]
    if return_eikonal:
      eikonal_term = out[3]

    fake_pred, fake_viewpoint_pred = discriminator(fake_img)
    if global_cfg.lambda_pose > 0:
      g_view_loss = global_cfg.lambda_pose * viewpoints_loss(fake_viewpoint_pred, curr_gt_viewpoints)
    else:
      g_view_loss = 0.

    if opt.with_sdf and global_cfg.lambda_eikonal > 0:
      g_eikonal, g_minimal_surface = eikonal_loss(eikonal_term,
                                                  sdf=sdf if return_sdf else None,
                                                  beta=global_cfg.min_surf_beta)
      g_eikonal = global_cfg.lambda_eikonal * g_eikonal
      if return_sdf > 0:
        g_minimal_surface = global_cfg.lambda_min_surf * g_minimal_surface
    else:
      g_eikonal = 0.
      g_minimal_surface = 0.

    g_gan_loss = g_nonsaturating_loss(fake_pred)
    g_loss = g_gan_loss + g_view_loss + g_eikonal + g_minimal_surface
    g_loss.backward()

  g_optim.step()
  generator.zero_grad()

  loss_dict['g_loss_gan']["g_loss_gan"] = g_gan_loss.item()
  loss_dict["g_loss_pose"]['g_loss_pose'] = g_view_loss.item() if torch.is_tensor(g_view_loss) else g_view_loss
  loss_dict["g_loss_eikonal"]['g_loss_eikonal'] = g_eikonal.item() if torch.is_tensor(g_eikonal) else g_eikonal
  loss_dict["g_loss_minimal_surface"]['g_loss_minimal_surface'] = \
    g_minimal_surface.item() if torch.is_tensor(g_minimal_surface) else g_minimal_surface
  loss_dict["g_loss_total"]['g_loss_total'] = g_loss.item()
  pass

@torch.no_grad()
def save_images(log_img_size,
                log_N_row,
                log_N_col,
                g_module,
                g_ema,
                saved_dir,
                device,
                sample_kwargs):
  sample_z = sample_kwargs['sample_z']
  sample_cam_extrinsics = sample_kwargs['sample_cam_extrinsics']
  sample_focals = sample_kwargs['sample_focals']
  sample_near = sample_kwargs['sample_near']
  sample_far = sample_kwargs['sample_far']

  samples = torch.Tensor(0, 3, log_img_size, log_img_size)
  step_size = 4
  # mean_latent = g_module.mean_latent(10000, device)
  mean_latent = g_ema.mean_latent(10000, device)
  for k in range(0, log_N_row * log_N_col, step_size):
    _, curr_samples = g_ema([sample_z[0][k:k + step_size]],
                            sample_cam_extrinsics[k:k + step_size],
                            sample_focals[k:k + step_size],
                            sample_near[k:k + step_size],
                            sample_far[k:k + step_size],
                            truncation=0.7,
                            truncation_latent=mean_latent)
    samples = torch.cat([samples, curr_samples.cpu()], 0)

  utils.save_image(samples,
                   f"{saved_dir}/0_G_ema.jpg",
                   nrow=log_N_col,
                   normalize=True,
                   value_range=(-1, 1), )
  pass

def save_models(g_module,
                d_module,
                g_ema,
                state_dict,
                info_msg,
                saved_dir=None):

  model_dict = {
    "G": g_module,
    "D": d_module,
    "G_ema": g_ema,
    "state_dict": state_dict,
  }

  if saved_dir is None:
    ckpt_max2keep = tl2_utils.MaxToKeep.get_named_max_to_keep(name='ckpt', use_circle_number=True)
    saved_dir = ckpt_max2keep.step_and_ret_circle_dir(global_cfg.tl_ckptdir)
  os.makedirs(saved_dir, exist_ok=True)

  global_cfg.dump_to_file_with_command(f"{saved_dir}/config_command.yaml", global_cfg.tl_command)

  torch_utils.save_models(save_dir=saved_dir, model_dict=model_dict)
  tl2_utils.write_info_msg(saved_dir, info_msg)

  return saved_dir

def train(opt,
          experiment_opt,
          loader,
          generator,
          discriminator,
          g_optim,
          d_optim,
          g_ema,
          device,
          sampler,
          state_dict):
  rank = get_rank()
  world_size = get_world_size()
  distributed = world_size > 1

  # data loader
  loader = sample_data(loader, distributed=distributed, sampler=sampler)# loss_dict = {}
  # ema
  accum = 0.5 ** (32 / (10 * 1000))

  if distributed:
    g_module = generator.module
    d_module = discriminator.module
  else:
    g_module = generator
    d_module = discriminator

  # fixed_z: N_row x N_col
  sample_z = [torch.randn(global_cfg.log_N_row, opt.style_dim, device=device)
                .repeat_interleave(global_cfg.log_N_col, dim=0)]
  sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = generate_camera_params(
    global_cfg.log_img_size,
    device,
    batch=global_cfg.log_N_row, # batch x 8
    sweep=True,
    uniform=opt.camera.uniform,
    azim_range=opt.camera.azim,
    elev_range=opt.camera.elev,
    fov_ang=opt.camera.fov,
    dist_radius=opt.camera.dist_radius)
  sample_kwargs = {
    'sample_z': sample_z,
    'sample_cam_extrinsics': sample_cam_extrinsics,
    'sample_focals': sample_focals,
    'sample_near': sample_near,
    'sample_far': sample_far,
  }

  if opt.with_sdf and opt.sphere_init and opt.start_iter == 0:
    sphere_init_func(opt=opt,
                     generator=generator,
                     g_module=g_module,
                     g_optim=g_optim,
                     g_ema=g_ema,
                     d_module=d_module,
                     device=device)

  start_iter = state_dict['iter']
  total_iters = global_cfg.total_iters
  if get_rank() == 0:
    pbar = tl2_utils.TL_tqdm(total=total_iters, start=start_iter)
  else:
    pbar = range(start_iter, total_iters)

  loss_dict = defaultdict(dict)

  # start training
  for idx in pbar:
    loss_dict.clear()

    update_D(opt=opt,
             generator=generator,
             discriminator=discriminator,
             d_optim=d_optim,
             loader=loader,
             device=device,
             loss_dict=loss_dict)

    update_G(opt=opt,
             generator=generator,
             discriminator=discriminator,
             g_optim=g_optim,
             device=device,
             loss_dict=loss_dict)

    # update ema
    if idx >= global_cfg.ema_start:
      accumulate(g_ema, g_module, accum)
    else:
      accumulate(g_ema, g_module, 0.)

    # log txt
    if rank == 0 and ((idx + 1) % global_cfg.log_txt_every == 0 or global_cfg.tl_debug):
      g_beta_val = g_module.renderer.sigmoid_beta.item() if opt.with_sdf else 0
      loss_dict['g_beta_val']['g_beta_val'] = g_beta_val
      loss_dict['lr']['G_lr'] = torch_utils.get_optimizer_lr(g_optim)
      loss_dict['lr']['D_lr'] = torch_utils.get_optimizer_lr(d_optim)
      loss_dict['lambda_gp']['lambda_gp'] = global_cfg.lambda_gp
      loss_dict['lambda_pose']['lambda_pose'] = global_cfg.lambda_pose
      loss_dict['lambda_eikonal']["lambda_eikonal"] = global_cfg.lambda_eikonal
      loss_dict['lambda_min_surf']["lambda_min_surf"] = global_cfg.lambda_min_surf
      loss_dict['min_surf_beta']["min_surf_beta"] = global_cfg.min_surf_beta
      loss_dict['batch']["batch"] = global_cfg.batch

      log_str = tl2_utils.get_print_dict_str(loss_dict, outdir=global_cfg.tl_outdir, suffix_str=pbar.get_string())
      print(log_str)
      if idx > 1000:
        summary_defaultdict2txtfig(default_dict=loss_dict, prefix='train', step=idx, textlogger=global_textlogger)

    state_dict['iter'] = idx
    # save ckpt and images
    if rank == 0 and ((idx + 1) % global_cfg.log_ckpt_every == 0 or global_cfg.tl_debug):
      saved_dir = save_models(g_module=g_module,
                              d_module=d_module,
                              g_ema=g_ema,
                              state_dict=state_dict,
                              info_msg=f"iter: {state_dict['iter']}",
                              saved_dir=None)
      save_images(log_img_size=global_cfg.log_img_size,
                  log_N_row=global_cfg.log_N_row,
                  log_N_col=global_cfg.log_N_col,
                  g_module=g_module,
                  g_ema=g_ema,
                  saved_dir=saved_dir,
                  device=device,
                  sample_kwargs=sample_kwargs)
      moxing_utils.modelarts_sync_results_dir(cfg=global_cfg, )

    synchronize()
  pass


def main():
  device = "cuda"
  opt = BaseOptions().parse()
  opt.model.freeze_renderer = False
  opt.model.no_viewpoint_loss = opt.training.view_lambda == 0.0
  opt.training.camera = opt.camera
  opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
  opt.training.style_dim = opt.model.style_dim
  opt.training.with_sdf = not opt.rendering.no_sdf
  opt.training.iter = 200001
  opt.rendering.no_features_output = True

  n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
  opt.training.distributed = n_gpu > 1

  if opt.training.distributed:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
  rank = get_rank()

  update_parser_defaults_from_yaml(parser=None, is_main_process=(rank == 0))
  if rank == 0:
    moxing_utils.setup_tl_outdir_obs(global_cfg)
    moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  synchronize()

  if opt.training.with_sdf and global_cfg.lambda_min_surf > 0:
    opt.rendering.return_sdf = True

  # create checkpoints directories
  # os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer'), exist_ok=True)
  # os.makedirs(os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', 'samples'),
  #             exist_ok=True)

  discriminator = VolumeRenderDiscriminator(opt.model).to(device)

  if 'G_cfg' in global_cfg:
    generator = build_model(cfg=global_cfg.G_cfg, kwargs_priority=True).to(device)
    g_ema = build_model(cfg=global_cfg.G_cfg, kwargs_priority=True,
                        ema=True, full_pipeline=False).to(device)
  else:
    generator = Generator(model_opt=opt.model, renderer_opt=opt.rendering, full_pipeline=False).to(device)
    g_ema = Generator(model_opt=opt.model, renderer_opt=opt.rendering, ema=True, full_pipeline=False).to(device)

  g_ema.eval()
  accumulate(g_ema, generator, 0)
  g_optim = optim.Adam(generator.parameters(), lr=2e-5, betas=(0, 0.9))
  d_optim = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0, 0.9))

  opt.training.start_iter = 0

  state_dict = {
    'iter': 0,
    'fid': float('inf'),
    'best_fid': float('inf')
  }

  # resume
  if opt.experiment.continue_training and opt.experiment.ckpt is not None:
    if get_rank() == 0:
      print("load model:", opt.experiment.ckpt)
    ckpt_path = os.path.join(opt.training.checkpoints_dir,
                             opt.experiment.expname,
                             'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    try:
      opt.training.start_iter = int(opt.experiment.ckpt) + 1

    except ValueError:
      pass

    generator.load_state_dict(ckpt["g"])
    discriminator.load_state_dict(ckpt["d"])
    g_ema.load_state_dict(ckpt["g_ema"])
    if "g_optim" in ckpt.keys():
      g_optim.load_state_dict(ckpt["g_optim"])
      d_optim.load_state_dict(ckpt["d_optim"])

  # sphere_init_path = './pretrained_renderer/sphere_init.pt'
  moxing_utils.copy_data(rank=rank, global_cfg=global_cfg,
                         datapath_obs=global_cfg.obs_sphere_init_path, datapath=global_cfg.sphere_init_path)
  sphere_init_path = global_cfg.sphere_init_path
  if opt.training.no_sphere_init:
    opt.training.sphere_init = False
  elif not opt.experiment.continue_training and opt.training.with_sdf and os.path.isfile(sphere_init_path):
    if get_rank() == 0:
      print("loading sphere inititialized model")
    ckpt = torch.load(sphere_init_path, map_location=lambda storage, loc: storage)
    # generator.load_state_dict(ckpt["g"])
    # discriminator.load_state_dict(ckpt["d"])
    # g_ema.load_state_dict(ckpt["g_ema"])
    Checkpointer(generator).load_state_dict(ckpt['g'])
    Checkpointer(discriminator).load_state_dict(ckpt['d'])
    Checkpointer(g_ema).load_state_dict(ckpt['g_ema'])

    opt.training.sphere_init = False
  else:
    opt.training.sphere_init = True

  if opt.training.distributed:
    generator = nn.parallel.DistributedDataParallel(
      generator,
      device_ids=[opt.training.local_rank],
      output_device=opt.training.local_rank,
      broadcast_buffers=True,
      find_unused_parameters=True,
    )

    discriminator = nn.parallel.DistributedDataParallel(
      discriminator,
      device_ids=[opt.training.local_rank],
      output_device=opt.training.local_rank,
      broadcast_buffers=False,
      find_unused_parameters=True
    )

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

  moxing_utils.copy_data(rank=rank, global_cfg=global_cfg, **global_cfg.obs_training_dataset)
  dataset = MultiResolutionDataset(path=global_cfg.obs_training_dataset.datapath,
                                   transform=transform,
                                   resolution=global_cfg.resolution,
                                   nerf_resolution=global_cfg.nerf_resolution)

  sampler = data_sampler(dataset, shuffle=True, distributed=opt.training.distributed)

  loader = data.DataLoader(
    dataset,
    batch_size=opt.training.batch,
    sampler=sampler,
    drop_last=True,
  )
  opt.training.dataset_name = opt.dataset.dataset_path.lower()

  # save options
  # opt_path = os.path.join(opt.training.checkpoints_dir, opt.experiment.expname, 'volume_renderer', f"opt.yaml")
  opt_path = f"{global_cfg.tl_outdir}/exp/opt.yaml"
  os.makedirs(os.path.dirname(opt_path), exist_ok=True)
  with open(opt_path, 'w') as f:
    yaml.safe_dump(opt, f)

  # set wandb environment
  if get_rank() == 0 and wandb is not None and opt.training.wandb:
    wandb.init(project="StyleSDF")
    wandb.run.name = opt.experiment.expname
    wandb.config.dataset = os.path.basename(opt.dataset.dataset_path)
    wandb.config.update(opt.training)
    wandb.config.update(opt.model)
    wandb.config.update(opt.rendering)

  train(opt.training,
        opt.experiment,
        loader,
        generator,
        discriminator,
        g_optim,
        d_optim,
        g_ema,
        device,
        sampler=sampler,
        state_dict=state_dict)
  pass


if __name__ == '__main__':

  main()

  moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)