import argparse
import collections
import math
import random
import os
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import moxing_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.pytorch import torch_utils
from tl2 import tl2_utils
from tl2.proj.logger.textlogger import summary_defaultdict2txtfig, global_textlogger
from tl2.proj.fvcore import build_model

from distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
from exp.stylesdf.losses import *
from exp.stylesdf.dataset import MultiResolutionDataset
from exp.stylesdf.models.model import Generator, Discriminator
from exp.stylesdf.options import BaseOptions
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


def update_D(opt,
             generator,
             discriminator,
             idx,
             loader,
             d_optim,
             loss_dict,
             device):

  requires_grad(generator, False)
  requires_grad(discriminator, True)

  real_imgs, _ = next(loader)
  real_imgs = real_imgs.to(device)

  noise = mixing_noise(opt.batch, opt.style_dim, opt.mixing, device)

  cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(
    resolution=global_cfg.renderer_output_size,
    device=device,
    batch=opt.batch,
    uniform=opt.camera.uniform,
    azim_range=opt.camera.azim,
    elev_range=opt.camera.elev,
    fov_ang=opt.camera.fov,
    dist_radius=opt.camera.dist_radius)

  discriminator.zero_grad()
  d_regularize = (idx + 1) % global_cfg.d_reg_every == 0

  for j in range(0, opt.batch, opt.chunk):
    curr_real_imgs = real_imgs[j:j + opt.chunk]
    curr_noise = [n[j:j + opt.chunk] for n in noise]
    gen_imgs, _ = generator(curr_noise,
                            cam_extrinsics[j:j + opt.chunk],
                            focal[j:j + opt.chunk],
                            near[j:j + opt.chunk],
                            far[j:j + opt.chunk])

    fake_pred = discriminator(gen_imgs.detach())

    if d_regularize:
      curr_real_imgs.requires_grad = True

    real_pred = discriminator(curr_real_imgs)
    d_gan_loss = d_logistic_loss(real_pred, fake_pred)

    if d_regularize:
      grad_penalty = d_r1_loss(real_pred, curr_real_imgs)
      r1_loss = global_cfg.lambda_gp * 0.5 * grad_penalty * global_cfg.d_reg_every
    else:
      r1_loss = 0.

    d_loss = d_gan_loss + r1_loss
    d_loss.backward()

  d_optim.step()
  discriminator.zero_grad(set_to_none=True)

  loss_dict["d_loss_gan"]['d_loss_gan'] = d_gan_loss.item()
  loss_dict["d_loss_gp"]['d_loss_gp'] = r1_loss.item() if torch.is_tensor(r1_loss) else r1_loss
  loss_dict["d_loss_total"]['d_loss_total'] = d_loss.item()
  loss_dict["d_logits"]['d_logits_real'] = real_pred.detach().mean().item()
  loss_dict["d_logits"]['d_logits_fake'] = fake_pred.detach().mean().item()

  pass

def update_G(opt,
             idx,
             generator,
             discriminator,
             g_optim,
             loss_dict,
             device
             ):

  requires_grad(generator, True)
  requires_grad(discriminator, False)

  generator.zero_grad()

  for j in range(0, opt.batch, opt.chunk):
    noise = mixing_noise(opt.chunk, opt.style_dim, opt.mixing, device)
    cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(
      resolution=global_cfg.renderer_output_size,
      device=device,
      batch=opt.chunk,
      uniform=opt.camera.uniform,
      azim_range=opt.camera.azim,
      elev_range=opt.camera.elev,
      fov_ang=opt.camera.fov,
      dist_radius=opt.camera.dist_radius)

    fake_img, _ = generator(noise, cam_extrinsics, focal, near, far)
    fake_pred = discriminator(fake_img)
    g_gan_loss = g_nonsaturating_loss(fake_pred)

    g_loss = g_gan_loss
    g_loss.backward()

  g_optim.step()
  # generator.zero_grad()

  loss_dict["g_loss_gan"]['g_loss_gan'] = g_gan_loss.item()

  # generator path regularization
  g_regularize = (global_cfg.g_reg_every > 0) and ((idx + 1) % global_cfg.g_reg_every == 0)
  if g_regularize:
    generator.zero_grad()

    path_batch_size = max(1, opt.batch // opt.path_batch_shrink)
    path_noise = mixing_noise(path_batch_size, opt.style_dim, opt.mixing, device)
    path_cam_extrinsics, path_focal, path_near, path_far, _ = generate_camera_params(
      resolution=global_cfg.renderer_output_size,
      device=device,
      batch=path_batch_size,
      uniform=opt.camera.uniform,
      azim_range=opt.camera.azim,
      elev_range=opt.camera.elev,
      fov_ang=opt.camera.fov,
      dist_radius=opt.camera.dist_radius)

    for j in range(0, path_batch_size, opt.chunk):
      path_fake_img, path_latents = generator(path_noise,
                                              path_cam_extrinsics,
                                              path_focal,
                                              path_near,
                                              path_far,
                                              return_latents=True)

      path_loss, mean_path_length, path_lengths = g_path_regularize(
        fake_img=path_fake_img,
        latents=path_latents,
        mean_path_length=0.
      )

      weighted_path_loss = global_cfg.path_regularize * global_cfg.g_reg_every * path_loss  # * opt.chunk / path_batch_size
      if opt.path_batch_shrink:
        weighted_path_loss += (0. * path_fake_img[0, 0, 0, 0])

      weighted_path_loss.backward()
      # mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())

    g_optim.step()

  else: # g_regularize = False
    weighted_path_loss = 0.
    path_lengths = 0.
    pass

  generator.zero_grad(set_to_none=True)

  loss_dict["g_loss_weighted_path"]['g_loss_weighted_path'] = weighted_path_loss.item() \
    if torch.is_tensor(weighted_path_loss) else weighted_path_loss
  loss_dict["path_length_mean"]['path_length_mean'] = path_lengths.detach().mean().item() \
    if torch.is_tensor(path_lengths) else path_lengths

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

@torch.no_grad()
def save_images(renderer_output_size,
                gen_output_size,
                log_N_row,
                log_N_col,
                g_module,
                g_ema,
                saved_dirs,
                device,
                sample_kwargs):
  sample_z = sample_kwargs['sample_z']
  sample_cam_extrinsics = sample_kwargs['sample_cam_extrinsics']
  sample_focals = sample_kwargs['sample_focals']
  sample_near = sample_kwargs['sample_near']
  sample_far = sample_kwargs['sample_far']

  samples = torch.Tensor(0, 3, gen_output_size, gen_output_size)
  thumbs_samples = torch.Tensor(0, 3, renderer_output_size, renderer_output_size)
  step_size = 8
  # mean_latent = g_module.mean_latent(10000, device)
  mean_latent = g_ema.mean_latent(10000, device)
  for k in range(0, log_N_row * log_N_col, step_size):
    curr_samples, curr_thumbs = g_ema([sample_z[0][k:k + step_size]],
                                      sample_cam_extrinsics[k:k + step_size],
                                      sample_focals[k:k + step_size],
                                      sample_near[k:k + step_size],
                                      sample_far[k:k + step_size],
                                      truncation=0.7,
                                      truncation_latent=mean_latent)
    samples = torch.cat([samples, curr_samples.cpu()], 0)
    thumbs_samples = torch.cat([thumbs_samples, curr_thumbs.cpu()], 0)

  for saved_dir in saved_dirs:
    utils.save_image(samples,
                     f"{saved_dir}/0_G_ema.jpg",
                     nrow=log_N_col,
                     normalize=True,
                     value_range=(-1, 1), )
    utils.save_image(thumbs_samples,
                     f"{saved_dir}/0_G_ema_nerf.jpg",
                     nrow=log_N_col,
                     normalize=True,
                     value_range=(-1, 1), )
  pass

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
          state_dict
          ):

  rank = get_rank()
  world_size = get_world_size()
  distributed = world_size > 1

  loader = sample_data(loader, distributed=distributed, sampler=sampler)

  if distributed:
    g_module = generator.module
    d_module = discriminator.module
  else:
    g_module = generator
    d_module = discriminator

  accum = 0.5 ** (32 / (10 * 1000))

  # fixed_z: N_row x N_col
  sample_z = [torch.randn(global_cfg.log_N_row, opt.style_dim, device=device)
                .repeat_interleave(global_cfg.log_N_col, dim=0)]
  sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = generate_camera_params(
    global_cfg.renderer_output_size,
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

  start_iter = state_dict['iter']
  total_iters = global_cfg.total_iters
  if get_rank() == 0:
    pbar = tl2_utils.TL_tqdm(total=total_iters, start=start_iter)
  else:
    pbar = range(start_iter, total_iters)

  loss_dict = collections.defaultdict(dict)

  for idx in pbar:
    loss_dict.clear()

    # update discriminator
    update_D(opt=opt,
             generator=generator,
             discriminator=discriminator,
             idx=idx,
             loader=loader,
             d_optim=d_optim,
             loss_dict=loss_dict,
             device=device)

    # update genearator
    update_G(opt=opt,
             idx=idx,
             generator=generator,
             discriminator=discriminator,
             g_optim=g_optim,
             loss_dict=loss_dict,
             device=device)

    # update ema
    if idx >= global_cfg.ema_start:
      accumulate(g_ema, g_module, accum)
    else:
      accumulate(g_ema, g_module, 0.)

    # log txt
    if rank == 0 and ((idx + 1) % global_cfg.log_txt_every == 0 or global_cfg.tl_debug):
      loss_dict['lr']['G_lr'] = torch_utils.get_optimizer_lr(g_optim, return_all=False)
      loss_dict['lr']['D_lr'] = torch_utils.get_optimizer_lr(d_optim)
      loss_dict['lambda_gp']['lambda_gp'] = global_cfg.lambda_gp
      loss_dict['d_reg_every']["d_reg_every"] = global_cfg.d_reg_every
      loss_dict['g_reg_every']["g_reg_every"] = global_cfg.g_reg_every
      loss_dict['batch']["batch"] = global_cfg.batch

      log_str = tl2_utils.get_print_dict_str(loss_dict, outdir=global_cfg.tl_outdir, suffix_str=pbar.get_string())
      print(log_str)
      if idx > 1000:
        summary_defaultdict2txtfig(default_dict=loss_dict, prefix='train', step=idx, textlogger=global_textlogger)

    state_dict['iter'] = idx
    # save ckpt and images
    if rank == 0 and ((idx + 1) % global_cfg.log_ckpt_every == 0 or global_cfg.tl_debug):
      saved_dirs = []
      saved_dir = save_models(g_module=g_module,
                  d_module=d_module,
                  g_ema=g_ema,
                  state_dict=state_dict,
                  info_msg=f"iter: {idx}",
                  saved_dir=None)
      saved_dirs.append(saved_dir)
      saved_dir = save_models(g_module=g_module,
                              d_module=d_module,
                              g_ema=g_ema,
                              state_dict=state_dict,
                              info_msg=f"iter: {idx}",
                              saved_dir=f"{global_cfg.tl_ckptdir}/resume")
      saved_dirs.append(saved_dir)

      save_images(renderer_output_size=global_cfg.renderer_output_size,
                  gen_output_size=global_cfg.gen_output_size,
                  log_N_row=global_cfg.log_N_row,
                  log_N_col=global_cfg.log_N_col,
                  g_module=g_module,
                  g_ema=g_ema,
                  saved_dirs=saved_dirs,
                  device=device,
                  sample_kwargs=sample_kwargs)
      moxing_utils.modelarts_sync_results_dir(cfg=global_cfg, )

    synchronize()

  pass


def create_models(opt,
                  device):

  discriminator = Discriminator(opt.model).to(device)

  if 'G_cfg' in global_cfg:
    generator = build_model(cfg=global_cfg.G_cfg, kwargs_priority=True).to(device)

    g_ema = build_model(cfg=global_cfg.G_cfg, kwargs_priority=True, ema=True).to(device)
    g_ema.eval()
  else:
    generator = Generator(opt.model, opt.rendering).to(device)

    g_ema = Generator(opt.model, opt.rendering, ema=True).to(device)
    g_ema.eval()

  return discriminator, generator, g_ema

def create_optims(opt,
                  generator,
                  discriminator):

  g_reg_ratio = global_cfg.g_reg_every / (global_cfg.g_reg_every + 1) if global_cfg.g_reg_every > 0 else 1
  d_reg_ratio = global_cfg.d_reg_every / (global_cfg.d_reg_every + 1)

  params_g = []
  params_dict_g = dict(generator.named_parameters())
  for key, value in params_dict_g.items():
    decoder_cond = ('decoder' in key)
    if decoder_cond:
      params_g += [{'params': [value], 'lr': opt.training.lr * g_reg_ratio}]

  g_optim = optim.Adam(params_g,  # generator.parameters(),
                       lr=opt.training.lr * g_reg_ratio,
                       betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
  d_optim = optim.Adam(discriminator.parameters(),
                       lr=opt.training.lr * d_reg_ratio,  # * g_d_ratio,
                       betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

  return g_optim, d_optim

def create_dataset(opt,
                   rank):
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])

  moxing_utils.copy_data(rank=rank, global_cfg=global_cfg, **global_cfg.obs_training_dataset)

  dataset = MultiResolutionDataset(global_cfg.obs_training_dataset.datapath,
                                   transform, # [-1, 1]
                                   resolution=global_cfg.resolution,
                                   nerf_resolution=global_cfg.nerf_resolution)

  sampler = data_sampler(dataset, shuffle=True, distributed=opt.training.distributed)

  loader = data.DataLoader(
    dataset,
    batch_size=opt.training.batch,
    sampler=sampler,
    drop_last=True,
  )
  return loader, sampler


def main():
  device = "cuda"
  opt = BaseOptions().parse()
  opt.training.camera = opt.camera
  opt.training.size = opt.model.size
  opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
  opt.training.style_dim = opt.model.style_dim
  opt.model.freeze_renderer = True

  n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
  opt.training.distributed = n_gpu > 1

  if opt.training.distributed:
    torch.cuda.set_device(opt.training.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
  rank = get_rank()

  update_parser_defaults_from_yaml(parser=None, is_main_process=(rank == 0))
  if rank == 0:
    moxing_utils.setup_tl_outdir_obs(global_cfg)
    moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  synchronize()

  discriminator, generator, g_ema = create_models(opt=opt, device=device)

  g_optim, d_optim = create_optims(opt=opt, generator=generator, discriminator=discriminator)

  opt.training.start_iter = 0

  state_dict = {
    'iter': 0,
    'fid': float('inf'),
    'best_fid': float('inf')
  }

  if global_cfg.tl_resume:
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

  elif global_cfg.tl_finetune:
    if rank == 0:
      print("finetuning pretrained renderer weights...")

    moxing_utils.copy_data(rank=0, global_cfg=global_cfg,
                           datapath_obs=global_cfg.finetune_dir, datapath=global_cfg.finetune_dir)
    model_dict = {
      "G": generator,
      "G_ema": g_ema,
    }
    torch_utils.load_models(global_cfg.finetune_dir, model_dict=model_dict, rank=rank)

    generator.load_state_dict(g_ema.state_dict())

  else:

    pass

  # save configuration
  opt_path = os.path.join(global_cfg.tl_outdir, 'exp', f"opt.yaml")
  if rank == 0:
    os.makedirs(os.path.dirname(opt_path), exist_ok=True)
    with open(opt_path, 'w') as f:
      yaml.safe_dump(opt, f)

  # initialize g_ema weights to generator weights
  accumulate(g_ema, generator, 0)

  # set distributed models
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

  loader, sampler = create_dataset(opt=opt, rank=rank)

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