import copy
import logging
import pprint
from itertools import chain
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
from tl2.proj.logger.textlogger import summary_defaultdict2txtfig, global_textlogger, summary_dict2txtfig
from tl2.proj.fvcore import build_model
from tl2.proj.pytorch.ddp import ddp_utils

from stylesdf.models.distributed import get_rank, synchronize, reduce_loss_dict, reduce_sum, get_world_size
from exp.stylesdf.losses import *
from exp.stylesdf.dataset import MultiResolutionDataset
from exp.stylesdf.options import BaseOptions
from exp.cips3d.utils import (data_sampler,
                              requires_grad,
                              accumulate,
                              sample_data,
                              make_noise,
                              mixing_noise,
  # generate_camera_params,
                              )
# from exp.cips3d.models.model import Generator, Discriminator
from exp.cips3d import nerf_utils
from exp.cips3d.scripts.setup_evaluation import setup_evaluation
from exp.cips3d.scripts.gen_images import gen_images
from exp.cips3d.scripts.eval_fid import eval_fid

try:
  import wandb
except ImportError:
  wandb = None


def update_D_render(discriminator,
                    gen_imgs,
                    gt_viewpoints,
                    real_imgs,
                    loss_dict,
                    alpha):
  if real_imgs.shape[-1] > gen_imgs.shape[-1]:
    real_imgs = F.interpolate(real_imgs, scale_factor=(gen_imgs.shape[-1] / real_imgs.shape[-1]),
                              recompute_scale_factor=False,
                              mode='bicubic', align_corners=False)

  fake_pred, fake_viewpoint_pred = discriminator(gen_imgs.detach(), alpha=alpha)
  if global_cfg.lambda_pose > 0:
    d_view_loss = global_cfg.lambda_pose * viewpoints_loss(fake_viewpoint_pred, gt_viewpoints)
  else:
    d_view_loss = 0.

  real_imgs.requires_grad = True
  real_pred, _ = discriminator(real_imgs, alpha=alpha)
  # GAN loss
  d_gan_loss = d_logistic_loss(real_pred, fake_pred)
  # gp loss
  grad_penalty = d_r1_loss(real_pred, real_imgs)
  r1_loss = global_cfg.lambda_gp * 0.5 * grad_penalty

  # total loss
  d_loss = d_gan_loss + r1_loss + d_view_loss
  d_loss.backward()

  loss_dict["d_loss_gan_render"]['d_loss_gan_render'] = d_gan_loss.item()
  loss_dict["d_loss_r1_render"]['d_loss_r1_render'] = r1_loss.item()
  loss_dict["d_loss_pose_render"]['d_loss_pose_render'] = d_view_loss.item() if torch.is_tensor(d_view_loss) else d_view_loss
  loss_dict["d_loss_total_render"]['d_loss_total_render'] = d_loss.item()
  loss_dict['d_logits_render']["d_logits_real_render"] = real_pred.detach().mean().item()
  loss_dict['d_logits_render']["d_logits_fake_render"] = fake_pred.detach().mean().item()

  pass

def update_D_decoder(discriminator,
                     gen_imgs,
                     d_regularize,
                     curr_real_imgs,
                     loss_dict,
                     alpha):

  fake_pred = discriminator(gen_imgs.detach(), alpha=alpha)

  if d_regularize:
    curr_real_imgs.requires_grad = True

  real_pred = discriminator(curr_real_imgs, alpha=alpha)
  d_gan_loss = d_logistic_loss(real_pred, fake_pred)

  if d_regularize:
    grad_penalty = d_r1_loss(real_pred, curr_real_imgs)
    r1_loss = global_cfg.lambda_gp * 0.5 * grad_penalty * global_cfg.d_reg_every
  else:
    r1_loss = 0.

  d_loss = d_gan_loss + r1_loss
  d_loss.backward()

  loss_dict["d_loss_gan_decoder"]['d_loss_gan_decoder'] = d_gan_loss.item()
  loss_dict["d_loss_gp_decoder"]['d_loss_gp_decoder'] = r1_loss.item() if torch.is_tensor(r1_loss) else r1_loss
  loss_dict["d_loss_total_decoder"]['d_loss_total_decoder'] = d_loss.item()
  loss_dict["d_logits_decoder"]['d_logits_real_decoder'] = real_pred.detach().mean().item()
  loss_dict["d_logits_decoder"]['d_logits_fake_decoder'] = fake_pred.detach().mean().item()

  pass

def update_D(opt,
             generator,
             discriminator,
             discriminator_render,
             g_module,
             idx,
             loader,
             d_optim,
             loss_dict,
             device,
             alpha):

  requires_grad(generator, False)
  requires_grad(discriminator, True)
  requires_grad(discriminator_render, True)

  enable_decoder = global_cfg.G_cfg.enable_decoder
  freeze_renderer = global_cfg.G_cfg.freeze_renderer
  d_regularize = (idx + 1) % global_cfg.d_reg_every == 0

  cam_img_size = global_cfg.cam_img_size
  gen_img_size = global_cfg.gen_img_size

  real_imgs, _ = next(loader)
  real_imgs = real_imgs.to(device)

  noise = mixing_noise(opt.batch, g_module.z_dim, opt.mixing, device)

  cam_extrinsics, focal, near, far, gt_viewpoints = nerf_utils.Camera.generate_camera_params(
    device=device,
    batch=opt.batch,
    **{**global_cfg.G_kwargs.cam_cfg,
       'img_size': cam_img_size})

  discriminator_render.zero_grad(set_to_none=True)
  discriminator.zero_grad(set_to_none=True)

  for j in range(0, opt.batch, opt.chunk):
    curr_real_imgs = real_imgs[j:j + opt.chunk]
    curr_noise = [n[j:j + opt.chunk] for n in noise]

    if gen_img_size < cam_img_size:
      if global_cfg.get('sample_mode', 'default') == 'patch':
        sample_idx_h = torch_utils.get_gather_sample_idx_patch(batch=opt.chunk, all_size=cam_img_size,
                                                               patch_size=gen_img_size, device=device)
        sample_idx_w = torch_utils.get_gather_sample_idx_patch(batch=opt.chunk, all_size=cam_img_size,
                                                               patch_size=gen_img_size, device=device)
        curr_real_imgs = torch_utils.sample_image_patch(images=curr_real_imgs,
                                                        patch_size_h=gen_img_size, patch_size_w=gen_img_size,
                                                        device=device,
                                                        sample_idx_h=sample_idx_h, sample_idx_w=sample_idx_w)

      else:
        sample_idx_h = torch_utils.get_gather_sample_idx(batch=opt.chunk, N_size=cam_img_size,
                                                         N_samples=gen_img_size, device=device)
        sample_idx_w = torch_utils.get_gather_sample_idx(batch=opt.chunk, N_size=cam_img_size,
                                                         N_samples=gen_img_size, device=device)
        curr_real_imgs = torch_utils.sample_image_sub_pixels(images=curr_real_imgs,
                                                             N_h_pixels=gen_img_size, N_w_pixels=gen_img_size,
                                                             device=device,
                                                             sample_idx_h=sample_idx_h, sample_idx_w=sample_idx_w)
    else:
      sample_idx_h = None
      sample_idx_w = None

    ret_maps = generator(zs=curr_noise,
                         cam_poses=cam_extrinsics[j:j + opt.chunk],
                         focals=focal[j:j + opt.chunk],
                         img_size=cam_img_size,
                         near=near[j:j + opt.chunk],
                         far=far[j:j + opt.chunk],
                         N_rays_forward=global_cfg.N_rays_forward ** 2 if global_cfg.N_rays_forward else None,
                         N_rays_grad=None,
                         N_samples_forward=global_cfg.N_samples_forward ** 2 if global_cfg.N_samples_forward else None,
                         eikonal_reg=False,
                         nerf_cfg=global_cfg.G_kwargs.nerf_cfg.to_dict(),
                         sample_idx_h=sample_idx_h,
                         sample_idx_w=sample_idx_w)

    if not freeze_renderer:
      update_D_render(discriminator=discriminator_render,
                      gen_imgs=ret_maps['thumb_rgb'],
                      gt_viewpoints=gt_viewpoints[j : j + opt.chunk],
                      real_imgs=curr_real_imgs.detach().clone(),
                      loss_dict=loss_dict,
                      alpha=alpha)
      D_grad_norm_renderer = torch.nn.utils.clip_grad_norm_(
        discriminator_render.parameters(), global_cfg.grad_clip)
      loss_dict['grad_norm']['D_grad_norm_renderer'] = D_grad_norm_renderer.item()

    if enable_decoder:
      update_D_decoder(discriminator=discriminator,
                       gen_imgs=ret_maps['rgb'],
                       d_regularize=d_regularize,
                       curr_real_imgs=curr_real_imgs.detach().clone(),
                       loss_dict=loss_dict,
                       alpha=alpha)
      D_grad_norm_decoder = torch.nn.utils.clip_grad_norm_(
        discriminator.parameters(), global_cfg.grad_clip)
      loss_dict['grad_norm']['D_grad_norm_decoder'] = D_grad_norm_decoder.item()

  d_optim.step()
  discriminator_render.zero_grad(set_to_none=True)
  discriminator.zero_grad(set_to_none=True)

  pass


def update_G_render(fake_img,
                    sdf,
                    eikonal_term,
                    discriminator,
                    curr_gt_viewpoints,
                    retain_graph,
                    loss_dict,
                    alpha):

  fake_pred, fake_viewpoint_pred = discriminator(fake_img, alpha=alpha)

  if global_cfg.lambda_pose > 0:
    g_view_loss = global_cfg.lambda_pose * viewpoints_loss(fake_viewpoint_pred, curr_gt_viewpoints)
  else:
    g_view_loss = 0.

  if global_cfg.lambda_eikonal > 0:
    g_eikonal, g_minimal_surface = eikonal_loss(eikonal_term,
                                                sdf=sdf,
                                                beta=global_cfg.min_surf_beta)
    g_eikonal = global_cfg.lambda_eikonal * g_eikonal
    g_minimal_surface = global_cfg.lambda_min_surf * g_minimal_surface

  else:
    g_eikonal = 0.
    g_minimal_surface = 0.

  g_gan_loss = g_nonsaturating_loss(fake_pred)
  g_loss = g_gan_loss + g_view_loss + g_eikonal + g_minimal_surface

  g_loss.backward(retain_graph=retain_graph)

  loss_dict['g_loss_gan_render']["g_loss_gan_render"] = g_gan_loss.item()
  loss_dict["g_loss_pose_render"]['g_loss_pose_render'] = g_view_loss.item() \
    if torch.is_tensor(g_view_loss) else g_view_loss
  loss_dict["g_loss_eikonal_render"]['g_loss_eikonal_render'] = g_eikonal.item() \
    if torch.is_tensor(g_eikonal) else g_eikonal
  loss_dict["g_loss_minimal_surface_render"]['g_loss_minimal_surface_render'] = \
    g_minimal_surface.item() if torch.is_tensor(g_minimal_surface) else g_minimal_surface
  loss_dict["g_loss_total_render"]['g_loss_total_render'] = g_loss.item()

  return g_loss

def update_G_decoder(discriminator,
                     fake_img,
                     loss_dict,
                     alpha):

  fake_pred = discriminator(fake_img, alpha=alpha)
  g_gan_loss = g_nonsaturating_loss(fake_pred)

  g_loss = g_gan_loss

  g_loss.backward()

  loss_dict["g_loss_gan_decoder"]['g_loss_gan_decoder'] = g_gan_loss.item()

  return g_loss

def update_G(opt,
             idx,
             generator,
             discriminator,
             discriminator_render,
             g_module,
             g_optim,
             loss_dict,
             device,
             alpha,
             world_size,
             renderer_detach):

  requires_grad(generator, True)
  requires_grad(discriminator, False)
  requires_grad(discriminator_render, False)

  enable_decoder = global_cfg.G_cfg.enable_decoder
  freeze_renderer = global_cfg.G_cfg.freeze_renderer
  g_regularize = (global_cfg.g_reg_every > 0) and ((idx + 1) % global_cfg.g_reg_every == 0)

  cam_img_size = global_cfg.cam_img_size
  gen_img_size = global_cfg.gen_img_size

  noise = mixing_noise(opt.batch, g_module.z_dim, opt.mixing, device)
  cam_extrinsics, focal, near, far, gt_viewpoints = nerf_utils.Camera.generate_camera_params(
    device=device,
    batch=opt.batch,
    **{**global_cfg.G_kwargs.cam_cfg,
       'img_size': cam_img_size})

  generator.zero_grad(set_to_none=True)

  for j in range(0, opt.batch, opt.chunk):
    curr_noise = [n[j:j + opt.chunk] for n in noise]

    if gen_img_size < cam_img_size:
      if global_cfg.get('sample_mode', 'default') == 'patch':
        sample_idx_h = torch_utils.get_gather_sample_idx_patch(batch=opt.chunk, all_size=cam_img_size,
                                                               patch_size=gen_img_size, device=device)
        sample_idx_w = torch_utils.get_gather_sample_idx_patch(batch=opt.chunk, all_size=cam_img_size,
                                                               patch_size=gen_img_size, device=device)

      else:
        sample_idx_h = torch_utils.get_gather_sample_idx(batch=opt.chunk, N_size=cam_img_size,
                                                         N_samples=gen_img_size, device=device)
        sample_idx_w = torch_utils.get_gather_sample_idx(batch=opt.chunk, N_size=cam_img_size,
                                                         N_samples=gen_img_size, device=device)
    else:
      sample_idx_h = None
      sample_idx_w = None

    ret_maps = generator(zs=curr_noise,
                         cam_poses=cam_extrinsics[j:j + opt.chunk],
                         focals=focal[j:j + opt.chunk],
                         img_size=cam_img_size,
                         near=near[j:j + opt.chunk],
                         far=far[j:j + opt.chunk],
                         N_rays_forward=global_cfg.N_rays_forward ** 2 if global_cfg.N_rays_forward else None,
                         N_rays_grad=global_cfg.N_rays_grad ** 2 if global_cfg.N_rays_grad else None,
                         N_samples_forward=global_cfg.N_samples_forward ** 2 if global_cfg.N_samples_forward else None,
                         path_reg=False,
                         eikonal_reg=not freeze_renderer and global_cfg.get('eikonal_reg', True),
                         return_sdf=not freeze_renderer and global_cfg.get('sdf_reg', True),
                         nerf_cfg=global_cfg.G_kwargs.nerf_cfg.to_dict(),
                         renderer_detach=renderer_detach,
                         sample_idx_h=sample_idx_h,
                         sample_idx_w=sample_idx_w)

    if not freeze_renderer:
      update_G_render(fake_img=ret_maps['thumb_rgb'],
                      sdf=ret_maps['sdf'],
                      eikonal_term=ret_maps['eikonal_term'],
                      discriminator=discriminator_render,
                      curr_gt_viewpoints=gt_viewpoints[j: j + opt.chunk],
                      retain_graph=True if enable_decoder else False,
                      loss_dict=loss_dict,
                      alpha=alpha)
      ddp_utils.sync_gradients(model=generator, world_size=world_size)

      # G_grad_norm_renderer = torch.nn.utils.clip_grad_norm_(
      #   chain(g_module.renderer.parameters(), g_module.style.parameters()),
      #   global_cfg.grad_clip)
      # loss_dict['grad_norm']['G_grad_norm_renderer'] = G_grad_norm_renderer.item()

    if enable_decoder:
      update_G_decoder(discriminator=discriminator,
                       fake_img=ret_maps['rgb'],
                       loss_dict=loss_dict,
                       alpha=alpha)
      ddp_utils.sync_gradients(model=generator, world_size=world_size)

  G_grad_norm_renderer = torch.nn.utils.clip_grad_norm_(
    chain(g_module.renderer.parameters(), g_module.style.parameters()),
    global_cfg.grad_clip)
  loss_dict['grad_norm']['G_grad_norm_renderer'] = G_grad_norm_renderer.item()
  G_grad_norm_decoder = torch.nn.utils.clip_grad_norm_(
    chain(g_module.decoder.parameters(), g_module.style_decoder.parameters()),
    global_cfg.grad_clip)
  loss_dict['grad_norm']['G_grad_norm_decoder'] = G_grad_norm_decoder.item()

  g_optim.step()
  generator.zero_grad(set_to_none=True)

  # generator path regularization
  if (enable_decoder and g_regularize) or global_cfg.tl_debug:
    # generator.zero_grad(set_to_none=True)
    g_module.zero_grad(set_to_none=True)

    path_batch_size = max(1, opt.batch // opt.path_batch_shrink)
    path_noise = mixing_noise(path_batch_size, g_module.z_dim, opt.mixing, device)
    path_cam_extrinsics, path_focal, path_near, path_far, _ = nerf_utils.Camera.generate_camera_params(
      device=device,
      batch=path_batch_size,
      **{**global_cfg.G_kwargs.cam_cfg,
         'img_size': cam_img_size})

    for j in range(0, path_batch_size, opt.chunk):
      curr_path_noise = [n[j:j + opt.chunk] for n in path_noise]

      if gen_img_size < cam_img_size:
        if global_cfg.get('sample_mode', 'default') == 'patch':
          sample_idx_h = torch_utils.get_gather_sample_idx_patch(batch=min(path_batch_size, opt.chunk),
                                                                 all_size=cam_img_size,
                                                                 patch_size=gen_img_size, device=device)
          sample_idx_w = torch_utils.get_gather_sample_idx_patch(batch=min(path_batch_size, opt.chunk),
                                                                 all_size=cam_img_size,
                                                                 patch_size=gen_img_size, device=device)
        else:
          sample_idx_h = torch_utils.get_gather_sample_idx(batch=min(path_batch_size, opt.chunk), N_size=cam_img_size,
                                                           N_samples=gen_img_size, device=device)
          sample_idx_w = torch_utils.get_gather_sample_idx(batch=min(path_batch_size, opt.chunk), N_size=cam_img_size,
                                                           N_samples=gen_img_size, device=device)
      else:
        sample_idx_h = None
        sample_idx_w = None

      ret_maps = g_module(zs=curr_path_noise,
                          cam_poses=path_cam_extrinsics[j:j + opt.chunk],
                          focals=path_focal[j:j + opt.chunk],
                          img_size=cam_img_size,
                          near=path_near[j:j + opt.chunk],
                          far=path_far[j:j + opt.chunk],
                          N_rays_forward=global_cfg.N_rays_forward ** 2 if global_cfg.N_rays_forward else None,
                          N_rays_grad=global_cfg.N_rays_grad ** 2 if global_cfg.N_rays_grad else None,
                          N_samples_forward=global_cfg.N_samples_forward ** 2 if global_cfg.N_samples_forward else None,
                          path_reg=True,
                          eikonal_reg=False,
                          nerf_cfg=global_cfg.G_kwargs.nerf_cfg.to_dict(),
                          renderer_detach=renderer_detach,
                          sample_idx_h=sample_idx_h,
                          sample_idx_w=sample_idx_w)

      path_loss, mean_path_length, path_lengths = g_path_regularize(
        fake_img=ret_maps['rgb'],
        latents=ret_maps['style_decoder'],
        mean_path_length=0.
      )

      weighted_path_loss = global_cfg.path_regularize * global_cfg.g_reg_every * path_loss  # * opt.chunk / path_batch_size
      # if opt.path_batch_shrink:
      #   weighted_path_loss += (0. * ret_maps['rgb'][0, 0, 0, 0])
      # mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())

      weighted_path_loss.backward()
      ddp_utils.sync_gradients(model=g_module, world_size=world_size)

    G_grad_norm_renderer_path_reg = torch.nn.utils.clip_grad_norm_(
      chain(g_module.renderer.parameters(), g_module.style.parameters()), 0)
    G_grad_norm_decoder_path_reg = torch.nn.utils.clip_grad_norm_(
      chain(g_module.decoder.parameters(), g_module.style_decoder.parameters()),
      global_cfg.grad_clip)
    loss_dict['grad_norm']['G_grad_norm_renderer_path_reg'] = G_grad_norm_renderer_path_reg.item()
    loss_dict['grad_norm']['G_grad_norm_decoder_path_reg'] = G_grad_norm_decoder_path_reg.item()

    g_optim.step()

    g_module.zero_grad(set_to_none=True)

  else: # g_regularize = False
    weighted_path_loss = 0.
    path_lengths = 0.
    pass

  # generator.zero_grad(set_to_none=True)

  loss_dict["g_loss_weighted_path"]['g_loss_weighted_path'] = weighted_path_loss.item() \
    if torch.is_tensor(weighted_path_loss) else weighted_path_loss
  loss_dict["path_length_mean"]['path_length_mean'] = path_lengths.detach().mean().item() \
    if torch.is_tensor(path_lengths) else path_lengths

  pass

def save_models(g_module,
                d_module,
                d_render_module,
                g_ema,
                state_dict,
                info_msg,
                saved_dir=None):

  model_dict = {
    "G": g_module,
    "D": d_module,
    "D_render": d_render_module,
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
def save_images(log_N_row,
                log_N_col,
                g_module,
                g_ema,
                saved_dirs,
                device,
                sample_kwargs):
  sample_z = sample_kwargs['zs']
  sample_cam_extrinsics = sample_kwargs['cam_poses']
  sample_focals = sample_kwargs['focals']
  sample_near = sample_kwargs['near']
  sample_far = sample_kwargs['far']
  img_size = sample_kwargs['img_size']

  nerf_cfg = global_cfg.G_kwargs.nerf_cfg.clone()
  nerf_cfg.perturb = True
  nerf_cfg.static_viewdirs = False

  # samples = torch.Tensor(0, 3, img_size, img_size)
  # thumbs_samples = torch.Tensor(0, 3, img_size, img_size)
  samples_list = []
  thumbs_samples_list = []
  step_size = 4
  # mean_latent = g_module.mean_latent(10000, device)
  # mean_latent = g_ema.mean_latent(10000, device)
  for k in range(0, log_N_row * log_N_col, step_size):
    curr_noise = [n[k:k + step_size] for n in sample_z]
    ret_maps = g_ema(zs=curr_noise,
                     cam_poses=sample_cam_extrinsics[k:k + step_size],
                     focals=sample_focals[k:k + step_size],
                     img_size=img_size,
                     near=sample_near[k:k + step_size],
                     far=sample_far[k:k + step_size],
                     N_rays_forward=global_cfg.N_rays_forward ** 2 if global_cfg.N_rays_forward else None,
                     N_rays_grad=None,
                     N_samples_forward=global_cfg.N_samples_forward ** 2 if global_cfg.N_samples_forward else None,
                     truncation=0.7,
                     nerf_cfg=nerf_cfg,
                     recompute_mean=True)
    curr_samples = ret_maps['rgb']
    curr_thumbs = ret_maps['thumb_rgb']
    samples_list.append(curr_samples.cpu())
    thumbs_samples_list.append(curr_thumbs.cpu())

    # samples = torch.cat([samples, curr_samples.cpu()], 0)
    # thumbs_samples = torch.cat([thumbs_samples, curr_thumbs.cpu()], 0)

  samples = torch.cat(samples_list, dim=0)
  thumbs_samples = torch.cat(thumbs_samples_list, dim=0)

  if not isinstance(saved_dirs, (list, tuple)):
    saved_dirs = [saved_dirs]

  for saved_dir in saved_dirs:
    os.makedirs(saved_dir, exist_ok=True)
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


def sphere_init_func(opt,
                     generator,
                     g_module,
                     g_optim,
                     g_ema,
                     device,
                     saved_path,
                     sample_kwargs,
                     rank,
                     world_size):

  N_total = 10000

  if rank == 0:
    pbar = tl2_utils.TL_tqdm(total=N_total, start=0, desc='sphere_init_func')
  else:
    pbar = range(0, N_total)

  bs = 4
  img_size = global_cfg.G_kwargs.cam_cfg.img_size
  nerf_cfg = global_cfg.G_kwargs.nerf_cfg.clone().to_dict()

  loss_dict = {}

  generator.zero_grad()
  ddp_utils.sync_models(rank=rank, world_size=world_size, sync_models=[generator])

  for idx in pbar:
    noise = mixing_noise(bs, opt.style_dim, opt.mixing, device)
    cam_extrinsics, focal, near, far, gt_viewpoints = nerf_utils.Camera.generate_camera_params(
      device=device,
      batch=bs,
      **global_cfg.G_kwargs.cam_cfg)

    sdf, target_values = generator.init_forward(zs=noise,
                                                cam_poses=cam_extrinsics,
                                                focals=focal,
                                                img_size=img_size,
                                                near=near,
                                                far=far,
                                                nerf_cfg=nerf_cfg)

    loss = F.l1_loss(sdf, target_values)
    loss_dict['l1_loss'] = loss.item()

    generator.zero_grad()

    loss.backward()
    ddp_utils.sync_gradients(model=generator, world_size=world_size)

    g_optim.step()

    if rank == 0 and (idx + 1) % 10 == 0:
      loss_str = tl2_utils.get_print_dict_str(loss_dict, outdir=global_cfg.tl_outdir, suffix_str=pbar.get_string())
      print(loss_str)
      summary_dict2txtfig(loss_dict, prefix='init', step=idx, textlogger=global_textlogger)

    if rank == 0 and (idx + 1) % 100 == 0 or global_cfg.tl_debug:
      save_images(log_N_row=global_cfg.log_N_row,
                  log_N_col=global_cfg.log_N_col,
                  g_module=g_module,
                  g_ema=g_module,
                  saved_dirs=f"{global_cfg.tl_ckptdir}/G_sdf_init",
                  device=device,
                  sample_kwargs=sample_kwargs)

    synchronize()
    if global_cfg.tl_debug:
      break

  # saved_path = f"{global_cfg.tl_ckptdir}/G_sdf_init.pth"
  if rank == 0:
    torch.save(g_module.state_dict(), saved_path)
  pass


def train(opt,
          experiment_opt,
          loader,
          generator,
          discriminator,
          discriminator_render,
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
  logger = logging.getLogger('tl')

  loader = sample_data(loader, distributed=distributed, sampler=sampler)

  g_module = generator
  if distributed:
    # g_module = generator.module
    d_module = discriminator.module
    d_render_module = discriminator_render.module
  else:
    d_module = discriminator
    d_render_module = discriminator_render

  accum = 0.5 ** (32 / (10 * 1000))

  # fixed_z: N_row x N_col
  sample_z = [
    torch.randn(global_cfg.log_N_row, g_module.z_dim, device=device)
      .repeat_interleave(global_cfg.log_N_col, dim=0),
    torch.randn(global_cfg.log_N_row, g_module.z_dim, device=device)
      .repeat_interleave(global_cfg.log_N_col, dim=0),
  ]
  sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
    device=device,
    batch=global_cfg.log_N_row, # batch x 8
    sweep=True,
    **{**global_cfg.G_kwargs.cam_cfg})
  sample_kwargs = {
    'zs': sample_z,
    'cam_poses': sample_cam_extrinsics,
    'focals': sample_focals,
    'img_size': global_cfg.G_kwargs.cam_cfg.img_size,
    'near': sample_near,
    'far': sample_far,
  }

  sphere_init_flag = global_cfg.init_renderer and not global_cfg.tl_finetune
  if sphere_init_flag:
    if global_cfg.init_renderer_ckpt is None:
      saved_path = f"{global_cfg.tl_ckptdir}/G_sdf_init.pth"
      # if rank == 0:
        # sphere_init_path = "cache_pretrained/pretrained/pretrained_renderer/sphere_init.pt"
        # loaded_g = torch.load(sphere_init_path)['g']
        # Checkpointer(g_module).load_state_dict(loaded_g)
      sphere_init_func(opt=opt,
                       generator=generator,
                       g_module=g_module,
                       g_optim=g_optim,
                       g_ema=g_ema,
                       device=device,
                       saved_path=saved_path,
                       sample_kwargs=sample_kwargs,
                       rank=rank,
                       world_size=world_size)
      synchronize()

    else:
      moxing_utils.copy_data(rank=rank, global_cfg=global_cfg,
                             datapath_obs=global_cfg.init_renderer_ckpt, datapath=global_cfg.init_renderer_ckpt)
      saved_path = global_cfg.init_renderer_ckpt

    Checkpointer(g_module).load_state_dict_from_file(saved_path, rank=rank)
    accumulate(g_ema, g_module, 0)

  # sync params
  sync_models = [g_module, g_ema, d_module, d_render_module]
  ddp_utils.sync_models(rank=rank, world_size=world_size, sync_models=sync_models)

  # pbar
  start_iter = state_dict['iter']
  total_iters = global_cfg.total_iters
  if get_rank() == 0:
    pbar = tl2_utils.TL_tqdm(total=total_iters, start=start_iter)
  else:
    pbar = range(start_iter, total_iters)

  loss_dict = collections.defaultdict(dict)

  for idx in pbar:
    loss_dict.clear()

    if global_cfg.fade_D:
      alpha = min(1, idx / global_cfg.fade_steps)
    else:
      alpha = 1.

    if idx < global_cfg.warmup_iters and sphere_init_flag:
      renderer_detach = True
    else:
      renderer_detach = None

    # update discriminator
    update_D(opt=opt,
             generator=generator,
             discriminator=discriminator,
             discriminator_render=discriminator_render,
             g_module=g_module,
             idx=idx,
             loader=loader,
             d_optim=d_optim,
             loss_dict=loss_dict,
             device=device,
             alpha=alpha)

    # update genearator
    update_G(opt=opt,
             idx=idx,
             generator=generator,
             discriminator=discriminator,
             discriminator_render=discriminator_render,
             g_module=g_module,
             g_optim=g_optim,
             loss_dict=loss_dict,
             device=device,
             alpha=alpha,
             world_size=world_size,
             renderer_detach=renderer_detach)

    # update ema
    if idx >= global_cfg.ema_start:
      accumulate(g_ema, g_module, accum)
    else:
      accumulate(g_ema, g_module, 0.)

    # log txt
    if rank == 0 and ((idx + 1) % global_cfg.log_txt_every == 0 or global_cfg.tl_debug):
      loss_dict['lr']['G_lr_render'] = global_cfg.G_lr_render
      loss_dict['lr']['G_lr_decoder'] = global_cfg.G_lr_decoder
      loss_dict['lr']['D_lr_render'] = global_cfg.D_lr_render
      loss_dict['lr']['D_lr_decoder'] = global_cfg.D_lr_decoder
      loss_dict['lambda_gp']['lambda_gp'] = global_cfg.lambda_gp
      loss_dict['lambda_pose']['lambda_pose'] = global_cfg.lambda_pose
      loss_dict['lambda_eikonal']["lambda_eikonal"] = global_cfg.lambda_eikonal
      loss_dict['lambda_min_surf']["lambda_min_surf"] = global_cfg.lambda_min_surf
      loss_dict['min_surf_beta']["min_surf_beta"] = global_cfg.min_surf_beta
      loss_dict['d_reg_every']["d_reg_every"] = global_cfg.d_reg_every
      loss_dict['g_reg_every']["g_reg_every"] = global_cfg.g_reg_every
      loss_dict['alpha']['alpha'] = alpha
      loss_dict['warmup_iters']['warmup_iters'] = global_cfg.warmup_iters
      g_beta_val = g_module.renderer.sigmoid_beta.item()
      loss_dict['g_beta_val']['g_beta_val'] = g_beta_val
      loss_dict['batch']["batch"] = global_cfg.batch

      log_str = tl2_utils.get_print_dict_str(loss_dict, outdir=global_cfg.tl_outdir, suffix_str=pbar.get_string())
      print(log_str)
      if idx > 1000:
        summary_defaultdict2txtfig(default_dict=loss_dict, prefix='train', step=idx, textlogger=global_textlogger)

    state_dict['iter'] = idx

    if ((idx + 1) % global_cfg.log_ckpt_every == 0 or idx == 0 or global_cfg.tl_debug):
      # sync weights
      ddp_utils.sync_models(rank=rank, world_size=world_size, sync_models=sync_models)

      # output real images
      real_dir = f"{global_cfg.tl_outdir}/exp/fid/real"
      setup_evaluation(rank=rank,
                       world_size=world_size,
                       batch_gpu=64,
                       distributed=world_size > 1,
                       real_dir=real_dir,
                       N_real_images_eval=global_cfg.N_real_images_eval,
                       del_fid_real_images=global_cfg.del_fid_real_images,
                       data_img_size=global_cfg.data_img_size,
                       img_size=global_cfg.G_kwargs.cam_cfg.img_size * global_cfg.G_cfg.get('scale_factor', 1))
      global_cfg.del_fid_real_images = False
      ddp_utils.d2_synchronize()
      # output fake images
      fake_dir = f"{global_cfg.tl_outdir}/exp/fid/fake"
      gen_images(rank=rank,
                 world_size=world_size,
                 generator=g_ema,
                 G_kwargs=global_cfg.G_kwargs.clone(),
                 fake_dir=fake_dir,
                 num_imgs=global_cfg.N_gen_images_eval,
                 batch_gpu=global_cfg.batch_gpu)
      ddp_utils.d2_synchronize()

      # debug ddp models
      save_models(g_module=g_module,
                  d_module=d_module,
                  d_render_module=d_render_module,
                  g_ema=g_ema,
                  state_dict=state_dict,
                  info_msg=f"iter: {idx}\n"
                           f"cur_fid: {state_dict.get('cur_fid', 0)}\n"
                           f"best_fid: {state_dict['best_fid']}",
                  saved_dir=f"{global_cfg.tl_ckptdir}/resume_{rank}")

      moxing_utils.copy_data(rank=rank, global_cfg=global_cfg, **global_cfg.obs_inception_v3)
      global_cfg.obs_inception_v3.disable = True
      if rank == 0:
        # eval
        metric_dict = eval_fid(real_dir=real_dir, fake_dir=fake_dir)
        logger.info(f"\nstep: {state_dict['iter']}, {pprint.pformat(metric_dict)}\n")
        summary_dict2txtfig(metric_dict, prefix='eval', step=state_dict['iter'], textlogger=global_textlogger)
        state_dict['cur_fid'] = metric_dict['FID']

        saved_dirs = []
        saved_dir = save_models(g_module=g_module,
                                d_module=d_module,
                                d_render_module=d_render_module,
                                g_ema=g_ema,
                                state_dict=state_dict,
                                info_msg=f"iter: {idx}"
                                         f"cur_fid: {state_dict['cur_fid']}\n"
                                         f"best_fid: {state_dict['best_fid']}",
                                saved_dir=None)
        saved_dirs.append(saved_dir)
        saved_dir = save_models(g_module=g_module,
                                d_module=d_module,
                                d_render_module=d_render_module,
                                g_ema=g_ema,
                                state_dict=state_dict,
                                info_msg=f"iter: {idx}"
                                         f"cur_fid: {state_dict['cur_fid']}\n"
                                         f"best_fid: {state_dict['best_fid']}",
                                saved_dir=f"{global_cfg.tl_ckptdir}/resume")
        saved_dirs.append(saved_dir)

        if state_dict['best_fid'] > metric_dict['FID']:
          state_dict['best_fid'] = metric_dict['FID']
          saved_dir = save_models(g_module=g_module,
                                  d_module=d_module,
                                  d_render_module=d_render_module,
                                  g_ema=g_ema,
                                  state_dict=state_dict,
                                  info_msg=f"iter: {idx}"
                                           f"cur_fid: {state_dict['cur_fid']}\n"
                                           f"best_fid: {state_dict['best_fid']}",
                                  saved_dir=f"{global_cfg.tl_ckptdir}/best_fid")
          saved_dirs.append(saved_dir)

        save_images(log_N_row=global_cfg.log_N_row,
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

  if 'D_cfg' in global_cfg:
    discriminator = build_model(cfg=global_cfg.D_cfg).to(device)
  else:
    # discriminator = Discriminator(opt.model).to(device)
    raise NotImplementedError

  discriminator_render = build_model(cfg=global_cfg.D_renderer_cfg).to(device)

  if 'G_cfg' in global_cfg:
    generator = build_model(cfg=global_cfg.G_cfg, kwargs_priority=True).to(device)

    g_ema = build_model(cfg=global_cfg.G_cfg, kwargs_priority=True, ema=True).to(device)
    g_ema.eval()
  else:
    # generator = Generator(opt.model, opt.rendering).to(device)
    # g_ema = Generator(opt.model, opt.rendering, ema=True).to(device)
    # g_ema.eval()
    raise NotImplementedError

  return discriminator, discriminator_render, generator, g_ema

def create_optims(opt,
                  generator,
                  discriminator,
                  discriminator_render):

  # g_reg_ratio = global_cfg.g_reg_every / (global_cfg.g_reg_every + 1) if global_cfg.g_reg_every > 0 else 1
  g_reg_ratio = 1.
  d_reg_ratio = global_cfg.d_reg_every / (global_cfg.d_reg_every + 1)

  params_dict_g = dict(generator.named_parameters())

  params_optim_g = []

  for name, value in params_dict_g.items():
    if name.startswith(('decoder', 'style_decoder.')):
      params_optim_g.append({'params': [value],
                             'lr': global_cfg.G_lr_decoder * g_reg_ratio,
                             'betas': (0 ** g_reg_ratio, 0.99 ** g_reg_ratio)})

    elif name.startswith(('renderer', 'style.')):
      params_optim_g.append({'params': [value],
                             'lr': global_cfg.G_lr_render,
                             'betas': (0, 0.9)})
    else:
      raise NotImplementedError

  assert len(params_optim_g) == len(params_dict_g)
  g_optim = optim.Adam(params=params_optim_g)

  params_optim_d = []
  for name, param in discriminator.named_parameters():
    params_optim_d.append({'params': [param],
                           'lr': global_cfg.D_lr_decoder * d_reg_ratio,
                           'betas': (0 ** d_reg_ratio, 0.99 ** d_reg_ratio)})
  for name, param in discriminator_render.named_parameters():
    params_optim_d.append({'params': [param],
                           'lr': global_cfg.D_lr_render,
                           'betas': (0, 0.9)})

  d_optim = optim.Adam(params=params_optim_d)

  return g_optim, d_optim

def create_dataset(opt,
                   rank,
                   img_size,
                   shuffle=True,
                   batch_gpu=None,
                   distributed=None,
                   hflip=True):
  if batch_gpu is None:
    batch_gpu = opt.training.batch
  if distributed is None:
    distributed = opt.training.distributed

  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])

  moxing_utils.copy_data(rank=rank, global_cfg=global_cfg, **global_cfg.obs_training_dataset)

  dataset = MultiResolutionDataset(global_cfg.obs_training_dataset.datapath,
                                   transform, # [-1, 1]
                                   resolution=img_size,
                                   nerf_resolution=img_size,
                                   hflip=hflip)

  sampler = data_sampler(dataset, shuffle=shuffle, distributed=distributed)

  loader = data.DataLoader(
    dataset,
    batch_size=batch_gpu,
    sampler=sampler,
    drop_last=True,
    num_workers=global_cfg.get('num_workers', 0)
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
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(opt.training.local_rank)
    synchronize()
  rank = get_rank()
  torch_utils.init_seeds(seed=0, rank=rank)

  update_parser_defaults_from_yaml(parser=None, is_main_process=(rank == 0))
  if rank == 0:
    moxing_utils.setup_tl_outdir_obs(global_cfg)
    moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  synchronize()

  discriminator, discriminator_render, generator, g_ema = create_models(opt=opt, device=device)

  g_optim, d_optim = create_optims(opt=opt,
                                   generator=generator,
                                   discriminator=discriminator,
                                   discriminator_render=discriminator_render)

  opt.training.start_iter = 0

  state_dict = {
    'iter': 0,
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
                           datapath_obs=global_cfg.finetune_dir, datapath=global_cfg.finetune_dir,
                           overwrite=(get_world_size() > 2))
    state_dict_loaded = copy.deepcopy(state_dict)
    model_dict = {
      "G": generator,
      "D": discriminator,
      "D_render": discriminator_render,
      "G_ema": g_ema,
      "state_dict": state_dict_loaded,
    }
    torch_utils.load_models(global_cfg.finetune_dir, model_dict=model_dict, rank=rank)
    logging.getLogger('tl').info(pprint.pformat(state_dict_loaded))

    generator.load_state_dict(g_ema.state_dict())

  else:

    pass

  # save configuration
  opt_path = os.path.join(global_cfg.tl_outdir, 'exp', f"opt.yaml")
  if rank == 0:
    os.makedirs(os.path.dirname(opt_path), exist_ok=True)
    with open(opt_path, 'w') as f:
      yaml.safe_dump(opt, f)

  # # initialize g_ema weights to generator weights
  # accumulate(g_ema, generator, 0)

  # set distributed models
  if opt.training.distributed:
    # generator = nn.parallel.DistributedDataParallel(
    #   generator,
    #   device_ids=[opt.training.local_rank],
    #   output_device=opt.training.local_rank,
    #   broadcast_buffers=True,
    #   find_unused_parameters=True,
    # )

    discriminator = nn.parallel.DistributedDataParallel(
      discriminator,
      device_ids=[opt.training.local_rank],
      output_device=opt.training.local_rank,
      broadcast_buffers=False,
      find_unused_parameters=True
    )

    discriminator_render = nn.parallel.DistributedDataParallel(
      discriminator_render,
      device_ids=[opt.training.local_rank],
      output_device=opt.training.local_rank,
      broadcast_buffers=False,
      find_unused_parameters=True
    )

  loader, sampler = create_dataset(opt=opt,
                                   rank=rank,
                                   img_size=global_cfg.data_img_size)

  if get_rank() == 0 and wandb is not None and opt.training.wandb:
    wandb.init(project="StyleSDF")
    wandb.run.name = opt.experiment.expname
    wandb.config.dataset = os.path.basename(opt.dataset.dataset_path)
    wandb.config.update(opt.training)
    wandb.config.update(opt.model)
    wandb.config.update(opt.rendering)

  train(opt=opt.training,
        experiment_opt=opt.experiment,
        loader=loader,
        generator=generator,
        discriminator=discriminator,
        discriminator_render=discriminator_render,
        g_optim=g_optim,
        d_optim=d_optim,
        g_ema=g_ema,
        device=device,
        sampler=sampler,
        state_dict=state_dict)
  pass


if __name__ == '__main__':
  main()
  moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)