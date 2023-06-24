import json
import collections
import pathlib
from PIL import Image
import tqdm
import itertools
import PIL.Image
import pickle
import copy
from time import perf_counter
import os
import streamlit as st
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as trans_f

from tl2.proj.streamlit import st_utils
from tl2.proj.fvcore import build_model, MODEL_REGISTRY
from tl2.launch.launch_utils import global_cfg
from tl2.proj.pil import pil_utils
from tl2.proj.cv2 import cv2_utils
from tl2.proj.skimage import skimage_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.logger import logger_utils
from tl2.proj.pytorch.ddp import ddp_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.modelarts import moxing_utils
from tl2.proj.tools3d.camera_pose_visualizer import CameraPoseVisualizer

from exp.cips3d import nerf_utils


def load_pil_crop_resize(image_path,
                         out_size=None):
  target_pil = pil_utils.pil_open_rgb(image_path)

  if target_pil.size == (out_size, out_size):
    return target_pil

  w, h = target_pil.size
  s = min(w, h)
  target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
  if out_size:
    target_pil = target_pil.resize((out_size, out_size), Image.LANCZOS)
  return target_pil


@MODEL_REGISTRY.register(name_prefix=__name__)
class StyleGAN2Projector(object):
  def __init__(self,
               G,
               device,
               perceptual_cfg={},
               rank=0,
               **kwargs
               ):
    self.perceptual_cfg = copy.deepcopy(perceptual_cfg)

    self.device = device

    # Load networks.
    self.G = G

    self.G_weight = copy.deepcopy(self.G.state_dict())

    self.preceptual_net = self.create_perceptual_net(perceptual_cfg=perceptual_cfg, rank=rank)

    pass

  def create_perceptual_net(self, perceptual_cfg, rank):
    preceptual_net = build_model(perceptual_cfg, rank=rank, cfg_to_kwargs=True).to(self.device)
    preceptual_net.requires_grad_(False)
    return preceptual_net

  def reset(self):
    self.G.load_state_dict(self.G_weight, strict=True)
    self.G = self.G.requires_grad_(False).to(self.device)
    pass

  def compute_w_stat(self,
                     G,
                     w_avg_samples,
                     device):
    # Compute w stats.
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')

    noises_renderer = torch.randn(w_avg_samples, G.z_dim, device=device)
    style_renders = G.style(noises_renderer)
    style_render_mean = style_renders.mean(0, keepdim=True)

    noises_decoder = torch.randn(w_avg_samples, G.z_dim, device=device)
    style_decoders = G.style_decoder(noises_decoder)
    style_decoder_mean = style_decoders.mean(0, keepdim=True)

    # w_samples = torch.cat([style_renders, style_decoders], dim=1)
    # w_avg = torch.cat([style_render_mean, style_decoder_mean], dim=1)
    # w_std = (((w_samples - w_avg) ** 2).sum() / w_avg_samples) ** 0.5

    return style_render_mean, style_decoder_mean

  def get_perceptual_fea(self,
                         image_tensor,
                         image_tensor_thumb=None,
                         img_size=1024,
                         **kwargs):
    """
    image_tensor: [-1, 1]
    """
    if image_tensor.dim() == 3:
      image_tensor = image_tensor.unsqueeze(0)

    features = self.preceptual_net(image_tensor, **kwargs)

    if image_tensor_thumb is None:
      image_tensor_thumb = F.interpolate(image_tensor, scale_factor=64/img_size,
                                         recompute_scale_factor=False,
                                         mode='bicubic', align_corners=False)

    features_thumb = self.preceptual_net(image_tensor_thumb, **kwargs)

    return features, features_thumb

  def _get_target_image(self,
                        image_path,
                        device):
    # Load target image.
    target_pil = load_pil_crop_resize(image_path, out_size=1024)
    # [0, 1]
    target_np = np.array(target_pil, dtype=np.float32).transpose([2, 0, 1]) / 255.
    # [-1, 1]
    target_images = (torch.from_numpy(target_np).to(device) - 0.5) * 2

    return target_pil, target_np, target_images

  def _get_noise_bufs(self,
                      G,
                      device,
                      start_size=64):

    noise_bufs = G.create_noise_bufs(start_size=start_size, device=device)

    return noise_bufs

  def _get_cur_lr(self,
                  step,
                  num_steps,
                  initial_learning_rate,
                  lr_rampdown_length=0.25,
                  lr_rampup_length=0.05,
                  ):
    t = step / num_steps
    lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
    lr = initial_learning_rate * lr_ramp
    return lr

  def _get_w_noise_scale(self,
                         w_std,
                         step,
                         num_steps,
                         initial_noise_factor = 0.05,
                         noise_ramp_length = 0.75,
                         ):
    w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - (step / num_steps) / noise_ramp_length) ** 2
    return w_noise_scale

  def _get_proj_w_name(self,
                       image_path,
                       optim_noise_bufs):
    image_path = pathlib.Path(image_path)

    if optim_noise_bufs:
      proj_w_name = f"{image_path.stem}_wn"
    else:
      proj_w_name = f"{image_path.stem}_w"
    return proj_w_name

  def _G_forward(self,
                 G,
                 style_render,
                 style_decoder,
                 noise_bufs,
                 cam_cfg,
                 nerf_cfg,
                 device,
                 style_render_noise=None,
                 style_decoder_noise=None,
                 rot=None,
                 trans=None,
                 return_mask=False,
                 flip_w_decoder=False):

    # trajectory = torch.zeros(1, 2, device=device)
    trajectory = torch.cat([rot, trans], dim=1)

    cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
      locations=trajectory,
      device=device,
      **{**cam_cfg})

    # if rot is not None and trans is not None:
    #   trans = F.normalize(trans, p=2, dim=1)
    #   cam_extrinsics = nerf_utils.Camera.get_camera2world(cam2world=rot, trans=trans)
    # else:
    #   cam_extrinsics = sample_cam_extrinsics

    if style_render_noise is not None:
      style_render = style_render + style_render_noise
    if style_decoder_noise is not None:
      style_decoder = style_decoder + style_decoder_noise

    if flip_w_decoder:
      style_decoder = style_decoder.detach() # only update decoder params
      style_decoder = style_decoder.flip(dims=(0, ))


    ret_maps = G(zs=[None, None],
                 style_render=style_render,
                 style_decoder=style_decoder,
                 cam_poses=cam_extrinsics,
                 focals=sample_focals,
                 img_size=cam_cfg.img_size,
                 near=sample_near,
                 far=sample_far,
                 noise_bufs=noise_bufs,
                 N_rays_forward=None,
                 N_rays_grad=None,
                 N_samples_forward=None,
                 eikonal_reg=False,
                 nerf_cfg=nerf_cfg,
                 recompute_mean=False,
                 return_xyz=False,
                 renderer_detach=False)

    if return_mask:
      mask_thumb = ret_maps['mask'].detach().expand_as(ret_maps['thumb_rgb'])
      mask_thumb = 1 - mask_thumb

      mask = F.interpolate(mask_thumb, scale_factor=(ret_maps['rgb'].shape[-1] / mask_thumb.shape[-1]),
                           recompute_scale_factor=False, mode='bicubic')

      return ret_maps['rgb'], ret_maps['thumb_rgb'], mask, mask_thumb, cam_extrinsics.detach().cpu().numpy()
    else:
      return ret_maps['rgb'], ret_maps['thumb_rgb']

  def _create_cam_optimizer(self,
                            optim_cam,
                            lr_cam,
                            bs=1):

    azim = torch.zeros(bs, 1).cuda()
    elev = torch.zeros(bs, 1).cuda()

    params_optim_cam = []
    if optim_cam:
      azim = nn.Parameter(azim, requires_grad=True)
      elev = nn.Parameter(elev, requires_grad=True)
      params_optim_cam.append(
        {
          'params': [azim, elev],
          'lr': lr_cam,
          'initial_lr': lr_cam,
          'betas': (0.9, 0.999),
        })

    optimizer_cam = torch.optim.Adam(params_optim_cam)

    return azim, elev, optimizer_cam

  def _create_render_optimizer(self,
                               G,
                               w_avg_samples,
                               optim_render_w,
                               lr_render_w,
                               device,
                               bs=1):

    # w_mean render
    noises_renderer = torch.randn(w_avg_samples, G.z_dim, device=device)
    style_renders = G.style(noises_renderer)
    style_render_mean = style_renders.mean(0, keepdim=True)

    # params
    w_render_opt = torch.zeros(bs, G.N_layers_renderer + 1, style_render_mean.shape[-1], device=device)
    w_render_opt.copy_(style_render_mean)

    params_optim_render = []
    if optim_render_w:
      w_render_opt = nn.Parameter(w_render_opt, requires_grad=True)
      params_optim_render.append(
        {
          'params': [w_render_opt], # + list(G.renderer.parameters()),
          'lr': lr_render_w,
          'initial_lr': lr_render_w,
          'betas': (0.9, 0.999),
        })

    optimizer_render = torch.optim.Adam(params_optim_render)

    return w_render_opt, optimizer_render, style_render_mean

  def _create_decoder_optimizer(self,
                                G,
                                w_avg_samples,
                                optim_decoder_w,
                                optim_decoder_params,
                                optim_noise_bufs,
                                zero_noise_bufs,
                                lr_decoder_w,
                                lr_decoder_params,
                                lr_noise,
                                device,
                                bs=1):

    # w_avg decoder
    noises_decoder = torch.randn(w_avg_samples, G.z_dim, device=device)
    style_decoders = G.style_decoder(noises_decoder)
    style_decoder_mean = style_decoders.mean(0, keepdim=True)

    w_decoder_opt = torch.zeros(bs, G.decoder.n_latent, style_decoder_mean.shape[-1], device=device)
    w_decoder_opt.copy_(style_decoder_mean)

    params_optim_decoder = []

    if optim_decoder_w:
      w_decoder_opt = nn.Parameter(w_decoder_opt, requires_grad=True)
      params_optim_decoder.append(
        {
          'params': [w_decoder_opt],
          'lr': lr_decoder_w,
          'initial_lr': lr_decoder_w,
          'betas': (0.9, 0.999),
        })

    if optim_decoder_params:
      params_optim_decoder.append(
        {
          'params': G.decoder.parameters(),
          'lr': lr_decoder_params,
          'initial_lr': lr_decoder_params,
          'betas': (0.9, 0.999),
        })

    # Setup noise inputs.
    noise_bufs = self._get_noise_bufs(G=G, device=device, start_size=64)
    if zero_noise_bufs:
      [buf.zero_() for buf in noise_bufs]

    if optim_noise_bufs:
      noise_bufs = [nn.Parameter(buf, requires_grad=True) for buf in noise_bufs]
      params_optim_decoder.append(
        {
          'params': noise_bufs,
          'lr': lr_noise,
          'initial_lr': lr_noise,
          'betas': (0.9, 0.999),
        })

    optimizer_decoder = torch.optim.Adam(params_optim_decoder)

    return w_decoder_opt, noise_bufs, optimizer_decoder, style_decoder_mean

  def project_wplus(
          self,
          outdir,
          image_path,
          cam_cfg,
          nerf_cfg,
          optim_cam,
          optim_render_w,
          optim_decoder_w,
          optim_decoder_params,
          zero_noise_bufs,
          N_steps_pose,
          N_steps_app,
          N_steps_multiview,
          lr_cam,
          lr_render_w,
          lr_decoder_w,
          lr_decoder_params,
          lr_noise,
          rgb_weight,
          thumb_weight,
          truncation_psi,
          perceptual_layers_multiview,
          w_avg_samples=10000,
          optim_noise_bufs=True,
          normalize_noise=True,
          regularize_noise_weight=1e5,
          mse_weight=0.,
          seed=123,
          lpips_metric=None,
          fps=10,
          hd_video=True,
          save_noise_bufs=False,
          st_log_every=100,
          st_web=False,
          **kwargs
  ):
    rank = ddp_utils.d2_get_rank()

    if optim_noise_bufs:
      save_noise_bufs = True

    device = self.device
    G = copy.deepcopy(self.G).eval().requires_grad_(False).to(self.device)  # type: ignore
    G.decoder.requires_grad_(True)

    image_path = pathlib.Path(image_path)
    image_name = image_path.stem
    proj_w_name = self._get_proj_w_name(image_path=image_path, optim_noise_bufs=optim_noise_bufs)

    target_pil, target_np, target_images = self._get_target_image(image_path=image_path, device=device)

    with torch.no_grad():
      target_features, target_features_thumb = self.get_perceptual_fea(image_tensor=target_images,
                                                                       image_tensor_thumb=None)

    params_optim_render = []
    params_optim_decoder = []

    # camera pose axis-angle
    # azim = torch.zeros(1, 3).cuda()
    # elev = torch.zeros(1, 3).cuda()
    # elev[:, 2] = 1
    # style_render_mean, style_decoder_mean = self.compute_w_stat(G=G, w_avg_samples=w_avg_samples, device=device)

    azim, elev, optimizer_cam = self._create_cam_optimizer(optim_cam=optim_cam, lr_cam=lr_cam)

    w_render_opt, optimizer_render, style_render_mean = self._create_render_optimizer(
      G=G, w_avg_samples=w_avg_samples,
      optim_render_w=optim_render_w, lr_render_w=lr_render_w, device=device)

    w_decoder_opt, noise_bufs, optimizer_decoder, style_decoder_mean = self._create_decoder_optimizer(
      G=G, w_avg_samples=w_avg_samples,
      optim_decoder_w=optim_decoder_w, optim_decoder_params=optim_decoder_params, optim_noise_bufs=optim_noise_bufs,
      zero_noise_bufs=zero_noise_bufs,
      lr_decoder_w=lr_decoder_w, lr_decoder_params=lr_decoder_params, lr_noise=lr_noise, device=device)


    if st_web:
      st_chart_lr = st_utils.LineChart(x_label='step', y_label='lr_mul')
      st_chart_noise_scale = st_utils.LineChart(x_label='step', y_label='st_chart_noise_scale')
      st_chart_percep_loss = st_utils.LineChart(x_label='step', y_label='percep_loss')
      st_chart_mse_loss = st_utils.LineChart(x_label='step', y_label='mse_loss')
      st_chart_reg_loss = st_utils.LineChart(x_label='step', y_label='reg_loss')
      st_chart_loss = st_utils.LineChart(x_label='step', y_label='loss')
      st_chart_psnr = st_utils.LineChart(x_label='step', y_label='psnr')
      st_chart_ssim = st_utils.LineChart(x_label='step', y_label='ssim')
      st_image = st.empty()
      st_text_rot = st.empty()
      st_text_trans = st.empty()

      video_f_target_proj = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_target_proj.mp4", fps=fps, hd_video=hd_video)
      video_f_inversion = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_inversion.mp4", fps=fps, hd_video=hd_video)

    N_steps = N_steps_pose + N_steps_app + N_steps_multiview
    if rank == 0:
      pbar = tqdm.tqdm(range(N_steps), desc=f"{image_path.stem}")
    else:
      pbar = range(N_steps)

    dummy_zero = torch.tensor(0., device=device)

    # save kwargs
    saved_state_dict = {
      'azim': azim,
      'elev': elev,
      'w_render_opt': w_render_opt,
      'w_decoder_opt': w_decoder_opt,
      'decoder_state_dict': G.decoder.state_dict(),
    }
    if save_noise_bufs:
      saved_state_dict['noise_bufs'] = noise_bufs

    for step in pbar:
      # Learning rate mul
      if step < N_steps_pose:
        lr_mul = self._get_cur_lr(step=step, num_steps=N_steps_pose, initial_learning_rate=1)
      elif step < N_steps_pose + N_steps_app:
        lr_mul = self._get_cur_lr(step=step - N_steps_pose, num_steps=N_steps_app,
                                  initial_learning_rate=1, lr_rampup_length=0.25)
      else:
        lr_mul = self._get_cur_lr(step=step - N_steps_pose - N_steps_app, num_steps=N_steps_multiview,
                                  initial_learning_rate=1, lr_rampup_length=0.25)

      torch_utils.mul_optimizer_lr(optimizer=optimizer_cam, lr_mul=lr_mul)
      torch_utils.mul_optimizer_lr(optimizer=optimizer_render, lr_mul=lr_mul)
      torch_utils.mul_optimizer_lr(optimizer=optimizer_decoder, lr_mul=lr_mul)

      if step < N_steps_pose: # optimizing cam, render
        torch_utils.set_optimizer_lr(optimizer=optimizer_decoder, lr=0)
        w_noise_scale = (1 - min(step / (N_steps_pose + 1e-5), 1)) * 0.5

      elif step < N_steps_pose + N_steps_app: # optimizing cam, decoder
        if step == N_steps_pose:  # select w_render
          with torch.no_grad():
            w_render_opt.copy_(torch.lerp(style_render_mean, w_render_opt, truncation_psi))

        # torch_utils.set_optimizer_lr(optimizer=optimizer_render, lr=0)
        w_noise_scale = 0

      else: # optimizing decoder (multi_view)
        if step == N_steps_pose + N_steps_app: # update perceptual net
          saved_state_dict.pop('azim')
          saved_state_dict.pop('elev')
          with torch.no_grad():
            saved_state_dict['azim'] = azim.detach().clone()
            saved_state_dict['elev'] = elev.detach().clone()
            azim.zero_()
            elev.zero_()
          # setup perceptual net
          perceptual_cfg = copy.deepcopy(self.perceptual_cfg)
          perceptual_cfg.layers = perceptual_layers_multiview
          self.preceptual_net = self.create_perceptual_net(perceptual_cfg=perceptual_cfg, rank=rank)
          with torch.no_grad():
            target_features, target_features_thumb = self.get_perceptual_fea(image_tensor=target_images,
                                                                             image_tensor_thumb=None)
          pass

        torch_utils.set_optimizer_lr(optimizer=optimizer_cam, lr=0)
        torch_utils.set_optimizer_lr(optimizer=optimizer_render, lr=0)
        w_noise_scale = 1

      # w_noise_render = torch.randn_like(w_render_opt) * w_noise_scale
      # w_noise_decoder = torch.randn_like(w_decoder_opt) * w_noise_scale
      w_noise_render = None
      w_noise_decoder = None

      if st_web and (step % st_log_every == 0 or step == N_steps - 1):
        st_chart_lr.write(step, lr_mul)
        st_chart_noise_scale.write(step, w_noise_scale)

      synth_images, synth_images_thumb = self._G_forward(
        G=G,
        style_render=w_render_opt,
        style_decoder=w_decoder_opt,
        noise_bufs=noise_bufs,
        cam_cfg=cam_cfg,
        nerf_cfg=nerf_cfg,
        device=device,
        style_render_noise=w_noise_render,
        style_decoder_noise=w_noise_decoder,
        rot=azim + w_noise_scale * 0.3 * torch.randn(1, device=device),
        trans=elev + w_noise_scale * 0.15 * torch.randn(1, device=device))


      if st_web and (step % st_log_every == 0 or step == N_steps - 1):
        with torch.no_grad():
          # img_pil = G.synthesis(w_opt.detach(), **self.synthesis_kwargs)
          # img_pil = stylegan_utils.to_pil(img_pil.detach())
          img_pil = torch_utils.img_tensor_to_pil(synth_images)

          psnr = skimage_utils.sk_psnr(image_true_pil=target_pil, image_test_pil=img_pil)
          ssim = skimage_utils.sk_ssim(image_true_pil=target_pil, image_test_pil=img_pil)

          # psnr line
          st_chart_psnr.write(step, psnr)
          st_chart_ssim.write(step, ssim)

        merged_pil = pil_utils.merge_image_pil([target_pil, img_pil], nrow=2)
        pil_utils.add_text(
          merged_pil, text=f"step: {step}, psnr: {psnr:.2f}dB, ssim: {ssim:.2f}", size=merged_pil.size[0] // 20)

        st_utils.st_image(merged_pil, caption=f"target {target_pil.size}, img_pil {img_pil.size}",
                          debug=global_cfg.tl_debug, st_empty=st_image)
        st_text_rot.write(f"{azim}")
        st_text_trans.write(f"{elev}")
        video_f_target_proj.write(merged_pil)
        video_f_inversion.write(img_pil)

      synth_features, synth_features_thumb = self.get_perceptual_fea(image_tensor=synth_images,
                                                                     image_tensor_thumb=synth_images_thumb)

      percep_loss = (target_features - synth_features).square().sum() * rgb_weight + \
                    (target_features_thumb - synth_features_thumb).square().sum() * thumb_weight

      if mse_weight > 0:
        mse_loss = F.mse_loss(synth_images, target=target_images.detach())
        mse_loss = mse_loss * mse_weight
      else:
        mse_loss = dummy_zero

      # Noise regularization.
      if optim_noise_bufs and regularize_noise_weight > 0:
        reg_loss = 0
        for v in noise_bufs:
          # noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
          noise = v
          while True:
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
            if noise.shape[2] <= 8:
              break
            noise = F.avg_pool2d(noise, kernel_size=2)
        reg_loss = reg_loss * regularize_noise_weight

        # Normalize noise.
        # if normalize_noise:
        #   with torch.no_grad():
        #     for buf in noise_bufs:
        #       buf -= buf.mean()
        #       buf *= buf.square().mean().rsqrt()

      else:
        reg_loss = dummy_zero

      loss = percep_loss + mse_loss + reg_loss

      # Step
      optimizer_cam.zero_grad(set_to_none=True)
      optimizer_render.zero_grad(set_to_none=True)
      optimizer_decoder.zero_grad(set_to_none=True)
      loss.backward()
      optimizer_cam.step()
      optimizer_render.step()
      optimizer_decoder.step()

      if st_web and step >= 50 and (step % st_log_every == 0 or step == N_steps - 1):
        st_chart_percep_loss.write(x=step, y=percep_loss.item())
        st_chart_mse_loss.write(x=step, y=mse_loss.item())
        st_chart_reg_loss.write(x=step, y=reg_loss.item())
        st_chart_loss.write(x=step, y=loss.item())

      if global_cfg.tl_debug:
        break

    if st_web:
      video_f_target_proj.release(st_video=True)
      video_f_inversion.release(st_video=True)

    # save ori image
    pil_utils.pil_save(target_pil, image_path=f'{outdir}/{image_name}.png', save_png=False)
    # save proj image
    with torch.no_grad():
      proj_images, _ = self._G_forward(G=G,
                                       style_render=w_render_opt,
                                       style_decoder=w_decoder_opt,
                                       noise_bufs=noise_bufs,
                                       cam_cfg=cam_cfg,
                                       nerf_cfg=nerf_cfg,
                                       device=device,
                                       style_render_noise=None,
                                       style_decoder_noise=None,
                                       rot=azim,
                                       trans=elev)
      proj_img_pil = torch_utils.img_tensor_to_pil(proj_images)
      pil_utils.pil_save(proj_img_pil, f"{outdir}/{proj_w_name}_proj.png", save_png=False)


    saved_path = f"{outdir}/{proj_w_name}.pth"
    torch.save(saved_state_dict, saved_path)
    st.write(saved_path)

    # proj_psnr image
    file_logger = logger_utils.get_file_logger(filename=f"{outdir}/{proj_w_name}.txt")
    psnr = skimage_utils.sk_psnr(image_true_pil=target_pil, image_test_pil=proj_img_pil)
    ssim = skimage_utils.sk_ssim(image_true_pil=target_pil, image_test_pil=proj_img_pil)
    if lpips_metric is None:
      lpips_metric = skimage_utils.LPIPS(device=device)
    lpips = lpips_metric.calc_lpips(target_pil, proj_img_pil)
    file_logger.info_msg(f"psnr: {psnr}\n"
                         f"ssim: {ssim}\n"
                         f"lpips: {lpips}")

    ret_dict = {'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips}
    return ret_dict

  def _get_w_n_decoded_image(self,
                             G,
                             synthesis_kwargs,
                             w_file_path,
                             ):

    w_tensor, ns_tensor = stylegan_utils.load_w_and_n_tensor(w_file=w_file_path, device=self.device)
    if len(ns_tensor):
      img_decoded = stylegan_utils_v1.G_w_ns(G=G, w=w_tensor, ns=ns_tensor, synthesis_kwargs=synthesis_kwargs)
    else:
      img_decoded = stylegan_utils_v1.G_w(G=G, w=w_tensor, synthesis_kwargs=synthesis_kwargs)
    img_decoded_pil = stylegan_utils.to_pil(img_decoded)

    return img_decoded_pil, w_tensor, ns_tensor

  def _lerp_ns(self, ns0, ns1, gamma):
    ret_ns = {}
    for name, value0 in ns0.items():
      value1 = ns1[name]
      ret_ns[name] = value0 + gamma * (value1 - value0)

    return ret_ns

  def lerp_image_list(
        self,
        outdir,
        w_file_list,
        num_interp,
        num_pause,
        resolution,
        author_name_list=None,
        fps=10,
        hd_video=False,
        st_web=False,
        **kwargs
  ):
    self.reset()
    device = self.device
    # G_c = copy.deepcopy(self.G_c).eval().requires_grad_(False).to(self.device)
    # G_s = copy.deepcopy(self.G_s).eval().requires_grad_(False).to(self.device)
    G = self.G

    decoded_pil_list = []
    w_list = []
    ns_list = []

    # load proj w and ns from npz file
    for w_file_path in w_file_list:
      img_decoded_pil, w_tensor, ns_tensor = self._get_w_n_decoded_image(
        G=G, synthesis_kwargs=self.synthesis_kwargs, w_file_path=w_file_path)
      decoded_pil_list.append(img_decoded_pil)
      w_list.append(w_tensor)
      ns_list.append(ns_tensor)

    if st_web:
      merged_pil = pil_utils.merge_image_pil(decoded_pil_list, nrow=4, dst_size=2048)
      st_utils.st_image(merged_pil, caption=img_decoded_pil.size, debug=global_cfg.tl_debug, )
      st_image_interp_model = st.empty()

    video_f_interp_model = cv2_utils.ImageioVideoWriter(
      outfile=f"{outdir}/author_list.mp4", fps=fps, hd_video=hd_video, save_gif=True)

    num_authors = len(w_list)
    if author_name_list is not None:
      assert len(author_name_list) == num_authors

    pbar = tqdm.tqdm(range(num_authors))
    for idx in pbar:
      pbar.update()
      pbar.set_description_str(w_file_list[idx])

      cur_w = w_list[idx]
      next_w = w_list[(idx + 1) % num_authors]
      cur_ns = ns_list[idx]
      next_ns = ns_list[(idx + 1) % num_authors]

      pbar_gama = tqdm.tqdm(np.linspace(0, 1, num_interp))
      for gama in pbar_gama:
        pbar_gama.update()

        # interp w
        w_ = cur_w + gama * (next_w - cur_w)
        ns_ = self._lerp_ns(cur_ns, next_ns, gamma=gama)

        with torch.no_grad():
          img_decoded = stylegan_utils_v1.G_w_ns(G=G, w=w_, ns=ns_,
                                                 synthesis_kwargs=self.synthesis_kwargs)
        img_decoded_pil = stylegan_utils.to_pil(img_decoded)

        if author_name_list is not None:
          if gama == 0:
            img_decoded_pil = img_decoded_pil.resize((resolution, resolution), Image.LANCZOS)
            author_name = f"{author_name_list[idx]}".replace("_", " ")
            pil_utils.add_text(img_decoded_pil, text=author_name,
                               size=img_decoded_pil.size[0] // 15, color=(0, 255, 0), xy=(4, 0))
            for _ in range(num_pause):
              if st_web:
                st_utils.st_image(img_decoded_pil, caption=f"{img_decoded_pil.size}", debug=global_cfg.tl_debug,
                                  st_empty=st_image_interp_model)
              video_f_interp_model.write(img_decoded_pil)
              if global_cfg.tl_debug: break

        if st_web:
          st_utils.st_image(img_decoded_pil, caption=f"{img_decoded_pil.size}", debug=global_cfg.tl_debug,
                            st_empty=st_image_interp_model)
        video_f_interp_model.write(img_decoded_pil, dst_size=resolution)
        if global_cfg.tl_debug: break
    video_f_interp_model.release(st_video=st_web)

    return



@MODEL_REGISTRY.register(name_prefix=__name__)
class StyleGAN2Projector_Flip(StyleGAN2Projector):

  def _get_target_image(self,
                        image_path,
                        device,
                        out_size=1024):
    # Load target image.
    target_pil = load_pil_crop_resize(image_path, out_size=out_size)
    # [0, 1]
    target_np = np.array(target_pil, dtype=np.float32).transpose([2, 0, 1]) / 255.
    # [-1, 1]
    target_images = (torch.from_numpy(target_np).to(device) - 0.5) * 2

    target_pil_flip = trans_f.hflip(target_pil)
    target_np_flip = np.array(target_pil_flip, dtype=np.float32).transpose([2, 0, 1]) / 255.
    target_images_flip = (torch.from_numpy(target_np_flip).to(device) - 0.5) * 2

    target_images = torch.stack([target_images, target_images_flip], dim=0)

    return target_pil, target_np, target_images, target_pil_flip

  def _create_render_optimizer(self,
                               G,
                               w_avg_samples,
                               optim_render_w,
                               optim_render_params,
                               lr_render_w,
                               device,
                               bs=1):

    # w_mean render
    noises_renderer = torch.randn(w_avg_samples, G.z_dim, device=device)
    style_renders = G.style(noises_renderer)
    style_render_mean = style_renders.mean(0, keepdim=True)

    # params
    w_render_opt = torch.zeros(bs, G.N_layers_renderer + 1, style_render_mean.shape[-1], device=device)
    w_render_opt.copy_(style_render_mean)

    params_optim_render = []
    if optim_render_w:
      w_render_opt = nn.Parameter(w_render_opt, requires_grad=True)
      params_optim_render.append(
        {
          'params': [w_render_opt],
          'lr': lr_render_w,
          'initial_lr': lr_render_w,
          'betas': (0.9, 0.999),
        })
    if optim_render_params:
      params_optim_render.append(
        {
          'params': list(G.renderer.parameters()),
          'lr': 0.0001,
          'initial_lr': 0.0001,
          'betas': (0.9, 0.999),
        })

    optimizer_render = torch.optim.Adam(params_optim_render)

    return w_render_opt, optimizer_render, style_render_mean

  def project_wplus(
          self,
          outdir,
          image_path,
          cam_cfg,
          nerf_cfg,
          optim_cam,
          optim_render_w,
          optim_render_params,
          optim_decoder_w,
          optim_decoder_params,
          zero_noise_bufs,
          bs_cam,
          bs_render,
          bs_decoder,
          N_steps_pose,
          N_steps_app,
          N_steps_multiview,
          lr_cam,
          lr_render_w,
          lr_decoder_w,
          lr_decoder_params,
          lr_noise,
          rgb_weight,
          thumb_weight,
          truncation_psi,
          mask_background,
          flip_w_decoder_every,
          w_avg_samples=10000,
          optim_noise_bufs=True,
          normalize_noise=True,
          regularize_noise_weight=1e5,
          mse_weight=0.,
          seed=123,
          lpips_metric=None,
          fps=10,
          hd_video=True,
          save_noise_bufs=False,
          st_log_every=100,
          st_web=False,
          **kwargs
  ):
    rank = ddp_utils.d2_get_rank()

    if optim_noise_bufs:
      save_noise_bufs = True

    device = self.device
    G = copy.deepcopy(self.G).eval().requires_grad_(False).to(self.device)  # type: ignore
    if optim_render_params:
      G.renderer.requires_grad_(True)
    G.decoder.requires_grad_(True)

    image_path = pathlib.Path(image_path)
    image_name = image_path.stem
    proj_w_name = self._get_proj_w_name(image_path=image_path, optim_noise_bufs=optim_noise_bufs)

    target_pil, target_np, target_images, target_pil_flip = self._get_target_image(
      image_path=image_path, device=device, out_size=global_cfg.get('img_size', 1024))

    with torch.no_grad():
      target_features, target_features_thumb = self.get_perceptual_fea(image_tensor=target_images,
                                                                       image_tensor_thumb=None,
                                                                       img_size=global_cfg.get('img_size', 1024))

    # camera pose axis-angle
    # azim = torch.zeros(1, 3).cuda()
    # elev = torch.zeros(1, 3).cuda()
    # elev[:, 2] = 1
    # style_render_mean, style_decoder_mean = self.compute_w_stat(G=G, w_avg_samples=w_avg_samples, device=device)

    azim, elev, optimizer_cam = self._create_cam_optimizer(optim_cam=optim_cam, lr_cam=lr_cam, bs=bs_cam)

    w_render_opt, optimizer_render, style_render_mean = self._create_render_optimizer(
      G=G, w_avg_samples=w_avg_samples,
      optim_render_w=optim_render_w, optim_render_params=optim_render_params,
      lr_render_w=lr_render_w,
      device=device, bs=bs_render)

    w_decoder_opt, noise_bufs, optimizer_decoder, style_decoder_mean = self._create_decoder_optimizer(
      G=G, w_avg_samples=w_avg_samples,
      optim_decoder_w=optim_decoder_w, optim_decoder_params=optim_decoder_params, optim_noise_bufs=optim_noise_bufs,
      zero_noise_bufs=zero_noise_bufs,
      lr_decoder_w=lr_decoder_w, lr_decoder_params=lr_decoder_params, lr_noise=lr_noise,
      device=device, bs=bs_decoder)

    if st_web:
      st_chart_lr = st_utils.LineChart(x_label='step', y_label='lr_mul')
      st_chart_noise_scale = st_utils.LineChart(x_label='step', y_label='st_chart_noise_scale')
      st_chart_percep_loss = st_utils.LineChart(x_label='step', y_label='percep_loss')
      st_chart_mse_loss = st_utils.LineChart(x_label='step', y_label='mse_loss')
      st_chart_reg_loss = st_utils.LineChart(x_label='step', y_label='reg_loss')
      st_chart_loss = st_utils.LineChart(x_label='step', y_label='loss')
      st_chart_psnr = st_utils.LineChart(x_label='step', y_label='psnr')
      st_chart_ssim = st_utils.LineChart(x_label='step', y_label='ssim')
      st_image_proj = st.empty()
      st_image = st.empty()
      st_text_rot = st.empty()
      st_text_trans = st.empty()

      video_f_target_proj = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_target_proj.mp4", fps=fps, hd_video=hd_video)
      video_f_inversion = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_inversion.mp4", fps=fps, hd_video=hd_video)
      video_f_cam_pose = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_cam_pose.mp4", fps=fps, hd_video=hd_video)

    N_steps = N_steps_pose + N_steps_app + N_steps_multiview

    cam_pose_vis = CameraPoseVisualizer(
      N_frames=N_steps,
      figsize=(global_cfg.get('img_size', 1024)/100, global_cfg.get('img_size', 1024)/100))

    if rank == 0:
      pbar = tqdm.tqdm(range(N_steps), desc=f"{image_path.stem}")
    else:
      pbar = range(N_steps)

    dummy_zero = torch.tensor(0., device=device)
    flip_w_decoder = False

    # save kwargs
    saved_state_dict = {
      'azim': azim,
      'elev': elev,
      'w_render_opt': w_render_opt,
      'w_decoder_opt': w_decoder_opt,
      'render_state_dict': G.renderer.state_dict(),
      'decoder_state_dict': G.decoder.state_dict(),
    }
    # if save_noise_bufs:
    saved_state_dict['noise_bufs'] = noise_bufs

    for step in pbar:
      # Learning rate mul
      if step < N_steps_pose:
        lr_mul = self._get_cur_lr(step=step, num_steps=N_steps_pose, initial_learning_rate=1)
      elif step < N_steps_pose + N_steps_app:
        lr_mul = self._get_cur_lr(step=step - N_steps_pose, num_steps=N_steps_app,
                                  initial_learning_rate=1, lr_rampup_length=0.25)
      else:
        lr_mul = self._get_cur_lr(step=step - N_steps_pose - N_steps_app, num_steps=N_steps_multiview,
                                  initial_learning_rate=1, lr_rampup_length=0.25)

      torch_utils.mul_optimizer_lr(optimizer=optimizer_cam, lr_mul=lr_mul)
      torch_utils.mul_optimizer_lr(optimizer=optimizer_render, lr_mul=lr_mul)
      torch_utils.mul_optimizer_lr(optimizer=optimizer_decoder, lr_mul=lr_mul)

      if step < N_steps_pose:  # optimizing cam, render
        torch_utils.set_optimizer_lr(optimizer=optimizer_decoder, lr=0)
        w_noise_scale = (1 - min(step / (N_steps_pose + 1e-5), 1)) * 0.5

      elif step < N_steps_pose + N_steps_app:  # optimizing cam, decoder
        if step == N_steps_pose:  # select w_render
          with torch.no_grad():
            w_render_opt.copy_(torch.lerp(style_render_mean, w_render_opt, truncation_psi))

        # torch_utils.set_optimizer_lr(optimizer=optimizer_render, lr=0)
        w_noise_scale = 0
        if (step + flip_w_decoder_every - 1) % flip_w_decoder_every == 0 and step != N_steps - 1:
          flip_w_decoder = True
          # optimizer_decoder.param_groups[0]['lr'] = 0.

        else:
          flip_w_decoder = False

      else:  # optimizing decoder (multi_view)
        if step == N_steps_pose + N_steps_app:  # update perceptual net
          raise NotImplementedError

        torch_utils.set_optimizer_lr(optimizer=optimizer_cam, lr=0)
        torch_utils.set_optimizer_lr(optimizer=optimizer_render, lr=0)
        w_noise_scale = 1

      # w_noise_render = torch.randn_like(w_render_opt) * w_noise_scale
      # w_noise_decoder = torch.randn_like(w_decoder_opt) * w_noise_scale
      w_noise_render = None
      w_noise_decoder = None

      if st_web and (step % st_log_every == 0 or step == N_steps - 1):
        st_chart_lr.write(step, lr_mul)
        st_chart_noise_scale.write(step, w_noise_scale)

      synth_images, synth_images_thumb, masks, masks_thumb, cam_extrinsics = self._G_forward(
        G=G,
        style_render=w_render_opt if w_render_opt.shape[0] == 2 else w_render_opt.repeat(2, 1, 1),
        style_decoder=w_decoder_opt if w_decoder_opt.shape[0] == 2 else w_decoder_opt.repeat(2, 1, 1),
        noise_bufs=noise_bufs,
        cam_cfg=cam_cfg,
        nerf_cfg=nerf_cfg,
        device=device,
        style_render_noise=w_noise_render,
        style_decoder_noise=w_noise_decoder,
        rot=azim + w_noise_scale * 0.3 * torch.randn(1, device=device),
        trans=elev + w_noise_scale * 0.15 * torch.randn(1, device=device),
        return_mask=True,
        flip_w_decoder=flip_w_decoder)

      if st_web and (step % st_log_every == 0 or step == N_steps - 1):
        with torch.no_grad():
          # img_pil = G.synthesis(w_opt.detach(), **self.synthesis_kwargs)
          # img_pil = stylegan_utils.to_pil(img_pil.detach())
          img_pil = torch_utils.img_tensor_to_pil(synth_images[0])
          img_pil_flip = torch_utils.img_tensor_to_pil(synth_images[1])
          mask_pil = torch_utils.img_tensor_to_pil(masks[0])
          mask_pil_flip = torch_utils.img_tensor_to_pil(masks[1])

          psnr = skimage_utils.sk_psnr(image_true_pil=target_pil, image_test_pil=img_pil)
          ssim = skimage_utils.sk_ssim(image_true_pil=target_pil, image_test_pil=img_pil)

          # psnr line
          st_chart_psnr.write(step, psnr)
          st_chart_ssim.write(step, ssim)

        cam_pose_vis.extrinsic2pyramid(extrinsic=cam_extrinsics[0], cur_frame=step)
        cam_pose_pil = cam_pose_vis.to_pil()

        merged_pil = pil_utils.merge_image_pil([target_pil, img_pil, mask_pil,
                                                target_pil_flip, img_pil_flip, mask_pil_flip,
                                                cam_pose_pil], nrow=3)
        pil_utils.add_text(
          merged_pil, text=f"step: {step}, psnr: {psnr:.2f}dB, ssim: {ssim:.2f}", size=merged_pil.size[0] // 20)

        st_utils.st_image(merged_pil, caption=f"target {target_pil.size}, img_pil {img_pil.size}",
                          debug=global_cfg.tl_debug, st_empty=st_image)
        st_text_rot.write(f"{azim}")
        st_text_trans.write(f"{elev}")
        video_f_target_proj.write(merged_pil)
        video_f_cam_pose.write(cam_pose_pil)

        pil_utils.add_text(
          img_pil, text=f"step: {step}", size=img_pil.size[0] // 10)
        st_utils.st_image(img_pil, caption=f"{img_pil.size}",
                          debug=global_cfg.tl_debug, st_empty=st_image_proj)
        video_f_inversion.write(img_pil)

      if mask_background and step >= N_steps_pose:
        # synth_images = synth_images * masks
        synth_images = synth_images * masks + synth_images.detach() * (1 - masks)
        # synth_images_thumb = synth_images_thumb * masks_thumb + synth_images_thumb.detach() * (1 - masks_thumb)

      # perceptual loss
      synth_features, synth_features_thumb = self.get_perceptual_fea(image_tensor=synth_images,
                                                                     image_tensor_thumb=synth_images_thumb)

      percep_loss = (target_features - synth_features).square().sum() * rgb_weight + \
                    (target_features_thumb - synth_features_thumb).square().sum() * thumb_weight

      # mse loss
      if mse_weight > 0:
        mse_loss = F.mse_loss(synth_images, target=target_images.detach())
        mse_loss = mse_loss * mse_weight
      else:
        mse_loss = dummy_zero

      # Noise regularization.
      if optim_noise_bufs and regularize_noise_weight > 0:
        reg_loss = 0
        for v in noise_bufs:
          # noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
          noise = v
          while True:
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
            if noise.shape[2] <= 8:
              break
            noise = F.avg_pool2d(noise, kernel_size=2)
        reg_loss = reg_loss * regularize_noise_weight

      else:
        reg_loss = dummy_zero

      loss = percep_loss + mse_loss + reg_loss

      # Step
      optimizer_cam.zero_grad(set_to_none=True)
      optimizer_render.zero_grad(set_to_none=True)
      optimizer_decoder.zero_grad(set_to_none=True)
      loss.backward()
      optimizer_cam.step()
      optimizer_render.step()
      optimizer_decoder.step()

      if st_web and step >= 50 and (step % st_log_every == 0 or step == N_steps - 1):
        st_chart_percep_loss.write(x=step, y=percep_loss.item())
        st_chart_mse_loss.write(x=step, y=mse_loss.item())
        st_chart_reg_loss.write(x=step, y=reg_loss.item())
        st_chart_loss.write(x=step, y=loss.item())

      if global_cfg.tl_debug:
        break

    if st_web:
      video_f_target_proj.release(st_video=True)
      video_f_inversion.release(st_video=True)
      video_f_cam_pose.release(st_video=True)

    # save ori image
    pil_utils.pil_save(target_pil, image_path=f'{outdir}/{image_name}.png', save_png=False)
    pil_utils.pil_save(target_pil_flip, image_path=f'{outdir}/{image_name}_flip.png', save_png=False)
    # save proj image
    with torch.no_grad():
      proj_images, _ = self._G_forward(
        G=G,
        style_render=w_render_opt if w_render_opt.shape[0] == 2 else w_render_opt.repeat(2, 1, 1),
        style_decoder=w_decoder_opt if w_decoder_opt.shape[0] == 2 else w_decoder_opt.repeat(2, 1, 1),
        noise_bufs=noise_bufs,
        cam_cfg=cam_cfg,
        nerf_cfg=nerf_cfg,
        device=device,
        style_render_noise=None,
        style_decoder_noise=None,
        rot=azim,
        trans=elev)
      proj_img_pil = torch_utils.img_tensor_to_pil(proj_images[0])
      pil_utils.pil_save(proj_img_pil, f"{outdir}/{proj_w_name}_proj.png", save_png=False)
      proj_img_pil_flip = torch_utils.img_tensor_to_pil(proj_images[1])
      pil_utils.pil_save(proj_img_pil_flip, f"{outdir}/{proj_w_name}_proj_flip.png", save_png=False)

      proj_img_pil_cam_extr = pil_utils.add_text(
        proj_img_pil, f"azimuth:".ljust(10) + f"{azim[0].item():.2f}\n"
                                              f"elevation:".ljust(14) + f"{elev[0].item():.2f}",
        size=proj_img_pil.size[0] // 15, color=(255, 0, 0), clone=True)
      pil_utils.pil_save(proj_img_pil_cam_extr, f"{outdir}/{proj_w_name}_proj_cam_extr.png", save_png=False)
      # pil_utils.imshow_pil(proj_img_pil_cam_extr)

      proj_img_pil_flip_cam_extr = pil_utils.add_text(
        proj_img_pil_flip, f"azimuth:".ljust(10) + f"{azim[1].item():.2f}\n"
                                              f"elevation:".ljust(14) + f"{elev[1].item():.2f}",
        size=proj_img_pil.size[0] // 15, color=(255, 0, 0), clone=True)
      pil_utils.pil_save(proj_img_pil_flip_cam_extr, f"{outdir}/{proj_w_name}_proj_flip_cam_extr.png", save_png=False)
      # pil_utils.imshow_pil(proj_img_pil_flip_cam_extr)


    saved_path = f"{outdir}/{proj_w_name}.pth"
    torch.save(saved_state_dict, saved_path)
    st.write(saved_path)

    # proj_psnr image
    file_logger = logger_utils.get_file_logger(filename=f"{outdir}/{proj_w_name}.txt")
    psnr = skimage_utils.sk_psnr(image_true_pil=target_pil, image_test_pil=proj_img_pil)
    ssim = skimage_utils.sk_ssim(image_true_pil=target_pil, image_test_pil=proj_img_pil)
    if lpips_metric is None:
      lpips_metric = skimage_utils.LPIPS(device=device)
    lpips = lpips_metric.calc_lpips(target_pil, proj_img_pil)
    file_logger.info_msg(f"psnr: {psnr}\n"
                         f"ssim: {ssim}\n"
                         f"lpips: {lpips}")

    ret_dict = {'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips}
    return ret_dict
