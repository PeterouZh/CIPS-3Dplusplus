import math
import random
import trimesh
import numpy as np
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F

from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.transforms import matrix_to_euler_angles

from tl2 import tl2_utils
from tl2.proj.fvcore import MODEL_REGISTRY
from tl2.proj.pytorch import torch_utils

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

from exp.cips3d.volume_renderer import VolumeFeatureRenderer
from exp.stylesdf.utils import (create_cameras, create_mesh_renderer, add_textures, create_depth_mesh_renderer)
from exp.cips3d import nerf_utils
from .model_v3 import Decoder
from .model_v3 import Generator as Generator_base


@MODEL_REGISTRY.register(name_prefix=__name__)
class Generator(Generator_base):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, prefix=__name__)

  def __init__(self,
               enable_decoder,
               freeze_renderer,
               freeze_decoder_mapping,
               renderer_detach=True,
               predict_rgb_residual=False,
               scale_factor=None,
               renderer_cfg={},
               mapping_renderer_cfg={},
               decoder_cfg={},
               mapping_decoder_cfg={},
               **kwargs):
    super(Generator_base, self).__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'enable_decoder': enable_decoder,
      'freeze_renderer': freeze_renderer,
      'freeze_decoder_mapping': freeze_decoder_mapping,
      'renderer_detach': renderer_detach,
      'predict_rgb_residual': predict_rgb_residual,
      'scale_factor': scale_factor,
    })

    self.enable_decoder = enable_decoder
    self.freeze_renderer = freeze_renderer
    self.freeze_decoder_mapping = freeze_decoder_mapping
    self.renderer_detach = renderer_detach
    self.predict_rgb_residual = predict_rgb_residual
    self.scale_factor = scale_factor

    self.renderer_cfg = renderer_cfg
    self.mapping_renderer_cfg = mapping_renderer_cfg
    self.decoder_cfg = decoder_cfg
    self.mapping_decoder_cfg = mapping_decoder_cfg

    self.module_name_list = []

    # nerf net
    self.renderer = VolumeFeatureRenderer(style_dim=mapping_renderer_cfg['style_dim'],
                                          **renderer_cfg)
    self.module_name_list.append('renderer')
    self.N_layers_renderer = self.renderer.N_layers_renderer

    self.create_mapping_nerf(**mapping_renderer_cfg)
    self.z_dim = mapping_renderer_cfg['z_dim']

    # fc net
    self.decoder = Decoder(style_dim=mapping_decoder_cfg['style_dim'],
                           **{**decoder_cfg,
                              'in_channel': renderer_cfg['hidden_dim']})
    self.module_name_list.append('decoder')

    self.create_mapping_decoder(z_dim=mapping_renderer_cfg['style_dim'],
                                **mapping_decoder_cfg)

    # if scale_factor > 1:
    #   # interp_mode = 'bilinear'
    #   interp_mode = 'bicubic'
    #   self.upsample_layer = nn.Upsample(scale_factor=scale_factor, mode=interp_mode, align_corners=False)
    # else:
    #   self.upsample_layer = None

    tl2_utils.print_repr(self)
    pass

  def forward(self,
              zs, # [(b, style_dim)]
              # camera kwargs
              cam_poses,
              focals,
              img_size,
              near=0.88,
              far=1.12,
              # mapping kwargs
              truncation=1,
              inject_index=None,
              path_reg=False,
              style_render=None,
              style_decoder=None,
              # noise bufs
              noise_bufs=None,
              randomize_noise=True,
              # aux
              eikonal_reg=False,
              return_sdf=False,
              return_xyz=False,
              N_rays_forward=None, # grad or no_grad
              N_rays_grad=None, # grad
              N_samples_forward=None, # no_grad
              nerf_cfg={},
              recompute_mean=False,
              # others
              project_noise=False,
              mesh_path=None,
              renderer_detach=None,
              sample_idx_h=None,
              sample_idx_w=None,
              **kwargs):
    if renderer_detach is None:
      renderer_detach = self.renderer_detach

    if eikonal_reg:
      if N_rays_grad is None:
        assert N_rays_forward is None
    assert len(zs) == 2

    # do not calculate renderer gradients if renderer weights are frozen
    if self.freeze_renderer:
      self.style.requires_grad_(False)
      self.renderer.requires_grad_(False)
    if self.freeze_decoder_mapping:
      self.style_decoder.requires_grad_(False)

    style_render, style_decoder = self.mapping_networks(zs=zs,
                                                        truncation=truncation,
                                                        inject_index=inject_index,
                                                        path_reg=path_reg,
                                                        style_render=style_render,
                                                        style_decoder=style_decoder,
                                                        recompute_mean=recompute_mean)
    noise_bufs = self.get_noise_bufs(noise_bufs=noise_bufs, randomize_noise=randomize_noise)

    pts, rays_d, viewdirs, z_vals = nerf_utils.Render.prepare_nerf_inputs(
      focal=focals,
      img_size=img_size,
      cam_poses=cam_poses,
      near=near,
      far=far,
      **nerf_cfg)

    if sample_idx_h is not None and sample_idx_w is not None:
      pts, rays_d, viewdirs, z_vals = self.sample_sub_pixels(pts=pts,
                                                             rays_d=rays_d,
                                                             viewdirs=viewdirs,
                                                             z_vals=z_vals,
                                                             sample_idx_h=sample_idx_h,
                                                             sample_idx_w=sample_idx_w)

    B, H, W, N, C = pts.shape
    pts = rearrange(pts, "b h w n c -> b (h w) n c")
    # normalized_pts = rearrange(normalized_pts, "b h w n c -> b (h w) n c")
    rays_d = rearrange(rays_d, "b h w c -> b (h w) c")
    viewdirs = rearrange(viewdirs, "b h w c -> b (h w) c")
    z_vals = rearrange(z_vals, "b h w n -> b (h w) n")

    # grad
    if N_rays_grad is not None and N_rays_grad < H * W:

      raise NotImplementedError

      N_points = H * W
      idx_grad, idx_no_grad = torch_utils.batch_random_split_indices(bs=B,
                                                                     num_points=N_points,
                                                                     grad_points=N_rays_grad,
                                                                     device=pts.device)
      thumb_rgb, sdf, mask, xyz, rgb, eikonal_term = self.part_grad_rays_forward(
        N_rays_forward=N_rays_forward,
        N_samples_forward=N_samples_forward,
        idx_grad=idx_grad,
        idx_no_grad=idx_no_grad,
        pts=pts,
        # normalized_pts=normalized_pts,
        rays_d=rays_d,
        viewdirs=viewdirs,
        z_vals=z_vals,
        near=near,
        far=far,
        style_render=style_render,
        style_decoder=style_decoder,
        noise_bufs=noise_bufs,
        eikonal_reg=eikonal_reg,
        cam_poses=cam_poses,
        project_noise=project_noise,
        mesh_path=mesh_path,
        renderer_detach=renderer_detach)

    else: # grad or no_grad

      thumb_rgb, sdf, mask, xyz, features, eikonal_term = self.rays_forward(
        N_rays_forward=N_rays_forward,
        N_samples_forward=N_samples_forward,
        pts=pts,
        # normalized_pts=normalized_pts,
        rays_d=rays_d,
        viewdirs=viewdirs,
        z_vals=z_vals,
        near=near,
        far=far,
        style_render=style_render,
        style_decoder=style_decoder,
        noise_bufs=noise_bufs,
        eikonal_reg=eikonal_reg,
        cam_poses=cam_poses,
        project_noise=project_noise,
        mesh_path=mesh_path,
        renderer_detach=renderer_detach)

    thumb_rgb = rearrange(thumb_rgb, "b (h w) c -> b c h w", h=H, w=W).contiguous()
    sdf = rearrange(sdf, "b (h w) n c -> b h w n c", h=H, w=W).contiguous()
    mask = rearrange(mask, "b (h w) c -> b c h w", h=H, w=W).contiguous()
    xyz = rearrange(xyz, "b (h w) c -> b c h w", h=H, w=W).contiguous()
    # if eikonal_term is not None:
    #   eikonal_term = rearrange(eikonal_term, "b (h w) n c -> b h w n c", h=H, w=W)

    # decoder
    if self.enable_decoder:
      features = rearrange(features, "b (h w) c -> b c h w", h=H, w=W).contiguous()  # bug

      if renderer_detach:
        features = features.detach()

      rgb = self.decoder(features=features,
                         styles=style_decoder,
                         rgbd_in=None,
                         transform=cam_poses if project_noise else None,
                         noise=noise_bufs,
                         mesh_path=mesh_path).contiguous()  # bug
    else:
      raise NotImplementedError
      rgb = thumb_rgb.clone()
      # rgb = rearrange(rgb, "b hw c -> b c hw 1").contiguous()  # bug
    # rgb = rearrange(rgb, "b c (h w) 1 -> b c h w", h=H, w=W).contiguous()

    ret_maps = {
      'rgb': rgb,
      'thumb_rgb': thumb_rgb,
      'style_decoder': style_decoder if path_reg else None,
      'eikonal_term': eikonal_term,
      'sdf': sdf if return_sdf else None,
      'xyz': xyz if return_xyz else None,
      'mask': mask[:, [0]],
      'depth': mask[:, [1]]
    }

    return ret_maps


