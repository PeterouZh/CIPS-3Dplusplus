import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from einops import rearrange

from tl2 import tl2_utils

from exp.cips3d.nerf_utils import Render


# Basic SIREN fully connected layer
class LinearLayer(nn.Module):
  def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False):
    super().__init__()
    if is_first:
      self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
    elif freq_init:
      self.weight = nn.Parameter(
        torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
    else:
      self.weight = nn.Parameter(
        0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

    self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1 / in_dim), b=np.sqrt(1 / in_dim)))

    self.bias_init = bias_init
    self.std_init = std_init

  def forward(self, input):
    out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init

    return out


# Siren layer with frequency modulation and offset
class FiLMSiren(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, prefix=__name__)

  def __init__(self,
               in_channel,
               out_channel,
               style_dim,
               is_first=False):
    super().__init__()

    self.repr_str = f"in_channel={in_channel}, out_channel={out_channel}, style_dim={style_dim}, is_first={is_first}"
    # self.module_name_list = []

    self.in_channel = in_channel
    self.out_channel = out_channel

    if is_first:
      self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
    else:
      self.weight = nn.Parameter(
        torch.empty(out_channel, in_channel).uniform_(-np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25))

    self.bias = nn.Parameter(
      nn.Parameter(nn.init.uniform_(torch.empty(out_channel), a=-np.sqrt(1 / in_channel), b=np.sqrt(1 / in_channel))))
    self.activation = torch.sin

    self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
    self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)
    pass

  def forward(self,
              input,
              style):
    batch, features = style.shape
    out = F.linear(input, self.weight, bias=self.bias)

    if out.ndim == 5:
      gamma = self.gamma(style).view(batch, 1, 1, 1, -1)
      beta = self.beta(style).view(batch, 1, 1, 1, -1)
    elif out.ndim == 4:
      gamma = self.gamma(style).view(batch, 1, 1, -1)
      beta = self.beta(style).view(batch, 1, 1, -1)

    out = self.activation(gamma * out + beta)

    return out


# Siren Generator Model
class SirenGenerator(nn.Module):
  def __init__(self,
               D=8,
               W=256,
               style_dim=256,
               input_ch=3,
               input_ch_views=3,
               output_features=True,
               emb_dim=0,
               **kwargs):
    super(SirenGenerator, self).__init__()

    self.D = D
    self.W = W
    self.style_dim = style_dim
    self.input_ch = input_ch
    self.input_ch_views = input_ch_views
    self.output_features = output_features

    self.pts_linears = nn.ModuleList(
      [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
      [FiLMSiren(W, W, style_dim=style_dim) for _ in range(D - 1)])

    self.views_linears = FiLMSiren(in_channel=W + emb_dim + input_ch_views,
                                   out_channel=W,
                                   style_dim=style_dim)
    self.rgb_linear = LinearLayer(W, 3, freq_init=True)
    self.sigma_linear = LinearLayer(W + emb_dim, 1, freq_init=True)
    pass

  def forward(self,
              x,
              styles,
              plane_emb):
    """

    :param x: (b, h, w, N_points, c) or (b, hw, N_points, c)
    :param styles: (b, style_dim)
    :return:
    """

    rgb, sdf, out_features = self.points_forward(x=x, styles=styles, plane_emb=plane_emb)

    return rgb, sdf, out_features

  def points_forward(self,
                     x,
                     styles,
                     plane_emb):
    """

    :param x: (b, h, w, N_points, c) or (b, hw, N_points, c)
    :param styles: (b, style_dim)
    :return:
    """

    input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
    mlp_out = input_pts.contiguous()
    for i in range(len(self.pts_linears)):
      mlp_out = self.pts_linears[i](mlp_out, styles[:, i])
    
    mlp_out = torch.cat([mlp_out, plane_emb], dim=-1)
    sdf = self.sigma_linear(mlp_out)

    # rgb branch
    mlp_out = torch.cat([mlp_out, input_views], -1)
    out_features = self.views_linears(mlp_out, styles[:, -1])

    rgb = self.rgb_linear(out_features)

    # outputs = torch.cat([rgb, sdf], -1)
    # if self.output_features:
    #   outputs = torch.cat([outputs, out_features], -1)

    return rgb, sdf, out_features

# Full volume renderer
class VolumeFeatureRenderer(nn.Module):
  def __init__(self,
               N_layers_renderer,
               input_dim,
               hidden_dim,
               style_dim,
               view_dim,
               with_sdf,
               output_features,
               triplane_cfg={},
               **kwargs):
    super().__init__()

    self.N_layers_renderer = N_layers_renderer
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.style_dim = style_dim
    self.view_dim = view_dim
    self.with_sdf = with_sdf
    self.output_features = output_features

    self.planes = torch.nn.Parameter(torch.randn([triplane_cfg.N_planes, triplane_cfg.emb_dim,
                                                  triplane_cfg.size, triplane_cfg.size]))
    
    self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))
    self.network = SirenGenerator(D=N_layers_renderer,
                                  W=hidden_dim,
                                  style_dim=style_dim,
                                  input_ch=input_dim,
                                  input_ch_views=view_dim,
                                  output_features=output_features,
                                  emb_dim=triplane_cfg.emb_dim)

    plane_axes = self.generate_planes(mode=triplane_cfg.get('plane_mode', 'xy_xz_zx'))
    self.register_buffer('plane_axes', plane_axes)
    
    pass

  def forward(self,
              pts,
              # normalized_pts,
              rays_d,
              viewdirs,
              z_vals,
              near,
              far,
              styles=None,
              return_eikonal=False,
              N_samples_forward=None):
    """

    :param pts: (b h w N_samples, 3) or (b hw N_samples, 3)
    :param normalized_pts: (b h w N_samples, 3)
    :param rays_d: (b h w 3)
    :param viewdirs: (b h w 3)
    :param z_vals: (b h w N_samples)
    :param styles:
    :param return_eikonal:
    :return

    - rgb_map: (b h w 3) or (b hw 3)
    - feature_map: (b h w c)
    - sdf: (b h w n 1)
    - mask: (b h w 1)
    - xyz: (b h w 3)
    - eikonal_term:

    """

    if return_eikonal:
      pts.requires_grad = True

    normalized_pts = Render.normalize_points(pts=pts, near=near, far=far)

    if N_samples_forward is not None: # no_grad
      raise NotImplementedError
      if normalized_pts.ndim == 4:
        rgb_map_list = []
        sdf_list = []
        feature_map_list = []
        xyz_list = []
        mask_list = []
        B, HW, N, _ = normalized_pts.shape
        for idx in range(0, HW, N_samples_forward):
          rgb_, sdf_, features_ = self.run_network(normalized_pts[:, idx : idx + N_samples_forward],
                                                   viewdirs[:, idx : idx + N_samples_forward],
                                                   styles=styles)
          rgb_map_, feature_map_, xyz_, mask_, eikonal_term = Render.volume_integration(
            rgb=rgb_,
            sdf=sdf_,
            features=features_,
            z_vals=z_vals[:, idx : idx + N_samples_forward],
            rays_d=rays_d[:, idx : idx + N_samples_forward],
            pts=pts[:, idx : idx + N_samples_forward],
            with_sdf=self.with_sdf,
            sigmoid_beta=self.sigmoid_beta,
            return_eikonal=return_eikonal)

          rgb_map_list.append(rgb_map_)
          sdf_list.append(sdf_)
          feature_map_list.append(feature_map_)
          xyz_list.append(xyz_)
          mask_list.append(mask_)

        rgb_map = torch.cat(rgb_map_list, dim=1)
        sdf = torch.cat(sdf_list, dim=1)
        feature_map = torch.cat(feature_map_list, dim=1)
        xyz = torch.cat(xyz_list, dim=1)
        mask = torch.cat(mask_list, dim=1)

      else:
        raise NotImplementedError

      pass
    else:
      
      rgb, sdf, features = self.run_network(normalized_pts, # (2, 4096, 24, 3)
                                            viewdirs,
                                            styles=styles)

      rgb_map, feature_map, xyz, mask, eikonal_term = Render.volume_integration(
        rgb=rgb,
        sdf=sdf,
        features=features,
        z_vals=z_vals,
        rays_d=rays_d,
        pts=pts,
        with_sdf=self.with_sdf,
        sigmoid_beta=self.sigmoid_beta,
        return_eikonal=return_eikonal)

    return rgb_map, feature_map, sdf, mask, xyz, eikonal_term

  def generate_planes(self,
                      mode="xy_xz_zx"):
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
  
    if mode == 'xy_xz_zx':
      plane_axes = torch.tensor([[[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 [[1, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]],
                                 [[0, 0, 1],
                                  [1, 0, 0],
                                  [0, 1, 0]]], dtype=torch.float32)
    elif mode == 'xy_xz_yz':
      plane_axes = torch.tensor([[[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 [[1, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]],
                                 [[0, 1, 0],
                                  [0, 0, 1],
                                  [1, 0, 0]]], dtype=torch.float32)
    else:
      raise NotImplementedError
  
    return plane_axes

  @staticmethod
  def _project_onto_planes(planes,
                           coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N * n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

  @staticmethod
  def sample_from_planes(plane_axes,
                         plane_features,
                         coordinates,
                         mode='bilinear',
                         padding_mode='zeros'):
    """

    :param plane_axes:
    :param plane_features:
    :param coordinates: (b, N, 3), [-1, 1]
    :param mode:
    :param padding_mode:
    :return

    - output_features: (b, 3, N, 32)

    """
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)
  
    # coordinates = (2 / box_warp) * coordinates  # TODO: add specific box bounds
  
    projected_coordinates = VolumeFeatureRenderer._project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features,
                                                      projected_coordinates.float(),
                                                      mode=mode,
                                                      padding_mode=padding_mode,
                                                      align_corners=False) \
      .permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

  def run_network(self,
                  inputs,
                  viewdirs,
                  styles=None):
    """

    :param inputs: (b h w N_samples 3) or (b, hw, N_points, c)
    :param viewdirs: (b h w 3)
    :param styles: (b style_dim)

    :return

    """
    input_dirs = viewdirs.unsqueeze(-2).expand(inputs.shape)
    net_inputs = torch.cat([inputs, input_dirs], -1)

    # merge planes
    if inputs.dim() == 4:
      B, N_rays, N_pts, C = inputs.shape
      pts = rearrange(inputs, "B N_rays N_pts C -> B (N_rays N_pts) C").contiguous()
      
    elif inputs.dim() == 5:
      B, H, W, N_pts, C = inputs.shape
      pts = rearrange(inputs, "B H W N_pts C -> B (H W N_pts) C").contiguous()
    
    else:
      raise NotImplementedError

    planes = self.planes.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()
    sampled_features = self.sample_from_planes(self.plane_axes,
                                               planes, # (b, 3, 32, 256, 256)
                                               pts.detach(),  # (b, N, 3)
                                               padding_mode='zeros')

    sampled_features = sampled_features.mean(1, keepdim=True)
    if inputs.dim() == 4:
      sampled_features = rearrange(sampled_features,
                                   "B N_plane (N_rays N_pts) C_plane -> B N_rays N_pts (N_plane C_plane)",
                                   N_rays=N_rays, N_pts=N_pts).contiguous()
    elif inputs.dim() == 5:
      sampled_features = rearrange(sampled_features,
                                   "B N_plane (H W N_pts) C_plane -> B H W N_pts (N_plane C_plane)",
                                   H=H, W=W, N_pts=N_pts).contiguous()
    
    
    rgb, sdf, out_features = self.network(net_inputs, styles=styles, plane_emb=sampled_features)

    return rgb, sdf, out_features

  # def create_buffers(self,
  #                    img_size,
  #                    device='cuda'):
  #
  #   # create meshgrid to generate rays
  #   i, j = torch.meshgrid(torch.linspace(0.5, img_size - 0.5, img_size, device=device),
  #                         torch.linspace(0.5, img_size - 0.5, img_size, device=device))
  #
  #   self.register_buffer('i', i.t().unsqueeze(0), persistent=False)
  #   self.register_buffer('j', j.t().unsqueeze(0), persistent=False)
  #
  #   # create integration values
  #   if self.offset_sampling:
  #     t_vals = torch.linspace(0., 1. - 1 / self.N_samples, steps=self.N_samples, device=device).view(1, 1, 1, -1)
  #   else:  # Original NeRF Stratified sampling
  #     t_vals = torch.linspace(0., 1., steps=self.N_samples, device=device).view(1, 1, 1, -1)
  #
  #   self.register_buffer('t_vals', t_vals, persistent=False)
  #   self.register_buffer('inf', torch.tensor([1e10], device=device), persistent=False)
  #   self.register_buffer('zero_idx', torch.tensor([0], dtype=torch.int64, device=device), persistent=False)
  #   pass
  #
  # def get_rays(self,
  #              focal,
  #              c2w):
  #
  #   dirs = torch.stack(
  #     tensors=[(self.i - self.img_size * .5) / focal,
  #              -(self.j - self.img_size * .5) / focal,
  #              -torch.ones_like(self.i).expand(focal.shape[0], self.img_size, self.img_size)],
  #     dim=-1)
  #
  #   # Rotate ray directions from camera frame to the world frame
  #   rays_d = torch.sum(input=dirs[..., None, :] * c2w[:, None, None, :3, :3],
  #                      dim=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
  #
  #   # Translate camera frame's origin to the world frame. It is the origin of all rays.
  #   rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)
  #   if self.static_viewdirs:
  #     viewdirs = dirs
  #   else:
  #     viewdirs = rays_d
  #
  #   return rays_o, rays_d, viewdirs
  #
  # def get_eikonal_term(self,
  #                      pts,
  #                      sdf):
  #   eikonal_term = autograd.grad(outputs=sdf,
  #                                inputs=pts,
  #                                grad_outputs=torch.ones_like(sdf),
  #                                create_graph=True)[0]
  #
  #   return eikonal_term
  #
  # def sdf_activation(self, input):
  #   sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta
  #
  #   return sigma
  #
  # def volume_integration(self,
  #                        raw,
  #                        z_vals,
  #                        rays_d,
  #                        pts,
  #                        return_eikonal=False):
  #   """
  #
  #   :param raw: (b, h, w, n, c)
  #   :param z_vals: (b, h, w, n)
  #   :param rays_d: (b, h, w, 3)
  #   :param pts: (b, h, w, n, 3)
  #   :param return_eikonal:
  #
  #   :return
  #
  #   """
  #   dists = z_vals[..., 1:] - z_vals[..., :-1]
  #   rays_d_norm = torch.norm(rays_d.unsqueeze(self.samples_dim), dim=self.channel_dim)
  #   # dists still has 4 dimensions here instead of 5, hence, in this case samples dim is actually the channel dim
  #   dists = torch.cat([dists, self.inf.expand(rays_d_norm.shape)], self.channel_dim)  # [N_rays, N_samples]
  #   dists = dists * rays_d_norm
  #
  #   # If sdf modeling is off, the sdf variable stores the
  #   # pre-integration raw sigma MLP outputs.
  #   if self.output_features:
  #     rgb, sdf, features = torch.split(raw, [3, 1, self.feature_out_size], dim=self.channel_dim)
  #   else:
  #     rgb, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)
  #
  #   noise = 0.
  #   if self.raw_noise_std > 0.:
  #     noise = torch.randn_like(sdf) * self.raw_noise_std
  #
  #   if self.with_sdf:
  #     sigma = self.sdf_activation(-sdf)
  #
  #     if return_eikonal:
  #       eikonal_term = self.get_eikonal_term(pts, sdf)
  #     else:
  #       eikonal_term = None
  #
  #     sigma = 1 - torch.exp(-sigma * dists.unsqueeze(self.channel_dim))
  #   else:
  #     sigma = sdf
  #     eikonal_term = None
  #
  #     sigma = 1 - torch.exp(-F.softplus(sigma + noise) * dists.unsqueeze(self.channel_dim))
  #
  #   visibility = torch.cumprod(
  #     input=torch.cat([torch.ones_like(torch.index_select(sigma, dim=self.samples_dim, index=self.zero_idx)),
  #                      1. - sigma + 1e-10], self.samples_dim),
  #     dim=self.samples_dim)
  #   visibility = visibility[..., :-1, :]
  #   weights = sigma * visibility
  #
  #   if self.return_sdf:
  #     sdf_out = sdf
  #   else:
  #     sdf_out = None
  #
  #   if self.force_background:
  #     weights[..., -1, :] = 1 - weights[..., :-1, :].sum(self.samples_dim)
  #
  #   rgb_map = -1 + 2 * torch.sum(weights * torch.sigmoid(rgb), self.samples_dim)  # switch to [-1,1] value range
  #
  #   if self.output_features:
  #     feature_map = torch.sum(weights * features, self.samples_dim)
  #   else:
  #     feature_map = None
  #
  #   # Return surface point cloud in world coordinates.
  #   # This is used to generate the depth maps visualizations.
  #   # We use world coordinates to avoid transformation errors between
  #   # surface renderings from different viewpoints.
  #   if self.return_xyz:
  #     xyz = torch.sum(weights * pts, self.samples_dim)
  #     mask = weights[..., -1, :]  # background probability map
  #   else:
  #     xyz = None
  #     mask = None
  #
  #   return rgb_map, feature_map, sdf_out, mask, xyz, eikonal_term
  #

  # def render_rays(self,
  #                 ray_batch,
  #                 styles=None,
  #                 return_eikonal=False,
  #                 forward_points=None):
  #   """
  #
  #   :param ray_batch: (b, h, w, 11) [rays_o(3), rays_d(3), near(1), far(1), view_d(3)]
  #   :param styles:
  #   :param return_eikonal:
  #
  #   :return
  #
  #   - rgb_map:
  #   - features:
  #   - sdf:
  #   - mask:
  #   - xyz:
  #   - eikonal_term:
  #
  #   """
  #
  #   batch, h, w, _ = ray_batch.shape
  #   split_pattern = [3, 3, 2]
  #   if ray_batch.shape[-1] > 8:
  #     split_pattern += [3]
  #     rays_o, rays_d, bounds, viewdirs = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
  #   else:
  #     rays_o, rays_d, bounds = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
  #     viewdirs = None
  #
  #   near, far = torch.split(bounds, [1, 1], dim=self.channel_dim)
  #   z_vals = near * (1. - self.t_vals) + far * (self.t_vals)
  #
  #   if self.perturb > 0.:
  #     if self.offset_sampling:
  #       # random offset samples
  #       upper = torch.cat([z_vals[..., 1:], far], -1)
  #       lower = z_vals.detach()
  #       t_rand = torch.rand(batch, h, w).unsqueeze(self.channel_dim).to(z_vals.device)
  #     else:
  #       # get intervals between samples
  #       mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
  #       upper = torch.cat([mids, z_vals[..., -1:]], -1)
  #       lower = torch.cat([z_vals[..., :1], mids], -1)
  #       # stratified samples in those intervals
  #       t_rand = torch.rand(z_vals.shape).to(z_vals.device)
  #
  #     z_vals = lower + (upper - lower) * t_rand
  #
  #   pts = rays_o.unsqueeze(self.samples_dim) + \
  #         rays_d.unsqueeze(self.samples_dim) * z_vals.unsqueeze(self.channel_dim)
  #
  #   if return_eikonal:
  #     pts.requires_grad = True
  #
  #   if self.z_normalize:
  #     normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))
  #   else:
  #     normalized_pts = pts
  #
  #   rgb_map, features, sdf, mask, xyz, eikonal_term = self.ray_forward(pts=pts,
  #                                                                      normalized_pts=normalized_pts,
  #                                                                      styles=styles,
  #                                                                      viewdirs=viewdirs,
  #                                                                      z_vals=z_vals,
  #                                                                      rays_d=rays_d,
  #                                                                      return_eikonal=return_eikonal)
  #
  #   return rgb_map, features, sdf, mask, xyz, eikonal_term
  #
  # def ray_forward(self,
  #                 pts,
  #                 normalized_pts,
  #                 styles,
  #                 viewdirs,
  #                 z_vals,
  #                 rays_d,
  #                 return_eikonal
  #                 ):
  #
  #   raw = self.run_network(normalized_pts,
  #                          viewdirs,
  #                          styles=styles)
  #   rgb_map, features, sdf, mask, xyz, eikonal_term = self.volume_integration(
  #     raw,
  #     z_vals,
  #     rays_d,
  #     pts,
  #     return_eikonal=return_eikonal)
  #
  #   return rgb_map, features, sdf, mask, xyz, eikonal_term
  #
  # def render(self,
  #            focal,
  #            c2w,
  #            near,
  #            far,
  #            styles,
  #            c2w_staticcam=None,
  #            return_eikonal=False,
  #            forward_points=None):
  #
  #   rays_o, rays_d, viewdirs = self.get_rays(focal, c2w)
  #   viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
  #
  #   # Create ray batch
  #   near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
  #   far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
  #   rays = torch.cat([rays_o, rays_d, near, far], -1)
  #   rays = torch.cat([rays, viewdirs], -1)
  #   rays = rays.float()
  #   rgb, features, sdf, mask, xyz, eikonal_term = self.render_rays(rays,
  #                                                                  styles=styles,
  #                                                                  return_eikonal=return_eikonal,
  #                                                                  forward_points=forward_points)
  #
  #   return rgb, features, sdf, mask, xyz, eikonal_term

  def mlp_init_pass(self,
                    cam_poses,
                    focals,
                    img_size,
                    near,
                    far,
                    styles,
                    nerf_cfg):
    # rays_o, rays_d, viewdirs = self.get_rays(focal, cam_poses)
    # viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    rays_o, rays_d, viewdirs = Render.get_rays_in_world(focal=focals,
                                                        img_size=img_size,
                                                        c2w=cam_poses)
    z_vals = Render.get_z_vals(near=near, far=far, rays_d=rays_d,
                               N_samples=nerf_cfg['N_samples'],
                               offset_sampling=False)

    # near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
    # far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
    # z_vals = near * (1. - self.t_vals) + far * (self.t_vals)
    # # get intervals between samples
    # mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    # upper = torch.cat([mids, z_vals[..., -1:]], -1)
    # lower = torch.cat([z_vals[..., :1], mids], -1)
    # # stratified samples in those intervals
    # t_rand = torch.rand(z_vals.shape).to(z_vals.device)
    #
    # z_vals = lower + (upper - lower) * t_rand


    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)

    normalized_pts = Render.normalize_points(pts=pts, near=near, far=far)
    # normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))

    rgb, sdf, _ = self.run_network(normalized_pts, viewdirs, styles=styles)
    # _, sdf = torch.split(raw, [3, 1], dim=self.channel_dim)
    sdf = sdf.squeeze(-1)
    view_shape = [1] * sdf.dim()
    view_shape[0] = -1
    target_values = pts.detach().norm(dim=-1) - ((far - near).view(*view_shape) / 4)
    # target_values = pts.detach().norm(dim=-1) - ((far - near) / 4)


    # pts, rays_d, viewdirs, z_vals = Render.prepare_nerf_inputs(
    #   focal=focals,
    #   img_size=img_size,
    #   cam_poses=cam_poses,
    #   near=near,
    #   far=far,
    #   **nerf_cfg)
    #
    # normalized_pts = Render.normalize_points(pts=pts, near=near, far=far)
    #
    # rgb, sdf, _ = self.run_network(normalized_pts,
    #                                viewdirs,
    #                                styles=styles)
    #
    # # (b h w n)
    # sdf = sdf.squeeze(-1)
    # view_shape = [1] * sdf.dim()
    # view_shape[0] = -1
    # target_values = pts.detach().norm(dim=-1) - ((far - near).view(*view_shape) / 4)

    return sdf, target_values

