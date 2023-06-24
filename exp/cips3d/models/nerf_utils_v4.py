import unittest
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from pytorch3d import transforms as tr3d


class Render(object):

  def __init__(self):

    pass

  @staticmethod
  def get_rays_in_world(focal,
                        img_size,
                        c2w,
                        static_viewdirs=False):
    """

    :param focal: (b, 1, 1)
    :param img_size:
    :param c2w: (b, 3, 4)
    :param static_viewdirs:
    :return

    - rays_o: (b h w 3)
    - rays_d: (b h w 3)
    - viewdirs: (b h w 3)

    """

    device = focal.device

    # create meshgrid to generate rays
    y, x = torch.meshgrid(torch.linspace(0.5, img_size - 0.5, img_size, device=device),
                          torch.linspace(0.5, img_size - 0.5, img_size, device=device))
    # (b, h, w)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    rays_d_cam = torch.stack(
      tensors=[(x - img_size * .5) / focal,
               - (y - img_size * .5) / focal,
               - torch.ones_like(x).expand(focal.shape[0], img_size, img_size)],
      dim=-1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(input=rays_d_cam[..., None, :] * c2w[:, None, None, :3, :3],
                       dim=-1)

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)

    if static_viewdirs:
      viewdirs = rays_d_cam
    else:
      viewdirs = rays_d

    viewdirs = F.normalize(viewdirs, p=2, dim=-1)
    # viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    return rays_o, rays_d, viewdirs

  @staticmethod
  def get_z_vals(near,
                 far,
                 rays_d, # (b, h, w, 3)
                 N_samples,
                 perturb=True,
                 offset_sampling=True):
    """

    :param near: (b, 1, 1)
    :param far: (b, 1, 1)
    :param rays_d: (b, h, w, 3)
    :param N_samples:
    :param perturb:
    :param offset_sampling:

    :return

    - z_vals: (b, h, w, N_samples)
    """
    device = rays_d.device
    batch, h, w, _ = rays_d.shape

    # (b, 1, 1) -> (b, h, w, 1)
    near = near.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])
    far = far.unsqueeze(-1) * torch.ones_like(rays_d[..., :1])

    if offset_sampling:
      # (1, 1, 1, N_samples)
      t_vals = torch.linspace(0., 1. - 1. / N_samples, steps=N_samples, device=device).view(1, 1, 1, -1)
    else:  # Original NeRF Stratified sampling
      t_vals = torch.linspace(0., 1., steps=N_samples, device=device).view(1, 1, 1, -1)

    # (b, h, w, N_samples)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    if perturb:
      if offset_sampling:
        # (b, h, w, N_samples)
        upper = torch.cat([z_vals[..., 1:], far], -1)
        lower = z_vals.detach()
        # random offset samples
        t_rand = torch.rand(batch, h, w, 1, device=device)
      else:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(z_vals.device)

      z_vals = lower + (upper - lower) * t_rand

    return z_vals

  @staticmethod
  def normalize_points(pts,
                       near,
                       far):

    view_shape = [1] * pts.dim()
    view_shape[0] = -1
    normalized_pts = pts * 2 / ((far - near).view(*view_shape))
    # normalized_pts = pts * 2 / (far - near)

    return normalized_pts

  @staticmethod
  def get_points(rays_o,
                 rays_d,
                 z_vals,
                 # far,
                 # near,
                 # points_requires_grad=False,
                 # normalize_pts=True,
                 ):
    """

    :param rays_o: (b h w 3)
    :param rays_d: (b h w 3)
    :param z_vals: (b h w N_samples)
    :param far: (b 1 1)
    :param near: (b 1 1)
    :param points_requires_grad:
    :param normalize_pts:
    :return

    - pts: (b h w N_samples, 3)
    - normalized_pts: (b h w N_samples, 3)
    """

    # (b h w N_samples 3)
    pts = rays_o.unsqueeze(-2) + \
          rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)

    # if points_requires_grad:
    #   pts.requires_grad = True
    # if normalize_pts:
    #   normalized_pts = pts * 2 / ((far - near).view(-1, 1, 1, 1, 1))
    # else:
    #   normalized_pts = pts

    return pts

  @staticmethod
  def prepare_nerf_inputs(focal,
                          img_size,
                          cam_poses,
                          near,
                          far,
                          N_samples,
                          perturb,
                          static_viewdirs=False,
                          **kwargs):
    """

    :param focal: (b 1 1)
    :param img_size:
    :param cam_poses: (b 3 4)
    :param near: (b 1 1)
    :param far: (b 1 1)
    :param N_samples:
    :param perturb:
    :param static_viewdirs:
    :return

    - pts: (b h w N_samples, 3)
    - normalized_pts: (b h w N_samples, 3)
    - rays_d: (b h w 3)
    - viewdirs: (b h w 3)
    - z_vals: (b h w N_samples)

    """

    # (b h w 3)
    rays_o, rays_d, viewdirs = Render.get_rays_in_world(
      focal=focal, img_size=img_size, c2w=cam_poses, static_viewdirs=static_viewdirs)

    # (b, h, w, N_samples)
    z_vals = Render.get_z_vals(near=near,
                               far=far,
                               rays_d=rays_d,
                               N_samples=N_samples,
                               perturb=perturb, offset_sampling=True)

    # (b h w N_samples, 3)
    pts = Render.get_points(rays_o=rays_o,
                            rays_d=rays_d,
                            z_vals=z_vals)

    return pts, rays_d, viewdirs, z_vals

  @staticmethod
  def get_eikonal_term(pts,
                       sdf):
    eikonal_term = autograd.grad(outputs=sdf,
                                 inputs=pts,
                                 grad_outputs=torch.ones_like(sdf),
                                 retain_graph=True,
                                 create_graph=True)[0]

    return eikonal_term

  @staticmethod
  def volume_integration(rgb,
                         sdf,
                         features,
                         z_vals,
                         rays_d,
                         pts,
                         # sdf
                         with_sdf=True,
                         sigmoid_beta=None,
                         return_eikonal=False,
                         # sigma noise
                         raw_noise_std=0.,
                         force_background=False):
    """

    :param rgb: (b, h, w, n, 3)
    :param sdf: (b h w n 1)
    :param features: (b h w n c)
    :param z_vals: (b, h, w, n)
    :param rays_d: (b, h, w, 3)
    :param pts: (b, h, w, n, 3)
    :param sigmoid_beta: nn.Parameter()
    :param return_eikonal:

    :return

    - rgb_map: (b, h, w, 3)
    - feature_map: (b h w c)
    - xyz: (b h w 3)
    - mask: (b h w 1)
    - eikonal_term:

    """
    device = rays_d.device

    # (b h w n-1)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # (b h w 1)
    rays_d_norm = torch.norm(rays_d, dim=-1, keepdim=True)
    # (b h w n)
    inf_tensor = torch.tensor([1e10], device=device)
    dists = torch.cat([dists, inf_tensor.expand_as(rays_d_norm)], -1)
    # (b h w n) * (b h w 1)
    dists = dists * rays_d_norm

    if with_sdf:
      # (b h w n 1) [0, )
      sigma = torch.sigmoid((- sdf) / sigmoid_beta) / sigmoid_beta

      if return_eikonal:
        eikonal_term = Render.get_eikonal_term(pts=pts, sdf=sdf)
      else:
        eikonal_term = None

      # (b h w n 1) * (b h w n 1) -> (b h w n 1)
      sigma = 1 - torch.exp(-sigma * dists.unsqueeze(-1))

    else:
      sigma = sdf
      eikonal_term = None

      noise = 0.
      if raw_noise_std > 0.:
        noise = torch.randn_like(sigma) * raw_noise_std

      sigma = 1 - torch.exp(-F.softplus(sigma + noise) * dists.unsqueeze(-1))

    # (b h w 1 1)
    ones_tensor = torch.ones_like(sigma.detach()[..., [0], :])
    # (b h w 1+n 1)
    visibility = torch.cumprod(input=torch.cat(tensors=[ones_tensor, 1. - sigma + 1e-10],
                                               dim=-2),
                               dim=-2)
    # (b h w n 1)
    visibility = visibility[..., :-1, :]
    # (b h w n 1) * (b h w n 1) -> (b h w n 1)
    weights = sigma * visibility

    if force_background:
      weights[..., -1, :] = 1 - weights[..., :-1, :].sum(dim=-2)

    # start integrating

    # (b, h, w, 3)
    rgb_map = -1 + 2 * torch.sum(input=weights * torch.sigmoid(rgb),
                                 dim=-2)  # switch to [-1,1] value range

    if features is not None:
      # (b h w c)
      feature_map = torch.sum(weights * features, dim=-2)
    else:
      feature_map = None

    # Return surface point cloud in world coordinates.
    # This is used to generate the depth maps visualizations.
    # We use world coordinates to avoid transformation errors between
    # surface renderings from different viewpoints.
    # (b, h, w, n, 3) -> (b h w 3)
    xyz = torch.sum(weights * pts, dim=-2)

    # (b h w n 1) * (b h w n 1) -> (b h w 1)
    mask = weights[..., -1, :]  # background probability map

    # depth = - xyz.norm(dim=-1, keepdim=True)
    depth = torch.sum(weights * z_vals.unsqueeze(-1), dim=-2)

    mask = torch.cat([mask, depth], dim=-1)

    return rgb_map, feature_map, xyz, mask, eikonal_term


class Camera(object):

  @staticmethod
  def generate_camera_params(img_size,
                             device,
                             batch=1,
                             locations=None,
                             sweep=False,
                             uniform=False,
                             azim_range=0.3,
                             elev_range=0.15,
                             fov_ang=6,
                             dist_radius=0.12):
    """
    ################# Camera parameters sampling ####################

    :param resolution:
    :param device:
    :param batch:
    :param locations:
    :param sweep:
    :param uniform:
    :param azim_range:
    :param elev_range:
    :param fov_ang:
    :param dist_radius:
    :return:
    """
    if locations is not None:
      azim = locations[:, 0].view(-1, 1)
      elev = locations[:, 1].view(-1, 1)

      # generate intrinsic parameters
      # fix distance to 1
      dist = torch.ones(azim.shape[0], 1, device=device)
      near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
      fov_angle = fov_ang * torch.ones(azim.shape[0], 1, device=device).view(-1, 1) * np.pi / 180
      focal = 0.5 * img_size / torch.tan(fov_angle).unsqueeze(-1)
    elif sweep:
      # generate camera locations on the unit sphere
      if isinstance(azim_range, list) and isinstance(elev_range, list):
        azim = (azim_range[0] + (azim_range[1] - azim_range[0]) / 7 * torch.arange(8, device=device)).view(-1, 1).repeat(batch, 1)
        elev = (elev_range[0] + (elev_range[1] - elev_range[0]) * torch.rand(batch, 1, device=device).repeat(1, 8).view(-1, 1))
      else:
        azim = (-azim_range + (2 * azim_range / 7) * torch.arange(8, device=device)).view(-1, 1).repeat(batch, 1)
        elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device).repeat(1, 8).view(-1, 1))

      # generate intrinsic parameters
      dist = (torch.ones(batch, 1, device=device)).repeat(1, 8).view(-1, 1)
      near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
      fov_angle = fov_ang * torch.ones(batch, 1, device=device).repeat(1, 8).view(-1, 1) * np.pi / 180
      focal = 0.5 * img_size / torch.tan(fov_angle).unsqueeze(-1)
    else:
      # sample camera locations on the unit sphere
      if uniform:
        if isinstance(azim_range, list) and isinstance(elev_range, list):
          azim = azim_range[0] + (azim_range[1] - azim_range[0]) * torch.rand(batch, 1, device=device)
          elev = elev_range[0] + (elev_range[1] - elev_range[0]) * torch.rand(batch, 1, device=device)
        else:
          azim = (-azim_range + 2 * azim_range * torch.rand(batch, 1, device=device))
          elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device))
      else:
        azim = (azim_range * torch.randn(batch, 1, device=device))
        elev = (elev_range * torch.randn(batch, 1, device=device))

      # generate intrinsic parameters
      dist = torch.ones(batch, 1, device=device)  # restrict camera position to be on the unit sphere
      near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
      fov_angle = fov_ang * torch.ones(batch, 1, device=device) * np.pi / 180  # full fov is 12 degrees
      focal = 0.5 * img_size / torch.tan(fov_angle).unsqueeze(-1)

    viewpoint = torch.cat([azim, elev], 1)

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).view(-1, 3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    up = torch.tensor([[0, 1, 0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(camera_dir, eps=1e-5)  # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
      replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
      x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = camera_loc[:, :, None]
    extrinsics = torch.cat((R.transpose(1, 2), T), -1)

    return extrinsics, focal, near, far, viewpoint

  @staticmethod
  def get_camera2world(cam2world,
                       trans,
                       homo=False):
    """

    :param cam2world: (b, 3)
    :param trans: (b, 3)

    :return:

    - cam_extrinsics: (b, 3, 4) or (b, 4, 4)

    """
    assert cam2world.shape[:-1] == trans.shape[:-1]
    prefix = cam2world.shape[:-1]

    rot_m = tr3d.axis_angle_to_matrix(cam2world)

    cam_extrinsics = torch.cat((rot_m.view(*prefix, 3, 3), trans.view(*prefix, 3, 1)), dim=-1)
    if homo:
      extend = torch.zeros(*prefix, 1, 4).to(cam2world.device)
      extend[..., 0, 3] = 1.
      cam_extrinsics = torch.cat((cam_extrinsics, extend), dim=-2)  # [...,4,4]

    return cam_extrinsics

  @staticmethod
  def generate_camera_params_v1(img_size,
                                device,
                                batch=1,
                                locations=None,
                                sweep=False,
                                uniform=False,
                                azim_range=0.3,
                                elev_range=0.15,
                                fov_ang=6,
                                dist_radius=0.12,
                                up=None):
    """
    ################# Camera parameters sampling ####################

    :param resolution:
    :param device:
    :param batch:
    :param locations:
    :param sweep:
    :param uniform:
    :param azim_range:
    :param elev_range:
    :param fov_ang:
    :param dist_radius:
    :param up: (b, 3)
    :return:
    """
    if locations is not None:
      azim = locations[:, 0].view(-1, 1)
      elev = locations[:, 1].view(-1, 1)

      # generate intrinsic parameters
      # fix distance to 1
      dist = torch.ones(azim.shape[0], 1, device=device)
      near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
      fov_angle = fov_ang * torch.ones(azim.shape[0], 1, device=device).view(-1, 1) * np.pi / 180
      focal = 0.5 * img_size / torch.tan(fov_angle).unsqueeze(-1)
    elif sweep:
      # generate camera locations on the unit sphere
      if isinstance(azim_range, list) and isinstance(elev_range, list):
        azim = (azim_range[0] + (azim_range[1] - azim_range[0]) / 7 * torch.arange(8, device=device)).view(-1,
                                                                                                           1).repeat(
          batch, 1)
        elev = (elev_range[0] + (elev_range[1] - elev_range[0]) * torch.rand(batch, 1, device=device).repeat(1, 8).view(
          -1, 1))
      else:
        azim = (-azim_range + (2 * azim_range / 7) * torch.arange(8, device=device)).view(-1, 1).repeat(batch, 1)
        elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device).repeat(1, 8).view(-1, 1))

      # generate intrinsic parameters
      dist = (torch.ones(batch, 1, device=device)).repeat(1, 8).view(-1, 1)
      near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
      fov_angle = fov_ang * torch.ones(batch, 1, device=device).repeat(1, 8).view(-1, 1) * np.pi / 180
      focal = 0.5 * img_size / torch.tan(fov_angle).unsqueeze(-1)
    else:
      # sample camera locations on the unit sphere
      if uniform:
        if isinstance(azim_range, list) and isinstance(elev_range, list):
          azim = azim_range[0] + (azim_range[1] - azim_range[0]) * torch.rand(batch, 1, device=device)
          elev = elev_range[0] + (elev_range[1] - elev_range[0]) * torch.rand(batch, 1, device=device)
        else:
          azim = (-azim_range + 2 * azim_range * torch.rand(batch, 1, device=device))
          elev = (-elev_range + 2 * elev_range * torch.rand(batch, 1, device=device))
      else:
        azim = (azim_range * torch.randn(batch, 1, device=device))
        elev = (elev_range * torch.randn(batch, 1, device=device))

      # generate intrinsic parameters
      dist = torch.ones(batch, 1, device=device)  # restrict camera position to be on the unit sphere
      near, far = (dist - dist_radius).unsqueeze(-1), (dist + dist_radius).unsqueeze(-1)
      fov_angle = fov_ang * torch.ones(batch, 1, device=device) * np.pi / 180  # full fov is 12 degrees
      focal = 0.5 * img_size / torch.tan(fov_angle).unsqueeze(-1)

    viewpoint = torch.cat([azim, elev], 1)

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).view(-1, 3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    if up is None:
      up = torch.tensor([[0, 1, 0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(camera_dir, eps=1e-5)  # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
      replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
      x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = camera_loc[:, :, None]
    extrinsics = torch.cat((R.transpose(1, 2), T), -1)

    return extrinsics, focal, near, far, viewpoint


class Testing_Camera(unittest.TestCase):

  def test_get_camera2world(self):

    bs = 1
    rot = torch.zeros(bs, 3).cuda()
    trans = torch.zeros(bs, 3).cuda()
    trans[:, 2] = 1

    rot = nn.Parameter(rot, requires_grad=True)
    trans = nn.Parameter(trans, requires_grad=True)

    rot_mat = Camera.get_camera2world(cam2world=rot, trans=trans)

    rot_mat.mean().backward()

    pass