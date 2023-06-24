import math
import copy
from time import perf_counter
import pathlib
import collections
import logging
import sys
import streamlit as st
import os
import trimesh
import numpy as np
import skvideo.io
from PIL import Image
from tqdm import tqdm
# from pdb import set_trace as st
from munch import *

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision import transforms

sys.path.insert(0, os.getcwd())

from pytorch3d.structures import Meshes

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import moxing_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.streamlit import SessionState
from tl2.proj.streamlit import st_utils
from tl2.proj.logger.logger_utils import get_file_logger
from tl2 import tl2_utils
from tl2.proj.streamlit import st_utils
from tl2.proj.fvcore import build_model, MODEL_REGISTRY, TLCfgNode
from tl2.proj.cv2 import cv2_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.pil import pil_utils
from tl2.proj.tools3d.camera_pose_visualizer import CameraPoseVisualizer

# sys.path.insert(0, f"{os.getcwd()}/DGP_lib")
# from DGP_lib import utils
# sys.path.pop(0)

# from exp.stylesdf.options import BaseOptions
from exp.cips3d.utils import (
  generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
  xyz2mesh, create_cameras, create_mesh_renderer, add_textures, mixing_noise
)
from exp.cips3d import nerf_utils
from .projector import StyleGAN2Projector, StyleGAN2Projector_Flip


def select_interp_state_dict(total_state_dict,
                             interp_indices=[0, 1, 2, 3, 4, 5, 6, 7],
                             interp_conv_in=False):
  interp_prefix = []
  if interp_conv_in:
    interp_prefix.extend(['conv1', 'to_rgb1'])
  for idx in interp_indices:
    interp_prefix.extend([f"convs.{idx * 2}.", f"convs.{idx * 2 + 1}.", f"to_rgbs.{idx}."])
  interp_prefix = tuple(interp_prefix)

  interp_state_dict = {}
  for name in total_state_dict.keys():
    if name.startswith(interp_prefix):
      interp_state_dict[name] = total_state_dict[name]

  return interp_state_dict


def render_video(cfg,
                 outdir,
                 g_ema,
                 device,
                 N_frames,
                 view_mode,
                 fps,
                 hd_video,
                 truncation,
                 N_samples,
                 zero_noise_bufs,
                 G_kwargs,
                 seed,
                 saved_w_file=None,
                 truncation_content=0,
                 interp_indices=[],
                 interp_conv_in=False,
                 light_location=(0.0, 1.0, 5.0),
                 project_noise=False,
                 **kwargs):
  g_ema.eval()

  # G_kwargs = cfg.G_kwargs.clone()

  nerf_cfg = G_kwargs.nerf_cfg.clone()
  nerf_cfg.perturb = False
  nerf_cfg.N_samples = N_samples

  cam_cfg = G_kwargs.cam_cfg.clone()
  # cam_cfg.img_size = img_size

  # Generate video trajectory
  trajectory = np.zeros((N_frames, 3), dtype=np.float32)

  # set camera trajectory
  # sweep azimuth angles (4 seconds)
  if view_mode == 'yaw':
    azim_range = cfg.get('azim_range', cam_cfg.azim_range)
    elev_range = cfg.get('elev_range', cam_cfg.elev_range)

    fov = cam_cfg.fov_ang
    t = np.linspace(0, 1, N_frames)
    if isinstance(elev_range, list) and isinstance(azim_range, list):
      elev = np.mean(elev_range)
      azim = azim_range[0] + (azim_range[1] - azim_range[0]) * np.sin(t * np.pi)

    else:
      elev = 0
      if cam_cfg.uniform:
        azim = azim_range * np.cos(t * 2 * np.pi)
      else:
        azim = 1 * azim_range * np.cos(t * 2 * np.pi)

    trajectory[:N_frames, 0] = azim
    trajectory[:N_frames, 1] = elev
    trajectory[:N_frames, 2] = fov

  # elipsoid sweep (4 seconds)
  elif view_mode == 'circle':
    t = np.linspace(0, 1, N_frames)
    fov = cam_cfg.fov_ang  # + 1 * np.sin(t * 2 * np.pi)

    if isinstance(cam_cfg.elev_range, list):
      cam_cfg.azim_range = 0.3
      cam_cfg.elev_range = 0.15

    if cam_cfg.uniform:
      elev = cam_cfg.elev_range / 2 + cam_cfg.elev_range / 2 * np.sin(t * 2 * np.pi)
      azim = cam_cfg.azim_range * np.cos(t * 2 * np.pi)
    else:
      elev = 1. * cam_cfg.elev_range * np.sin(t * 2 * np.pi)
      azim = 1. * cam_cfg.azim_range * np.cos(t * 2 * np.pi)

    trajectory[:N_frames, 0] = azim
    trajectory[:N_frames, 1] = elev
    trajectory[:N_frames, 2] = fov

  trajectory = torch.from_numpy(trajectory).to(device)

  if saved_w_file and os.path.isfile(saved_w_file):
    loaded_kwargs = torch_utils.torch_load(saved_w_file, rank=0)
    loaded_azim = loaded_kwargs['azim'][0].item()
    loaded_elev = loaded_kwargs['elev'][0].item()
    end_azim = trajectory[0, 0].item()
    end_elev = trajectory[0, 1].item()
    pre_azims = torch.linspace(loaded_azim, end_azim, N_frames // 2, device=device)
    pre_elev = torch.linspace(loaded_elev, end_elev, N_frames // 2, device=device)
    pre_fov = torch.empty(N_frames // 2, device=device).fill_(fov)
    pre_traj = torch.stack([pre_azims, pre_elev, pre_fov], dim=1)
    trajectory = torch.cat([pre_traj, trajectory, pre_traj.flip(dims=(0, ))], dim=0)

    N_frames += N_frames

    w_render_opt = loaded_kwargs['w_render_opt'].data[[0]]
    w_decoder_opt = loaded_kwargs['w_decoder_opt'].data[[0]]

    interp_state_dict = select_interp_state_dict(total_state_dict=loaded_kwargs['decoder_state_dict'],
                                                 interp_indices=interp_indices,
                                                 interp_conv_in=interp_conv_in)
    # in-place interp
    torch_utils.ema_accumulate(interp_state_dict,
                               g_ema.decoder,
                               truncation_content)
    Checkpointer(g_ema.decoder).load_state_dict(loaded_kwargs['decoder_state_dict'])


    loaded_noise_bufs = loaded_kwargs.get('noise_bufs', None)
    if loaded_noise_bufs is not None:
      loaded_noise_bufs = [buf.requires_grad_(False) for buf in loaded_noise_bufs]

    nerf_cfg.static_viewdirs = True

  else:
    w_render_opt = None
    w_decoder_opt = None
    loaded_noise_bufs = None

  # generate input parameters for the camera trajectory
  sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
    locations=trajectory[:, :2],
    device=device,
    **{**cam_cfg,
       'fov_ang': trajectory[:, 2:]})

  # In case of noise projection, generate input parameters for the frontal position.
  # The reference mesh for the noise projection is extracted from the frontal position.
  # For more details see section C.1 in the supplementary material.
  if project_noise:
    frontal_pose = torch.tensor([[0.0, 0.0, opt.camera.fov]]).to(device)
    # frontal_cam_pose, frontal_focals, frontal_near, frontal_far = \
    # generate_camera_params(frontal_pose, opt.surf_extraction_output_size, device, dist_radius=opt.camera.dist_radius)
    frontal_cam_pose, frontal_focals, frontal_near, frontal_far, _ = \
      generate_camera_params(resolution=opt.surf_extraction_output_size,
                             device=device,
                             location=frontal_pose[:, :2],
                             fov_ang=frontal_pose[:, 2:],
                             dist_radius=opt.camera.dist_radius)

  # create geometry renderer (renders the depth maps)
  cameras = create_cameras(azim=np.rad2deg(trajectory[0, 0].cpu().numpy()),
                           elev=np.rad2deg(trajectory[0, 1].cpu().numpy()),
                           dist=1, device=device)
  renderer = create_mesh_renderer(cameras=cameras,
                                  image_size=512,
                                  specular_color=((0, 0, 0),),
                                  ambient_color=((0.1, .1, .1),),
                                  diffuse_color=((0.75, .75, .75),),
                                  device=device)

  # generate videos
  chunk = 1
  # sample_z = torch.randn(1, g_ema.z_dim, device=device).repeat(chunk, 1)
  # sample_z = mixing_noise(chunk, g_ema.z_dim, 0.9, device)
  sample_z = torch_utils.sample_noises(bs=chunk, noise_dim=g_ema.z_dim, device=device, N_samples=2, seed=seed)

  video_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video.mp4", fps=fps, hd_video=hd_video)
  video_rgb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_rgb.mp4", fps=fps, hd_video=hd_video)
  video_thumb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_thumb.mp4", fps=fps, hd_video=hd_video)
  video_mesh_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_mesh.mp4", fps=fps, hd_video=hd_video)

  image_st = st.empty()


  ####################### Extract initial surface mesh from the frontal viewpoint #############
  # For more details see section C.1 in the supplementary material.
  if project_noise:
    with torch.no_grad():
      frontal_surface_out = surface_g_ema([sample_z],
                                          frontal_cam_pose,
                                          frontal_focals,
                                          frontal_near,
                                          frontal_far,
                                          truncation=opt.truncation_ratio,
                                          truncation_latent=surface_mean_latent,
                                          return_sdf=True)
      frontal_sdf = frontal_surface_out[2].cpu()
    # print('Extracting Identity {} Frontal view Marching Cubes for consistent video rendering'.format(i))

    frostum_aligned_frontal_sdf = align_volume(frontal_sdf)
    del frontal_sdf

    try:
      frontal_marching_cubes_mesh = extract_mesh_with_marching_cubes(frostum_aligned_frontal_sdf)
    except ValueError:
      frontal_marching_cubes_mesh = None

    if frontal_marching_cubes_mesh != None:
      # frontal_marching_cubes_mesh_filename = os.path.join(opt.results_dst_dir,
      #                                                     'sample_{}_frontal_marching_cubes_mesh{}.obj'.format(i, suffix))
      frontal_marching_cubes_mesh_filename = f"{outdir}/frontal_marching_cubes_mesh.obj"
      with open(frontal_marching_cubes_mesh_filename, 'w') as f:
        frontal_marching_cubes_mesh.export(f, file_type='obj')

    del frontal_surface_out
    torch.cuda.empty_cache()
  #############################################################################################

  N_rays_forward = None
  N_samples_forward = None

  if loaded_noise_bufs is None:
    if hasattr(g_ema, 'create_noise_bufs'):
      noise_bufs = g_ema.create_noise_bufs(start_size=cam_cfg.img_size, device=device)
    else:
      noise_bufs = None
      N_rays_forward = 128**2
      N_samples_forward = 128**2

  else:
    noise_bufs = loaded_noise_bufs

  if zero_noise_bufs:
    [buf.zero_() for buf in noise_bufs]

  for j in tqdm(range(0, N_frames, chunk)):
    with torch.no_grad():
      ret_maps = g_ema(zs=sample_z,
                       style_render=w_render_opt,
                       style_decoder=w_decoder_opt,
                       cam_poses=sample_cam_extrinsics[j:j + chunk],
                       focals=sample_focals[j:j + chunk],
                       img_size=cam_cfg.img_size,
                       near=sample_near[j:j + chunk],
                       far=sample_far[j:j + chunk],
                       noise_bufs=noise_bufs,
                       truncation=truncation,
                       N_rays_forward=N_rays_forward,
                       N_rays_grad=None,
                       N_samples_forward=N_samples_forward,
                       eikonal_reg=False,
                       nerf_cfg=nerf_cfg,
                       recompute_mean=False,
                       return_xyz=True,
                       mesh_path=frontal_marching_cubes_mesh_filename if project_noise else None)
      rgb = ret_maps['rgb']
      rgb_pil = torch_utils.img_tensor_to_pil(rgb)

      rgb_pil = pil_utils.add_text(
        rgb_pil, f"azimuth:".ljust(10) + f"{trajectory[j, 0].item():.2f}\n"
                                         f"elevation:".ljust(14) + f"{trajectory[j, 1].item():.2f}",
        size=rgb_pil.size[0] // 15, color=(255, 0, 0), clone=False)
      # pil_utils.imshow_pil(rgb_pil)

      aux_imgs = ret_maps['thumb_rgb']
      aux_imgs_pil = torch_utils.img_tensor_to_pil(aux_imgs)
      aux_imgs_pil = pil_utils.pil_resize(aux_imgs_pil, rgb.shape[-2:])


      ########## Extract surface ##########
      xyz = ret_maps['xyz'].cpu()

      # Render mesh for video
      depth_mesh = xyz2mesh(xyz)
      mesh = Meshes(
        verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
        faces=[torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
        textures=None,
        verts_normals=[
          torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
      )
      mesh = add_textures(mesh)
      cameras = create_cameras(azim=np.rad2deg(trajectory[j, 0].cpu().numpy()),
                               elev=np.rad2deg(trajectory[j, 1].cpu().numpy()),
                               fov=2 * trajectory[j, 2].cpu().numpy(),
                               dist=1,
                               device=device)
      renderer = create_mesh_renderer(cameras=cameras,
                                      image_size=512,
                                      light_location=(light_location,),
                                      specular_color=((0.2, 0.2, 0.2),),
                                      ambient_color=((0.1, 0.1, 0.1),),
                                      diffuse_color=((0.65, .65, .65),),
                                      device=device)

      # mesh_image = 255 * renderer(mesh).cpu().numpy()
      # mesh_image = mesh_image[..., :3]
      mesh_image = renderer(mesh)[..., :3].squeeze().cpu().numpy()
      mesh_image_pil = pil_utils.np_to_pil(mesh_image, range01=True)
      mesh_image_pil = pil_utils.pil_resize(mesh_image_pil, rgb.shape[-2:])

      if cfg.show_aux_img:
        merged_pil = pil_utils.merge_image_pil([rgb_pil, aux_imgs_pil, mesh_image_pil], nrow=3)
      else:
        merged_pil = pil_utils.merge_image_pil([rgb_pil, mesh_image_pil], nrow=2)
      video_f.write(merged_pil)
      video_rgb_f.write(rgb_pil)
      video_thumb_f.write(aux_imgs_pil)
      video_mesh_f.write(mesh_image_pil)
      st_utils.st_image(merged_pil, caption=f"{rgb.shape}", debug=global_cfg.tl_debug, st_empty=image_st)

  # Close video writers
  # writer.close()
  # if not opt.no_surface_renderings:
  #   depth_writer.close()
  video_f.release(st_video=True)
  video_rgb_f.release(st_video=True)
  video_thumb_f.release(st_video=True)
  video_mesh_f.release(st_video=True)
  pass


def interpolate_sphere(z1,
                       z2,
                       t):
  p = (z1 * z2).sum(dim=-1, keepdim=True)
  p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
  p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
  omega = torch.acos(p)
  s1 = torch.sin((1 - t) * omega) / torch.sin(omega)
  s2 = torch.sin(t * omega) / torch.sin(omega)
  z = s1 * z1 + s2 * z2
  return z


@MODEL_REGISTRY.register(name_prefix=__name__)
class STModel(object):
  def __init__(self):

    pass

  def _render_video_web(self,
                        cfg,
                        outdir,
                        saved_suffix_state=None,
                        **kwargs):
    from tl2.proj.streamlit import st_utils

    st_utils.st_set_sep('video')
    N_frames = st_utils.number_input('N_frames', cfg.N_frames, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    view_mode = st_utils.selectbox('view_mode', cfg.view_mode, default_value=cfg.default_view_mode, sidebar=True)
    seed = st_utils.get_seed(cfg.seeds)

    st_utils.st_set_sep('gen')
    truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)
    light_location = st_utils.parse_list_from_st_text_input(
      'light_location', cfg.get('light_location', [0.0, 1.0, 5.0]), sidebar=True)
    cfg.show_aux_img = st_utils.checkbox('show_aux_img', True, sidebar=True)

    st_utils.st_set_sep('Loading proj w')
    saved_w_file = st_utils.text_input('saved_w_file', cfg.saved_w_file, sidebar=True)
    w_gallery = st_utils.selectbox_v1('w_gallery', cfg.w_gallery, default_key=cfg.default_w_gallery, sidebar=True)
    use_w_gallery = st_utils.checkbox('use_w_gallery', cfg.use_w_gallery, sidebar=True)
    if use_w_gallery:
      saved_w_file = w_gallery

    truncation_content = st_utils.number_input(
      'truncation_content', cfg.truncation_content, sidebar=True, format="%.2f")
    interp_indices = st_utils.parse_list_from_st_text_input('interp_indices', cfg.interp_indices, sidebar=True)
    interp_conv_in = st_utils.checkbox('interp_conv_in', cfg.interp_conv_in, sidebar=True)

    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.cam_cfg.img_size = st_utils.number_input('img_size', G_kwargs.cam_cfg.img_size, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.get('upsample_list', []), sidebar=True)
    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', G_kwargs.nerf_cfg.static_viewdirs, sidebar=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # torch_utils.init_seeds(seed)

    device = "cuda"


    g_ema = build_model(cfg=G_cfg).cuda()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")

    # cfg.G_kwargs.cam_cfg.azim_range = 0.3
    # cfg.G_kwargs.cam_cfg.elev_range = 0.15

    render_video(cfg=cfg,
                 outdir=outdir,
                 g_ema=g_ema,
                 device=device,
                 N_frames=N_frames,
                 view_mode=view_mode,
                 fps=fps,
                 hd_video=hd_video,
                 N_samples=N_samples,
                 zero_noise_bufs=zero_noise_bufs,
                 G_kwargs=G_kwargs,
                 light_location=light_location,
                 truncation=truncation_ratio,
                 saved_w_file=saved_w_file,
                 truncation_content=truncation_content,
                 interp_indices=interp_indices,
                 interp_conv_in=interp_conv_in,
                 seed=seed)

    torch.cuda.empty_cache()
    pass

  def _interpolate_z_web(self,
                         cfg,
                         outdir,
                         saved_suffix_state=None,
                         **kwargs):
    from tl2.proj.streamlit import st_utils

    N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    N_step = st_utils.number_input('N_step', cfg.N_step, sidebar=True)
    z_mode = st_utils.selectbox('z_mode', options=cfg.z_mode, default_value=cfg.default_z_mode, sidebar=True)
    interp_mode = st_utils.selectbox('interp_mode', options=cfg.interp_mode, sidebar=True)
    azim = st_utils.number_input('azim', cfg.azim, sidebar=True, format="%.3f")
    elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    fov = st_utils.number_input('fov', cfg.fov, sidebar=True, format="%.3f")
    truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    seed = st_utils.get_seed(cfg.seeds)

    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.cam_cfg.img_size = st_utils.number_input('img_size', G_kwargs.cam_cfg.img_size, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.upsample_list, sidebar=True)
    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', G_kwargs.nerf_cfg.static_viewdirs, sidebar=True)
    G_kwargs.nerf_cfg.N_samples = st_utils.number_input('N_samples', 120, sidebar=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    torch_utils.init_seeds(seed)

    device = "cuda"

    g_ema = build_model(cfg=G_cfg).cuda().eval()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")


    cam_cfg = G_kwargs.cam_cfg
    nerf_cfg = G_kwargs.nerf_cfg

    trajectory = np.zeros((1, 2), dtype=np.float32)
    trajectory = torch.from_numpy(trajectory).to(device)

    trajectory[0, 0] = azim
    trajectory[0, 1] = elev
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
      locations=trajectory,
      device=device,
      **{**cam_cfg,
         'fov_ang': fov})

    zs = mixing_noise(1, g_ema.z_dim, 0.9, device, N_noise=3)
    zs = list(zs)
    zs_list = [mixing_noise(1, g_ema.z_dim, 0.9, device, N_noise=3) for _ in range(N_samples)]

    z_mode_dict = {
      'shape': 0,
      'app': 1,
      'style': 2,
    }
    noise_bufs = g_ema.create_noise_bufs(start_size=cam_cfg.img_size, device=device)
    style_render_mean, style_decoder_mean = g_ema.get_mean_latent(N_noises=10000, device=zs[0].device)

    st_image = st.empty()
    video_f = cv2_utils.ImageioVideoWriter(f"{outdir}/video.mp4", fps=fps, hd_video=hd_video)

    for idx in range(N_samples):
      zs1 = zs_list[idx]
      zs2 = zs_list[(idx + 1) % N_samples]
      ts = np.linspace(0, 1, N_step)
      for t in ts:
        z1_ = zs1[z_mode_dict[z_mode]]
        z2_ = zs2[z_mode_dict[z_mode]]

        if interp_mode == 'slerp':
          z_interp_ = interpolate_sphere(z1_, z2_, t)
        elif interp_mode == 'lerp':
          z_interp_ = torch.lerp(z1_, z2_, t)
        else:
          assert 0
        zs[z_mode_dict[z_mode]] = z_interp_

        w_shape = g_ema.style(zs[0])
        w_shape = torch.lerp(style_render_mean, w_shape, truncation_ratio)\
          .view(1, 1, -1).repeat(1, g_ema.N_layers_renderer, 1)

        w_app = g_ema.style(zs[1])
        w_app = torch.lerp(style_render_mean, w_app, truncation_ratio).view(1, 1, -1)
        style_render = torch.cat([w_shape, w_app], dim=1)

        w_style = g_ema.style_decoder(zs[2])
        w_style = torch.lerp(style_decoder_mean, w_style, truncation_ratio)\
          .view(1, 1, -1).repeat(1, g_ema.decoder.n_latent, 1)
        style_decoder = w_style

        img_list = []
        with torch.set_grad_enabled(False):
          ret_maps = g_ema(zs=zs[:2],
                           style_render=style_render,
                           style_decoder=style_decoder,
                           cam_poses=sample_cam_extrinsics,
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
                           return_xyz=True)
          rgb = ret_maps['rgb']
          rgb_pil = torch_utils.img_tensor_to_pil(rgb)

          aux_imgs = ret_maps['thumb_rgb']
          aux_imgs_pil = torch_utils.img_tensor_to_pil(aux_imgs)
          aux_imgs_pil = pil_utils.pil_resize(aux_imgs_pil, rgb.shape[-2:])


        merged_pil = pil_utils.merge_image_pil([rgb_pil, aux_imgs_pil, ], nrow=2)
        img_str = f"{idx}-{(idx + 1) % N_samples}/{N_samples}, t={t:.2f}"
        pil_utils.add_text(merged_pil, img_str, size=merged_pil.size[0] // 18)

        st_utils.st_image(merged_pil, caption=f"{rgb.shape}",
                          debug=global_cfg.tl_debug, st_empty=st_image)
        video_f.write(merged_pil)

    video_f.release(st_video=True)


    pass

  def _inversion_web(self,
                     cfg,
                     outdir,
                     saved_suffix_state=None,
                     **kwargs):

    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, show_image=False)
      image_list_kwargs[data_k]['image_path'] = image_path
    data_k = st_utils.radio('source', options=image_list_kwargs.keys(), index=0, sidebar=True)
    image_path = image_list_kwargs[data_k]['image_path']
    st_utils.st_show_image(image_path)

    # N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    # N_step = st_utils.number_input('N_step', cfg.N_step, sidebar=True)
    # z_mode = st_utils.selectbox('z_mode', options=cfg.z_mode, default_value=cfg.default_z_mode, sidebar=True)
    # interp_mode = st_utils.selectbox('interp_mode', options=cfg.interp_mode, sidebar=True)
    # azim = st_utils.number_input('azim', cfg.azim, sidebar=True, format="%.3f")
    # elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    # fov = st_utils.number_input('fov', cfg.fov, sidebar=True, format="%.3f")
    # truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    # fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    # hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_log_every = st_utils.number_input('st_log_every', 10, sidebar=True)

    cfg.perceptual_cfg.layers = st_utils.parse_list_from_st_text_input(
      'layers', cfg.perceptual_cfg.layers, sidebar=True)

    st_utils.st_set_sep("learning rate")
    cfg.lr_cam = st_utils.number_input('lr_cam', cfg.lr_cam, sidebar=True, format="%.6f")
    cfg.lr_render_w = st_utils.number_input('lr_render_w', cfg.lr_render_w, sidebar=True, format="%.6f")
    cfg.lr_decoder_w = st_utils.number_input('lr_decoder_w', cfg.lr_decoder_w, sidebar=True, format="%.6f")
    cfg.lr_decoder_params = st_utils.number_input('lr_decoder_params', cfg.lr_decoder_params, sidebar=True, format="%.6f")
    cfg.lr_noise = st_utils.number_input('lr_noise', cfg.lr_noise, sidebar=True, format="%.6f")

    st_utils.st_set_sep("loss weights")
    cfg.rgb_weight = st_utils.number_input('rgb_weight', cfg.rgb_weight, sidebar=True)
    cfg.thumb_weight = st_utils.number_input('thumb_weight', cfg.thumb_weight, sidebar=True)
    cfg.mse_weight = st_utils.number_input('mse_weight', cfg.mse_weight, sidebar=True)
    cfg.regularize_noise_weight = st_utils.number_input(
      'regularize_noise_weight', cfg.regularize_noise_weight, sidebar=True)

    st_utils.st_set_sep("whether optimizing")
    cfg.optim_cam = st_utils.checkbox('optim_cam', cfg.optim_cam, sidebar=True)
    cfg.optim_render_w = st_utils.checkbox('optim_render_w', cfg.optim_render_w, sidebar=True)
    cfg.optim_decoder_w = st_utils.checkbox('optim_decoder_w', cfg.optim_decoder_w, sidebar=True)
    cfg.optim_decoder_params = st_utils.checkbox('optim_decoder_params', cfg.optim_decoder_params, sidebar=True)
    cfg.optim_noise_bufs = st_utils.checkbox('optim_noise_bufs', cfg.optim_noise_bufs, sidebar=True)
    cfg.zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)

    cfg.N_steps_pose = st_utils.number_input('N_steps_pose', cfg.N_steps_pose, sidebar=True)
    cfg.N_steps_app = st_utils.number_input('N_steps_app', cfg.N_steps_app, sidebar=True)
    cfg.N_steps_multiview = st_utils.number_input('N_steps_multiview', cfg.N_steps_multiview, sidebar=True)
    cfg.truncation_psi = st_utils.number_input('truncation_psi', cfg.truncation_psi, sidebar=True, format="%.2f")
    cfg.perceptual_layers_multiview = st_utils.parse_list_from_st_text_input(
      'perceptual_layers_multiview', cfg.perceptual_layers_multiview, sidebar=True)

    seed = st_utils.get_seed(cfg.seeds)

    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', True, sidebar=True)
    G_kwargs.nerf_cfg.N_samples = st_utils.number_input('N_samples', G_kwargs.nerf_cfg.N_samples, sidebar=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    torch_utils.init_seeds(seed)

    device = "cuda"

    g_ema = build_model(cfg=G_cfg).cuda().eval()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")

    cam_cfg = G_kwargs.cam_cfg
    nerf_cfg = G_kwargs.nerf_cfg

    projector = StyleGAN2Projector(
      G=g_ema,
      device=device,
      perceptual_cfg=cfg.perceptual_cfg)

    start_time = perf_counter()
    projector.project_wplus(
      outdir=outdir,
      image_path=image_path,
      cam_cfg=cam_cfg,
      nerf_cfg=nerf_cfg,
      st_log_every=st_log_every,
      st_web=True,
      **cfg
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')


    pass

  def _flip_inversion_web(self,
                          cfg,
                          outdir,
                          saved_suffix_state=None,
                          **kwargs):

    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, show_image=False)
      image_list_kwargs[data_k]['image_path'] = image_path
    data_k = st_utils.radio('source', options=image_list_kwargs.keys(), index=0, sidebar=True)
    image_path = image_list_kwargs[data_k]['image_path']
    st_utils.st_show_image(image_path)
    if data_k == 'disney':
      cfg.default_network_pkl = 'FFHQ_disney'
      # cfg.optim_decoder_params = False

    # N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    # N_step = st_utils.number_input('N_step', cfg.N_step, sidebar=True)
    # z_mode = st_utils.selectbox('z_mode', options=cfg.z_mode, default_value=cfg.default_z_mode, sidebar=True)
    # interp_mode = st_utils.selectbox('interp_mode', options=cfg.interp_mode, sidebar=True)
    # azim = st_utils.number_input('azim', cfg.azim, sidebar=True, format="%.3f")
    # elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    # fov = st_utils.number_input('fov', cfg.fov, sidebar=True, format="%.3f")
    # truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    # fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    # hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_log_every = st_utils.number_input('st_log_every', 10, sidebar=True)

    cfg.perceptual_cfg.layers = st_utils.parse_list_from_st_text_input(
      'layers', cfg.perceptual_cfg.layers, sidebar=True)

    st_utils.st_set_sep("learning rate")
    cfg.lr_cam = st_utils.number_input('lr_cam', cfg.lr_cam, sidebar=True, format="%.6f")
    cfg.lr_render_w = st_utils.number_input('lr_render_w', cfg.lr_render_w, sidebar=True, format="%.6f")
    cfg.lr_decoder_w = st_utils.number_input('lr_decoder_w', cfg.lr_decoder_w, sidebar=True, format="%.6f")
    cfg.lr_decoder_params = st_utils.number_input('lr_decoder_params', cfg.lr_decoder_params, sidebar=True,
                                                  format="%.6f")
    cfg.lr_noise = st_utils.number_input('lr_noise', cfg.lr_noise, sidebar=True, format="%.6f")

    st_utils.st_set_sep("loss weights")
    cfg.rgb_weight = st_utils.number_input('rgb_weight', cfg.rgb_weight, sidebar=True)
    cfg.thumb_weight = st_utils.number_input('thumb_weight', cfg.thumb_weight, sidebar=True)
    cfg.mse_weight = st_utils.number_input('mse_weight', cfg.mse_weight, sidebar=True)
    cfg.regularize_noise_weight = st_utils.number_input(
      'regularize_noise_weight', cfg.regularize_noise_weight, sidebar=True)

    st_utils.st_set_sep("whether optimizing")
    cfg.optim_cam = st_utils.checkbox('optim_cam', cfg.optim_cam, sidebar=True)
    cfg.optim_render_w = st_utils.checkbox('optim_render_w', cfg.optim_render_w, sidebar=True)
    cfg.optim_render_params = st_utils.checkbox('optim_render_params', cfg.optim_render_params, sidebar=True)
    cfg.optim_decoder_w = st_utils.checkbox('optim_decoder_w', cfg.optim_decoder_w, sidebar=True)
    cfg.optim_decoder_params = st_utils.checkbox('optim_decoder_params', cfg.optim_decoder_params, sidebar=True)
    cfg.optim_noise_bufs = st_utils.checkbox('optim_noise_bufs', cfg.optim_noise_bufs, sidebar=True)
    cfg.zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)

    cfg.bs_cam = st_utils.number_input('bs_cam', cfg.bs_cam, sidebar=True)
    cfg.bs_render = st_utils.number_input('bs_render', cfg.bs_render, sidebar=True)
    cfg.bs_decoder = st_utils.number_input('bs_decoder', cfg.bs_decoder, sidebar=True)

    cfg.N_steps_pose = st_utils.number_input('N_steps_pose', cfg.N_steps_pose, sidebar=True)
    cfg.N_steps_app = st_utils.number_input('N_steps_app', cfg.N_steps_app, sidebar=True)
    cfg.N_steps_multiview = st_utils.number_input('N_steps_multiview', cfg.N_steps_multiview, sidebar=True)
    cfg.truncation_psi = st_utils.number_input('truncation_psi', cfg.truncation_psi, sidebar=True, format="%.2f")
    cfg.mask_background = st_utils.checkbox('mask_background', cfg.mask_background, sidebar=True)
    cfg.flip_w_decoder_every = st_utils.number_input('flip_w_decoder_every', cfg.flip_w_decoder_every, sidebar=True)

    seed = st_utils.get_seed(cfg.seeds)

    network_pkl_choice = st_utils.selectbox('network_pkl_choice', cfg.network_pkl_choice,
                                            default_value=cfg.default_network_pkl, sidebar=True)
    network_pkl = cfg.network_pkl[network_pkl_choice]
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', True, sidebar=True)
    G_kwargs.nerf_cfg.N_samples = st_utils.number_input('N_samples', G_kwargs.nerf_cfg.N_samples, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.get('upsample_list', []), sidebar=True)
    global_cfg.img_size = st_utils.number_input('img_size', 1024, sidebar=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    torch_utils.init_seeds(seed)

    device = "cuda"

    g_ema = build_model(cfg=G_cfg).cuda().eval()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")

    cam_cfg = G_kwargs.cam_cfg
    nerf_cfg = G_kwargs.nerf_cfg

    projector = StyleGAN2Projector_Flip(
      G=g_ema,
      device=device,
      perceptual_cfg=cfg.perceptual_cfg)

    start_time = perf_counter()
    projector.project_wplus(
      outdir=outdir,
      image_path=image_path,
      cam_cfg=cam_cfg,
      nerf_cfg=nerf_cfg,
      st_log_every=st_log_every,
      st_web=True,
      **cfg
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    torch.cuda.empty_cache()
    pass

  def __load_proj_w(self,
                    saved_w_file,
                    w_idx=0):
    loaded_kwargs = torch_utils.torch_load(saved_w_file, rank=0)

    loaded_azim = loaded_kwargs['azim'][w_idx].item()
    loaded_elev = loaded_kwargs['elev'][w_idx].item()
    w_render_opt = loaded_kwargs['w_render_opt'].data[[0]]
    w_decoder_opt = loaded_kwargs['w_decoder_opt'].data[[w_idx]]

    render_state_dict = loaded_kwargs.get('render_state_dict', None)
    decoder_state_dict = loaded_kwargs['decoder_state_dict']

    loaded_noise_bufs = loaded_kwargs.get('noise_bufs', None)
    if loaded_noise_bufs is not None:
      loaded_noise_bufs = [buf.requires_grad_(False) for buf in loaded_noise_bufs]

    return loaded_azim, loaded_elev, w_render_opt, w_decoder_opt, decoder_state_dict, \
           loaded_noise_bufs, render_state_dict

  def __get_img_pil(self,
                    ret_maps,
                    name, # rgb,
                    trajectory,
                    idx,
                    show_trajectory=True):
    img_tensor = ret_maps[name]
    img_tensor_pil = torch_utils.img_tensor_to_pil(img_tensor)
    if show_trajectory:
      img_tensor_pil = pil_utils.add_text(
        img_tensor_pil, f"{'azimuth:'.ljust(10)}{trajectory[idx, 0].item():.2f}\n"
                        f"elevation:".ljust(14) + f"{trajectory[idx, 1].item():.2f}",
        size=img_tensor_pil.size[0] // 15, color=(255, 0, 0), clone=False)

    return img_tensor_pil

  def __interp_state_dict_decoder(self,
                                  source_state_dict,
                                  target_state_dict,
                                  interp_conv_dict,
                                  interp_to_rgb_dict):

    # interp_prefix = []
    # if interp_conv_in:
    #   interp_prefix.extend(['conv1', 'to_rgb1'])
    # for idx in interp_indices:
    #   interp_prefix.extend([f"convs.{idx * 2}.", f"convs.{idx * 2 + 1}.", f"to_rgbs.{idx}."])
    # interp_prefix = tuple(interp_prefix)

    interp_dict = {}
    for name in interp_conv_dict.keys():
      if name.isdigit():
        interp_dict[f"convs.{name}."] = interp_conv_dict[name]
      else:
        interp_dict[name] = interp_conv_dict[name]
    for name in interp_to_rgb_dict.keys():
      if name.isdigit():
        interp_dict[f"to_rgbs.{name}."] = interp_to_rgb_dict[name]
      else:
        interp_dict[name] = interp_to_rgb_dict[name]

    temp = {}
    for prefix_name, value in interp_dict.items():
      for name in source_state_dict.keys():
        if name.startswith(prefix_name):
          temp[name] = value
    interp_dict = temp

    ret_state_dict = {}
    for name in source_state_dict.keys():
      with torch.no_grad():
        ret_state_dict[name] = source_state_dict[name] + \
                               (target_state_dict[name] - source_state_dict[name]) * interp_dict[name]
      pass

    return ret_state_dict

  def _render_multi_view_web(self,
                             cfg,
                             outdir,
                             saved_suffix_state=None,
                             **kwargs):
    from tl2.proj.streamlit import st_utils

    st_utils.st_set_sep('video')
    N_frames = st_utils.number_input('N_frames', cfg.N_frames, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_utils.st_set_sep('Camera pose')
    view_mode = st_utils.selectbox('view_mode', cfg.view_mode, default_value=cfg.default_view_mode, sidebar=True)
    if view_mode == 'yaw':
      azim_range = st_utils.parse_list_from_st_text_input('azim_range', cfg.azim_range, sidebar=True)
      elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    elif view_mode == 'circle':
      azim = st_utils.number_input('azim', cfg.circle.azim, sidebar=True, format="%.3f")
      elev = st_utils.number_input('elev', cfg.circle.elev, sidebar=True, format="%.3f")

    st_utils.st_set_sep('gen')
    seed = st_utils.get_seed(cfg.seeds)
    target_truncation = st_utils.number_input('target_truncation', cfg.target_truncation, format="%.3f", sidebar=True)
    zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)

    st_utils.st_set_sep('Update network_pkl')
    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()
    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox('static_viewdirs', True, sidebar=True)
    G_kwargs.nerf_cfg.N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.get('upsample_list', []), sidebar=True)

    st_utils.st_set_sep('Loading proj w (source)')
    saved_w_file = st_utils.text_input('saved_w_file', cfg.saved_w_file, sidebar=True)
    w_gallery = st_utils.selectbox_v1('w_gallery', cfg.w_gallery, default_key=cfg.default_w_gallery, sidebar=True)
    use_w_gallery = st_utils.checkbox('use_w_gallery', cfg.use_w_gallery, sidebar=True)
    if use_w_gallery:
      saved_w_file = w_gallery
    w_idx = st_utils.selectbox('w_idx', [0, 1], default_value=0, sidebar=True)

    show_trajectory = st_utils.checkbox('show_trajectory', True, sidebar=True)
    #######################################################
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1
    #######################################################

    device = "cuda"
    # torch_utils.init_seeds(seed)

    # kwargs
    nerf_cfg = G_kwargs.nerf_cfg.clone()
    nerf_cfg.perturb = False
    # nerf_cfg.N_samples = N_samples
    cam_cfg = G_kwargs.cam_cfg.clone()

    # load target model
    g_ema_source = build_model(cfg=G_cfg).cuda()
    Checkpointer(g_ema_source).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")
    g_ema_source.eval()

    # Generate video trajectory
    trajectory = np.zeros((N_frames, 3), dtype=np.float32)
    fov = cam_cfg.fov_ang
    if view_mode == 'yaw':
      t = np.linspace(0, 1, N_frames)
      azim = azim_range[0] + (azim_range[1] - azim_range[0]) * np.sin(t * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

    elif view_mode == 'circle':
      t = np.linspace(0, 1, N_frames)

      elev = 1. * elev * np.sin(t * 4 * np.pi)
      azim = 1. * azim * np.cos(t * 4 * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)

    # append pre and post trajectory
    loaded_azim, loaded_elev, \
    w_render_opt, w_decoder_opt, \
    decoder_state_dict, loaded_noise_bufs, \
    render_state_dict = self.__load_proj_w(saved_w_file=saved_w_file, w_idx=w_idx)
    # load w_2DNet
    Checkpointer(g_ema_source.decoder).load_state_dict(decoder_state_dict)
    if render_state_dict is not None:
      Checkpointer(g_ema_source.renderer).load_state_dict(render_state_dict)

    end_azim = trajectory[0, 0].item()
    end_elev = trajectory[0, 1].item()
    pre_azims = torch.linspace(loaded_azim, end_azim, N_frames // 2, device=device)
    pre_elev = torch.linspace(loaded_elev, end_elev, N_frames // 2, device=device)
    pre_fov = torch.empty(N_frames // 2, device=device).fill_(fov)

    pre_traj = torch.stack([pre_azims, pre_elev, pre_fov], dim=1)
    trajectory = torch.cat([pre_traj, trajectory, pre_traj.flip(dims=(0,))], dim=0)

    N_frames += N_frames

    # generate input parameters for the camera trajectory
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
      locations=trajectory[:, :2],
      device=device,
      **{**cam_cfg,
         'fov_ang': trajectory[:, 2:]})

    frontal_trajectory = torch.tensor([[0, 0, fov]], device=device)
    frontal_cam_extrinsics, _, _, _, _ = nerf_utils.Camera.generate_camera_params(
      locations=frontal_trajectory[:, :2],
      device=device,
      **{**cam_cfg,
         'fov_ang': frontal_trajectory[:, 2:]})

    # generate videos
    chunk = 1
    # sample_z = mixing_noise(chunk, g_ema_source.z_dim, 0.9, device)

    if loaded_noise_bufs is None:
      noise_bufs = g_ema_source.create_noise_bufs(start_size=cam_cfg.img_size, device=device)
    else:
      noise_bufs = loaded_noise_bufs
    if zero_noise_bufs:
      [buf.zero_() for buf in noise_bufs]

    image_st = st.empty()

    video_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video.mp4", fps=fps, hd_video=hd_video)
    video_rgb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_rgb.mp4", fps=fps, hd_video=hd_video)
    video_thumb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_thumb.mp4", fps=fps, hd_video=hd_video)
    video_mesh_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_mesh.mp4", fps=fps, hd_video=hd_video)
    video_cam_pose_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_cam_pose.mp4", fps=fps, hd_video=hd_video)



    for j in tqdm(range(0, N_frames, chunk)):
      with torch.no_grad():
        ret_maps_source = g_ema_source(zs=[None, None],
                                       style_render=w_render_opt,
                                       style_decoder=w_decoder_opt,
                                       cam_poses=sample_cam_extrinsics[j:j + chunk],
                                       focals=sample_focals[j:j + chunk],
                                       img_size=cam_cfg.img_size,
                                       near=sample_near[j:j + chunk],
                                       far=sample_far[j:j + chunk],
                                       noise_bufs=noise_bufs,
                                       truncation=1,
                                       N_rays_forward=None,
                                       N_rays_grad=None,
                                       N_samples_forward=None,
                                       eikonal_reg=False,
                                       nerf_cfg=nerf_cfg,
                                       recompute_mean=False,
                                       return_xyz=True)
        rgb_source_pil = self.__get_img_pil(ret_maps=ret_maps_source,
                                            name='rgb', trajectory=trajectory, idx=j,
                                            show_trajectory=show_trajectory)

        if j == 0:
          cam_pose_vis = CameraPoseVisualizer(figsize=(rgb_source_pil.size[0]/100, rgb_source_pil.size[1]/100))

        aux_imgs = ret_maps_source['thumb_rgb']
        aux_imgs_pil = torch_utils.img_tensor_to_pil(aux_imgs)
        aux_imgs_pil = pil_utils.pil_resize(aux_imgs_pil, rgb_source_pil.size)

        ########## Extract surface ##########
        xyz = ret_maps_source['xyz'].cpu()

        # Render mesh for video
        depth_mesh = xyz2mesh(xyz)
        mesh = Meshes(
          verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
          faces=[torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
          textures=None,
          verts_normals=[
            torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
        )
        mesh = add_textures(mesh)
        cameras = create_cameras(azim=np.rad2deg(trajectory[j, 0].cpu().numpy()),
                                 elev=np.rad2deg(trajectory[j, 1].cpu().numpy()),
                                 fov=2 * trajectory[j, 2].cpu().numpy(),
                                 dist=1,
                                 device=device)
        renderer = create_mesh_renderer(cameras=cameras,
                                        image_size=512,
                                        light_location=([0, 0, 5],),
                                        specular_color=((0.2, 0.2, 0.2),),
                                        ambient_color=((0.1, 0.1, 0.1),),
                                        diffuse_color=((0.65, .65, .65),),
                                        device=device)

        mesh_image = renderer(mesh)[..., :3].squeeze().cpu().numpy()
        mesh_image_pil = pil_utils.np_to_pil(mesh_image, range01=True)
        mesh_image_pil = pil_utils.pil_resize(mesh_image_pil, rgb_source_pil.size)

        cam_pose_vis.extrinsic2pyramid(extrinsic=sample_cam_extrinsics[j].cpu().numpy(), color=cam_pose_vis.cmap(1.))
        cam_pose_pil = cam_pose_vis.to_pil()

        merged_pil = pil_utils.merge_image_pil([rgb_source_pil,
                                                aux_imgs_pil,
                                                mesh_image_pil,
                                                cam_pose_pil], nrow=2)
        st_utils.st_image(merged_pil, caption=f"{rgb_source_pil.size}", debug=global_cfg.tl_debug, st_empty=image_st)

        video_f.write(merged_pil)
        video_rgb_f.write(rgb_source_pil)
        video_thumb_f.write(aux_imgs_pil)
        video_mesh_f.write(mesh_image_pil)
        video_cam_pose_f.write(cam_pose_pil)

    video_f.release(st_video=True)
    video_rgb_f.release(st_video=True)
    video_thumb_f.release(st_video=True)
    video_mesh_f.release(st_video=True)
    video_cam_pose_f.release(st_video=True)

    torch.cuda.empty_cache()
    pass

  def _interpolate_decoder_web(self,
                               cfg,
                               outdir,
                               saved_suffix_state=None,
                               **kwargs):
    from tl2.proj.streamlit import st_utils

    st_utils.st_set_sep('video')
    N_frames = st_utils.number_input('N_frames', cfg.N_frames, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_utils.st_set_sep('Camera pose')
    view_mode = st_utils.selectbox('view_mode', cfg.view_mode, default_value=cfg.default_view_mode, sidebar=True)
    if view_mode == 'yaw':
      azim_range = st_utils.parse_list_from_st_text_input('azim_range', cfg.azim_range, sidebar=True)
      elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    elif view_mode == 'circle':
      azim = st_utils.number_input('azim', cfg.circle.azim, sidebar=True, format="%.3f")
      elev = st_utils.number_input('elev', cfg.circle.elev, sidebar=True, format="%.3f")

    st_utils.st_set_sep('gen')
    seed = st_utils.get_seed(cfg.seeds)
    target_truncation = st_utils.number_input('target_truncation', cfg.target_truncation, format="%.3f", sidebar=True)
    zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)

    st_utils.st_set_sep('Update network_pkl')
    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()
    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox('static_viewdirs', True, sidebar=True)
    G_kwargs.nerf_cfg.N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.get('upsample_list', []), sidebar=True)

    st_utils.st_set_sep('Loading proj w (source)')
    saved_w_file = st_utils.text_input('saved_w_file', cfg.saved_w_file, sidebar=True)
    w_gallery = st_utils.selectbox_v1('w_gallery', cfg.w_gallery, default_key=cfg.default_w_gallery, sidebar=True)
    use_w_gallery = st_utils.checkbox('use_w_gallery', cfg.use_w_gallery, sidebar=True)
    if use_w_gallery:
      saved_w_file = w_gallery
    w_idx = st_utils.selectbox('w_idx', [0, 1], default_value=0, sidebar=True)

    st_utils.st_set_sep('Loading proj w (target)')
    target_saved_w_file = st_utils.text_input('target_saved_w_file', cfg.target_saved_w_file, sidebar=True)
    target_w_gallery = st_utils.selectbox_v1('target_w_gallery', cfg.target_w_gallery,
                                             default_key=cfg.target_default_w_gallery, sidebar=True)
    target_use_w_gallery = st_utils.checkbox('target_use_w_gallery', cfg.target_use_w_gallery, sidebar=True)
    if target_use_w_gallery:
      target_saved_w_file = target_w_gallery
    target_w_idx = st_utils.selectbox('target_w_idx', [0, 1], default_value=0, sidebar=True)
    target_inverted_truncation = st_utils.number_input(
      'target_inverted_truncation', cfg.target_inverted_truncation, format="%.3f", sidebar=True)

    show_trajectory = st_utils.checkbox('show_trajectory', True, sidebar=True)

    st_utils.st_set_sep('Interpolation Shape', sidebar=False)
    interp_w_shape_weights_dict = st_utils.parse_dict_from_st_text_input(
      'interp_w_shape_weights_dict', cfg.interp_w_shape_weights_dict)

    st_utils.st_set_sep('Interpolation Appearance', sidebar=False)
    interp_w_app_weights_dict0 = st_utils.parse_dict_from_st_text_input(
      'interp_w_app_weights_dict0', cfg.interp_w_app_weights_dict0)
    interp_w_app_weights_dict1 = st_utils.parse_dict_from_st_text_input(
      'interp_w_app_weights_dict1', cfg.interp_w_app_weights_dict1)
    interp_w_app_weights_dict = {**interp_w_app_weights_dict0,
                                 **interp_w_app_weights_dict1}

    st_utils.st_set_sep('Interpolation 2DNet', sidebar=False)
    interp_conv_weights_dict0 = st_utils.parse_dict_from_st_text_input(
      'interp_conv_weights_dict0', cfg.interp_conv_weights_dict0)
    interp_conv_weights_dict1 = st_utils.parse_dict_from_st_text_input(
      'interp_conv_weights_dict1', cfg.interp_conv_weights_dict1)
    interp_to_rgb_weights_dict = st_utils.parse_dict_from_st_text_input(
      'interp_to_rgb_weights_dict', cfg.interp_to_rgb_weights_dict)
    interp_conv_weights_dict = {**interp_conv_weights_dict0,
                                **interp_conv_weights_dict1}

    #######################################################
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1
    #######################################################

    device = "cuda"
    # torch_utils.init_seeds(seed)

    # kwargs
    nerf_cfg = G_kwargs.nerf_cfg.clone()
    nerf_cfg.perturb = False
    # nerf_cfg.N_samples = N_samples
    cam_cfg = G_kwargs.cam_cfg.clone()

    # load target model
    g_ema_target = build_model(cfg=G_cfg).cuda()
    Checkpointer(g_ema_target).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")

    g_ema_source = copy.deepcopy(g_ema_target)
    g_ema_interp = copy.deepcopy(g_ema_target)

    g_ema_target.eval()
    g_ema_source.eval()
    g_ema_interp.eval()

    # Generate video trajectory
    trajectory = np.zeros((N_frames, 3), dtype=np.float32)
    fov = cam_cfg.fov_ang
    if view_mode == 'yaw':
      t = np.linspace(0, 1, N_frames)
      azim = azim_range[0] + (azim_range[1] - azim_range[0]) * np.sin(t * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

    elif view_mode == 'circle':
      t = np.linspace(0, 1, N_frames)

      elev = 1. * elev * np.sin(t * 4 * np.pi)
      azim = 1. * azim * np.cos(t * 4 * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)

    # append pre and post trajectory
    loaded_azim, loaded_elev,\
    w_render_opt, w_decoder_opt, \
    decoder_state_dict, loaded_noise_bufs, \
    render_state_dict = self.__load_proj_w(saved_w_file=saved_w_file, w_idx=w_idx)
    # load w_2DNet
    Checkpointer(g_ema_source.decoder).load_state_dict(decoder_state_dict)
    if render_state_dict is not None:
      Checkpointer(g_ema_source.renderer).load_state_dict(render_state_dict)

    end_azim = trajectory[0, 0].item()
    end_elev = trajectory[0, 1].item()
    pre_azims = torch.linspace(loaded_azim, end_azim, N_frames // 2, device=device)
    pre_elev = torch.linspace(loaded_elev, end_elev, N_frames // 2, device=device)
    pre_fov = torch.empty(N_frames // 2, device=device).fill_(fov)

    pre_traj = torch.stack([pre_azims, pre_elev, pre_fov], dim=1)
    trajectory = torch.cat([pre_traj, trajectory, pre_traj.flip(dims=(0,))], dim=0)

    N_frames += N_frames

    # sample target ws
    if os.path.isfile(target_saved_w_file): # use inverted target w_app
      _, _, \
      target_w_render_plus, target_w_decoder_plus_inverted, \
      target_decoder_state_dict, _, \
      target_render_state_dict = self.__load_proj_w(saved_w_file=target_saved_w_file, w_idx=target_w_idx)
      Checkpointer(g_ema_target.decoder).load_state_dict(target_decoder_state_dict)
      if target_render_state_dict is not None:
        Checkpointer(g_ema_target.renderer).load_state_dict(target_render_state_dict)

      with torch.no_grad():
        _, target_w_decoder_mean = g_ema_target.get_mean_latent(N_noises=10000, device=device)
        target_w_decoder_plus = target_w_decoder_mean[:, None, :] + \
                                (target_w_decoder_plus_inverted - target_w_decoder_mean[:, None, :]) * target_inverted_truncation

    else: # random sample target w_app
      sample_z = torch_utils.sample_noises(bs=1, noise_dim=g_ema_source.z_dim, device=device,
                                           N_samples=2, seed=seed)
      with torch.no_grad():
        target_w_render_plus, target_w_decoder_plus = g_ema_target.get_ws(
          zs=sample_z, truncation=target_truncation, device=device)
        # target_w_render_plus = w_render_opt.clone()
        target_w_decoder_plus_inverted = None

    # interp w_shape
    with torch.no_grad():
      interp_w_shape_weights_list = []
      for idx in range(3):
        interp_w_shape_weights_list.append(interp_w_shape_weights_dict[f"{idx}"])
      interp_w_shape_weights = torch.tensor(interp_w_shape_weights_list, device=device).view(1, -1, 1)

      interp_w_shape = w_render_opt + (target_w_render_plus - w_render_opt) * interp_w_shape_weights

    # interp w_appearance
    with torch.no_grad():
      interp_w_app_weights_list = []
      for idx in range(18):
        interp_w_app_weights_list.append(interp_w_app_weights_dict[f"{idx}"])
      interp_w_app_weights = torch.tensor(interp_w_app_weights_list, device=device).view(1, -1, 1)
      interp_w_app = w_decoder_opt + (target_w_decoder_plus - w_decoder_opt) * interp_w_app_weights
      # the mean w_app looks good
      # interp_w_app = w_decoder_opt + (target_w_decoder_plus - w_decoder_opt) * interp_w_app_weight

    # interp w_2DNet
    source_state_dict = copy.deepcopy(g_ema_source.decoder.state_dict())
    target_state_dict = copy.deepcopy(g_ema_target.decoder.state_dict())

    interp_state_dict = self.__interp_state_dict_decoder(
      source_state_dict=source_state_dict,
      target_state_dict=target_state_dict,
      interp_conv_dict=interp_conv_weights_dict,
      interp_to_rgb_dict=interp_to_rgb_weights_dict)
    Checkpointer(g_ema_interp.decoder).load_state_dict(interp_state_dict)

    # generate input parameters for the camera trajectory
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
      locations=trajectory[:, :2],
      device=device,
      **{**cam_cfg,
         'fov_ang': trajectory[:, 2:]})

    frontal_trajectory = torch.tensor([[0, 0, fov]], device=device)
    frontal_cam_extrinsics, _, _, _, _ = nerf_utils.Camera.generate_camera_params(
      locations=frontal_trajectory[:, :2],
      device=device,
      **{**cam_cfg,
         'fov_ang': frontal_trajectory[:, 2:]})

    # generate videos
    chunk = 1
    # sample_z = mixing_noise(chunk, g_ema_source.z_dim, 0.9, device)

    if loaded_noise_bufs is None:
      noise_bufs = g_ema_source.create_noise_bufs(start_size=cam_cfg.img_size, device=device)
    else:
      noise_bufs = loaded_noise_bufs
    if zero_noise_bufs:
      [buf.zero_() for buf in noise_bufs]

    image_st = st.empty()

    video_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video.mp4", fps=fps, hd_video=hd_video)
    video_rgb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_rgb.mp4", fps=fps, hd_video=hd_video)
    video_interp_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_interp.mp4", fps=fps, hd_video=hd_video)
    video_thumb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_thumb.mp4", fps=fps, hd_video=hd_video)
    video_mesh_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_mesh.mp4", fps=fps, hd_video=hd_video)
    video_cam_pose_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_cam_pose.mp4", fps=fps, hd_video=hd_video)

    cam_pose_vis = CameraPoseVisualizer()

    with torch.no_grad():
      ret_maps_target_frontal = g_ema_target(zs=[None, None],
                                             style_render=target_w_render_plus,
                                             style_decoder=target_w_decoder_plus,
                                             cam_poses=frontal_cam_extrinsics[[0]],
                                             focals=sample_focals[[0]],
                                             img_size=cam_cfg.img_size,
                                             near=sample_near[[0]],
                                             far=sample_far[[0]],
                                             noise_bufs=noise_bufs,
                                             truncation=1,
                                             N_rays_forward=None,
                                             N_rays_grad=None,
                                             N_samples_forward=None,
                                             eikonal_reg=False,
                                             nerf_cfg=nerf_cfg,
                                             recompute_mean=False,
                                             return_xyz=True)
      rgb_target_frontal_pil = self.__get_img_pil(ret_maps=ret_maps_target_frontal,
                                                  name='rgb', trajectory=frontal_trajectory, idx=0,
                                                  show_trajectory=show_trajectory)
      rgb_target_frontal_pil.save(f"{outdir}/target_frontal.jpg")
      st.image(rgb_target_frontal_pil, f"{outdir}/target_frontal.jpg")
      if global_cfg.tl_debug:
        pil_utils.imshow_pil(rgb_target_frontal_pil)


    for j in tqdm(range(0, N_frames, chunk)):
      with torch.no_grad():
        ret_maps_source = g_ema_source(zs=[None, None],
                                       style_render=w_render_opt,
                                       style_decoder=w_decoder_opt,
                                       cam_poses=sample_cam_extrinsics[j:j + chunk],
                                       focals=sample_focals[j:j + chunk],
                                       img_size=cam_cfg.img_size,
                                       near=sample_near[j:j + chunk],
                                       far=sample_far[j:j + chunk],
                                       noise_bufs=noise_bufs,
                                       truncation=1,
                                       N_rays_forward=None,
                                       N_rays_grad=None,
                                       N_samples_forward=None,
                                       eikonal_reg=False,
                                       nerf_cfg=nerf_cfg,
                                       recompute_mean=False,
                                       return_xyz=True)
        ret_maps_target = g_ema_target(zs=[None, None],
                                       style_render=target_w_render_plus,
                                       style_decoder=target_w_decoder_plus,
                                       cam_poses=sample_cam_extrinsics[j:j + chunk],
                                       focals=sample_focals[j:j + chunk],
                                       img_size=cam_cfg.img_size,
                                       near=sample_near[j:j + chunk],
                                       far=sample_far[j:j + chunk],
                                       noise_bufs=noise_bufs,
                                       truncation=1,
                                       N_rays_forward=None,
                                       N_rays_grad=None,
                                       N_samples_forward=None,
                                       eikonal_reg=False,
                                       nerf_cfg=nerf_cfg,
                                       recompute_mean=False,
                                       return_xyz=True)
        if target_w_decoder_plus_inverted is not None:
          ret_maps_target_inverted = g_ema_target(zs=[None, None],
                                                  style_render=target_w_render_plus,
                                                  style_decoder=target_w_decoder_plus_inverted,
                                                  cam_poses=sample_cam_extrinsics[j:j + chunk],
                                                  focals=sample_focals[j:j + chunk],
                                                  img_size=cam_cfg.img_size,
                                                  near=sample_near[j:j + chunk],
                                                  far=sample_far[j:j + chunk],
                                                  noise_bufs=noise_bufs,
                                                  truncation=1,
                                                  N_rays_forward=None,
                                                  N_rays_grad=None,
                                                  N_samples_forward=None,
                                                  eikonal_reg=False,
                                                  nerf_cfg=nerf_cfg,
                                                  recompute_mean=False,
                                                  return_xyz=True)
          rgb_target_inverted_pil = self.__get_img_pil(ret_maps=ret_maps_target_inverted,
                                                       name='rgb', trajectory=trajectory, idx=j)
        else:
          rgb_target_inverted_pil = None

        ret_maps_interp = g_ema_interp(zs=[None, None],
                                       style_render=interp_w_shape,
                                       style_decoder=interp_w_app,
                                       cam_poses=sample_cam_extrinsics[j:j + chunk],
                                       focals=sample_focals[j:j + chunk],
                                       img_size=cam_cfg.img_size,
                                       near=sample_near[j:j + chunk],
                                       far=sample_far[j:j + chunk],
                                       noise_bufs=noise_bufs,
                                       truncation=1,
                                       N_rays_forward=None,
                                       N_rays_grad=None,
                                       N_samples_forward=None,
                                       eikonal_reg=False,
                                       nerf_cfg=nerf_cfg,
                                       recompute_mean=False,
                                       return_xyz=True)

        rgb_source_pil = self.__get_img_pil(ret_maps=ret_maps_source,
                                            name='rgb', trajectory=trajectory, idx=j, show_trajectory=show_trajectory)
        rgb_target_pil = self.__get_img_pil(ret_maps=ret_maps_target,
                                            name='rgb', trajectory=trajectory, idx=j)
        rgb_interp_pil = self.__get_img_pil(ret_maps=ret_maps_interp,
                                            name='rgb', trajectory=trajectory, idx=j, show_trajectory=show_trajectory)

        aux_imgs = ret_maps_source['thumb_rgb']
        aux_imgs_pil = torch_utils.img_tensor_to_pil(aux_imgs)
        aux_imgs_pil = pil_utils.pil_resize(aux_imgs_pil, rgb_source_pil.size)

        ########## Extract surface ##########
        xyz = ret_maps_source['xyz'].cpu()

        # Render mesh for video
        depth_mesh = xyz2mesh(xyz)
        mesh = Meshes(
          verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
          faces=[torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
          textures=None,
          verts_normals=[
            torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
        )
        mesh = add_textures(mesh)
        cameras = create_cameras(azim=np.rad2deg(trajectory[j, 0].cpu().numpy()),
                                 elev=np.rad2deg(trajectory[j, 1].cpu().numpy()),
                                 fov=2 * trajectory[j, 2].cpu().numpy(),
                                 dist=1,
                                 device=device)
        renderer = create_mesh_renderer(cameras=cameras,
                                        image_size=512,
                                        light_location=([0, 0, 5],),
                                        specular_color=((0.2, 0.2, 0.2),),
                                        ambient_color=((0.1, 0.1, 0.1),),
                                        diffuse_color=((0.65, .65, .65),),
                                        device=device)

        mesh_image = renderer(mesh)[..., :3].squeeze().cpu().numpy()
        mesh_image_pil = pil_utils.np_to_pil(mesh_image, range01=True)
        mesh_image_pil = pil_utils.pil_resize(mesh_image_pil, rgb_source_pil.size)

        cam_pose_vis.extrinsic2pyramid(extrinsic=sample_cam_extrinsics[j].cpu().numpy(), color=cam_pose_vis.cmap(1.))
        cam_pose_pil = cam_pose_vis.to_pil()

        if rgb_target_inverted_pil is not None:
          merged_pil = pil_utils.merge_image_pil([rgb_source_pil, rgb_target_pil, rgb_interp_pil,
                                                  aux_imgs_pil, mesh_image_pil, rgb_target_inverted_pil,
                                                  cam_pose_pil], nrow=3)
        else:
          merged_pil = pil_utils.merge_image_pil([rgb_source_pil, rgb_target_pil, rgb_interp_pil,
                                                  aux_imgs_pil, mesh_image_pil,
                                                  cam_pose_pil], nrow=3)
        st_utils.st_image(merged_pil, caption=f"{rgb_source_pil.size}", debug=global_cfg.tl_debug, st_empty=image_st)


        video_f.write(merged_pil)
        video_rgb_f.write(rgb_source_pil)
        video_interp_f.write(rgb_interp_pil)
        video_thumb_f.write(aux_imgs_pil)
        video_mesh_f.write(mesh_image_pil)
        video_cam_pose_f.write(cam_pose_pil)

    video_f.release(st_video=True)
    video_rgb_f.release(st_video=True)
    video_interp_f.release(st_video=True)
    video_thumb_f.release(st_video=True)
    video_mesh_f.release(st_video=True)
    video_cam_pose_f.release(st_video=True)

    torch.cuda.empty_cache()
    pass

  def __get_trans_rotation_cams(self,
                                N_frames,
                                trans_max,
                                cam_cfg,
                                device):
    fov = cam_cfg.fov_ang

    t = torch.linspace(0, 1, N_frames)

    # translation
    cam_extrinsics_trans_x = torch.zeros(N_frames, 3, 4, device=device)
    # rotate
    cam_pose_eyes = torch.eye(3, device=device)
    cam_extrinsics_trans_x[:, :, :3].copy_(cam_pose_eyes)
    # trans
    trans_x = trans_max * torch.sin(t * 2 * torch.pi)
    cam_extrinsics_trans_x[:, 0, 3].copy_(trans_x)
    cam_extrinsics_trans_x[:, 2, 3].fill_(1)

    # cam_extrinsics_trans_y = torch.zeros(N_frames, 3, 4, device=device)
    # cam_extrinsics_trans_y[:, :, :3].copy_(cam_pose_eyes)
    # cam_extrinsics_trans_y[:, 1, 3].copy_(trans_x)
    # cam_extrinsics_trans_y[:, 2, 3].fill_(1)
    # total_frames += N_frames
    # cam_extrinsics_trans = torch.cat([cam_extrinsics_trans_x, cam_extrinsics_trans_y], dim=0)

    cam_extrinsics_trans = cam_extrinsics_trans_x

    trajectory_trans = torch.zeros(cam_extrinsics_trans.shape[0], 3, dtype=torch.float32, device=device)
    trajectory_trans[:, 2] = fov

    _, sample_focals_trans, sample_near_trans, sample_far_trans, _ = nerf_utils.Camera.generate_camera_params(
      locations=trajectory_trans[:, :2],
      device=device,
      **{**cam_cfg,
         'fov_ang': trajectory_trans[:, 2:]})

    # rotation
    alpha = t * 2 * torch.pi + 0.5 * torch.pi
    ups = torch.zeros(N_frames, 3, device=device)
    x = torch.cos(alpha)
    y = torch.sin(alpha)
    ups[:, 0] = x
    ups[:, 1] = y
    ups[:, 2].fill_(0)

    trajectory_rot = torch.zeros(N_frames, 3, dtype=torch.float32, device=device)
    trajectory_rot[:, 2] = fov

    cam_extrinsics_rot, sample_focals_rot, sample_near_rot, sample_far_rot, _ = nerf_utils.Camera.generate_camera_params_v1(
      locations=trajectory_rot[:, :2],
      up=ups,
      device=device,
      **{**cam_cfg,
         'fov_ang': trajectory_rot[:, 2:]})

    sample_cam_extrinsics = torch.cat([cam_extrinsics_trans, cam_extrinsics_rot], dim=0)
    trajectory = torch.cat([trajectory_trans, trajectory_rot], dim=0)
    sample_focals = torch.cat([sample_focals_trans, sample_focals_rot], dim=0)
    sample_near = torch.cat([sample_near_trans, sample_near_rot], dim=0)
    sample_far = torch.cat([sample_far_trans, sample_far_rot], dim=0)

    return sample_cam_extrinsics, trajectory, sample_focals, sample_near, sample_far

  def _sample_multi_view_web(self,
                             cfg,
                             outdir,
                             saved_suffix_state=None,
                             **kwargs):
    from tl2.proj.streamlit import st_utils

    st_utils.st_set_sep('video')
    N_frames = st_utils.number_input('N_frames', cfg.N_frames, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_utils.st_set_sep('G kwargs')
    truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)
    light_location = st_utils.parse_list_from_st_text_input(
      'light_location', cfg.get('light_location', [0.0, 0.0, 5.0]), sidebar=True)
    show_trajectory = st_utils.checkbox('show_trajectory', cfg.show_trajectory, sidebar=True)

    st_utils.st_set_sep('Sampling kwargs')
    seed = st_utils.get_seed(cfg.seeds)
    view_mode = st_utils.selectbox('view_mode', cfg.view_mode, default_value=cfg.default_view_mode, sidebar=True)
    if view_mode == 'yaw':
      azim_range = st_utils.parse_list_from_st_text_input('azim_range', cfg.azim_range, sidebar=True)
      elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    elif view_mode == 'translate_rotate':
      trans_max = st_utils.number_input('trans_max', cfg.trans_max, sidebar=True, format="%.3f")
    elif view_mode == 'circle':
      truncation_ratio = st_utils.number_input('truncation_ratio', 0.7, format="%.3f", sidebar=True)

      azim_range = st_utils.number_input('azim_range', cfg.circle.azim_range, sidebar=True)
      elev_range = st_utils.number_input('elev_range', cfg.circle.elev_range, sidebar=True)
      fov_range = st_utils.parse_list_from_st_text_input('fov_range', cfg.circle.fov_range, sidebar=True)

      pass

    st_utils.st_set_sep('network_pkl kwargs')
    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.cam_cfg.img_size = st_utils.number_input('img_size', G_kwargs.cam_cfg.img_size, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.get('upsample_list', []), sidebar=True)
    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', G_kwargs.nerf_cfg.static_viewdirs, sidebar=True)
    G_kwargs.cam_cfg.fov_ang = st_utils.number_input('fov_ang', G_kwargs.cam_cfg.fov_ang, sidebar=True)

    ############################################################################
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1
    ############################################################################

    # torch_utils.init_seeds(seed)

    device = "cuda"

    g_ema = build_model(cfg=G_cfg).cuda()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")
    g_ema.eval()

    nerf_cfg = G_kwargs.nerf_cfg.clone()
    nerf_cfg.perturb = False
    nerf_cfg.N_samples = N_samples

    cam_cfg = G_kwargs.cam_cfg.clone()

    fov = cam_cfg.fov_ang
    if view_mode == 'yaw':
      trajectory = np.zeros((N_frames, 3), dtype=np.float32)
      t = np.linspace(0, 1, N_frames)
      azim = azim_range[0] + (azim_range[1] - azim_range[0]) * np.sin(t * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

      trajectory = torch.from_numpy(trajectory).to(device)

      sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
        locations=trajectory[:, :2],
        device=device,
        **{**cam_cfg,
           'fov_ang': trajectory[:, 2:]})

    elif view_mode == 'translate_rotate':
      show_trajectory = False

      sample_cam_extrinsics, trajectory, sample_focals, sample_near, sample_far = \
        self.__get_trans_rotation_cams(N_frames=N_frames, trans_max=trans_max, cam_cfg=cam_cfg, device=device)
      N_frames = sample_cam_extrinsics.shape[0]

    elif view_mode == 'circle':
      trajectory = np.zeros((N_frames, 3), dtype=np.float32)

      t = np.linspace(0, 1, N_frames)

      azim = 1. * azim_range * np.sin(t * 2 * np.pi)
      elev = elev_range
      fov = fov_range[0] + (fov_range[1] - fov_range[0]) * np.sin(t * np.pi)


      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

      trajectory = torch.from_numpy(trajectory).to(device)

      sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
        locations=trajectory[:, :2],
        device=device,
        **{**cam_cfg,
           'fov_ang': trajectory[:, 2:]})

    # generate input parameters for the camera trajectory
    # sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
    #   locations=trajectory[:, :2],
    #   device=device,
    #   **{**cam_cfg,
    #      'fov_ang': trajectory[:, 2:]})

    noise_bufs = g_ema.create_noise_bufs(start_size=cam_cfg.img_size, device=device)
    if zero_noise_bufs:
      [buf.zero_() for buf in noise_bufs]

    chunk = 1
    sample_z = torch_utils.sample_noises(bs=chunk, noise_dim=g_ema.z_dim, device=device, N_samples=2, seed=seed)

    video_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video.mp4", fps=fps, hd_video=hd_video)
    video_rgb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_rgb.mp4", fps=fps, hd_video=hd_video)
    video_thumb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_thumb.mp4", fps=fps, hd_video=hd_video)
    video_mesh_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_mesh.mp4", fps=fps, hd_video=hd_video)

    image_st = st.empty()

    for j in tqdm(range(0, N_frames, chunk)):
      with torch.no_grad():
        ret_maps = g_ema(zs=sample_z,
                         style_render=None,
                         style_decoder=None,
                         cam_poses=sample_cam_extrinsics[j:j + chunk],
                         focals=sample_focals[j:j + chunk],
                         img_size=cam_cfg.img_size,
                         near=sample_near[j:j + chunk],
                         far=sample_far[j:j + chunk],
                         noise_bufs=noise_bufs,
                         truncation=truncation_ratio,
                         N_rays_forward=None,
                         N_rays_grad=None,
                         N_samples_forward=None,
                         eikonal_reg=False,
                         nerf_cfg=nerf_cfg,
                         recompute_mean=False,
                         return_xyz=True)
        rgb = ret_maps['rgb']
        rgb_pil = torch_utils.img_tensor_to_pil(rgb)

        if show_trajectory:
          rgb_pil = pil_utils.add_text(
            rgb_pil, f"azimuth:".ljust(10) + f"{trajectory[j, 0].item():.2f}\n"
                                             f"elevation:".ljust(14) + f"{trajectory[j, 1].item():.2f}",
            size=rgb_pil.size[0] // 15, color=(255, 0, 0), clone=False)
          # pil_utils.imshow_pil(rgb_pil)

        aux_imgs = ret_maps['thumb_rgb']
        aux_imgs_pil = torch_utils.img_tensor_to_pil(aux_imgs)
        aux_imgs_pil = pil_utils.pil_resize(aux_imgs_pil, rgb.shape[-2:])

        ########## Extract surface ##########
        xyz = ret_maps['xyz'].cpu()

        depth_mesh = xyz2mesh(xyz)
        mesh = Meshes(
          verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
          faces=[torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
          textures=None,
          verts_normals=[
            torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
        )
        mesh = add_textures(mesh)
        cameras = create_cameras(azim=np.rad2deg(trajectory[j, 0].cpu().numpy()),
                                 elev=np.rad2deg(trajectory[j, 1].cpu().numpy()),
                                 fov=2 * trajectory[j, 2].cpu().numpy(),
                                 dist=1,
                                 device=device)
        renderer = create_mesh_renderer(cameras=cameras,
                                        image_size=512,
                                        light_location=(light_location,),
                                        specular_color=((0.2, 0.2, 0.2),),
                                        ambient_color=((0.1, 0.1, 0.1),),
                                        diffuse_color=((0.65, .65, .65),),
                                        device=device)

        mesh_image = renderer(mesh)[..., :3].squeeze().cpu().numpy()
        mesh_image_pil = pil_utils.np_to_pil(mesh_image, range01=True)
        mesh_image_pil = pil_utils.pil_resize(mesh_image_pil, rgb.shape[-2:])

        merged_pil = pil_utils.merge_image_pil([rgb_pil, aux_imgs_pil, mesh_image_pil], nrow=3)
        st_utils.st_image(merged_pil, caption=f"{rgb.shape}", debug=global_cfg.tl_debug, st_empty=image_st)

        video_f.write(merged_pil)
        video_rgb_f.write(rgb_pil)
        video_thumb_f.write(aux_imgs_pil)
        video_mesh_f.write(mesh_image_pil)


    video_f.release(st_video=True)
    video_rgb_f.release(st_video=True)
    video_thumb_f.release(st_video=True)
    video_mesh_f.release(st_video=True)

    torch.cuda.empty_cache()
    pass

  def _style_mixing_web(self,
                        cfg,
                        outdir,
                        saved_suffix_state=None,
                        **kwargs):
    from tl2.proj.streamlit import st_utils

    st_utils.st_set_sep('video')
    N_frames = st_utils.number_input('N_frames', cfg.N_frames, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_utils.st_set_sep('G kwargs')
    truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)
    light_location = st_utils.parse_list_from_st_text_input(
      'light_location', cfg.get('light_location', [0.0, 0.0, 5.0]), sidebar=True)
    show_trajectory = st_utils.checkbox('show_trajectory', cfg.show_trajectory, sidebar=True)

    st_utils.st_set_sep('Sampling kwargs')
    seed = st_utils.get_seed(cfg.seeds)
    view_mode = st_utils.selectbox('view_mode', cfg.view_mode, default_value=cfg.default_view_mode, sidebar=True)
    if view_mode == 'yaw':
      azim_range = st_utils.parse_list_from_st_text_input('azim_range', cfg.azim_range, sidebar=True)
      elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    elif view_mode == 'translate_rotate':
      trans_max = st_utils.number_input('trans_max', cfg.trans_max, sidebar=True, format="%.3f")

    st_utils.st_set_sep('network_pkl kwargs')
    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.cam_cfg.img_size = st_utils.number_input('img_size', G_kwargs.cam_cfg.img_size, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.get('upsample_list', []), sidebar=True)
    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', G_kwargs.nerf_cfg.static_viewdirs, sidebar=True)

    chunk = st_utils.number_input('chunk', cfg.chunk, sidebar=True)
    N_rows = st_utils.number_input('N_rows', cfg.N_rows, sidebar=True)
    seed_list1 = np.random.randint(0, 1e6, size=N_rows)
    seed_list1 = [*cfg.seed_list1, *seed_list1][:N_rows]
    seed_list1 = st_utils.parse_list_from_st_text_input('seed_list1', seed_list1, sidebar=True)

    N_cols = st_utils.number_input('N_cols', cfg.N_cols, sidebar=True)
    seed_list2 = np.random.randint(0, 1e6, size=N_cols)
    seed_list2 = [*cfg.seed_list2, *seed_list2][:N_cols]
    seed_list2 = st_utils.parse_list_from_st_text_input('seed_list2', seed_list2, sidebar=True)

    ############################################################################
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1
    ############################################################################

    torch_utils.init_seeds(seed)

    device = "cuda"

    g_ema = build_model(cfg=G_cfg).cuda()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")
    g_ema.eval()

    nerf_cfg = G_kwargs.nerf_cfg.clone()
    nerf_cfg.perturb = False
    nerf_cfg.N_samples = N_samples

    cam_cfg = G_kwargs.cam_cfg.clone()

    fov = cam_cfg.fov_ang
    if view_mode == 'yaw':
      trajectory = np.zeros((N_frames, 3), dtype=np.float32)
      t = np.linspace(0, 1, N_frames)
      azim = azim_range[0] + (azim_range[1] - azim_range[0]) * np.sin(t * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

      trajectory = torch.from_numpy(trajectory).to(device)

      sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
        locations=trajectory[:, :2],
        device=device,
        **{**cam_cfg,
           'fov_ang': trajectory[:, 2:]})

    elif view_mode == 'circle':
      raise NotImplementedError

      t = np.linspace(0, 1, N_frames)
      fov = cam_cfg.fov_ang  # + 1 * np.sin(t * 2 * np.pi)

      if isinstance(cam_cfg.elev_range, list):
        cam_cfg.azim_range = 0.3
        cam_cfg.elev_range = 0.15

      if cam_cfg.uniform:
        elev = cam_cfg.elev_range / 2 + cam_cfg.elev_range / 2 * np.sin(t * 2 * np.pi)
        azim = cam_cfg.azim_range * np.cos(t * 2 * np.pi)
      else:
        elev = 1. * cam_cfg.elev_range * np.sin(t * 2 * np.pi)
        azim = 1. * cam_cfg.azim_range * np.cos(t * 2 * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

    noise_bufs = g_ema.create_noise_bufs(start_size=cam_cfg.img_size, device=device)
    if zero_noise_bufs:
      [buf.zero_() for buf in noise_bufs]

    # chunk = 1
    # sample_z = torch_utils.sample_noises(bs=chunk, noise_dim=g_ema.z_dim, device=device, N_samples=2, seed=seed)

    image_st = st.empty()

    # shape pose
    trajectory = np.zeros((N_rows, 3), dtype=np.float32)
    t = np.linspace(0, 1, N_rows)
    azim = azim_range[0] + (azim_range[1] - azim_range[0]) * t

    trajectory[:, 0] = azim
    trajectory[:, 1] = elev
    trajectory[:, 2] = fov
    # trajectory = torch.from_numpy(trajectory).to(device)

    shape_styles = []
    for seed in seed_list1:
      z_ = torch_utils.sample_noises(bs=1, noise_dim=g_ema.z_dim, device=device, N_samples=2, seed=seed)
      w_render_plus, w_decoder_plus = g_ema.get_ws(z_, truncation=truncation_ratio, device=device)
      shape_styles.append([w_render_plus, w_decoder_plus])

    app_styles = []
    for seed in seed_list2:
      z_ = torch_utils.sample_noises(bs=1, noise_dim=g_ema.z_dim, device=device, N_samples=2, seed=seed)
      w_render_plus, w_decoder_plus = g_ema.get_ws(z_, truncation=truncation_ratio, device=device)
      app_styles.append([w_render_plus, w_decoder_plus])

    style_render_list = []
    style_decoder_list = []
    trajectory_list = []

    for idx_r in range(N_rows + 1):
      for idx_c in range(N_cols + 1):
        if idx_r == 0 and idx_c == 0:
          continue
        elif idx_r == 0:
          style_render_list.append(app_styles[idx_c - 1][0])
          style_decoder_list.append(app_styles[idx_c - 1][1])
          trajectory_list.append([0, elev, fov])
        elif idx_c == 0:
          style_render_list.append(shape_styles[idx_r - 1][0])
          style_decoder_list.append(shape_styles[idx_r - 1][1])
          trajectory_list.append(list(trajectory[idx_r - 1]))
        else:
          style_render_list.append(shape_styles[idx_r - 1][0])
          style_decoder_list.append(app_styles[idx_c - 1][1])
          trajectory_list.append(list(trajectory[idx_r - 1]))
          pass

    style_renders = torch.cat(style_render_list, dim=0)
    style_decoders = torch.cat(style_decoder_list, dim=0)
    trajectory = torch.tensor(trajectory_list, device=device)

    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
      locations=trajectory[:, :2],
      device=device,
      **{**cam_cfg,
         'fov_ang': trajectory[:, 2:]})

    rgb_list = []

    for j in tqdm(range(0, len(style_renders), chunk)):
      with torch.no_grad():
        ret_maps = g_ema(zs=[None, None],
                         style_render=style_renders[j:j + chunk],
                         style_decoder=style_decoders[j:j + chunk],
                         cam_poses=sample_cam_extrinsics[j:j + chunk],
                         focals=sample_focals[j:j + chunk],
                         img_size=cam_cfg.img_size,
                         near=sample_near[j:j + chunk],
                         far=sample_far[j:j + chunk],
                         noise_bufs=noise_bufs,
                         truncation=truncation_ratio,
                         N_rays_forward=None,
                         N_rays_grad=None,
                         N_samples_forward=None,
                         eikonal_reg=False,
                         nerf_cfg=nerf_cfg,
                         recompute_mean=False,
                         return_xyz=True)
        rgbs = ret_maps['rgb']
        for rgb in rgbs:
          rgb_pil = torch_utils.img_tensor_to_pil(rgb)
          rgb_list.append(rgb_pil)

    pad_img = Image.new(mode='RGB', size=rgb_pil.size, color='white')
    rgb_list.insert(0, pad_img)
    merged_pil = pil_utils.merge_image_pil(rgb_list, nrow=N_cols + 1, dst_size=2048)
    st_utils.st_image(merged_pil, caption=f"{rgb.shape}", debug=global_cfg.tl_debug, st_empty=image_st)

    outdir = f"{outdir}/imgdir"
    os.makedirs(outdir, exist_ok=True)
    st.write(outdir)

    for idx_r in range(N_rows + 1):
      for idx_c in range(N_cols + 1):
        cur_idx = idx_r * (N_cols + 1) + idx_c

        if rgb_pil.size[0] > 512:
          img_pil = pil_utils.pil_resize(rgb_list[cur_idx], (512, 512))
        else:
          img_pil = rgb_list[cur_idx]

        img_pil.save(f"{outdir}/style_mixing_{idx_r}_{idx_c}.jpg")

    torch.cuda.empty_cache()
    pass

  def _fixed_zs_multi_view_web(self,
                               cfg,
                               outdir,
                               saved_suffix_state=None,
                               **kwargs):
    from tl2.proj.streamlit import st_utils

    st_utils.st_set_sep('video')
    N_frames = st_utils.number_input('N_frames', cfg.N_frames, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_utils.st_set_sep('G kwargs')
    truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)
    light_location = st_utils.parse_list_from_st_text_input(
      'light_location', cfg.get('light_location', [0.0, 0.0, 5.0]), sidebar=True)
    show_trajectory = st_utils.checkbox('show_trajectory', cfg.show_trajectory, sidebar=True)

    st_utils.st_set_sep('Sampling kwargs')
    seed = st_utils.get_seed(cfg.seeds)
    view_mode = st_utils.selectbox('view_mode', cfg.view_mode, default_value=cfg.default_view_mode, sidebar=True)
    if view_mode == 'yaw':
      azim_range = st_utils.parse_list_from_st_text_input('azim_range', cfg.azim_range, sidebar=True)
      elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    elif view_mode == 'translate_rotate':
      trans_max = st_utils.number_input('trans_max', cfg.trans_max, sidebar=True, format="%.3f")
    elif view_mode == 'circle':
      truncation_ratio = st_utils.number_input('truncation_ratio', 0.7, format="%.3f", sidebar=True)

      azim_range = st_utils.number_input('azim_range', cfg.circle.azim_range, sidebar=True)
      elev_range = st_utils.number_input('elev_range', cfg.circle.elev_range, sidebar=True)
      fov_range = st_utils.parse_list_from_st_text_input('fov_range', cfg.circle.fov_range, sidebar=True)
    elif view_mode == 'elev_circle':
      truncation_ratio = st_utils.number_input('truncation_ratio', 0.7, format="%.3f", sidebar=True)

      azim_range = st_utils.parse_list_from_st_text_input('azim_range', cfg.circle.azim_range, sidebar=True)
      elev_range = st_utils.number_input('elev_range', cfg.circle.elev_range, sidebar=True)
      fov_range = st_utils.parse_list_from_st_text_input('fov_range', cfg.circle.fov_range, sidebar=True)

      pass

    st_utils.st_set_sep('network_pkl kwargs')
    network_pkl = st_utils.selectbox_v1('network_pkl', cfg.network_pkl, default_key=cfg.default_network_pkl,
                                        sidebar=True)
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.cam_cfg.img_size = st_utils.number_input('img_size', G_kwargs.cam_cfg.img_size, sidebar=True)
    G_cfg.decoder_cfg.upsample_list = st_utils.parse_list_from_st_text_input(
      'upsample_list', G_cfg.decoder_cfg.get('upsample_list', []), sidebar=True)
    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', G_kwargs.nerf_cfg.static_viewdirs, sidebar=True)
    G_kwargs.cam_cfg.fov_ang = st_utils.number_input('fov_ang', G_kwargs.cam_cfg.fov_ang, sidebar=True)

    ############################################################################
    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1
    ############################################################################

    # torch_utils.init_seeds(seed)

    device = "cuda"

    g_ema = build_model(cfg=G_cfg).cuda()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")
    g_ema.eval()

    loaded_state = torch_utils.torch_load(f"{network_pkl}/state_dict.pth", rank=0)

    nerf_cfg = G_kwargs.nerf_cfg.clone()
    nerf_cfg.perturb = False
    nerf_cfg.N_samples = N_samples

    cam_cfg = G_kwargs.cam_cfg.clone()

    if view_mode == 'circle':
      trajectory = np.zeros((N_frames, 3), dtype=np.float32)

      t = np.linspace(0, 1, N_frames)

      azim = 1. * azim_range * np.sin(t * 2 * np.pi)
      elev = elev_range
      fov = fov_range[0] + (fov_range[1] - fov_range[0]) * np.sin(t * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

      trajectory = torch.from_numpy(trajectory).to(device)

      sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
        locations=trajectory[:, :2],
        device=device,
        **{**cam_cfg,
           'fov_ang': trajectory[:, 2:]})
    elif view_mode == 'elev_circle':
      N_frames = N_frames // 2

      trajectory_elev = np.zeros((N_frames, 3), dtype=np.float32)

      t = np.linspace(0, 1, N_frames)
      elev = elev_range * t
      trajectory_elev[:N_frames, 1] = elev
      trajectory_elev[:N_frames, 2] = fov_range[0]
      trajectory_elev = torch.from_numpy(trajectory_elev).to(device)

      trajectory = np.zeros((N_frames, 3), dtype=np.float32)

      t = np.linspace(0, 1, N_frames)

      azim = azim_range[0] + (azim_range[1] - azim_range[0]) * t
      elev = elev_range
      fov = fov_range[0] + (fov_range[1] - fov_range[0]) * np.sin(t * np.pi)

      trajectory[:N_frames, 0] = azim
      trajectory[:N_frames, 1] = elev
      trajectory[:N_frames, 2] = fov

      trajectory = torch.from_numpy(trajectory).to(device)

      trajectory = torch.cat([trajectory_elev, trajectory], dim=0)
      N_frames = len(trajectory)

      sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = nerf_utils.Camera.generate_camera_params(
        locations=trajectory[:, :2],
        device=device,
        **{**cam_cfg,
           'fov_ang': trajectory[:, 2:]})
    else:
      raise NotImplementedError

    noise_bufs = g_ema.create_noise_bufs(start_size=cam_cfg.img_size, device=device)
    if zero_noise_bufs:
      [buf.zero_() for buf in noise_bufs]

    chunk = 4
    # sample_z = torch_utils.sample_noises(bs=chunk, noise_dim=g_ema.z_dim, device=device, N_samples=2, seed=seed)
    sample_z = loaded_state['sample_kwargs_random']['zs']

    video_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video.mp4", fps=fps, hd_video=hd_video)

    image_st = st.empty()

    for idx_f in tqdm(range(0, N_frames)):
      samples_list = []

      for j in range(0, len(sample_z[0]), chunk):

        cur_zs = [zs[j:j+chunk] for zs in sample_z]
        cur_bs = len(cur_zs[0])
        with torch.no_grad():
          ret_maps = g_ema(zs=cur_zs,
                           style_render=None,
                           style_decoder=None,
                           cam_poses=sample_cam_extrinsics[[idx_f]].repeat(cur_bs, 1, 1),
                           focals=sample_focals[[idx_f]].repeat(cur_bs, 1, 1),
                           img_size=cam_cfg.img_size,
                           near=sample_near[[idx_f]].repeat(cur_bs, 1, 1),
                           far=sample_far[[idx_f]].repeat(cur_bs, 1, 1),
                           noise_bufs=noise_bufs,
                           truncation=truncation_ratio,
                           N_rays_forward=None,
                           N_rays_grad=None,
                           N_samples_forward=None,
                           eikonal_reg=False,
                           nerf_cfg=nerf_cfg,
                           recompute_mean=False,
                           return_xyz=True)
          rgb = ret_maps['rgb']
          samples_list.append(rgb)
          # rgb_pil = torch_utils.img_tensor_to_pil(rgb)

      samples = torch.cat(samples_list, dim=0)
      N_row = int(math.sqrt(len(samples)))
      samples = utils.make_grid(samples,
                                padding=0,
                                nrow=N_row,
                                normalize=True,
                                value_range=(-1, 1), )
      samples_pil = torch_utils.img_tensor_to_pil(samples, low=0, high=1)
      st_utils.st_image(samples_pil, caption=f"{samples_pil.size}", debug=global_cfg.tl_debug, st_empty=image_st)

      video_f.write(samples_pil)

    video_f.release(st_video=True)

    torch.cuda.empty_cache()
    pass

  def _flip_inversion_axis_angle_web(self,
                                     cfg,
                                     outdir,
                                     saved_suffix_state=None,
                                     **kwargs):

    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, show_image=False)
      image_list_kwargs[data_k]['image_path'] = image_path
    data_k = st_utils.radio('source', options=image_list_kwargs.keys(), index=0, sidebar=True)
    image_path = image_list_kwargs[data_k]['image_path']
    st_utils.st_show_image(image_path)
    if data_k == 'disney':
      cfg.default_network_pkl = 'FFHQ_disney'
      cfg.optim_decoder_params = False

    # N_samples = st_utils.number_input('N_samples', cfg.N_samples, sidebar=True)
    # N_step = st_utils.number_input('N_step', cfg.N_step, sidebar=True)
    # z_mode = st_utils.selectbox('z_mode', options=cfg.z_mode, default_value=cfg.default_z_mode, sidebar=True)
    # interp_mode = st_utils.selectbox('interp_mode', options=cfg.interp_mode, sidebar=True)
    # azim = st_utils.number_input('azim', cfg.azim, sidebar=True, format="%.3f")
    # elev = st_utils.number_input('elev', cfg.elev, sidebar=True, format="%.3f")
    # fov = st_utils.number_input('fov', cfg.fov, sidebar=True, format="%.3f")
    # truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    # fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    # hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    st_log_every = st_utils.number_input('st_log_every', 10, sidebar=True)

    cfg.perceptual_cfg.layers = st_utils.parse_list_from_st_text_input(
      'layers', cfg.perceptual_cfg.layers, sidebar=True)

    st_utils.st_set_sep("learning rate")
    cfg.lr_cam = st_utils.number_input('lr_cam', cfg.lr_cam, sidebar=True, format="%.6f")
    cfg.lr_render_w = st_utils.number_input('lr_render_w', cfg.lr_render_w, sidebar=True, format="%.6f")
    cfg.lr_decoder_w = st_utils.number_input('lr_decoder_w', cfg.lr_decoder_w, sidebar=True, format="%.6f")
    cfg.lr_decoder_params = st_utils.number_input('lr_decoder_params', cfg.lr_decoder_params, sidebar=True,
                                                  format="%.6f")
    cfg.lr_noise = st_utils.number_input('lr_noise', cfg.lr_noise, sidebar=True, format="%.6f")

    st_utils.st_set_sep("loss weights")
    cfg.rgb_weight = st_utils.number_input('rgb_weight', cfg.rgb_weight, sidebar=True)
    cfg.thumb_weight = st_utils.number_input('thumb_weight', cfg.thumb_weight, sidebar=True)
    cfg.mse_weight = st_utils.number_input('mse_weight', cfg.mse_weight, sidebar=True)
    cfg.regularize_noise_weight = st_utils.number_input(
      'regularize_noise_weight', cfg.regularize_noise_weight, sidebar=True)

    st_utils.st_set_sep("whether optimizing")
    cfg.optim_cam = st_utils.checkbox('optim_cam', cfg.optim_cam, sidebar=True)
    cfg.optim_render_w = st_utils.checkbox('optim_render_w', cfg.optim_render_w, sidebar=True)
    cfg.optim_render_params = st_utils.checkbox('optim_render_params', cfg.optim_render_params, sidebar=True)
    cfg.optim_decoder_w = st_utils.checkbox('optim_decoder_w', cfg.optim_decoder_w, sidebar=True)
    cfg.optim_decoder_params = st_utils.checkbox('optim_decoder_params', cfg.optim_decoder_params, sidebar=True)
    cfg.optim_noise_bufs = st_utils.checkbox('optim_noise_bufs', cfg.optim_noise_bufs, sidebar=True)
    cfg.zero_noise_bufs = st_utils.checkbox('zero_noise_bufs', cfg.zero_noise_bufs, sidebar=True)

    cfg.bs_cam = st_utils.number_input('bs_cam', cfg.bs_cam, sidebar=True)
    cfg.bs_render = st_utils.number_input('bs_render', cfg.bs_render, sidebar=True)
    cfg.bs_decoder = st_utils.number_input('bs_decoder', cfg.bs_decoder, sidebar=True)

    cfg.N_steps_pose = st_utils.number_input('N_steps_pose', cfg.N_steps_pose, sidebar=True)
    cfg.N_steps_app = st_utils.number_input('N_steps_app', cfg.N_steps_app, sidebar=True)
    cfg.N_steps_multiview = st_utils.number_input('N_steps_multiview', cfg.N_steps_multiview, sidebar=True)
    cfg.truncation_psi = st_utils.number_input('truncation_psi', cfg.truncation_psi, sidebar=True, format="%.2f")
    cfg.mask_background = st_utils.checkbox('mask_background', cfg.mask_background, sidebar=True)
    cfg.flip_w_decoder_every = st_utils.number_input('flip_w_decoder_every', cfg.flip_w_decoder_every, sidebar=True)

    seed = st_utils.get_seed(cfg.seeds)

    network_pkl_choice = st_utils.selectbox('network_pkl_choice', cfg.network_pkl_choice,
                                            default_value=cfg.default_network_pkl, sidebar=True)
    network_pkl = cfg.network_pkl[network_pkl_choice]
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{network_pkl}/config_command.yaml").values())[0]
    G_cfg = loaded_cfg.G_cfg.clone()
    G_kwargs = loaded_cfg.G_kwargs.clone()

    G_kwargs.nerf_cfg.static_viewdirs = st_utils.checkbox(
      'static_viewdirs', True, sidebar=True)
    G_kwargs.nerf_cfg.N_samples = st_utils.number_input('N_samples', G_kwargs.nerf_cfg.N_samples, sidebar=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    torch_utils.init_seeds(seed)

    device = "cuda"

    g_ema = build_model(cfg=G_cfg).cuda().eval()
    Checkpointer(g_ema).load_state_dict_from_file(f"{network_pkl}/G_ema.pth")

    cam_cfg = G_kwargs.cam_cfg
    nerf_cfg = G_kwargs.nerf_cfg

    from .projector_axis_angle import StyleGAN2Projector_Flip

    projector = StyleGAN2Projector_Flip(
      G=g_ema,
      device=device,
      perceptual_cfg=cfg.perceptual_cfg)

    start_time = perf_counter()
    projector.project_wplus(
      outdir=outdir,
      image_path=image_path,
      cam_cfg=cam_cfg,
      nerf_cfg=nerf_cfg,
      st_log_every=st_log_every,
      st_web=True,
      **cfg
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    torch.cuda.empty_cache()
    pass