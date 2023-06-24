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
from tl2.proj.fvcore import build_model, MODEL_REGISTRY
from tl2.proj.cv2 import cv2_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.pil import pil_utils

# sys.path.insert(0, f"{os.getcwd()}/DGP_lib")
# from DGP_lib import utils
# sys.path.pop(0)

from exp.stylesdf.options import BaseOptions
from model import Generator
from exp.stylesdf.utils import (
  generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
  xyz2mesh, create_cameras, create_mesh_renderer, add_textures,
)
from exp.cips3d import nerf_utils


def _get_trans_rotation_cams(N_frames,
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

def render_video(outdir,
                 opt,
                 g_ema,
                 surface_g_ema,
                 device,
                 mean_latent,
                 surface_mean_latent,
                 N_frames,
                 view_mode,
                 fps,
                 hd_video,
                 seed,
                 ):
  g_ema.eval()
  if not opt.no_surface_renderings or opt.project_noise:
    surface_g_ema.eval()

  # Generate video trajectory
  trajectory = np.zeros((N_frames, 3), dtype=np.float32)

  # set camera trajectory
  show_trajectory = global_cfg.show_trajectory
  only_rotate = global_cfg.only_rotate
  
  if view_mode == 'circle':
    t = np.linspace(0, 1, N_frames)
  
    fov = global_cfg.fov + global_cfg.fov_std * np.sin(t * np.pi)
    azim = global_cfg.azim_mean + global_cfg.azim_std * np.sin(t * 2 * np.pi)
    elev = global_cfg.elev_mean + global_cfg.elev_std * np.sin(t * 2 * np.pi)
  
    trajectory[:N_frames, 0] = azim
    trajectory[:N_frames, 1] = elev
    trajectory[:N_frames, 2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)

    # generate input parameters for the camera trajectory
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
      generate_camera_params(resolution=opt.renderer_output_size,
                             device=device,
                             locations=trajectory[:, :2],
                             fov_ang=trajectory[:, 2:],
                             dist_radius=opt.camera.dist_radius)
    
  elif view_mode == 'translate_rotate':
    show_trajectory = False
  
    sample_cam_extrinsics, trajectory, sample_focals, sample_near, sample_far = \
      _get_trans_rotation_cams(N_frames=N_frames, trans_max=global_cfg.trans_max,
                               cam_cfg=global_cfg.cam_cfg, device=device)
    if only_rotate:
      sample_cam_extrinsics = sample_cam_extrinsics.chunk(2)[1]
      trajectory = trajectory.chunk(2)[1]
      sample_focals = sample_focals.chunk(2)[1]
      sample_near = sample_near.chunk(2)[1]
      sample_far = sample_far.chunk(2)[1]
    
    N_frames = sample_cam_extrinsics.shape[0]
    
  elif view_mode == 'yaw':
    # t = np.linspace(0, 1, N_frames)
    # elev = 0
    # fov = opt.camera.fov
    # if opt.camera.uniform:
    #   azim = opt.camera.azim * np.cos(t * 2 * np.pi)
    # else:
    #   azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)
    #
    # trajectory[:N_frames, 0] = azim
    # trajectory[:N_frames, 1] = elev
    # trajectory[:N_frames, 2] = fov
    trajectory = np.zeros((N_frames, 3), dtype=np.float32)
    t = np.linspace(0, 1, N_frames)
    azim = - global_cfg.azim_std + (global_cfg.azim_std * 2) * np.sin(t * np.pi)

    trajectory[:N_frames, 0] = azim
    trajectory[:N_frames, 1] = 0
    trajectory[:N_frames, 2] = global_cfg.fov

    trajectory = torch.from_numpy(trajectory).to(device)

    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
      nerf_utils.Camera.generate_camera_params(
        locations=trajectory[:, :2],
        device=device,
        **{**global_cfg.cam_cfg,
           'fov_ang': trajectory[:, 2:]})

  

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
  sample_z = np.random.RandomState(seed).randn(1, opt.style_dim)
  sample_z = torch.from_numpy(sample_z).to(device).to(torch.float32).repeat(chunk, 1)

  video_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video.mp4", fps=fps, hd_video=hd_video)
  video_rgb_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_rgb.mp4", fps=fps, hd_video=hd_video)
  video_mesh_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video_mesh.mp4", fps=fps, hd_video=hd_video)
  image_st = st.empty()

  for j in tqdm(range(0, N_frames, chunk)):
    with torch.no_grad():
      out = g_ema([sample_z],
                  sample_cam_extrinsics[j:j + chunk],
                  sample_focals[j:j + chunk],
                  sample_near[j:j + chunk],
                  sample_far[j:j + chunk],
                  truncation=opt.truncation_ratio,
                  truncation_latent=mean_latent,
                  randomize_noise=False,
                  project_noise=opt.project_noise,
                  mesh_path=None)

      rgb = out[0]
      rgb_pil = torch_utils.img_tensor_to_pil(rgb)

      if show_trajectory:
        azimuth = trajectory[j, 0]
        elevation = trajectory[j, 1]
        pil_utils.add_text(rgb_pil, f"azimuth: {azimuth:.2f}\n"
                                    f"elevation: {elevation:.2f}", size=rgb_pil.size[0]//15, )
      
      del out
      torch.cuda.empty_cache()

      ########## Extract surface ##########
      if not opt.no_surface_renderings:
        scale = surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
        surface_sample_focals = sample_focals * scale
        surface_out = surface_g_ema([sample_z],
                                    sample_cam_extrinsics[j:j + chunk],
                                    surface_sample_focals[j:j + chunk],
                                    sample_near[j:j + chunk],
                                    sample_far[j:j + chunk],
                                    truncation=opt.truncation_ratio,
                                    truncation_latent=surface_mean_latent,
                                    return_xyz=True)
        xyz = surface_out[2].cpu()

        del surface_out
        torch.cuda.empty_cache()

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
                                        light_location=((0.0, 0, 5.0),),
                                        specular_color=((0.2, 0.2, 0.2),),
                                        ambient_color=((0.1, 0.1, 0.1),),
                                        diffuse_color=((0.65, .65, .65),),
                                        device=device)

        mesh_image = renderer(mesh)[..., :3].squeeze().cpu().numpy()
        mesh_image_pil = pil_utils.np_to_pil(mesh_image, range01=True)
        mesh_image_pil = pil_utils.pil_resize(mesh_image_pil, rgb.shape[-2:])

        merged_pil = pil_utils.merge_image_pil([rgb_pil, mesh_image_pil], nrow=2)
        st_utils.st_image(merged_pil, caption=f"{rgb.shape}", debug=global_cfg.tl_debug, st_empty=image_st)
        video_f.write(merged_pil)
        video_rgb_f.write(rgb_pil)
        video_mesh_f.write(mesh_image_pil)

  video_f.release(st_video=True)
  video_rgb_f.release(st_video=True)
  video_mesh_f.release(st_video=True)
  pass



@MODEL_REGISTRY.register(name_prefix=__name__)
class STModel(object):
  def __init__(self):

    pass

  def render_video_ffhq_r1024_web(self,
                                  cfg,
                                  outdir,
                                  saved_suffix_state=None,
                                  **kwargs):
    from tl2.proj.streamlit import st_utils

    st_utils.st_set_sep('video')
    N_frames = st_utils.number_input('N_frames', cfg.N_frames, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', cfg.hd_video, sidebar=True)

    truncation_ratio = st_utils.number_input('truncation_ratio', cfg.truncation_ratio, format="%.3f", sidebar=True)
    view_mode = st_utils.selectbox('view_mode', cfg.view_mode, default_value=cfg.default_view_mode, sidebar=True)
    global_cfg.azim_mean = st_utils.number_input('azim_mean', cfg.azim_mean, sidebar=True, format="%.2f")
    global_cfg.elev_mean = st_utils.number_input('elev_mean', cfg.elev_mean, sidebar=True, format="%.2f")
    global_cfg.azim_std = st_utils.number_input('azim_std', cfg.azim_std, sidebar=True, format="%.2f")
    global_cfg.elev_std = st_utils.number_input('elev_std', cfg.elev_std, sidebar=True, format="%.2f")
    global_cfg.fov = st_utils.number_input('fov', cfg.fov, sidebar=True, format="%.2f")
    global_cfg.fov_std = st_utils.number_input('fov_std', cfg.fov_std, sidebar=True, format="%.2f")

    if view_mode == 'translate_rotate':
      global_cfg.trans_max = st_utils.number_input('trans_max', cfg.trans_max, sidebar=True, format="%.3f")

    global_cfg.show_trajectory = st_utils.checkbox('show_trajectory', cfg.show_trajectory, sidebar=True)
    global_cfg.only_rotate = st_utils.checkbox('only_rotate', cfg.only_rotate, sidebar=True)
    
    seed = st_utils.get_seed(cfg.seeds)

    opt = BaseOptions().parse()
    opt.model.is_test = True
    opt.model.style_dim = 256
    opt.model.freeze_renderer = False
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.rendering.perturb = 0
    opt.rendering.force_background = True
    opt.rendering.static_viewdirs = True
    opt.rendering.return_sdf = True
    opt.rendering.N_samples = 64

    # opt.inference.camera.fov = st_utils.number_input('fov', opt.inference.camera.fov, sidebar=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # torch_utils.init_seeds(seed)

    device = "cuda"


    moxing_utils.copy_data(rank=0, global_cfg=global_cfg, **global_cfg.obs_checkpoint_path)
    checkpoint_path = global_cfg.obs_checkpoint_path.datapath
    checkpoint = torch.load(checkpoint_path)
    pretrained_weights_dict = checkpoint["g_ema"]

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering).to(device)
    Checkpointer(g_ema).load_state_dict(pretrained_weights_dict)

    # load a the volume renderee to a second that extracts surfaces at 128x128x128
    opt['surf_extraction'] = Munch()
    opt.surf_extraction.rendering = opt.rendering
    opt.surf_extraction.model = opt.model.copy()
    opt.surf_extraction.model.renderer_spatial_output_dim = 128
    opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
    opt.surf_extraction.rendering.return_xyz = True
    opt.surf_extraction.rendering.return_sdf = True
    opt.inference.surf_extraction_output_size = opt.surf_extraction.model.renderer_spatial_output_dim
    surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(device)
    Checkpointer(surface_g_ema).load_state_dict(pretrained_weights_dict)

    # Checkpointer(g_ema).load_state_dict_from_file(
    #   '../bucket_3690/results/StyleSDF-exp/train_full_pipeline_ffhq/train_full_pipeline-20220328_095648_700/ckptdir/resume/G_ema.pth')
    # Checkpointer(surface_g_ema).load_state_dict_from_file(
    #   '../bucket_3690/results/StyleSDF-exp/train_cips3d_ffhq_v3/train_r1024_r64_ks1-20220412_201401_070/ckptdir/resume/G_ema.pth')

    # get the mean latent vector for g_ema
    opt.inference.truncation_ratio = truncation_ratio
    if opt.inference.truncation_ratio < 1:
      with torch.no_grad():
        mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
      mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings or opt.model.project_noise:
      surface_mean_latent = mean_latent[0]
    else:
      surface_mean_latent = None

    render_video(outdir,
                 opt.inference,
                 g_ema,
                 surface_g_ema,
                 device,
                 mean_latent,
                 surface_mean_latent,
                 N_frames=N_frames,
                 view_mode=view_mode,
                 fps=fps,
                 hd_video=hd_video,
                 seed=seed)
    pass
