import copy
import tqdm
import shutil
from PIL import Image
import logging
import argparse
import os

import torch
import torch.distributed as dist
import torch.utils.data as data_utils
import torchvision.transforms as tv_trans
from torchvision.utils import save_image

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import modelarts_utils, moxing_utils
from tl2.proj.pil import pil_utils
from tl2.proj.pytorch.ddp import ddp_utils
from tl2.tl2_utils import AverageMeter
from tl2.proj.logger.textlogger import summary_dict2txtfig, global_textlogger
from tl2.proj.fvcore import build_model
from tl2.proj.pytorch.examples.multi_process_main.dataset import ImageListDataset
from tl2.proj.argparser import argparser_utils
from tl2.proj.pytorch import torch_utils
from tl2 import tl2_utils
from tl2.proj.fvcore import build_model
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.fvcore import TLCfgNode

from exp.cips3d.utils import make_noise, mixing_noise
from exp.cips3d import nerf_utils

def gen_images(rank,
               world_size,
               generator,
               G_kwargs,
               fake_dir,
               num_imgs,
               batch_gpu,
               truncation=1):

  if rank ==0:
    os.makedirs(fake_dir, exist_ok=True)
  ddp_utils.d2_synchronize()

  G_kwargs = copy.deepcopy(G_kwargs)

  img_size = G_kwargs.cam_cfg.img_size
  batch_size = batch_gpu * world_size
  device = torch.device(rank)

  generator.eval()

  if rank == 0:
    pbar = tqdm.tqdm(desc=f"Generating feature maps at {img_size}x{img_size}", total=num_imgs)

  with torch.no_grad():
    for idx_b in range((num_imgs + batch_size - 1) // batch_size):
      if rank == 0:
        pbar.update(batch_size)

      noise = mixing_noise(batch_gpu, generator.z_dim, 0, device)
      cam_extrinsics, focal, near, far, gt_viewpoints = nerf_utils.Camera.generate_camera_params(
        device=device,
        batch=batch_gpu,
        **{**G_kwargs.cam_cfg})
      ret_maps = generator(zs=noise,
                           cam_poses=cam_extrinsics,
                           focals=focal,
                           img_size=img_size,
                           near=near,
                           far=far,
                           truncation=truncation,
                           N_rays_forward=global_cfg.N_rays_forward ** 2 if global_cfg.N_rays_forward else None,
                           N_rays_grad=None,
                           N_samples_forward=global_cfg.N_samples_forward ** 2 if global_cfg.N_samples_forward else None,
                           eikonal_reg=False,
                           nerf_cfg=G_kwargs.nerf_cfg.to_dict())

      generated_imgs = ret_maps['rgb']

      for idx_i, img in enumerate(generated_imgs):
        saved_path  = f"{fake_dir}/{idx_b * batch_size + idx_i * world_size + rank:0>5}.jpg"
        save_image(img, saved_path, normalize=True, value_range=(-1, 1))

      if global_cfg.tl_debug:
        break

  if rank == 0:
    pbar.close()
  ddp_utils.d2_synchronize()
  pass



def build_parser():
  ## runtime arguments
  parser = argparse.ArgumentParser(description='Training configurations.')

  argparser_utils.add_argument_int(parser, name='local_rank', default=0)
  argparser_utils.add_argument_int(parser, name='seed', default=0)
  argparser_utils.add_argument_int(parser, name='num_workers', default=0)

  return parser


def main():

  parser = build_parser()
  args, _ = parser.parse_known_args()

  rank, world_size = ddp_utils.ddp_init(seed=args.seed)
  # torch_utils.init_seeds(seed=args.seed, rank=rank)
  device = torch.device('cuda')

  is_main_process = (rank == 0)

  update_parser_defaults_from_yaml(parser, is_main_process=is_main_process)
  logger = logging.getLogger('tl')

  loaded_cfg = list(TLCfgNode.load_yaml_file(f"{global_cfg.network_pkl[global_cfg.default_network_pkl]}/config_command.yaml").values())[0]
  G_cfg = loaded_cfg.G_cfg.clone()
  G_kwargs = loaded_cfg.G_kwargs.clone()
  global_cfg.G_cfg.update(G_cfg)
  global_cfg.G_kwargs.update(G_kwargs)
  if global_cfg.get('frontal', False):
    global_cfg.G_kwargs.cam_cfg.azim_range = 0
    global_cfg.G_kwargs.cam_cfg.elev_range = 0

  G_ema = build_model(global_cfg.G_cfg).to(device)
  ckpt_path = f"{global_cfg.network_pkl[global_cfg.default_network_pkl]}/G_ema.pth"
  Checkpointer(G_ema).load_state_dict_from_file(ckpt_path, rank=rank)

  # global_cfg.tl_debug = False

  gen_images(rank=rank,
             world_size=world_size,
             generator=G_ema,
             G_kwargs=global_cfg.G_kwargs,
             fake_dir=f"{global_cfg.tl_outdir}/exp/fid/fake",
             num_imgs=global_cfg.N_gen_images_eval,
             batch_gpu=global_cfg.batch_gpu,
             truncation=global_cfg.truncation)

  print(f"Save images in {global_cfg.tl_outdir}/exp/fid/fake")
  pass


if __name__ == '__main__':
  main()
