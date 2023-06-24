import tqdm
import shutil
from PIL import Image
import logging
import argparse
import os

import torch
import torch.distributed as dist
import torch.utils.data as data_utils
import torch.nn.functional as F
import torchvision.transforms as tv_trans
from torchvision.utils import save_image
from torchvision.transforms import functional as trans_fn

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
from tl2.proj.pytorch.examples.dataset_stylegan3.dataset import get_training_dataloader, to_norm_tensor
from tl2.proj.fvcore import TLCfgNode


def setup_evaluation(rank,
                     world_size,
                     batch_gpu,
                     distributed,
                     real_dir,
                     N_real_images_eval,
                     del_fid_real_images,
                     data_img_size=None,
                     img_size=None):
  """
  Output real images.

  :return:
  """
  logger = logging.getLogger('tl')

  if rank == 0 and del_fid_real_images:
    if os.path.exists(real_dir):
      shutil.rmtree(real_dir)
  ddp_utils.d2_synchronize()

  if os.path.exists(real_dir):
    logger.info("Real images exist.")
    return
  else:

    from exp.cips3d.scripts.train_v2 import create_dataset, sample_data
    loader, sampler = create_dataset(opt=None,
                                     rank=rank,
                                     img_size=global_cfg.G_kwargs.cam_cfg.img_size if data_img_size is None else data_img_size,
                                     shuffle=False,
                                     batch_gpu=batch_gpu,
                                     distributed=distributed,
                                     hflip=False)
    loader = sample_data(loader, distributed=(world_size > 1), sampler=sampler)
    N_real_images = len(sampler) * world_size
    if img_size is None:
      img_size = global_cfg.G_kwargs.cam_cfg.img_size

    if rank == 0:
      os.makedirs(real_dir, exist_ok=True)
    ddp_utils.d2_synchronize()

    N_images = min(N_real_images, N_real_images_eval)
    batch_size = batch_gpu * world_size

    if rank == 0:
      pbar = tqdm.tqdm(desc=f"Output real images ({data_img_size}) at {img_size}x{img_size}", total=N_images)

    for idx_b in range((N_images + batch_size - 1) // batch_size):
      if rank == 0:
        pbar.update(batch_size)

      real_imgs, _ = next(loader)

      ddp_utils.d2_synchronize()

      # imgs_norm = to_norm_tensor(imgs, device=device)
      for idx_i, img in enumerate(real_imgs):
        saved_path = f"{real_dir}/{idx_b * batch_size + idx_i * world_size + rank:0>5}.jpg"

        if img_size != data_img_size:
          # real_imgs = F.interpolate(real_imgs, scale_factor=(img_size / data_img_size), recompute_scale_factor=False,
          #                           mode='bilinear', align_corners=False)
          img_pil = torch_utils.img_tensor_to_pil(img)
          img_pil = trans_fn.resize(img_pil, img_size, trans_fn.InterpolationMode.LANCZOS)
          img_pil.save(saved_path)
        else:
          save_image(img, saved_path, normalize=True, value_range=(-1, 1))

      if global_cfg.tl_debug:
        break
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
  torch_utils.init_seeds(seed=args.seed, rank=rank)
  device = torch.device('cuda')

  is_main_process = (rank == 0)

  update_parser_defaults_from_yaml(parser, is_main_process=is_main_process)
  logger = logging.getLogger('tl')

  opt = TLCfgNode()
  distributed = (world_size > 1)
  batch_gpu = 4

  setup_evaluation(rank=rank,
                   world_size=world_size,
                   batch_gpu=batch_gpu,
                   distributed=distributed,
                   real_dir=f"{global_cfg.tl_outdir}/exp/fid/real",
                   N_real_images_eval=global_cfg.N_real_images_eval,
                   del_fid_real_images=global_cfg.del_fid_real_images)
  global_cfg.del_fid_real_images = False
  pass


if __name__ == '__main__':
  main()
