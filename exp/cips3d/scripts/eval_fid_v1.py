import pprint
import copy
import tqdm
import shutil
from PIL import Image
import logging
import argparse
import os
import torch_fidelity
from torch_fidelity import calculate_metrics

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
from tl2.proj.fvcore import build_model, TLCfgNode
from tl2.proj.fvcore.checkpoint import Checkpointer

from exp.cips3d.scripts.setup_evaluation import setup_evaluation
from exp.cips3d.scripts.gen_images import gen_images


def eval_fid(real_dir,
             fake_dir,
             kid=False):

  metric_dict = {}

  metrics_dict = calculate_metrics(input1=real_dir,
                                   input2=fake_dir,
                                   cuda=True,
                                   isc=False,
                                   fid=True,
                                   kid=kid,
                                   verbose=False)

  metric_dict['FID'] = metrics_dict[torch_fidelity.KEY_METRIC_FID]
  if kid:
    metric_dict['KID_mean'] = metrics_dict[torch_fidelity.KEY_METRIC_KID_MEAN]
    metric_dict['KID_std'] = metrics_dict[torch_fidelity.KEY_METRIC_KID_STD]

  torch.cuda.empty_cache()

  return metric_dict


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

  if rank == 0:
    moxing_utils.setup_tl_outdir_obs(global_cfg)
    moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

  ddp_utils.d2_synchronize()

  G_ema = build_model(global_cfg.G_cfg).to(device)

  moxing_utils.copy_data(rank=rank, global_cfg=global_cfg,
                         datapath_obs=global_cfg.network_pkl, datapath=global_cfg.network_pkl)

  ckpt_path = f"{global_cfg.network_pkl}/resume_{rank}/G_ema.pth"
  Checkpointer(G_ema).load_state_dict_from_file(ckpt_path, rank=rank)

  if global_cfg.real_dir is None:
    real_dir = f"{global_cfg.tl_outdir}/exp/fid/real"
    setup_evaluation(rank=rank,
                     world_size=world_size,
                     batch_gpu=64,
                     distributed=world_size > 1,
                     real_dir=real_dir,
                     N_real_images_eval=global_cfg.N_real_images_eval,
                     del_fid_real_images=global_cfg.del_fid_real_images)
    logger.info(f"=== End setup_evaluation.")
  else:
    real_dir = global_cfg.real_dir

  if global_cfg.fake_dir is None:
    gen_images(rank=rank,
               world_size=world_size,
               generator=G_ema,
               G_kwargs=global_cfg.G_kwargs,
               fake_dir=f"{global_cfg.tl_outdir}/exp/fid/temp",
               num_imgs=global_cfg.N_gen_images_eval_pre,
               batch_gpu=global_cfg.batch_gpu)
    ddp_utils.d2_synchronize()
    if rank == 0: shutil.rmtree(f"{global_cfg.tl_outdir}/exp/fid/temp")

    fake_dir = f"{global_cfg.tl_outdir}/exp/fid/fake"
    gen_images(rank=rank,
               world_size=world_size,
               generator=G_ema,
               G_kwargs=global_cfg.G_kwargs,
               fake_dir=fake_dir,
               num_imgs=global_cfg.N_gen_images_eval,
               batch_gpu=global_cfg.batch_gpu)
    ddp_utils.d2_synchronize()
  else:
    fake_dir = global_cfg.fake_dir

  moxing_utils.copy_data(rank=rank, global_cfg=global_cfg, **global_cfg.obs_inception_v3)
  global_cfg.obs_inception_v3.disable = True
  if rank == 0:
    metric_dict = eval_fid(real_dir=real_dir, fake_dir=fake_dir, kid=global_cfg.kid)
    # logger.info(global_cfg.dump())
    logger.info(pprint.pformat(metric_dict))

    # shutil.rmtree(real_dir, ignore_errors=True)
    # shutil.rmtree(fake_dir, ignore_errors=True)
    moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

  logger.info(global_cfg.tl_outdir)
  ddp_utils.d2_synchronize()
  pass


if __name__ == '__main__':
  main()
