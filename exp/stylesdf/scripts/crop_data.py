import os
import argparse
import os.path
import shutil
from io import BytesIO
import multiprocessing
from functools import partial
from PIL import Image
import lmdb
from tqdm import tqdm
from pdb import set_trace as st

from torchvision import datasets
from torchvision.transforms import functional as trans_fn

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import moxing_utils
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.pytorch import torch_utils
from tl2.proj.pil import pil_utils


def resize_worker(img_file,
                  crop_size,
                  resize_size,
                  resample):
  i, file = img_file

  img_pil = pil_utils.pil_open_rgb(file)
  W, H = img_pil.size

  crop_w, crop_h = crop_size

  left = (W - crop_w) // 2
  upper = (H - crop_h) // 2
  right = left + crop_w
  lower = upper + crop_h

  img_pil = img_pil.crop((left, upper, right, lower))
  if global_cfg.tl_debug:
    pil_utils.imshow_pil(img_pil)

  if crop_size != resize_size:
    img_pil = trans_fn.resize(img_pil, resize_size, resample)

  return i, img_pil


def prepare(dataset,
            n_worker,
            crop_size,
            resize_size,
            resample,
            out_path):
  if isinstance(crop_size, int):
    crop_size = (crop_size, crop_size)
  if isinstance(resize_size, int):
    resize_size = (resize_size, resize_size)

  resize_fn = partial(resize_worker, crop_size=crop_size, resize_size=resize_size, resample=resample)

  files = sorted(dataset.imgs, key=lambda x: x[0])
  files = [(i, file) for i, (file, label) in enumerate(files)]
  total = 0

  with multiprocessing.Pool(n_worker) as pool:
    for i, img_pil in tqdm(pool.imap_unordered(resize_fn, files), total=len(files)):

      img_pil.save(f"{out_path}/{i:08d}.png")

      total += 1

  print(f"Num images: {total}")

def main():

  resample_map = {
    "lanczos": trans_fn.InterpolationMode.LANCZOS,
    "bilinear": trans_fn.InterpolationMode.BILINEAR,
  }
  resample = resample_map[global_cfg.resample]

  print(f"Make dataset of image sizes: crop_size={global_cfg.crop_size}, resize_size={global_cfg.resize_size}")

  moxing_utils.copy_data(rank=0, global_cfg=global_cfg, **global_cfg.obs_dataset_root)

  imgset = torch_utils.ImageFolder(global_cfg.in_path)
  if global_cfg.tl_debug:
    imgset.sample_partial_samples(N_samples=10)

  saved_name = os.path.basename(global_cfg.in_path)
  out_path = f"{global_cfg.out_path}/{saved_name}_crop{global_cfg.crop_size}_r{global_cfg.resize_size}"
  if os.path.exists(out_path):
    shutil.rmtree(out_path, ignore_errors=True)
  os.makedirs(out_path, exist_ok=True)

  prepare(imgset,
          global_cfg.n_worker,
          crop_size=global_cfg.crop_size,
          resize_size=global_cfg.resize_size,
          resample=resample,
          out_path=out_path)

  moxing_utils.copy_data(rank=0, global_cfg=global_cfg,
                         datapath_obs=f"{global_cfg.obs_out_path}/{os.path.basename(out_path)}",
                         datapath=out_path,
                         download=False)
  pass


if __name__ == '__main__':
  update_parser_defaults_from_yaml(parser=None)

  moxing_utils.setup_tl_outdir_obs(global_cfg)
  moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

  main()

  moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
