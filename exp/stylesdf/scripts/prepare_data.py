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


def resize_and_convert(img, size, resample):
  # img = pil_utils.pad2square(img)
  # W, H = img.size
  # padding_mode = 'reflect'
  # padding_mode = 'edge'
  # if W > H:
  #   padding = (W - H) // 2
  #   img = trans_fn.pad(img, padding=(0, padding, 0, padding), padding_mode=padding_mode)
  # else:
  #   padding = (H - W) // 2
  #   img = trans_fn.pad(img, padding=(padding, 0, padding, 0), padding_mode=padding_mode)

  img = trans_fn.center_crop(img, min(img.size))
  # pil_utils.imshow_pil(img, img.size)

  if img.size != (size, size):
    img = trans_fn.resize(img, size, resample)
  if img.size != (size, size):
    img = trans_fn.center_crop(img, size)

  buffer = BytesIO()
  img.save(buffer, format="png")
  val = buffer.getvalue()

  return val


def resize_multiple(
        img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS):
  imgs = []

  for size in sizes:
    imgs.append(resize_and_convert(img, size, resample))

  return imgs


def resize_worker(img_file, sizes, resample):
  i, file = img_file
  img = Image.open(file)
  img = img.convert("RGB")
  out = resize_multiple(img, sizes=sizes, resample=resample)

  return i, out


def prepare(
        env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
  resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

  files = sorted(dataset.imgs, key=lambda x: x[0])
  files = [(i, file) for i, (file, label) in enumerate(files)]
  total = 0

  img_pil = Image.open(files[0][1])
  img_size = img_pil.size

  with multiprocessing.Pool(n_worker) as pool:
    for i, imgs in tqdm(pool.imap_unordered(resize_fn, files), total=len(files), desc=f"img_size {img_size}"):
      for size, img in zip(sizes, imgs):
        key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

        with env.begin(write=True) as txn:
          txn.put(key, img)

      total += 1

    with env.begin(write=True) as txn:
      txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


def main():
  parser = argparse.ArgumentParser(description="Preprocess images for model training")
  # parser.add_argument("--size", type=str, default="64,512,1024", help="resolutions of images for the dataset")
  parser.add_argument("--n_worker", type=int, default=32, help="number of workers for preparing dataset")
  parser.add_argument("--resample", type=str, default="lanczos", help="resampling methods for resizing images")
  parser.add_argument("--out_path", type=str, default="datasets/FFHQ", help="Target path of the output lmdb dataset")
  parser.add_argument("--in_path", type=str, default="", help="path to the input image dataset")
  args, _ = parser.parse_known_args()

  if global_cfg.tl_debug:
    args.n_worker = 1

  # resample_map = {
  #   "lanczos": Image.LANCZOS,
  #   "bilinear": Image.BILINEAR
  # }
  resample_map = {
    "lanczos": trans_fn.InterpolationMode.LANCZOS,
    "bilinear": trans_fn.InterpolationMode.BILINEAR,
  }
  resample = resample_map[args.resample]

  # sizes = [int(s.strip()) for s in args.size.split(",")]
  sizes = global_cfg.size

  print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

  moxing_utils.copy_data(rank=0, global_cfg=global_cfg, **global_cfg.obs_dataset_root)
  imgset = torch_utils.ImageFolder(args.in_path)
  if global_cfg.tl_debug:
    imgset.sample_partial_samples(N_samples=50)

  if 'saved_name' in global_cfg:
    saved_name = global_cfg.saved_name
  else:
    saved_name = os.path.basename(args.in_path)
  out_path = f"{args.out_path}/{saved_name}_lmdb_{'_'.join(list(map(str, sizes)))}"
  if os.path.exists(out_path):
    shutil.rmtree(out_path, ignore_errors=True)

  os.makedirs(out_path, exist_ok=True)
  with lmdb.open(out_path, map_size=1024 ** 4, readahead=False) as env:
    prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)

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
