import os
import sys
import unittest
import argparse

import torch
from torchvision.utils import make_grid, save_image

from tl2.proj.pytorch import torch_utils
from tl2.proj.pil import pil_utils


class Testing_discriminator(unittest.TestCase):

  def test__build_D_StyleGAN(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib:./stylesdf_lib
        python -c "from exp.tests.test_stylesdf import Testing_train_volume_render_ffhq;\
          Testing_train_volume_render_ffhq().test_train_volume_render(debug=False)" \
          --tl_opts batch 2 chunk 2 log_ckpt_every 100

          # --tl_outdir results/$resume_dir

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    if 'RUN_NUM' not in os.environ:
      os.environ['RUN_NUM'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    if debug:
      # sys.argv.extend(['--tl_outdir', 'results/train_ffhq_256/train_ffhq_256-test'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d/models/discriminator.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from tl2.proj.fvcore import build_model, global_cfg
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from exp.cips3d.models.discriminator import D_StyleGAN

    torch_utils.init_seeds()

    D = D_StyleGAN(**cfg).cuda()

    ckpt_path = "../bucket_3690/results/StyleSDF-exp/train_full_pipeline_ffhq/train_full_pipeline-20220328_095648_700/ckptdir/resume/D.pth"
    Checkpointer(D).load_state_dict_from_file(ckpt_path)

    bs = 8
    img_size = D.input_size

    x = torch.randn(bs, 3, img_size, img_size).cuda()
    out = D(x)

    pass

  def test__build_D_StyleGAN_Progressive(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib:./stylesdf_lib
        python -c "from exp.tests.test_stylesdf import Testing_train_volume_render_ffhq;\
          Testing_train_volume_render_ffhq().test_train_volume_render(debug=False)" \
          --tl_opts batch 2 chunk 2 log_ckpt_every 100

          # --tl_outdir results/$resume_dir

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    if 'RUN_NUM' not in os.environ:
      os.environ['RUN_NUM'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    if debug:
      # sys.argv.extend(['--tl_outdir', 'results/train_ffhq_256/train_ffhq_256-test'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d/models/discriminator.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from tl2.proj.fvcore import build_model, global_cfg
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from exp.cips3d.models.discriminator import D_StyleGAN_Progressive

    torch_utils.init_seeds()

    cfg.pretrained_size = None
    D = D_StyleGAN_Progressive(**cfg).cuda()

    # ckpt_path = "../bucket_3690/results/StyleSDF-exp/train_full_pipeline_ffhq/train_full_pipeline-20220328_095648_700/ckptdir/resume/D.pth"
    # Checkpointer(D).load_state_dict_from_file(ckpt_path)

    bs = 8
    img_size = 256

    x = torch.randn(bs, 3, img_size, img_size).cuda()
    out = D(x, alpha=0.7)

    pass


class Testing_discriminator_multi_scale(unittest.TestCase):

  def test__build_Discriminator_MultiScale(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib:./stylesdf_lib
        python -c "from exp.tests.test_stylesdf import Testing_train_volume_render_ffhq;\
          Testing_train_volume_render_ffhq().test_train_volume_render(debug=False)" \
          --tl_opts batch 2 chunk 2 log_ckpt_every 100

          # --tl_outdir results/$resume_dir

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    if 'RUN_NUM' not in os.environ:
      os.environ['RUN_NUM'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    if debug:
      # sys.argv.extend(['--tl_outdir', 'results/train_ffhq_256/train_ffhq_256-test'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d/models/discriminator_multi_scale.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from tl2.proj.fvcore import build_model, global_cfg
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from exp.cips3d.models.discriminator_multi_scale import Discriminator_MultiScale

    torch_utils.init_seeds()

    D = Discriminator_MultiScale(**cfg).cuda()

    # ckpt_path = "../bucket_3690/results/StyleSDF-exp/train_full_pipeline_ffhq/train_full_pipeline-20220328_095648_700/ckptdir/resume/D.pth"
    # Checkpointer(D).load_state_dict_from_file(ckpt_path)

    bs = 8
    img_size = D.max_size
    alpha = 0.8

    x = torch.randn(bs, 3, img_size, img_size).cuda()
    out = D(x, alpha=alpha)

    pass


class Testing_discriminator_pose(unittest.TestCase):

  def test__build_VolumeRenderDiscriminator(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib:./stylesdf_lib
        python -c "from exp.tests.test_stylesdf import Testing_train_volume_render_ffhq;\
          Testing_train_volume_render_ffhq().test_train_volume_render(debug=False)" \
          --tl_opts batch 2 chunk 2 log_ckpt_every 100

          # --tl_outdir results/$resume_dir

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    if 'RUN_NUM' not in os.environ:
      os.environ['RUN_NUM'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    if debug:
      # sys.argv.extend(['--tl_outdir', 'results/train_ffhq_256/train_ffhq_256-test'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d/models/discriminator_pose.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from tl2.proj.fvcore import build_model, global_cfg
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from exp.cips3d.models.discriminator_pose import VolumeRenderDiscriminator

    torch_utils.init_seeds()

    D = VolumeRenderDiscriminator(**cfg).cuda()

    # ckpt_path = "../bucket_3690/results/StyleSDF-exp/train_full_pipeline_ffhq/train_full_pipeline-20220328_095648_700/ckptdir/resume/D.pth"
    # Checkpointer(D).load_state_dict_from_file(ckpt_path)

    bs = 8
    img_size = D.input_size
    alpha = 0.8

    x = torch.randn(bs, 3, img_size, img_size).cuda()
    out = D(x)

    pass

  def test__build_D_VolumeRender_Progressive(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib:./stylesdf_lib
        python -c "from exp.tests.test_stylesdf import Testing_train_volume_render_ffhq;\
          Testing_train_volume_render_ffhq().test_train_volume_render(debug=False)" \
          --tl_opts batch 2 chunk 2 log_ckpt_every 100

          # --tl_outdir results/$resume_dir

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    if 'RUN_NUM' not in os.environ:
      os.environ['RUN_NUM'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    if debug:
      # sys.argv.extend(['--tl_outdir', 'results/train_ffhq_256/train_ffhq_256-test'])
      pass
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    resume = os.path.isdir(f"{outdir}/ckptdir/resume") and \
             tl2_utils.parser_args_from_list(name="--tl_outdir", argv_list=sys.argv, type='str') is not None
    argv_str = f"""
                --tl_config_file exp/cips3d/models/discriminator_pose.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    from tl2.proj.fvcore import build_model, global_cfg
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from exp.cips3d.models.discriminator_pose import D_VolumeRender_Progressive

    torch_utils.init_seeds()

    cfg.pretrained_size = 64
    D = D_VolumeRender_Progressive(**cfg).cuda()

    # ckpt_path = "../bucket_3690/results/StyleSDF-exp/train_full_pipeline_ffhq/train_full_pipeline-20220328_095648_700/ckptdir/resume/D.pth"
    # Checkpointer(D).load_state_dict_from_file(ckpt_path)

    bs = 8
    img_size = 512
    alpha = 0.8

    x = torch.randn(bs, 3, img_size, img_size).cuda()
    out = D(x, alpha=alpha)

    pass