import os
import sys
import unittest
import argparse


class Testing_preparing_dataset(unittest.TestCase):
  
  def test_prepare_data_ffhq_r512(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_preparing_dataset;\
          Testing_preparing_dataset().test_prepare_data_ffhq_r512(debug=False)" \
          --tl_opts n_worker 8

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
                --tl_config_file exp/stylesdf/configs/preparing_dataset.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    
    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 8888)
    
    cmd_str = f"""
        python
        exp/stylesdf/scripts/prepare_data.py
        --out_path {cfg.out_path}
        --n_worker {cfg.n_worker}
        --in_path {cfg.in_path}
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts
                  """
    else:
      cmd_str += f"""
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils
    
    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
    
    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
    
    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass
  
  def test_prepare_data_ffhq_r1024(self, debug=True):
    """
    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_preparing_dataset;\
          Testing_preparing_dataset().test_prepare_data_ffhq_r1024(debug=False)" \
          --tl_opts n_worker 8

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
                --tl_config_file exp/stylesdf/configs/preparing_dataset.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 8888)

    cmd_str = f"""
        python
        exp/stylesdf/scripts/prepare_data.py
        --out_path {cfg.out_path}
        --n_worker {cfg.n_worker}
        --in_path {cfg.in_path}
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts
                  """
    else:
      cmd_str += f"""
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils

    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)

    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)


    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_prepare_data_disney_r1024(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_preparing_dataset;\
          Testing_preparing_dataset().test_prepare_data_disney_r1024(debug=False)" \
          --tl_opts n_worker 32

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
                --tl_config_file exp/stylesdf/configs/preparing_dataset.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 8888)
  
    cmd_str = f"""
        python
        exp/stylesdf/scripts/prepare_data.py
        --out_path {cfg.out_path}
        --in_path {cfg.in_path}
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --n_worker 1
                  --tl_opts
                  """
    else:
      cmd_str += f"""
                  --n_worker {cfg.n_worker}
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils
  
    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  
    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  
    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass
  

class Testing_train_cips3d_ffhq_v10(unittest.TestCase):
  
  def test__sample_multi_view_web(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test__sample_multi_view_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
    
    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
    
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    
    import importlib
    
    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test__flip_inversion_web(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test__flip_inversion_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
  
    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
  
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    import importlib
  
    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --

            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test__render_multi_view_web(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test__render_multi_view_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
  
    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
  
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    import importlib
  
    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --

            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test__interpolate_decoder_web(self, debug=True):
    """
    Usage:

        export TIME_STR=0
        export PYTHONPATH=.:./tl2_lib:./stylesdf_lib
        export CUDA_VISIBLE_DEVICES=0
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test__interpolate_decoder_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
  
    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
  
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    import importlib
  
    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --

            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test__style_mixing_web(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test__style_mixing_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
  
    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
  
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v3.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    import importlib
  
    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --
            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test__rendering_time(self, debug=True):
    """
    all (repeat=1000): 21.30794372712262 s, all/repeat: 0.02130794372712262 s, fps: 46.93085418313323

    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test__rendering_time(debug=False)"

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
                --tl_config_file none
                --tl_command none
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
    
    import time
    import tqdm
    
    import torch
    from torchvision.utils import make_grid
    
    from tl2.proj.fvcore import build_model, TLCfgNode
    from tl2.proj.fvcore.checkpoint import Checkpointer
    from tl2.proj.pytorch import torch_utils
    from tl2.proj.pil import pil_utils
    
    from exp.cips3d.utils import mixing_noise
    from exp.cips3d import nerf_utils

  
    debug = tl2_utils.is_debugging() and debug
  
    torch_utils.init_seeds(seed=12345)
    device = 'cuda'
  
    ckpt_path = "pretrained/ffhq_r1024_inversion"
  
    loaded_cfg = list(TLCfgNode.load_yaml_file(f"{ckpt_path}/config_command.yaml").values())[0]
    G_ema = build_model(loaded_cfg.G_cfg).to(device)
    Checkpointer(G_ema).load_state_dict_from_file(f"{ckpt_path}/G_ema.pth")
  
    N_times = 1000
    batch = 1
    style_dim = G_ema.z_dim
    noise = mixing_noise(batch, style_dim, 0.9, device)
  
    G_kwargs = loaded_cfg.G_kwargs.clone()
    locations = torch.zeros(batch, 2, device=device)
    cam_extrinsics, focal, near, far, gt_viewpoints = nerf_utils.Camera.generate_camera_params(
      locations=locations,
      device=device,
      batch=batch,
      **{**G_kwargs.cam_cfg})
  
    time_start = time.perf_counter()
    for j in tqdm.tqdm(range(0, N_times)):
    
      with torch.set_grad_enabled(False):
        ret_maps = G_ema(zs=noise,
                         cam_poses=cam_extrinsics,
                         focals=focal,
                         img_size=G_kwargs.cam_cfg.img_size,
                         near=near,
                         far=far,
                         truncation=1,
                         N_rays_forward=None,
                         N_rays_grad=None,
                         N_samples_forward=None,
                         eikonal_reg=False,
                         nerf_cfg=G_kwargs.nerf_cfg)
      
        gen_imgs = ret_maps['rgb']
    
      if debug:
        imgs_tensor = make_grid(gen_imgs, nrow=batch, value_range=(-1, 1))
        imgs_pil = torch_utils.img_tensor_to_pil(imgs_tensor)
        pil_utils.imshow_pil(imgs_pil, gen_imgs.shape)
        imgs_pil.save(f"{outdir}/img.png")
        break
  
    time_str = tl2_utils.time_us2string(time_start=time_start, repeat_times=N_times)
    print(time_str)
  
    pass
  
  def test_train_r1024_r64_ks1(self, debug=True):
    """
    train nerf and decoder;

    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test_train_r1024_r64_ks1(debug=False)"

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
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()
  
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 8888)
  
    if n_gpus > 1:
      python_str = f'python -m torch.distributed.launch --nproc_per_node {n_gpus}'
    else:
      python_str = f'python '
  
    cmd_str = f"""
        {python_str}
        exp/cips3d/scripts/train_v10.py
        --batch {cfg.batch}
        --chunk {cfg.chunk}
        --expname {cfg.expname}
        --size {cfg.size}
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --batch 4
                  --chunk 4
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --batch {cfg.batch}
                  --chunk {cfg.chunk}
                  --tl_opts {tl_opts} num_workers {n_gpus}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils
  
    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  
    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  
    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass
  
  def test_eval_fid_r1024(self, debug=True):
    """
    

    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test_eval_fid_r1024(debug=False)" \
          --tl_opts batch_gpu 8

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
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
    
    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)
    
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
    
    if n_gpus > 1:
      python_string = f"python -m torch.distributed.launch --nproc_per_node={n_gpus} --master_port={PORT} "
    else:
      python_string = "python "
    
    cmd_str = f"""
        {python_string}
        exp/cips3d/scripts/eval_fid.py
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts
                  """
    else:
      cmd_str += f"""
                  --num_workers {n_gpus}
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import modelarts_utils
    # update_parser_defaults_from_yaml(parser)
    
    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_eval_fid_r512(self, debug=True):
    """

    Usage:

        # export CUDA_VISIBLE_DEVICES=$cuda_devices
        # export RUN_NUM=$run_num

        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test_eval_fid_r512(debug=False)" \
          --tl_opts batch_gpu 8

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
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 12345)
  
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    if n_gpus > 1:
      python_cmd = f"python -m torch.distributed.launch --nproc_per_node={n_gpus} --master_port={PORT} "
    else:
      python_cmd = "python "
  
    cmd_str = f"""
        {python_cmd}
        exp/cips3d/scripts/eval_fid.py
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --tl_opts
                  """
    else:
      cmd_str += f"""
                  --num_workers {n_gpus}
                  --tl_opts {tl_opts}
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import modelarts_utils
    # update_parser_defaults_from_yaml(parser)
  
    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass

  def test_finetune_r1024_r64_ks1_disney(self, debug=True):
    """

    Usage:
        export CUDA_VISIBLE_DEVICES=0
        export PORT=12345
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
          Testing_train_cips3d_ffhq_v10().test_finetune_r1024_r64_ks1_disney(debug=False)"

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
                --tl_config_file exp/cips3d/configs/train_cips3d_ffhq_v3.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                {"--tl_resume --tl_resumedir " + outdir if resume else ""}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    if int(os.environ['RUN_NUM']) > 0:
      run_command = f"""
                  python -c "from tl2.modelarts.tests.test_run import TestingRun;\
                        TestingRun().test_run_v2(number={os.environ['RUN_NUM']}, )" \
                        --tl_opts root_obs {cfg.root_obs}
                  """
      p = tl2_utils.Worker(name='Run', args=(run_command,))
      p.start()
  
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    PORT = os.environ.get('PORT', 8888)
  
    if n_gpus > 1:
      python_str = f'python -m torch.distributed.launch --nproc_per_node {n_gpus}'
    else:
      python_str = f'python '
  
    cmd_str = f"""
        {python_str}
        exp/cips3d/scripts/train_v3.py
        --batch {cfg.batch}
        --chunk {cfg.chunk}
        --expname {cfg.expname}
        --size {cfg.size}
        {get_append_cmd_str(args)}
        """
    if debug:
      cmd_str += f"""
                  --tl_debug
                  --batch 4
                  --chunk 4
                  --tl_opts num_workers 0
                  """
    else:
      cmd_str += f"""
                  --batch {cfg.batch}
                  --chunk {cfg.chunk}
                  --tl_opts {tl_opts} num_workers 0
                  """
    start_cmd_run(cmd_str)
    # from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
    # from tl2.modelarts import moxing_utils
  
    # update_parser_defaults_from_yaml(parser)
    # if rank == 0:
    #   moxing_utils.setup_tl_outdir_obs(global_cfg)
    #   moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  
    # moxing_utils.modelarts_sync_results_dir(global_cfg, join=True)
  
    # modelarts_utils.setup_tl_outdir_obs(global_cfg)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)
    #
    # modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    # modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    pass
  

class Testing_train_cips3d_compcars_v10(unittest.TestCase):
  
  def test__sample_multi_view_web(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_compcars_v10;\
          Testing_train_cips3d_compcars_v10().test__sample_multi_view_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_compcars_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"

    import importlib

    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --

            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test__flip_inversion_web(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_compcars_v10;\
          Testing_train_cips3d_compcars_v10().test__flip_inversion_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
  
    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
  
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_compcars_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    import importlib
  
    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --

            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

  def test__render_multi_view_web(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0
        export TIME_STR=0
        export PYTHONPATH=.:exp:tl2_lib
        python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_compcars_v10;\
          Testing_train_cips3d_compcars_v10().test__render_multi_view_web(debug=False)" \
          --tl_opts port 8501

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'
    from tl2 import tl2_utils
    from tl2.launch.launch_utils import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)
  
    tl_opts_list = tl2_utils.parser_args_from_list(name="--tl_opts", argv_list=sys.argv, type='list')
    tl_opts = ' '.join(tl_opts_list)
    print(f'tl_opts:\n {tl_opts}')
  
    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file exp/cips3d/configs/train_cips3d_compcars_v10.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)
  
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    os.environ['DNNLIB_CACHE_DIR'] = "cache_dnnlib"
    os.environ['TORCH_EXTENSIONS_DIR'] = "cache_torch_extensions"
    os.environ['PATH'] = f"{os.path.dirname(sys.executable)}:{os.environ['PATH']}"
    os.environ['MAX_JOBS '] = "8"
  
    import importlib
  
    # script = "tl2_lib/tl2/proj/streamlit/scripts/run_web.py"
    script = importlib.import_module('tl2.proj.streamlit.scripts.run_web').__file__
    if debug:
      cmd_str = f"""
          python
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      cmd_str_prefix = f"""
              {os.path.dirname(sys.executable)}/streamlit run
              --server.port {cfg.port}
              --server.headless true
              {script}
              --

            """
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass










