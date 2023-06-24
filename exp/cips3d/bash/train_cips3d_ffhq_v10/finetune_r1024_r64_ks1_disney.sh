set -x
# bash exp/cips3d/bash/train_cips3d_ffhq_v10/finetune_r1024_r64_ks1_disney.sh


#cuda_devices=${1:-0,1,2,3,4,5,6,7}
cuda_devices=`python -c "import torch;print(','.join([str(i) for i in range(torch.cuda.device_count())]), end='')"`
export CUDA_VISIBLE_DEVICES=$cuda_devices

export ANSI_COLORS_DISABLED=0
#export TIME_STR=1
export TIME_STR=0
export PORT=12345
export PYTHONPATH=.:exp:tl2_lib


python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
  Testing_train_cips3d_ffhq_v10().test_finetune_r1024_r64_ks1_disney(debug=False)" \
  --tl_opts batch 4 chunk 4 log_txt_every 10 log_ckpt_every 50 \
      total_iters 5000 ema_start 0 \
      G_cfg.freeze_decoder_mapping True G_kwargs.nerf_cfg.static_viewdirs False \
      d_reg_every 15 g_reg_every 5 \
      G_cfg.decoder_cfg.kernel_size 1 \
      tl_finetune True \
      finetune_dir pretrained/ffhq_r1024_inversion










