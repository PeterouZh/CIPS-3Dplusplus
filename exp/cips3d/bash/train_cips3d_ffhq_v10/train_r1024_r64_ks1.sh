set -x
# bash exp/cips3d/bash/train_cips3d_ffhq_v10/train_r1024_r64_ks1.sh


#cuda_devices=${1:-0,1,2,3,4,5,6,7}
cuda_devices=`python -c "import torch;print(','.join([str(i) for i in range(torch.cuda.device_count())]), end='')"`
export CUDA_VISIBLE_DEVICES=$cuda_devices

export ANSI_COLORS_DISABLED=0
#export TIME_STR=1
export TIME_STR=0
export PORT=12345
export PYTHONPATH=.:exp:tl2_lib


python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
  Testing_train_cips3d_ffhq_v10().test_train_r1024_r64_ks1(debug=False)"\
  --tl_opts batch 4 chunk 4 log_ckpt_every 500 total_iters 800000 \
      d_reg_every 15 g_reg_every 5 \
      G_cfg.enable_decoder True G_cfg.freeze_renderer False \
      G_cfg.renderer_detach False G_cfg.predict_rgb_residual False \
      G_cfg.renderer_cfg.N_layers_renderer 2 \
      G_cfg.decoder_cfg.upsample_list "128,256,512,1024" G_cfg.decoder_cfg.kernel_size 1 \
      fade_D False D_cfg.pretrained_size None D_renderer_cfg.pretrained_size None \
      warmup_iters 10000 init_renderer True \
      tl_finetune False \
      finetune_dir xxx/ckptdir/resume









