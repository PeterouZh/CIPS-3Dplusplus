set -x

# bash exp/cips3d/bash/train_cips3d_ffhq_v10/eval_fid_r512.sh


export CUDA_VISIBLE_DEVICES=0
export PORT=12345
export TIME_STR=0
export PYTHONPATH=.:exp:tl2_lib


python -c "from exp.tests.test_cips3dpp import Testing_train_cips3d_ffhq_v10;\
  Testing_train_cips3d_ffhq_v10().test_eval_fid_r512(debug=False)" \
  --tl_opts batch_gpu 8 default_network_pkl ffhq_r512

