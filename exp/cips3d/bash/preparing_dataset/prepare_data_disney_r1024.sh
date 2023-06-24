set -x

# bash exp/cips3d/bash/preparing_dataset/prepare_data_disney_r1024.sh


export CUDA_VISIBLE_DEVICES=0
export PORT=12345
export TIME_STR=0
export PYTHONPATH=.:exp:tl2_lib


python -c "from exp.tests.test_cips3dpp import Testing_preparing_dataset;\
  Testing_preparing_dataset().test_prepare_data_disney_r1024(debug=False)" \
  --tl_opts n_worker 8

