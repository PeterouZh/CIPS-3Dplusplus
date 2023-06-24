set -x
# conda activate py310
# bash exp/tests/setup_env_debug.sh

pip install tl2

# torch
pip3 install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install  -r requirements.txt
pip install torch-fidelity
#pip install --no-cache-dir ninja
#pip install -e torch_fidelity_lib

if [ -d "tl2_lib" ]
then
    pip uninstall -y tl2
fi



