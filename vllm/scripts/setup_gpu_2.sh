curl -LsSf https://astral.sh/uv/install.sh | sh

bash
source $HOME/.local/bin/env

uv venv --python 3.12 --seed --managed-python

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8 ccache

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

cd /holly
git clone git@github.com:hcasalet/villum.git
cd villum
source .venv/bin/activate

uv pip install -U pip
uv pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install packaging setuptools wheel ninja cmake numpy nixl
uv pip install -U "triton==3.5"
