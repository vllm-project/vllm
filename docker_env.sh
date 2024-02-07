#for 4090 nolink
export NCCL_P2P_DISABLE=1 
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:./
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers
pip install tiktoken