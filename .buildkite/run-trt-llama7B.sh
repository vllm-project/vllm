#!/bin/bash

# This script should be run inside the trt-llm docker container, command:
# docker run -it --net host -e HF_TOKEN=<your HF TOKEN> --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia --gpus all --entrypoint /bin/bash nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
# (please modify `<your HF TOKEN>` to your own huggingface token in the above command
# Then, copy-paste this file into the docker and execute it using bash.

set -xe
TRT_LLM_VERSION=r24.04
model_path=meta-llama/llama-2-7b-chat-hf
model_name=llama-2-7b-chat-hf
model_type=llama
model_dtype=float16
model_tp_size=1
max_batch_size=233
max_input_len=15000
max_output_len=15000
cd ~
mkdir models
cd models
models_dir=`pwd`
trt_model_path=${models_dir}/${model_name}-trt-ckpt
trt_engine_path=${models_dir}/${model_name}-trt-engine



cd ~
git clone https://github.com/neuralmagic/tensorrt-demo.git
cd tensorrt-demo
tensorrt_demo_dir=`pwd`

# make sure the parameter inside tensorrt_demo is consistent to envvar
sed -i.bak "/key: \"tokenizer_dir\"/,/string_value:/s|string_value: \".*\"|string_value: \"$model_path\"|" ./triton_model_repo/postprocessing/config.pbtxt
sed -i.bak "/key: \"tokenizer_dir\"/,/string_value:/s|string_value: \".*\"|string_value: \"$model_path\"|" ./triton_model_repo/preprocessing/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/ensemble/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/preprocessing/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/postprocessing/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/tensorrt_llm_bls/config.pbtxt


cd /
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
git lfs install
cd tensorrtllm_backend
git checkout $TRT_LLM_VERSION
tensorrtllm_backend_dir=`pwd`

git submodule update --init --recursive
cp -r ${tensorrt_demo_dir}/triton_model_repo ${tensorrtllm_backend_dir}/

cd /tensorrtllm_backend
cd ./tensorrt_llm/examples/${model_type}

python3 convert_checkpoint.py \
    --model_dir ${model_path} \
    --dtype ${model_dtype} \
    --tp_size ${model_tp_size} \
    --output_dir ${trt_model_path}
    
trtllm-build \
    --checkpoint_dir=${trt_model_path} \
    --gpt_attention_plugin=${model_dtype} \
    --gemm_plugin=${model_dtype} \
    --remove_input_padding=enable \
    --paged_kv_cache=enable \
    --tp_size=${model_tp_size} \
    --max_batch_size=${max_batch_size} \
    --max_input_len=${max_input_len} \
    --max_output_len=${max_output_len} \
    --max_num_tokens=${max_output_len} \
    --opt_num_tokens=${max_output_len} \
    --output_dir=${trt_engine_path} 
    
cd /tensorrtllm_backend/triton_model_repo
cp -r ${trt_engine_path}/* ./tensorrt_llm/1
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py --world_size=${model_tp_size} --model_repo=/tensorrtllm_backend/triton_model_repo &


# sleep for 20 seconds, to make sure the server is launched 
sleep 30


# install vllm inside conda, for benchmarking.
(which wget && which curl) || (apt-get update && apt-get install -y wget curl)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
eval "$(cat ~/.bashrc | tail -n +15)"
conda create -n vllm python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate vllm
pip install vllm

# clone vllm's benchmark_serving script
cd ~
git clone https://github.com/vllm-project/vllm.git
cd vllm/benchmarks/

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python benchmark_serving.py --backend tensorrt-llm --endpoint /v2/models/ensemble/generate_stream --port 8000 --model $model_path --save-result --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 2>&1 | tee benchmark_serving.txt
