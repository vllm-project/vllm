#!/bin/bash


server_params=$1
common_params=$2



model_path=$(echo "$common_params" | jq -r '.model')
model_name="${model_path#*/}"
model_type=$(echo "$server_params" | jq -r '.model_type')
model_dtype=$(echo "$server_params" | jq -r '.model_dtype')
model_tp_size=$(echo "$common_params" | jq -r '.tp')
max_batch_size=$(echo "$server_params" | jq -r '.max_batch_size')
max_input_len=$(echo "$server_params" | jq -r '.max_input_len')
max_output_len=$(echo "$server_params" | jq -r '.max_output_len')
trt_llm_version=$(echo "$server_params" | jq -r '.trt_llm_version')

cd ~
rm -rf models
mkdir -p models
cd models
models_dir=$(pwd)
trt_model_path=${models_dir}/${model_name}-trt-ckpt
trt_engine_path=${models_dir}/${model_name}-trt-engine

cd ~
rm -rf tensorrt-demo
git clone https://github.com/neuralmagic/tensorrt-demo.git
cd tensorrt-demo
tensorrt_demo_dir=$(pwd)

# make sure the parameter inside tensorrt_demo is consistent to envvar
sed -i.bak "/key: \"tokenizer_dir\"/,/string_value:/s|string_value: \".*\"|string_value: \"$model_path\"|" ./triton_model_repo/postprocessing/config.pbtxt
sed -i.bak "/key: \"tokenizer_dir\"/,/string_value:/s|string_value: \".*\"|string_value: \"$model_path\"|" ./triton_model_repo/preprocessing/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/ensemble/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/preprocessing/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/postprocessing/config.pbtxt
sed -i.bak "s|\(max_batch_size:\s*\)[0-9]*|\1$max_batch_size|g" ./triton_model_repo/tensorrt_llm_bls/config.pbtxt


cd /
rm -rf tensorrtllm_backend
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
git lfs install
cd tensorrtllm_backend
git checkout $trt_llm_version
tensorrtllm_backend_dir=$(pwd)
git submodule update --init --recursive
cp -r ${tensorrt_demo_dir}/triton_model_repo ${tensorrtllm_backend_dir}/

cd /tensorrtllm_backend
cd ./tensorrt_llm/examples/${model_type}


if echo "$common_params" | jq -e 'has("fp8")' > /dev/null; then

    echo "Key 'fp8' exists in common params. Use quantize.py instead of convert_checkpoint.py"
    echo "Reference: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/README.md"
    python ../quantization/quantize.py \
        --model_dir ${model_path} \
        --dtype ${model_dtype} \
        --tp_size ${model_tp_size} \
        --output_dir ${trt_model_path} \
        --qformat fp8 \
        --kv_cache_dtype fp8 \
        --calib_size 2

else

    echo "Key 'fp8' does not exist in common params. Use convert_checkpoint.py"
    python3 convert_checkpoint.py \
        --model_dir ${model_path} \
        --dtype ${model_dtype} \
        --tp_size ${model_tp_size} \
        --output_dir ${trt_model_path}

fi



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
rm -rf ./tensorrt_llm/1/*
cp -r ${trt_engine_path}/* ./tensorrt_llm/1
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py \
--world_size=${model_tp_size} \
--model_repo=/tensorrtllm_backend/triton_model_repo &