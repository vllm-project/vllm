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
max_seq_len=$(echo "$server_params" | jq -r '.max_seq_len')
max_num_tokens=$(echo "$server_params" | jq -r '.max_num_tokens')
trt_llm_version=$(echo "$server_params" | jq -r '.trt_llm_version')

cd ~
rm -rf models
mkdir -p models
cd models
models_dir=$(pwd)
trt_model_path=${models_dir}/${model_name}-trt-ckpt
trt_engine_path=${models_dir}/${model_name}-trt-engine




cd /
rm -rf tensorrtllm_backend
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
git lfs install
cd tensorrtllm_backend
git checkout $trt_llm_version
tensorrtllm_backend_dir=$(pwd)
git submodule update --init --recursive


mkdir triton_model_repo
cp -r all_models/inflight_batcher_llm/* triton_model_repo/
python3 ../tools/fill_template.py -i tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,engine_dir:OUTPUT,decoupled_mode:true,batching_strategy:inflight_fused_batching,batch_scheduler_policy:guaranteed_no_evict,exclude_input_in_output:true,triton_max_batch_size:2048,max_queue_delay_microseconds:0,max_beam_width:1,max_queue_size:2048,enable_kv_cache_reuse:false
python3 ../tools/fill_template.py -i preprocessing/config.pbtxt triton_max_batch_size:2048,tokenizer_dir:$model_path,preprocessing_instance_count:5
python3 ../tools/fill_template.py -i postprocessing/config.pbtxt triton_max_batch_size:2048,tokenizer_dir:$model_path,postprocessing_instance_count:5,skip_special_tokens:false
python3 ../tools/fill_template.py -i ensemble/config.pbtxt triton_max_batch_size:$max_batch_size
python3 ../tools/fill_template.py -i tensorrt_llm_bls/config.pbtxt triton_max_batch_size:$max_batch_size,decoupled_mode:true,accumulate_tokens:"False",bls_instance_count:1


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
--use_fused_mlp \
--reduce_fusion disable \
--workers 8 \
--gpt_attention_plugin ${model_dtype} \
--gemm_plugin ${model_dtype} \
--tp_size ${model_tp_size} \
--max_batch_size ${max_batch_size} \
--max_input_len ${max_input_len} \
--max_seq_len ${max_seq_len} \
--max_num_tokens ${max_num_tokens} \
--output_dir ${trt_engine_path} 



cd /tensorrtllm_backend/triton_model_repo
rm -rf ./tensorrt_llm/1/*
cp -r ${trt_engine_path}/* ./tensorrt_llm/1
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py \
--world_size=${model_tp_size} \
--model_repo=/tensorrtllm_backend/triton_model_repo &