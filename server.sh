export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export port=12301
# export model_dir='/root/.cache/huggingface/hub/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/74391478bce6dc10b6d1ea323aa591273de23fcd/'
# export tensor_parallel_size=2
python3 -m vllm.entrypoints.api_switch --host 0.0.0.0 --port 12304
# python3  -m vllm.entrypoints.api_server --model $model_dir --host 0.0.0.0 --port $port --trust-remote-code --tensor-parallel-size $tensor_parallel_size
# python3 -m vllm.entrypoints.api_server_multi --modeltype "Baichuan2-13B" --host 0.0.0.0 --port $port --trust-remote-code
# python3 -m vllm.entrypoints.api_server_multi --modeltype "Llama2-13B" --host 0.0.0.0 --port $port --trust-remote-code
# python3 -m vllm.entrypoints.api_server_multi --modeltype "Qwen-14B" --host 0.0.0.0 --port $port --trust-remote-code
# python3 -m vllm.entrypoints.api_server_multi --modeltype "Mixtral-8x7B-v0.1" --host 0.0.0.0 --port $port --trust-remote-code
