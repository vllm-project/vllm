export NCCL_IGNORE_DISABLED_P2P=1
CUDA_VISIBLE_DEVICES=YOUR_GPU_LIST python api_server.py --model /PATH/TO/chatglm2-6b --port 23889 --gpu-memory-utilization=0.95  --host YOUR_IP --max-num-batched-tokens=4096 --tensor-parallel-size=2 
