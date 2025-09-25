PREFILL_GPUS=0 DECODE_GPUS=1 PREFILL_PORTS=20003 DECODE_PORTS=20005 \
MODEL=Qwen/Qwen3-1.7B \
bash /home/ubuntu/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh