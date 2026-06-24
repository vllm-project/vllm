#ROCR_VISIBLE_DEVICE=0 \
MODEL="${MODEL:-/data/models/gpt-oss-120b-w-mxfp4-a-fp8}" \
HSA_ENABLE_SDMA=0 USE_SVM=0 HSA_XNACK=0 \
VLLM_ROCM_AITER_FUSED_MOE_TRITON_GEMM_A4W4=1 \
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
VLLM_ROCM_USE_SKINNY_GEMM=0 VLLM_ROCM_USE_AITER_RMSNORM=0 \
vllm serve --model ${MODEL} --host localhost --port 8000 --tensor-parallel-size 1 --gpu_memory_utilization 0.7 #--compilation-config '{"mode":"None","cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1]}'
