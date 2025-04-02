find /tmp -iname "*attn.pt" 2>/dev/null | cut -d'/' -f1,2,3 | uniq | xargs rm -r

VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=1 python3 prefill_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=1 python3 decode_example.py