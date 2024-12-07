"""V1 vLLM engine test utils"""
import os

def assert_vllm_use_v1():
    if os.getenv("VLLM_USE_V1") != "1":
        raise OSError("Test requires VLLM_USE_V1=\"1\"")