# Adapted from llama.py
"""Inference-only Phi3 model code inherit from Llama.py"""

from vllm.model_executor.models.llama import LlamaForCausalLM


class Phi3ForCausalLM(LlamaForCausalLM):

    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }
