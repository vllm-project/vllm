from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from .modeling import (
    LlamaAttention_IC_init,
    LlamaAttention_IC_forward,
    Qwen2Attention_IC_init,
    Qwen2Attention_IC_forward,
    CausalLM_IC_forward,
)


def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        LlamaAttention_IC_init(self, config, layer_idx, compression_config)

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = LlamaAttention_IC_forward
    modeling_llama.LlamaForCausalLM.forward = CausalLM_IC_forward


def replace_qwen2(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen2Attention_IC_init(self, config, layer_idx, compression_config)

    modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_IC_forward
    modeling_qwen2.Qwen2ForCausalLM.forward = CausalLM_IC_forward
