# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


class model_aware_kv_ops_helper:

    def __init__(self, config: VllmConfig):
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE
        self.tp_size = config.parallel_config.tensor_parallel_size

    def get_model_args(self, model_executable: torch.nn.Module):

        model_config = model_executable.model.config
        self.model_executable = model_executable
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads

        # Deepseek's MLA (Multi-head Latent Attention) uses two different
        # kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
        # When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
        # resulting in a kv_cache shape of [num_blks, blk_size, 1,
        # kv_lora_rank + qk_rope_head_dim].
        # When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
        # to a kv_cache shape of [2, num_blks, blk_size,
        # num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
        # For more details, see vllm/attention/backends/mla/common.py.
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = model_config.kv_lora_rank + \
                model_config.qk_rope_head_dim
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = model_config.qk_nope_head_dim + \
                model_config.qk_rope_head_dim
        else:
            head_size = getattr(model_config, "head_dim", None)
            if head_size is None:
                head_size = int(hidden_size // num_attention_heads)

        return num_heads, head_size

    def get_kv_from_cache(self, kv_cache, num_heads, head_size):
        if self.is_deepseek_mla and self.use_mla_opt:
            key_cache = kv_cache.reshape(-1, num_heads, head_size)
            value_cache = kv_cache.reshape(-1, num_heads, head_size)
        else:
            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)
        return key_cache, value_cache

    def put_kv_to_cache(self, model_executable: torch.nn.Module, keys, values,
                        layer, kv_cache, slot_mapping, start_pos, end_pos):

        model_config = model_executable.model.config

        if self.is_deepseek_mla and self.use_mla_opt:
            layer.self_attn.attn = layer.self_attn.mla_attn
            k_c_normed_k_pe = keys.squeeze(1)
            k_c_normed = k_c_normed_k_pe[:, :model_config.kv_lora_rank]
            k_pe = k_c_normed_k_pe[:, model_config.kv_lora_rank:]
            ops.concat_and_cache_mla(
                k_c_normed.to(kv_cache.device),
                k_pe.to(kv_cache.device),
                kv_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
            )
        else:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            ops.reshape_and_cache_flash(
                keys.to(key_cache.device),
                values.to(value_cache.device),
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )


def get_kv_connector_cache_layout():
    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    if vllm_config.model_config is None:
        logger.warning("Unable to detect current VLLM config. " \
        "Defaulting to NHD kv cache layout.")
    else:
        use_mla = vllm_config.model_config.use_mla
        if not use_mla and kv_config.kv_connector == "NixlConnector":
            logger.info("NixlConnector detected. Setting KV cache " \
            "layout to HND for better xfer performance.")
            return "HND"
    return "NHD"
