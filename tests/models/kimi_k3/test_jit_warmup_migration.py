# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.kimi_k3.common.mtp import _FUSED_MTP_INPUT_KERNEL
from vllm.models.kimi_k3.nvidia.ops.attn_res import _ATTN_RES_KERNEL
from vllm.triton_utils import triton


def _vllm_config():
    text_config = SimpleNamespace(
        attn_res_block_size=8,
        hidden_size=7168,
        num_hidden_layers=64,
        rms_norm_eps=1e-5,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=text_config,
        )
    )


def test_attn_res_warmup_covers_runtime_dispatch_classes(monkeypatch):
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.ops.attn_res."
        "current_platform.is_arch_support_pdl",
        lambda: False,
    )
    vllm_config = _vllm_config()
    config = vllm_config.model_config.hf_text_config
    block_size = config.attn_res_block_size
    max_blocks = triton.cdiv(config.num_hidden_layers, block_size)
    warmed_keys = set(_ATTN_RES_KERNEL.get_warmup_keys(vllm_config))

    for num_tokens in (1, 256):
        for layer_idx in range(config.num_hidden_layers):
            is_block_write = layer_idx % block_size == 0
            previous_blocks = triton.cdiv(layer_idx, block_size)
            pre_key = _ATTN_RES_KERNEL.dispatch(
                dtype=vllm_config.model_config.dtype,
                num_tokens=num_tokens,
                num_blocks=previous_blocks,
                hidden_size=config.hidden_size,
                max_blocks=max_blocks,
                block_write_idx=layer_idx // block_size if is_block_write else -1,
                eps=config.rms_norm_eps,
                output_norm_eps=config.rms_norm_eps,
                has_delta=layer_idx > 0,
                apply_output_norm=True,
                launch_pdl=False,
            )
            assert pre_key in warmed_keys

            post_key = _ATTN_RES_KERNEL.dispatch(
                dtype=vllm_config.model_config.dtype,
                num_tokens=num_tokens,
                num_blocks=previous_blocks + is_block_write,
                hidden_size=config.hidden_size,
                max_blocks=max_blocks,
                block_write_idx=-1,
                eps=config.rms_norm_eps,
                output_norm_eps=config.rms_norm_eps,
                has_delta=not is_block_write,
                apply_output_norm=True,
                launch_pdl=False,
            )
            assert post_key in warmed_keys

        for has_delta in (False, True):
            final_key = _ATTN_RES_KERNEL.dispatch(
                dtype=vllm_config.model_config.dtype,
                num_tokens=num_tokens,
                num_blocks=max_blocks,
                hidden_size=config.hidden_size,
                max_blocks=max_blocks,
                block_write_idx=-1,
                eps=config.rms_norm_eps,
                output_norm_eps=0.0,
                has_delta=has_delta,
                apply_output_norm=False,
                launch_pdl=False,
            )
            assert final_key in warmed_keys


def test_fused_mtp_input_warmup_matches_runtime_dispatch():
    vllm_config = _vllm_config()
    config = vllm_config.model_config.hf_text_config
    warmed_keys = _FUSED_MTP_INPUT_KERNEL.get_warmup_keys(vllm_config)
    runtime_key = _FUSED_MTP_INPUT_KERNEL.dispatch(
        positions_dtype=torch.int64,
        dtype=vllm_config.model_config.dtype,
        inputs_embeds_stride=config.hidden_size,
        previous_hidden_states_stride=config.hidden_size,
        output_stride=2 * config.hidden_size,
        hidden_size=config.hidden_size,
    )

    assert warmed_keys == [runtime_key]
