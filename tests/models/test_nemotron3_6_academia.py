# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.nemotron3_6_academia import (
    AcademiaLoopConditioning,
    AcademiaLoopSeam,
)
from vllm.transformers_utils.configs.nemotron3_6_academia import (
    Nemotron3_6AcademiaConfig,
)


@pytest.mark.parametrize("seam", [False, True])
@torch.no_grad()
def test_loop_table_reload_preserves_captured_storage(seam):
    config = Nemotron3_6AcademiaConfig(
        hidden_size=8,
        loop_conditioning_hidden_size=4,
        loop_adapter_rank=0,
        n_recurrent_loops=2,
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        module = (
            AcademiaLoopSeam(config)
            if seam
            else AcademiaLoopConditioning(config, "test")
        )
    conds = [torch.randn(4) for _ in range(2 if seam else 3)]
    module.update(conds)
    captured_table = module.table.view_as(module.table)
    before = captured_table.clone()
    pointer = captured_table.data_ptr()
    head = module.entry_beta if seam else module.beta_head
    head.bias.add_(1)
    module.update(conds)
    assert module.table.data_ptr() == pointer
    torch.testing.assert_close(captured_table, module.table)
    torch.testing.assert_close(captured_table[:, 8:16], before[:, 8:16] + 1)
    assert "table" not in module.state_dict()


def _cache_layout(model):
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    return (
        sum(isinstance(m, Attention) for m in model.modules()),
        sum(isinstance(m, MambaBase) for m in model.modules()),
    )


@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
def test_loop_cache_modes(vllm_runner, tmp_path, cache_mode):
    """Last-loop attention must retain independent GDP state through decoding."""
    from vllm import SamplingParams

    Nemotron3_6AcademiaConfig(
        architectures=["Nemotron3_6AcademiaForCausalLM"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        n_input_layers=0,
        n_recurrent_layers=2,
        n_output_layers=0,
        n_recurrent_loops=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        loop_conditioning_hidden_size=8,
        loop_adapter_rank=0,
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=2,
        enable_prefix_caching=False,
        hf_overrides={"recurrent_kv_cache_mode": cache_mode},
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5]}],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1),
        )
        for output in outputs:
            completion = output.outputs[0]
            assert len(completion.token_ids) == 4
            assert all(
                torch.isfinite(torch.tensor(step[token].logprob))
                for token, step in zip(completion.token_ids, completion.logprobs)
            )
        attention_caches = 2 if cache_mode == "per_loop" else 1
        assert runner.apply_model(_cache_layout) == [(attention_caches, 2)]
