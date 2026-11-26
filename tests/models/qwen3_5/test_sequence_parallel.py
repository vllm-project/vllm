# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for dense Qwen3.5 sequence parallelism (`--enable-residual-sp`)."""

import pytest
import torch
from torch import nn

from tests.models.registry import _HfExamplesInfo
from tests.models.utils import check_logprobs_close
from tests.utils import create_new_process_for_each_test
from vllm.models.qwen3_5.common import sequence_parallel as qwen3_5_sp
from vllm.platforms import current_platform

MODEL_ID = "vllm-project/Qwen3.5-0.8B-text-only"


@pytest.mark.parametrize(("sp_active", "expected_gathers"), [(True, 1), (False, 0)])
def test_gdn_input_gathered_once_only_while_sp_active(
    monkeypatch: pytest.MonkeyPatch, sp_active: bool, expected_gathers: int
):
    """`in_proj_qkvz` and `in_proj_ba` both read the GDN input, so a pre-hook
    gathers it once instead of a wrapped linear."""
    gathers = []

    def fake_gather(x):
        gathers.append(x)
        return torch.cat([x, x])

    monkeypatch.setattr(qwen3_5_sp, "mark_sp_region", lambda *a: None)
    monkeypatch.setattr(qwen3_5_sp, "sp_all_gather", fake_gather)

    class _GDN(nn.Module):
        out_proj = None

        def forward(self, hidden_states):
            return hidden_states

    layer = nn.Module()
    layer.layer_type = "linear_attention"
    layer.linear_attn = _GDN()
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = layer.mlp.down_proj = None
    # The gathered shards carry one padding row past the 3 real tokens.
    qwen3_5_sp.mark_residual_sp_regions([layer], lambda: 3 if sp_active else None)

    out = layer.linear_attn(hidden_states=torch.ones(2, 4))

    assert len(gathers) == expected_gathers
    assert out.shape[0] == (3 if sp_active else 2)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Dense Qwen3.5 SP is only wired into the CUDA model.",
)
@create_new_process_for_each_test()
def test_tp_sp_generation(vllm_runner, monkeypatch, num_gpus_available):
    """SP generation matches the plain TP baseline, on token counts that
    don't divide by TP and on a prefill long enough for the fused kernels."""
    if num_gpus_available < 2:
        pytest.skip("Need at least 2 GPUs")
    model_info = _HfExamplesInfo(MODEL_ID)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")
    monkeypatch.setenv("VLLM_RESIDUAL_SP_MIN_TOKENS", "1")

    prompts = [
        "Hello, my name is",
        "Summarize: " + "The quick brown fox jumps over the lazy dog. " * 260,
    ]
    outputs = []
    for enable_residual_sp in (False, True):
        with vllm_runner(
            MODEL_ID,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=8,
            enforce_eager=True,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            enable_residual_sp=enable_residual_sp,
        ) as llm:
            outputs.append(
                llm.generate_greedy_logprobs(prompts, max_tokens=8, num_logprobs=5)
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="tp",
        name_1="tp+sp",
    )
