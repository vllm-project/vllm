# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config.vllm import _should_auto_enable_breakable_cudagraph


def _model_config(*architectures: str):
    return SimpleNamespace(architectures=list(architectures))


def test_deepseek_v4_does_not_auto_enable_breakable_cudagraph():
    # Breakable cudagraph disables torch.compile and is 1.5-3.8x slower for MTP
    # decode on SM12x (measured); DeepSeek-V4 defaults to FULL_AND_PIECEWISE.
    assert not _should_auto_enable_breakable_cudagraph(
        _model_config("DeepseekV4ForCausalLM")
    )
    assert not _should_auto_enable_breakable_cudagraph(
        _model_config("DeepSeekV4MTPModel")
    )


def test_minimax_m3_auto_enables_breakable_cudagraph():
    # MiniMax M3 retains upstream's unconditional auto-enable.
    assert _should_auto_enable_breakable_cudagraph(
        _model_config("MiniMaxM3SparseForCausalLM")
    )
    assert _should_auto_enable_breakable_cudagraph(
        _model_config("MiniMaxM3SparseForConditionalGeneration")
    )


def test_other_models_do_not_auto_enable_breakable_cudagraph():
    assert not _should_auto_enable_breakable_cudagraph(
        _model_config("Qwen3ForCausalLM")
    )
