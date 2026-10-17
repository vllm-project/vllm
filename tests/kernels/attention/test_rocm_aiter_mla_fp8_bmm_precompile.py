# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.layers.attention.mla_attention import (
    _get_aiter_fp8_bmm_precompile_batch_sizes,
)


def _vllm_config(
    max_num_seqs: int,
    max_num_batched_tokens: int = 2048,
    max_num_scheduled_tokens: int | None = None,
    num_speculative_tokens: int = 0,
    max_cudagraph_capture_size: int | None = 512,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
        ),
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=max_cudagraph_capture_size,
        ),
        num_speculative_tokens=num_speculative_tokens,
    )


def test_aiter_fp8_bmm_precompile_uses_graph_limit() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=1024,
        max_cudagraph_capture_size=512,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(range(1, 513))


def test_aiter_fp8_bmm_precompile_uses_scheduler_batch_size_below_graph_limit() -> None:
    vllm_config = _vllm_config(max_num_seqs=128)

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(range(1, 129))


def test_aiter_fp8_bmm_precompile_uses_scheduled_token_limit() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=128,
        max_num_scheduled_tokens=64,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(range(1, 65))


def test_aiter_fp8_bmm_precompile_uses_batched_token_limit() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=128,
        max_num_batched_tokens=32,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(range(1, 33))


def test_aiter_fp8_bmm_precompile_accounts_for_spec_decode_tokens() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=16,
        num_speculative_tokens=3,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(range(1, 65))


def test_aiter_fp8_bmm_precompile_caps_spec_decode_at_graph_limit() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=256,
        max_num_batched_tokens=2048,
        max_cudagraph_capture_size=512,
        num_speculative_tokens=3,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(range(1, 513))


def test_aiter_fp8_bmm_precompile_caps_spec_decode_at_token_limit() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=128,
        max_num_batched_tokens=256,
        num_speculative_tokens=3,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(range(1, 257))


def test_aiter_fp8_bmm_precompile_caps_at_old_limit() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=2048,
        max_cudagraph_capture_size=None,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(
        range(1, 1025)
    )


def test_aiter_fp8_bmm_precompile_defaults_to_old_limit() -> None:
    vllm_config = _vllm_config(
        max_num_seqs=0,
        max_cudagraph_capture_size=None,
    )

    assert _get_aiter_fp8_bmm_precompile_batch_sizes(vllm_config) == list(
        range(1, 1025)
    )
