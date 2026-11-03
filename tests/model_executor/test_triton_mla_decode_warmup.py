# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Triton MLA decode warmup."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.warmup import triton_mla_decode_warmup
from vllm.model_executor.warmup.triton_mla_decode_warmup import (
    _num_kv_splits_values,
)
from vllm.v1.attention.backends.mla import triton_mla
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl


@pytest.fixture(autouse=True)
def _triton_next_power_of_2(monkeypatch: pytest.MonkeyPatch) -> None:
    # The TritonPlaceholder used when Triton is not installed has no
    # next_power_of_2; provide the same helper real Triton ships.
    monkeypatch.setattr(
        triton_mla.triton,
        "next_power_of_2",
        lambda n: 1 << max(0, n - 1).bit_length(),
        raising=False,
    )


@pytest.mark.skip_global_cleanup
def test_num_kv_splits_values_cover_runtime_selection() -> None:
    """NUM_KV_SPLITS is constexpr: every value _compute_num_kv_splits can
    return across batches must appear, and nothing else. The SM-based cap is
    reachable itself for long sequences, even when not a power of two.
    """
    # max_model_len=8192, sm_count=8: cap min(16, 16) = 16, all powers of two.
    assert _num_kv_splits_values(8192, 8) == (1, 2, 4, 8, 16)
    # Short models never need more than one split.
    assert _num_kv_splits_values(100, 148) == (1,)
    # Long context on a small GPU: cap 200 cuts the power-of-2 ladder and is
    # reachable itself.
    assert _num_kv_splits_values(1_000_000, 100) == (1, 2, 4, 8, 16, 32, 64, 128, 200)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("kv_cache_dtype", "expected_dtype"),
    [("auto", torch.bfloat16), ("fp8", torch.float8_e4m3fn)],
)
def test_decode_warmup_uses_runtime_kv_cache_dtype(
    monkeypatch: pytest.MonkeyPatch, kv_cache_dtype: str, expected_dtype: torch.dtype
) -> None:
    impl = object.__new__(TritonMLAImpl)
    impl.num_heads = 16
    impl.dcp_world_size = 1
    impl.kv_lora_rank = 512
    impl.scale = 1.0
    layer: Any = object.__new__(MLAAttention)
    layer.impl = impl
    layer.layer_name = "layer.0"
    layer.kv_cache_dtype = kv_cache_dtype
    layer._k_scale = torch.ones(1)
    # Like the runtime allocation: fp8 caches are stored as uint8.
    layer.kv_cache = torch.zeros(
        (2, 16, 576), dtype=torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16
    )
    num_kv_splits_values: list[int] = []

    def decode_attention_fwd(
        q: torch.Tensor,
        cache: torch.Tensor,
        kv_c_cache: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_logits: torch.Tensor,
        num_kv_splits: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert attn_logits.shape[2] == num_kv_splits
        num_kv_splits_values.append(num_kv_splits)
        assert cache.dtype == (
            torch.float8_e4m3fn if kv_cache_dtype == "fp8" else torch.bfloat16
        )

    monkeypatch.setattr(
        triton_mla_decode_warmup, "decode_attention_fwd", decode_attention_fwd
    )
    monkeypatch.setattr(
        triton_mla_decode_warmup,
        "target_and_draft_models",
        lambda worker: [SimpleNamespace(modules=lambda: [layer])],
    )
    monkeypatch.setattr(
        triton_mla_decode_warmup,
        "kv_cache_group_ids",
        lambda worker: {"layer.0": 0},
    )
    monkeypatch.setattr(
        triton_mla_decode_warmup,
        "block_table",
        lambda worker, gid, num_reqs: torch.zeros((1, 8), dtype=torch.int32),
    )
    # max_model_len=8192, sm_count=8 -> warmup ladder (1, 2, 4, 8, 16).
    monkeypatch.setattr(
        triton_mla_decode_warmup,
        "current_platform",
        SimpleNamespace(
            num_compute_units=lambda: 8, fp8_dtype=lambda: torch.float8_e4m3fn
        ),
    )
    worker: Any = SimpleNamespace(
        model_runner=SimpleNamespace(
            is_pooling_model=False,
            device="cpu",
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192)),
    )

    triton_mla_decode_warmup.triton_mla_decode_warmup(worker)

    # One call per (constexpr split value, seq_lens dtype); the fp8 view of a
    # uint8 cache must reach the kernel on every call.
    assert num_kv_splits_values == [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]
