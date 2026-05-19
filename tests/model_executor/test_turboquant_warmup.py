# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

fake_flash_attn: Any = ModuleType("vllm.vllm_flash_attn")
fake_flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
fake_flash_attn.get_scheduler_metadata = lambda *args, **kwargs: None
sys.modules.setdefault("vllm.vllm_flash_attn", fake_flash_attn)

fake_flash_attn_interface: Any = ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
fake_flash_attn_interface.is_fa_version_supported = lambda fa_version: False
fake_flash_attn_interface.fa_version_unsupported_reason = lambda fa_version: "test"
sys.modules.setdefault(
    "vllm.vllm_flash_attn.flash_attn_interface",
    fake_flash_attn_interface,
)

from vllm.model_executor.warmup import kernel_warmup, turboquant_warmup  # noqa: E402


class _FakeTQConfig:
    key_mse_bits = 4
    key_packed_size = 10
    effective_value_quant_bits = 4
    key_fp8 = False
    norm_correction = True
    slot_size_aligned = 24


class _FakeTurboQuantAttentionImpl:
    def __init__(
        self,
        *,
        num_heads: int = 4,
        head_size: int = 8,
        num_kv_heads: int = 2,
        max_num_kv_splits: int = 32,
        scale: float = 0.125,
        tq_config: _FakeTQConfig | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.max_num_kv_splits = max_num_kv_splits
        self.scale = scale
        self.tq_config = tq_config or _FakeTQConfig()
        self.ensure_calls = 0
        self.decode_calls: list[dict[str, Any]] = []
        self.continuation_calls: list[dict[str, Any]] = []

    def _ensure_on_device(self, layer: torch.nn.Module, device: torch.device) -> None:
        self.ensure_calls += 1
        layer._tq_Pi = torch.eye(self.head_size, dtype=torch.float32, device=device)
        layer._tq_PiT = torch.eye(self.head_size, dtype=torch.float32, device=device)
        layer._tq_centroids = torch.zeros(16, dtype=torch.float32, device=device)

    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        self.decode_calls.append(
            {
                "query": query,
                "kv_cache": kv_cache,
                "attn_metadata": attn_metadata,
                "Pi": Pi,
                "centroids": centroids,
                "PiT": PiT,
                "layer": layer,
            }
        )
        return torch.empty_like(query)

    def _continuation_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_chunk: torch.Tensor,
        val_chunk: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        self.continuation_calls.append(
            {
                "layer": layer,
                "query": query,
                "key_chunk": key_chunk,
                "val_chunk": val_chunk,
                "kv_cache": kv_cache,
                "block_table": block_table,
                "cached_len": cached_len,
                "seq_len": seq_len,
                "Pi": Pi,
                "centroids": centroids,
            }
        )
        return torch.empty_like(query)


class _FakeAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        kv_cache_dtype: str = "turboquant_4bit_nc",
        impl: _FakeTurboQuantAttentionImpl | None = None,
    ) -> None:
        super().__init__()
        self.kv_cache_dtype = kv_cache_dtype
        self.impl = impl or _FakeTurboQuantAttentionImpl()


@pytest.fixture(autouse=True)
def patch_turboquant_types(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(turboquant_warmup, "Attention", _FakeAttention)
    monkeypatch.setattr(
        turboquant_warmup,
        "TurboQuantAttentionImpl",
        _FakeTurboQuantAttentionImpl,
    )
    monkeypatch.setattr(
        turboquant_warmup.torch.accelerator,
        "synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        turboquant_warmup,
        "is_workspace_manager_initialized",
        lambda: False,
    )


def test_turboquant_decode_warmup_skips_non_tq_layers() -> None:
    layer = _FakeAttention(kv_cache_dtype="fp8")
    model = torch.nn.Sequential(layer)

    turboquant_warmup.turboquant_decode_warmup(
        model,
        device=torch.device("cpu"),
        block_size=16,
        block_table_stride=8,
        max_num_decode_tokens=4,
        model_dtype=torch.bfloat16,
    )

    assert layer.impl.decode_calls == []
    assert layer.impl.ensure_calls == 0


def test_turboquant_decode_warmup_builds_runtime_shaped_inputs() -> None:
    impl = _FakeTurboQuantAttentionImpl(max_num_kv_splits=64)
    model = torch.nn.Sequential(_FakeAttention(impl=impl))

    turboquant_warmup.turboquant_decode_warmup(
        model,
        device=torch.device("cpu"),
        block_size=32,
        block_table_stride=17,
        max_num_decode_tokens=4,
        model_dtype=torch.bfloat16,
    )

    calls = impl.decode_calls
    assert len(calls) == 1
    call = calls[0]
    assert call["query"].shape == (4, impl.num_heads, impl.head_size)
    assert call["query"].dtype == torch.bfloat16
    assert call["kv_cache"].shape == (
        2,
        32,
        impl.num_kv_heads,
        impl.tq_config.slot_size_aligned,
    )
    assert call["kv_cache"].dtype == torch.uint8
    metadata = call["attn_metadata"]
    assert metadata.block_table.shape == (4, 17)
    assert metadata.block_table.tolist()[0][:2] == [1, 0]
    assert metadata.block_table.tolist()[3][:2] == [1, 0]
    assert metadata.seq_lens.tolist() == [1, 1, 1, 1]
    assert metadata.query_start_loc.tolist() == [0, 1, 2, 3, 4]
    assert metadata.num_decodes == 4
    assert metadata.num_decode_tokens == 4
    assert call["Pi"].shape == (impl.head_size, impl.head_size)
    assert call["layer"] is model[0]
    assert impl.ensure_calls == 1
    assert impl.continuation_calls == []


def test_turboquant_decode_warmup_also_warms_full_dequant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        turboquant_warmup,
        "is_workspace_manager_initialized",
        lambda: True,
    )
    impl = _FakeTurboQuantAttentionImpl()
    model = torch.nn.Sequential(_FakeAttention(impl=impl))

    turboquant_warmup.turboquant_decode_warmup(
        model,
        device=torch.device("cpu"),
        block_size=32,
        block_table_stride=17,
        max_num_decode_tokens=4,
        model_dtype=torch.bfloat16,
    )

    assert len(impl.decode_calls) == 1
    assert len(impl.continuation_calls) == 1
    call = impl.continuation_calls[0]
    assert call["layer"] is model[0]
    assert call["query"].shape == (1, impl.num_heads, impl.head_size)
    assert call["query"].dtype == torch.bfloat16
    assert call["key_chunk"].shape == (1, impl.num_kv_heads, impl.head_size)
    assert call["val_chunk"].shape == (1, impl.num_kv_heads, impl.head_size)
    assert call["kv_cache"] is impl.decode_calls[0]["kv_cache"]
    assert call["block_table"].shape == (1, 17)
    assert call["block_table"].tolist()[0][:2] == [1, 0]
    assert call["cached_len"] == 32
    assert call["seq_len"] == 33
    assert call["Pi"].shape == (impl.head_size, impl.head_size)


def test_turboquant_decode_warmup_deduplicates_compile_key() -> None:
    first = _FakeAttention(impl=_FakeTurboQuantAttentionImpl())
    second = _FakeAttention(impl=_FakeTurboQuantAttentionImpl())
    model = torch.nn.Sequential(first, second)

    turboquant_warmup.turboquant_decode_warmup(
        model,
        device=torch.device("cpu"),
        block_size=16,
        block_table_stride=8,
        max_num_decode_tokens=4,
        model_dtype=torch.float16,
    )

    assert len(first.impl.decode_calls) == 1
    assert second.impl.decode_calls == []
    assert first.impl.continuation_calls == []
    assert second.impl.continuation_calls == []
    assert first.impl.ensure_calls == 1
    assert second.impl.ensure_calls == 0


def test_turboquant_decode_warmup_keeps_distinct_compile_keys() -> None:
    first = _FakeAttention(
        impl=_FakeTurboQuantAttentionImpl(num_heads=4, num_kv_heads=2)
    )
    second = _FakeAttention(
        impl=_FakeTurboQuantAttentionImpl(num_heads=8, num_kv_heads=2)
    )
    model = torch.nn.Sequential(first, second)

    turboquant_warmup.turboquant_decode_warmup(
        model,
        device=torch.device("cpu"),
        block_size=16,
        block_table_stride=8,
        max_num_decode_tokens=4,
        model_dtype=torch.float16,
    )

    assert len(first.impl.decode_calls) == 1
    assert len(second.impl.decode_calls) == 1


def test_kernel_warmup_passes_turboquant_runtime_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_tq_warmup(model_arg, **kwargs):
        calls.append({"model": model_arg, **kwargs})

    monkeypatch.setattr(kernel_warmup, "turboquant_decode_warmup", fake_tq_warmup)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: False)

    model = torch.nn.Linear(1, 1)
    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=1024,
            max_num_seqs=7,
        ),
        cache_config=SimpleNamespace(block_size=48),
        model_runner=SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            input_batch=SimpleNamespace(
                block_table=SimpleNamespace(
                    block_tables=[
                        SimpleNamespace(block_size=16, max_num_blocks_per_req=257)
                    ]
                )
            ),
            is_pooling_model=False,
            attn_groups=[],
        ),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False)
        ),
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == [
        {
            "model": model,
            "device": torch.device("cpu"),
            "block_size": 16,
            "block_table_stride": 257,
            "max_num_decode_tokens": 7,
            "model_dtype": torch.bfloat16,
        }
    ]


def test_kernel_warmup_reads_v2_block_table_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_tq_warmup(model_arg, **kwargs):
        calls.append({"model": model_arg, **kwargs})

    monkeypatch.setattr(kernel_warmup, "turboquant_decode_warmup", fake_tq_warmup)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: False)

    model = torch.nn.Linear(1, 1)
    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=1024,
            max_num_seqs=7,
        ),
        cache_config=SimpleNamespace(block_size=48),
        model_runner=SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            block_tables=SimpleNamespace(
                kernel_block_sizes=[32],
                input_block_tables=[torch.zeros((2, 129), dtype=torch.int32)],
            ),
            is_pooling_model=False,
            attn_groups=[],
        ),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False)
        ),
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == [
        {
            "model": model,
            "device": torch.device("cpu"),
            "block_size": 32,
            "block_table_stride": 129,
            "max_num_decode_tokens": 7,
            "model_dtype": torch.bfloat16,
        }
    ]


def test_turboquant_kernels_do_not_specialize_runtime_strides() -> None:
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_decode_stage1,
        _tq_full_dequant_kv,
    )

    assert set(_tq_decode_stage1.do_not_specialize) == {
        "stride_qb",
        "stride_qh",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_bt_b",
        "stride_mid_b",
        "stride_mid_h",
        "stride_mid_s",
    }
    assert set(_tq_full_dequant_kv.do_not_specialize) == {
        "stride_ko_b",
        "stride_ko_h",
        "stride_ko_s",
        "stride_vo_b",
        "stride_vo_h",
        "stride_vo_s",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_bt_b",
    }
