# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the FA4 MLA prefill warmup."""

import math
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.warmup import fa4_combine_warmup, fa4_mla_prefill_warmup


def _fake_layer_config(num_heads: Any, mla_dims: Any, window_size: Any) -> Any:
    # Stand-in for the CUDA-only FA4MLALayerConfig dataclass.
    return SimpleNamespace(
        num_heads=num_heads, mla_dims=mla_dims, window_size=window_size
    )


@pytest.mark.skip_global_cleanup
def test_split_kv_combine_warmup_matches_runtime_compile_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_keys: list[tuple[Any, ...]] = []

    def _flash_attn_fwd_combine(
        out_partial: torch.Tensor,
        lse_partial: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        seqused: torch.Tensor | None = None,
        num_splits_dynamic_ptr: torch.Tensor | None = None,
        virtual_batch_idx: torch.Tensor | None = None,
        semaphore_to_reset: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        # Compile-only: no real tensor is allocated and no kernel launches.
        assert isinstance(out_partial, FakeTensor)
        # compile_key of flash-attention 9cd61de cute/interface.py
        # _flash_attn_fwd_combine, without the arch and the head-dim-derived
        # tile_m=8 / k_block_size=128 (which raise log_max_splits to >= 5).
        compile_keys.append(
            (
                out.dtype,
                out_partial.dtype,
                out_partial.shape[-1],
                out_partial.shape[-2],
                max(math.ceil(math.log2(out_partial.shape[0])), 5),
                cu_seqlens is not None,
                seqused is not None,
                lse is not None,
                virtual_batch_idx is not None,
                num_splits_dynamic_ptr is not None,
                semaphore_to_reset is not None,
                output_scale is not None,
            )
        )

    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.cute.interface",
        SimpleNamespace(_flash_attn_fwd_combine=_flash_attn_fwd_combine),
    )
    monkeypatch.setattr(
        fa4_combine_warmup,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    keys: list[Any] = [
        SimpleNamespace(
            q_dtype=torch.bfloat16,
            q_shape=(8, num_heads, 192),
            v_shape=(8, num_heads, 128),
            num_splits=num_splits,
        )
        for num_heads, num_splits in ((16, 0), (32, 0), (64, 1))
    ]

    fa4_combine_warmup.warm_split_kv_combine(keys)

    # _flash_attn_fwd's split-KV combine call: varlen, scheduler-metadata
    # num_splits_dynamic_ptr, no seqused / virtual_batch_idx / semaphore /
    # output_scale, LSE iff requested, num_head = q heads, auto splits < 32.
    # num_splits=1 keys never combine.
    assert sorted(compile_keys, key=str) == sorted(
        (
            (
                torch.bfloat16,
                torch.float32,
                128,
                num_heads,
                5,
                True,
                False,
                return_lse,
                False,
                True,
                False,
                False,
            )
            for num_heads in (16, 32)
            for return_lse in (True, False)
        ),
        key=str,
    )


@pytest.mark.skip_global_cleanup
def test_prefill_warmup_collects_distinct_layer_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep must feed one config per distinct (heads, dims, window)
    FA4 layer across the target and draft models, and skip non-FA4 layers.
    """

    def _layer(
        num_heads: int,
        window: int | None,
        fa_version: int | None,
    ) -> Any:
        layer: Any = object.__new__(MLAAttention)
        layer.num_heads = num_heads
        layer.qk_nope_head_dim = 128
        layer.qk_rope_head_dim = 64
        layer.v_head_dim = 128
        layer.q_lora_rank = 1536
        layer.kv_lora_rank = 512
        layer.sliding_window = window
        layer.prefill_backend = SimpleNamespace(vllm_flash_attn_version=fa_version)
        return layer

    full_attn = _layer(32, None, 4)
    swa_attn = _layer(32, 1024, 4)
    # Same config as full_attn: must dedupe to one entry.
    duplicate = _layer(32, None, 4)
    drafter = _layer(8, None, 4)
    non_fa4 = _layer(32, None, 3)

    recorded: dict[str, Any] = {}

    class _FakeKernel:
        def get_warmup_keys(self, vllm_config: Any, configs: Any) -> Any:
            recorded["configs"] = configs
            return ["key-0", "key-1"]

        def compile_many(self, keys: Any) -> None:
            recorded["keys"] = keys

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.backends.mla.prefill.flash_attn",
        SimpleNamespace(
            FA4MLALayerConfig=_fake_layer_config,
            _FA4_MLA_PREFILL_KERNEL=_FakeKernel(),
        ),
    )
    target = SimpleNamespace(modules=lambda: [full_attn, swa_attn, duplicate, non_fa4])
    draft = SimpleNamespace(modules=lambda: [drafter])
    monkeypatch.setattr(
        fa4_mla_prefill_warmup,
        "target_and_draft_models",
        lambda worker: [target, draft],
    )
    worker: Any = SimpleNamespace(
        model_runner=SimpleNamespace(is_pooling_model=False),
        vllm_config=SimpleNamespace(),
    )

    fa4_mla_prefill_warmup.fa4_mla_prefill_warmup(worker)

    configs = recorded["configs"]
    assert [(c.num_heads, c.window_size) for c in configs] == [
        (32, None),
        (32, (1023, 0)),
        (8, None),
    ]
    assert recorded["keys"] == ["key-0", "key-1"]
