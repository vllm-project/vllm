# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.mamba_triton_warmup import (
    _has_mamba_style_cache,
    _warm_batch_memcpy_kernel,
    mamba_triton_warmup,
)
from vllm.v1.kv_cache_interface import MambaSpec


def test_has_mamba_style_cache_detects_mamba_spec() -> None:
    spec = MambaSpec(block_size=1, shapes=((1,),), dtypes=(torch.float32,))
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
        )
    )
    assert _has_mamba_style_cache(runner) is True


def test_has_mamba_style_cache_skips_non_mamba_runner() -> None:
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    assert _has_mamba_style_cache(runner) is False
    assert _has_mamba_style_cache(SimpleNamespace()) is False


def test_mamba_triton_warmup_skips_without_mamba_cache(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("memcpy must not run without a Mamba-style cache")

    monkeypatch.setattr(
        "vllm.model_executor.warmup.mamba_triton_warmup._warm_batch_memcpy_kernel",
        fail,
    )
    mamba_triton_warmup(
        SimpleNamespace(
            device=torch.device("cuda"),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        )
    )


def test_mamba_triton_warmup_runs_for_mamba_cache(monkeypatch) -> None:
    calls: list[str] = []
    spec = MambaSpec(block_size=1, shapes=((1,),), dtypes=(torch.float32,))
    monkeypatch.setattr(
        "vllm.model_executor.warmup.mamba_triton_warmup._warm_batch_memcpy_kernel",
        lambda device: calls.append(str(device.type)),
    )
    mamba_triton_warmup(
        SimpleNamespace(
            device=torch.device("cuda"),
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
            ),
        )
    )
    assert calls == ["cuda"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mamba_batch_memcpy_kernel_compiles_on_gpu() -> None:
    _warm_batch_memcpy_kernel(torch.device("cuda"))
    torch.accelerator.synchronize("cuda")
