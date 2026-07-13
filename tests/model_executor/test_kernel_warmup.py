# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.model_executor.warmup.kernel_warmup as kernel_warmup_module


class _Backend:
    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


def _group(backend_name: str):
    return SimpleNamespace(backend=_Backend(backend_name))


def _make_worker(attn_groups):
    runner = SimpleNamespace(
        attn_groups=attn_groups,
        is_pooling_model=False,
        _dummy_run=Mock(),
    )
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=5120),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
            kernel_config=SimpleNamespace(
                enable_flashinfer_autotune=True,
                enable_cutedsl_warmup=False,
            ),
            model_config=SimpleNamespace(),
        ),
        get_model=lambda: SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _disable_unrelated_warmups(monkeypatch):
    minimax_module_name = "vllm.model_executor.warmup.minimax_m3_msa_warmup"
    minimax_warmup = ModuleType(minimax_module_name)
    minimax_warmup.minimax_m3_msa_warmup = (  # type: ignore[attr-defined]
        lambda *_args: None
    )
    monkeypatch.setitem(sys.modules, minimax_module_name, minimax_warmup)

    monkeypatch.setattr(kernel_warmup_module, "qwen_triton_warmup", lambda *args: None)
    monkeypatch.setattr(
        kernel_warmup_module, "deepseek_v4_mhc_warmup", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        kernel_warmup_module,
        "sparse_mla_triton_warmup_if_needed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        kernel_warmup_module,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        kernel_warmup_module,
        "deepseek_v4_sparse_mla_attention_warmup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(kernel_warmup_module.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(kernel_warmup_module, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        kernel_warmup_module.current_platform,
        "has_device_capability",
        lambda *_args, **_kwargs: True,
    )


@pytest.mark.parametrize("attn_groups", [[], [[]], [[_group("FLASH_ATTN")]]])
def test_kernel_warmup_skips_flashinfer_autotune_without_flashinfer_attention(
    monkeypatch, attn_groups
):
    autotune = Mock()
    monkeypatch.setattr(kernel_warmup_module, "flashinfer_autotune", autotune)

    kernel_warmup_module.kernel_warmup(_make_worker(attn_groups))

    autotune.assert_not_called()


@pytest.mark.parametrize(
    "backend_name",
    ["FLASHINFER", "FLASHINFER_MLA", "FLASHINFER_MLA_SPARSE_SM120"],
)
def test_kernel_warmup_runs_flashinfer_autotune_for_flashinfer_attention(
    monkeypatch, backend_name
):
    autotune = Mock()
    monkeypatch.setattr(kernel_warmup_module, "flashinfer_autotune", autotune)
    worker = _make_worker([[_group(backend_name)]])

    kernel_warmup_module.kernel_warmup(worker)

    autotune.assert_called_once_with(worker.model_runner)
