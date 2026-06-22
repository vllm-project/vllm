# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.warmup import kernel_warmup, minimax_m3_msa_warmup


def _make_worker(model_runner):
    return SimpleNamespace(
        model_runner=model_runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False),
        ),
    )


def test_kernel_warmup_invokes_private_kv_block_zeroer(monkeypatch):
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(minimax_m3_msa_warmup, "minimax_m3_msa_warmup", Mock())

    zeroer = Mock()
    model_runner = SimpleNamespace(
        _kv_block_zeroer=zeroer,
        is_pooling_model=True,
        attn_groups=[],
    )

    kernel_warmup.kernel_warmup(_make_worker(model_runner))

    zeroer.warmup.assert_called_once_with()


def test_kernel_warmup_invokes_public_kv_block_zeroer(monkeypatch):
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(minimax_m3_msa_warmup, "minimax_m3_msa_warmup", Mock())

    zeroer = Mock()
    model_runner = SimpleNamespace(
        kv_block_zeroer=zeroer,
        is_pooling_model=True,
        attn_groups=[],
    )

    kernel_warmup.kernel_warmup(_make_worker(model_runner))

    zeroer.warmup.assert_called_once_with()
