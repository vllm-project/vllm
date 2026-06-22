# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup import kernel_warmup


def test_kernel_warmup_runs_hybrid_and_zeroer_warmups(monkeypatch) -> None:
    calls: list[str] = []

    model = object()

    def fake_hybrid_warmup(*args, **kwargs) -> None:
        assert args == (model,)
        assert kwargs == {"model_dtype": torch.bfloat16}
        calls.append("hybrid")

    class FakeZeroer:
        def warmup(self) -> None:
            calls.append("zeroer")

    def fake_dummy_run(**kwargs) -> None:
        assert kwargs == {
            "num_tokens": 16,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        }
        calls.append("dummy")

    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: False)
    monkeypatch.setattr(
        kernel_warmup,
        "hybrid_gdn_mamba_mrope_warmup",
        fake_hybrid_warmup,
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.minimax_m3_msa_warmup.minimax_m3_msa_warmup",
        lambda worker: None,
    )

    worker = SimpleNamespace(
        get_model=lambda: model,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False)
        ),
        model_runner=SimpleNamespace(
            dtype=torch.bfloat16,
            _kv_block_zeroer=FakeZeroer(),
            _dummy_run=fake_dummy_run,
            is_pooling_model=True,
            attn_groups=[],
        ),
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == ["hybrid", "zeroer"]
