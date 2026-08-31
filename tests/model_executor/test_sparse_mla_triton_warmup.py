# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.warmup import sparse_mla_triton_warmup as warmup_module


def test_b12x_sparse_warms_prefill_chunk_metadata(monkeypatch) -> None:
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=SimpleNamespace(name="B12X_MLA_SPARSE")
        )
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        attn_groups=(),
        vllm_config=vllm_config,
    )
    worker = SimpleNamespace(
        model_runner=runner,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=3072,
            max_num_seqs=8,
        ),
    )
    warmed = []
    monkeypatch.setattr(
        warmup_module,
        "_compile_prefill_chunk_metadata_kernel",
        warmed.append,
    )
    executed = []
    monkeypatch.setattr(
        warmup_module,
        "_execute_prefill_chunk_metadata_kernel",
        lambda value: executed.append(value) or 2,
    )

    warmup_module.sparse_mla_triton_warmup(worker)

    assert warmed == [vllm_config]
    assert executed == [worker]
