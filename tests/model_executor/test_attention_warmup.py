# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.model_executor.warmup import attention_warmup, kernel_warmup
from vllm.model_executor.warmup.attention_warmup import mixed_batch_attention_warmup
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import AttentionGroup


def _worker(*, use_v2: bool, max_num_batched_tokens: int, max_model_len: int = 4096):
    calls = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(_dummy_run=lambda **kw: calls.append(kw)),
        use_v2_model_runner=use_v2,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        execute_model=object(),
        sample_tokens=object(),
    )
    return worker, calls


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "max_model_len", "expected"),
    [(8192, 4096, 16), (6, 4096, 6), (8192, 8, 8)],
)
def test_v1_runs_mixed_dummy_batch_within_limits(
    max_num_batched_tokens, max_model_len, expected
):
    worker, calls = _worker(
        use_v2=False,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
    )
    mixed_batch_attention_warmup(worker)
    assert calls == [
        dict(
            num_tokens=expected,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    ]


@pytest.mark.parametrize(("max_num_batched_tokens", "expected"), [(8192, 16), (6, 6)])
@pytest.mark.parametrize("ran", [True, False])
def test_v2_runs_mixed_prefill_decode_step(
    monkeypatch, max_num_batched_tokens, expected, ran
):
    worker, dummy_calls = _worker(
        use_v2=True, max_num_batched_tokens=max_num_batched_tokens
    )
    mixed_calls = []

    def fake_mixed_warmup(runner, execute_model, sample_tokens, num_tokens):
        mixed_calls.append((runner, execute_model, sample_tokens, num_tokens))
        return ran

    monkeypatch.setattr(
        attention_warmup, "run_mixed_prefill_decode_warmup", fake_mixed_warmup
    )
    mixed_batch_attention_warmup(worker)

    assert mixed_calls == [
        (worker.model_runner, worker.execute_model, worker.sample_tokens, expected)
    ]
    # No dummy-run fallback when the mixed step is skipped.
    assert dummy_calls == []


class _PPGroup:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, timeout=5)
        self._lock = threading.Lock()
        self._tensors: list[torch.Tensor] = []

    def cpu_all_reduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        with self._lock:
            self._tensors.append(tensor)

        if self._barrier.wait() == 0:
            reduced = torch.stack(self._tensors).sum(dim=0)
            for other in self._tensors:
                other.copy_(reduced)
        self._barrier.wait()
        return tensor


@dataclass
class _ModelRunner:
    attn_groups: list[list[AttentionGroup]]
    is_pooling_model: bool = False


@dataclass
class _Backend:
    name: str

    def get_name(self) -> str:
        return self.name


@dataclass
class _Worker:
    model_runner: _ModelRunner
    use_v2_model_runner: bool = False


def _workers(
    backend_names_by_rank: tuple[tuple[str, ...], ...], *, use_v2: bool
) -> tuple[_Worker, ...]:
    workers = []
    for backend_names in backend_names_by_rank:
        attention_groups = []
        for backend_name in backend_names:
            attention_groups.append(
                AttentionGroup(
                    backend=_Backend(backend_name),
                    layer_names=["layer"],
                    kv_cache_spec=AttentionSpec(
                        block_size=16,
                        num_kv_heads=1,
                        head_size=128,
                        dtype=torch.float16,
                    ),
                    kv_cache_group_id=0,
                )
            )
        workers.append(
            _Worker(_ModelRunner([attention_groups]), use_v2_model_runner=use_v2)
        )
    return tuple(workers)


@pytest.mark.parametrize(
    ("backend_names_by_rank", "use_v2", "expected_warmup"),
    [
        ((("FLASHINFER",),), False, True),
        ((("FLASHINFER",), ("FLASHINFER",)), False, True),
        ((("FLASHINFER",), ("FLASHINFER",)), True, True),
        ((("FLASHINFER", "OTHER_ATTENTION"),), False, False),
        ((("FLASHINFER", "OTHER_ATTENTION"),), True, True),
        ((("FLASHINFER", "OTHER_ATTENTION"), ("FLASHINFER",)), False, False),
        ((("FLASHINFER", "OTHER_ATTENTION"), ("OTHER_ATTENTION",)), True, True),
    ],
)
@pytest.mark.cpu_test
def test_flashinfer_attention_warmup_gate(
    monkeypatch, backend_names_by_rank, use_v2, expected_warmup
):
    workers = _workers(backend_names_by_rank, use_v2=use_v2)
    pp_group = _PPGroup(len(workers))
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)
    warmup_workers = []
    monkeypatch.setattr(
        kernel_warmup,
        "mixed_batch_attention_warmup",
        lambda worker: warmup_workers.append(worker),
    )

    with ThreadPoolExecutor(max_workers=pp_group.world_size) as executor:
        futures = [
            executor.submit(kernel_warmup._warmup_flashinfer_attention, worker)
            for worker in workers
        ]
        for future in futures:
            future.result()

    expected_warmup_calls = pp_group.world_size if expected_warmup else 0
    assert len(warmup_workers) == expected_warmup_calls
