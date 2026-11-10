# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two TP ranks build a real GPU Sampler. They must agree on the sampler backend
even when only one rank can use FlashInfer, and refuse to start when they were
configured with different seeds, since each rank feeds its own sampled tokens into
the next step.

Run `pytest tests/distributed/test_tp_sampler_consistency.py`.
"""

from types import SimpleNamespace

import pytest
import ray
import torch

from ..utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)


def _build_sampler(rank: int, seed: int):
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device(f"cuda:{rank}")
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=128,
        device=device,
    )
    vllm_config = SimpleNamespace(
        reasoning_config=None,
        model_config=SimpleNamespace(seed=seed),
        parallel_config=SimpleNamespace(data_parallel_index=0),
    )
    return Sampler(
        vllm_config=vllm_config,  # type: ignore[arg-type]
        max_num_reqs=4,
        vocab_size=128,
        device=device,
        req_states=req_states,
    )


def _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, case):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch.accelerator.set_device_index(torch.device(f"cuda:{rank}"))
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    from vllm.v1.worker.gpu.sample import sampler as sampler_mod

    if case == "backend":
        # Only rank 0 can use FlashInfer (as after a JIT failure on rank 1).
        monkeypatch.setattr(
            sampler_mod, "flashinfer_sampler_supported", lambda: rank == 0
        )
        sampler = _build_sampler(rank, seed=0)
        assert sampler.use_flashinfer is False  # on both ranks
        monkeypatch.setattr(sampler_mod, "flashinfer_sampler_supported", lambda: True)
        assert _build_sampler(rank, seed=0).use_flashinfer is True
    else:  # "seed"
        with pytest.raises(RuntimeError, match=r"seed per TP rank = \[0, 1\]"):
            _build_sampler(rank, seed=rank)


@ray.remote(num_gpus=1, max_calls=1)
def backend_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, "backend")


@ray.remote(num_gpus=1, max_calls=1)
def seed_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port, "seed")


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_target", [backend_worker, seed_worker])
def test_tp_sampler_consistency(monkeypatch: pytest.MonkeyPatch, test_target):
    multi_process_parallel(monkeypatch, 2, 1, test_target)
