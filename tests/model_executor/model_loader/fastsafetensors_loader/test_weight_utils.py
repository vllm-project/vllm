# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


def _download_gpt2(tmpdir):
    huggingface_hub.constants.HF_HUB_OFFLINE = False
    download_weights_from_hf(
        "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
    )
    safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
    assert len(safetensors) > 0
    return safetensors


def _assert_matches_safetensors(tmpdir):
    safetensors = _download_gpt2(tmpdir)

    fastsafetensors_tensors = {
        name: tensor
        for name, tensor in fastsafetensors_weights_iterator(safetensors, True)
    }
    hf_safetensors_tensors = {
        name: tensor for name, tensor in safetensors_weights_iterator(safetensors, True)
    }

    assert len(fastsafetensors_tensors) == len(hf_safetensors_tensors)
    for name, fastsafetensors_tensor in fastsafetensors_tensors.items():
        fastsafetensors_tensor = fastsafetensors_tensor.to("cpu")
        assert fastsafetensors_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert fastsafetensors_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(fastsafetensors_tensor.eq(hf_safetensors_tensors[name]))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
@pytest.mark.parametrize("queue_size", [0, 1])
def test_fastsafetensors_model_loader(monkeypatch, queue_size):
    monkeypatch.setenv("VLLM_FASTSAFETENSORS_QUEUE_SIZE", str(queue_size))
    with tempfile.TemporaryDirectory() as tmpdir:
        _assert_matches_safetensors(tmpdir)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_sub_shard_chunking(monkeypatch):
    """A budget below the shard size must still yield identical tensors.

    GPT-2 is a single 522 MiB shard whose largest tensor (wte.weight) is
    147 MiB. The planner's floor is *twice* that, not once: it double-buffers
    the transient copy, and does so even after collapsing the pipeline to
    fully serial, so the smallest satisfiable budget is 2 x 147 = 294 MiB.

    On unified-memory systems (DGX Spark) the copier first reserves 16 MiB per
    DMA worker, 128 MiB by default. 512 MiB leaves 384 MiB there and 512 MiB
    elsewhere, both above the floor and below the shard, forcing the planner
    to split the shard. Below 294 MiB the plan is infeasible and the loader
    raises rather than falling back, which test_fastsafetensors_budget_infeasible
    covers.
    """
    monkeypatch.setenv(
        "VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", str(512 * 1024 * 1024)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _assert_matches_safetensors(tmpdir)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_budget_infeasible(monkeypatch):
    """A budget below the largest tensor must raise, not silently fall back.

    The planner's floor is set by the largest single tensor, not the shard: it
    can always split a shard, but never a tensor, and it needs two transient
    buffers for it. GPT-2's largest tensor (wte.weight) is 147 MiB, so no plan
    can satisfy a 1 MiB budget; the error reports needing >= 2 x 147 MiB.

    Falling back to whole-shard staging here would need a buffer at least as
    large as the tensor the plan could not place, so it is more likely to run
    out of memory, not less. The loader therefore raises with guidance naming
    the way out. Note the iterator is a generator, so the plan is only built
    once iteration starts.
    """
    from fastsafetensors import BudgetInfeasibleError

    monkeypatch.setenv("VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", str(1024 * 1024))
    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors = _download_gpt2(tmpdir)
        with pytest.raises(BudgetInfeasibleError) as excinfo:
            next(iter(fastsafetensors_weights_iterator(safetensors, True)))

    message = str(excinfo.value)
    assert "VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET" in message
    assert "does not fit in the device memory" in message


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_accumulate_resident_skips_derived_budget(monkeypatch):
    """A consumer that keeps less than it reads must not be planned for.

    The planner charges every byte read as resident, so a consumer whose
    parameters materialize during the load -- online quantization stores a
    smaller quantized parameter than the checkpoint bytes it consumes -- would
    be over-charged and refused a load that fits. The derived budget is
    therefore skipped for those; the same 1 MiB that makes
    test_fastsafetensors_budget_infeasible raise must load here.
    """
    monkeypatch.delenv("VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors = _download_gpt2(tmpdir)
        names = {
            name
            for name, _ in fastsafetensors_weights_iterator(
                safetensors, True, accumulate_resident=True
            )
        }
    assert names


def _spy_parallel_loader(monkeypatch):
    """Record the keyword arguments each ParallelLoader is built with."""
    from fastsafetensors import parallel_loader

    calls = []
    real = parallel_loader.ParallelLoader

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(parallel_loader, "ParallelLoader", spy)
    return calls


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_expert_filter_needs_per_expert_names(monkeypatch):
    """The expert filter applies only to checkpoints with per-expert names.

    Filtering gives up broadcast loading, so a checkpoint with nothing to
    filter must keep it. GPT-2 has no expert tensors: a rank that owns only
    expert 0 is handed every tensor and the loader is left as configured. A
    checkpoint that names its experts drops the weights of the others.
    """
    from safetensors.torch import save_file

    monkeypatch.delenv("VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", raising=False)
    monkeypatch.setenv("VLLM_FASTSAFETENSORS_ALL_LOCAL", "0")
    calls = _spy_parallel_loader(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors = _download_gpt2(tmpdir)
        unfiltered = {
            name
            for name, _ in fastsafetensors_weights_iterator(
                safetensors, True, local_expert_ids={0}
            )
        }
        expected = {name for name, _ in safetensors_weights_iterator(safetensors, True)}
        assert unfiltered == expected
        assert calls[-1]["tensor_filter"] is None
        assert calls[-1]["all_local"] is False

        per_expert = f"{tmpdir}/per_expert.safetensors"
        save_file(
            {
                "layers.0.mlp.experts.0.w1.weight": torch.ones(4, 4),
                "layers.0.mlp.experts.1.w1.weight": torch.ones(4, 4),
                "layers.0.attn.weight": torch.ones(4, 4),
            },
            per_expert,
        )
        filtered = {
            name
            for name, _ in fastsafetensors_weights_iterator(
                [per_expert], True, local_expert_ids={0}
            )
        }
    assert filtered == {"layers.0.mlp.experts.0.w1.weight", "layers.0.attn.weight"}
    assert calls[-1]["all_local"] is True


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_uninspected_consumer_is_not_planned(monkeypatch):
    """A consumer that was never inspected gets no derived budget.

    Reload paths build a loader and call get_all_weights without load_weights,
    so nothing established whether parameters materialize during the load. The
    loader passes None for those: no budget is derived, and an explicit budget
    still bounds staging without charging the bytes read as resident, which
    would refuse a reload of any model larger than the budget.
    """
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    loader = DefaultModelLoader(LoadConfig(load_format="fastsafetensors"))
    assert loader.params_materialize_during_load is None

    calls = _spy_parallel_loader(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        safetensors = _download_gpt2(tmpdir)

        monkeypatch.delenv("VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", raising=False)
        assert list(
            fastsafetensors_weights_iterator(
                safetensors, True, accumulate_resident=None
            )
        )
        assert "device_memory_budget" not in calls[-1]

        budget = 512 * 1024 * 1024
        monkeypatch.setenv("VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", str(budget))
        assert list(
            fastsafetensors_weights_iterator(
                safetensors, True, accumulate_resident=None
            )
        )
        assert calls[-1]["device_memory_budget"] == budget
        assert calls[-1]["accumulate_resident"] is False


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fastsafetensors requires NVIDIA/AMD GPUs",
)
def test_fastsafetensors_planned_load_requests_gds(monkeypatch):
    """A memory budget must not cost a single-rank load its GDS reads.

    The GDS copier reads sub-shard chunks, so planning does not imply buffered
    reads.
    """
    calls = _spy_parallel_loader(monkeypatch)
    monkeypatch.setenv(
        "VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", str(512 * 1024 * 1024)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _assert_matches_safetensors(tmpdir)
    assert calls[0]["nogds"] is False
