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

    384 MiB therefore sits above the floor and below the shard, forcing the
    planner to split the shard -- it loads as 4 chunks. Below 294 MiB the plan
    is infeasible and the loader raises rather than falling back, which
    test_fastsafetensors_budget_infeasible covers.
    """
    monkeypatch.setenv(
        "VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET", str(384 * 1024 * 1024)
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
