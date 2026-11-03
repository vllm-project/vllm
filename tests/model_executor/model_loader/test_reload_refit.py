# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit equals fresh: an engine that loaded checkpoint A, with CUDA graphs
captured, reloads B the way NCCL/IPC weight transfer does (device tensors in
batches through the model's own `load_weights`, between `start_reload` and
`finish_reload`). Afterwards every tensor the model holds must equal a fresh
load of B and none may have moved, appeared or disappeared: captured graphs
read the old addresses.

"Every tensor" is the registered parameters and buffers plus the device
tensors reachable from module attributes (quant methods, kernels, quant
configs). Checksums, not logits, are compared: logits of separately started
engines can differ by kernel selection, weights cannot.
"""

import gc
import glob
import hashlib
import os
from itertools import islice

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api

from .test_reload import _fp8_reload_unsupported

BATCH_SIZE = 16  # tensors per `load_weights` call, as one transfer chunk


def _model_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Registered parameters and buffers, then the device tensors reachable
    from other module attributes, keyed by name or attribute path."""
    tensors = dict(model.named_parameters())
    tensors.update(model.named_buffers())
    seen = {id(t) for t in tensors.values()}

    def visit(obj, path: str, depth: int) -> None:
        if depth > 5 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and "kv_cache" not in path:  # runtime KV/Mamba state
                tensors[path] = obj
        elif isinstance(obj, torch.nn.Module):
            return  # reached through `named_modules`
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]", depth + 1)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}[{k!r}]", depth + 1)
        elif hasattr(obj, "__dict__") and not type(obj).__module__.startswith(
            ("torch", "builtins")
        ):
            for k, v in vars(obj).items():
                visit(v, f"{path}.{k}", depth + 1)

    for name, module in model.named_modules():
        for attr, value in vars(module).items():
            if attr not in ("_parameters", "_buffers", "_modules"):
                visit(value, f"{name}.{attr}", 0)
    return tensors


def _digest(t: torch.Tensor) -> str:
    if t.is_meta:
        return f"meta{tuple(t.shape)}"
    data = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return f"{t.dtype}{tuple(t.shape)}:{hashlib.sha256(data.tobytes()).hexdigest()}"


def _checksums(worker) -> dict[str, str]:
    torch.accelerator.synchronize()
    tensors = _model_tensors(worker.model_runner.get_model())
    return {name: _digest(t) for name, t in tensors.items()}


def _ptrs(worker) -> dict[str, int]:
    tensors = _model_tensors(worker.model_runner.get_model())
    return {name: t.data_ptr() for name, t in tensors.items() if t.is_cuda}


def _stream_reload(worker, model_path: str) -> int:
    """Reload `model_path` in device-tensor batches; return the bytes of device
    memory still held afterwards."""
    folder = hf_api().snapshot_download(
        model_path, allow_patterns=["*.safetensors", "*.json"]
    )
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    files = filter_duplicate_safetensors_files(
        files, folder, "model.safetensors.index.json"
    )
    weights = safetensors_weights_iterator(files, use_tqdm_on_load=False)
    model = worker.model_runner.get_model()
    device = worker.device

    gc.collect()
    torch.accelerator.synchronize()
    base = torch.accelerator.memory_allocated()
    with set_current_vllm_config(worker.vllm_config):
        start_reload(model)
        while batch := [(n, t.to(device)) for n, t in islice(weights, BATCH_SIZE)]:
            # one call per transfer chunk, so no call sees the whole checkpoint
            with disable_mtp_completeness_check():
                model.load_weights(iter(batch))
            del batch
        finish_reload(model, worker.model_config, device)
    gc.collect()
    torch.accelerator.synchronize()
    return torch.accelerator.memory_allocated() - base


def _diff(before: dict, after: dict) -> dict:
    """Entries that differ, as (before, after); None where one side lacks it."""
    return {
        k: (before.get(k), after.get(k))
        for k in sorted(before.keys() | after.keys())
        if before.get(k) != after.get(k)
    }


MODELS = {
    "bf16-dense-tied": (
        "Qwen/Qwen3-0.6B",
        "inference-optimization/Qwen3-0.6B-debug-multiply",
        {},
    ),
    "fp8-block-dense": (
        "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
        "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
        {},
    ),
    "mla-moe": (
        "inference-optimization/DeepSeek-V3-debug-empty",
        "inference-optimization/DeepSeek-V3-debug-multiply",
        {},
    ),
    "mamba-hybrid": (
        "tiiuae/Falcon-H1-0.5B-Base",
        "tiiuae/Falcon-H1-0.5B-Instruct",
        {},
    ),
    "online-fp8-moe": (
        "inference-optimization/DeepSeek-V3-debug-empty",
        "inference-optimization/DeepSeek-V3-debug-multiply",
        {"quantization": "fp8_per_tensor"},
    ),
}

CASES = [
    pytest.param(mode, *MODELS[case], id=f"{case}-{mode}")
    for case in MODELS
    for mode in ("layerwise", "direct")
    # direct reload refuses online quantization at engine construction
    if not (mode == "direct" and case.startswith("online"))
]


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA graphs and device tensors"
)
@pytest.mark.parametrize("mode,model_a,model_b,kwargs", CASES)
def test_refit_equals_fresh(vllm_runner, monkeypatch, mode, model_a, model_b, kwargs):
    if ("FP8" in model_a or "quantization" in kwargs) and _fp8_reload_unsupported():
        pytest.skip("Requires FP8 support")
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    engine_kwargs = dict(
        enable_prefix_caching=False,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        # autotuning off keeps kernel choice stable across engines
        kernel_config={"enable_flashinfer_autotune": False},
        **kwargs,
    )
    with vllm_runner(model_b, **engine_kwargs) as fresh:
        expected = fresh.collective_rpc(_checksums)

    with vllm_runner(model_a, **engine_kwargs) as llm:
        ptrs = llm.collective_rpc(_ptrs)
        for _ in range(2):  # the second reload runs on already-reloaded kernels
            held = llm.collective_rpc(_stream_reload, args=(model_b,))
        moved = [_diff(b, a) for b, a in zip(ptrs, llm.collective_rpc(_ptrs))]
        got = llm.collective_rpc(_checksums)

    # one assert, so a failure reports all three
    result = {
        "differ_from_fresh": [_diff(e, g) for e, g in zip(expected, got)],
        "moved": moved,
        "held_bytes": held,
    }
    ranks = len(got)
    assert result == {
        "differ_from_fresh": [{}] * ranks,
        "moved": [{}] * ranks,
        "held_bytes": [0] * ranks,
    }
