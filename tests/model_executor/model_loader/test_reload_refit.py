# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit equals fresh for direct weight reload (`VLLM_RELOAD_MODE=direct`).

An engine that cold-started on checkpoint A, with CUDA graphs captured,
reloads B and then A again the way NCCL/IPC weight transfer does (device
tensors in batches through the model's own `load_weights`, between
`start_reload` and `finish_reload`), so every reload writes new values. After
reloading B every tensor the model holds must equal a fresh engine on B, and
after reloading A its own cold start. After each reload no tensor may have
moved, appeared or disappeared (captured graphs read the old addresses), and
no PyTorch-allocated device memory may be left held.

"Every tensor" is the registered parameters and buffers plus the tensors
reachable from module attributes (quant methods, kernels and the modules they
hold, quant configs), and host copies of scales (`*_scale_float`). Checksums,
not logits, are compared: logits of separately started engines can differ by
kernel selection, weights cannot. All cases run on one GPU (TP=1).
"""

import gc
import glob
import hashlib
import os
from itertools import islice

import pytest
import torch

import vllm.envs as envs
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
MAX_DEPTH = 8  # attribute hops from a registered module


def _model_state(model: torch.nn.Module) -> dict[str, object]:
    """Registered parameters and buffers, then the tensors and host scale
    copies reachable from other module attributes, keyed by name or path."""
    state: dict[str, object] = dict(model.named_parameters())
    state.update(model.named_buffers())
    registered = {id(m) for m in model.modules()}
    seen = {id(t) for t in state.values()}

    def visit(obj, path: str, depth: int) -> None:
        if depth > MAX_DEPTH or id(obj) in seen or id(obj) in registered:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if "kv_cache" not in path:  # runtime KV/Mamba state
                state[path] = obj
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]", depth + 1)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}[{k!r}]", depth + 1)
        elif hasattr(obj, "__dict__") and (
            # modules outside the tree, e.g. experts held by a MoE kernel
            isinstance(obj, torch.nn.Module)
            or not type(obj).__module__.startswith(("torch", "builtins"))
        ):
            for k, v in vars(obj).items():
                visit(v, f"{path}.{k}", depth + 1)

    for name, module in model.named_modules():
        for attr, value in vars(module).items():
            if attr in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(value, float) and attr.endswith("_scale_float"):
                state[f"{name}.{attr}"] = value  # host copy of a scale
            else:
                visit(value, f"{name}.{attr}", 0)
    return state


def _digest(v) -> str:
    if not isinstance(v, torch.Tensor):
        return repr(v)
    if v.is_meta:
        return f"meta{tuple(v.shape)}"
    data = v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    digest = hashlib.sha256(data).hexdigest()
    return f"{v.device.type}:{v.dtype}{tuple(v.shape)}:{digest}"


def _checksums(worker) -> dict[str, str]:
    state = _model_state(worker.model_runner.get_model())
    return {name: _digest(v) for name, v in state.items()}


def _ptrs(worker) -> dict[str, tuple]:
    state = _model_state(worker.model_runner.get_model())
    return {
        name: (t.data_ptr(), t.stride(), t.storage_offset())
        for name, t in state.items()
        if isinstance(t, torch.Tensor) and t.is_cuda
    }


def _stream_reload(worker, model_path: str) -> tuple[str, int]:
    """Reload `model_path` in device-tensor batches; return the worker's reload
    mode and the bytes of device memory still held afterwards."""
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
        finish_reload(model, worker.model_config, device)
    gc.collect()
    torch.accelerator.synchronize()
    return envs.VLLM_RELOAD_MODE, torch.accelerator.memory_allocated() - base


def _diff(before: dict, after: dict) -> dict:
    """Entries that differ, as (before, after); None where one side lacks it."""
    return {
        k: (before.get(k), after.get(k))
        for k in sorted(before.keys() | after.keys())
        if before.get(k) != after.get(k)
    }


# (case id, checkpoint A, checkpoint B): A and B share an architecture and
# quantization and differ in values. Add a case per model family or quant
# method here; CI steps can select families with `-k`.
REFIT_CASES = [
    (
        "fp8-block-dense-tied",
        "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
        "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
    ),
    (
        "w4a16-dense-marlin",
        "inference-optimization/Qwen3-0.6B-W4A16-G128",
        "inference-optimization/Qwen3-0.6B-debug-multiply-W4A16-G128",
    ),
    (
        "fp8-mla-moe",
        "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC",
        "inference-optimization/DeepSeek-V3-debug-multiply-FP8_DYNAMIC",
    ),
    (
        "nvfp4a16-mla-moe",
        "inference-optimization/DeepSeek-V3-debug-empty-NVFP4A16",
        "inference-optimization/DeepSeek-V3-debug-multiply-NVFP4A16",
    ),
    (
        "mamba-hybrid",
        "tiiuae/Falcon-H1-0.5B-Base",
        "tiiuae/Falcon-H1-0.5B-Instruct",
    ),
]


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA graphs and device tensors"
)
@pytest.mark.parametrize(
    "model_a,model_b", [pytest.param(a, b, id=case) for case, a, b in REFIT_CASES]
)
def test_refit_equals_fresh(vllm_runner, monkeypatch, model_a, model_b):
    if "FP8" in model_a and _fp8_reload_unsupported():
        pytest.skip("Requires FP8 support")
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    engine_kwargs = dict(
        enable_prefix_caching=False,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        # autotuning off keeps kernel choice stable across engines
        kernel_config={"enable_flashinfer_autotune": False},
    )
    with vllm_runner(model_b, **engine_kwargs) as fresh:
        fresh_b = fresh.collective_rpc(_checksums)

    with vllm_runner(model_a, **engine_kwargs) as llm:
        ptrs = llm.collective_rpc(_ptrs)
        cold_a = llm.collective_rpc(_checksums)
        # each reload changes every tensor that differs between A and B, and
        # the second runs on what the first left behind
        result = {}
        steps = (("b", model_b, fresh_b), ("a", model_a, cold_a))
        for name, path, expected in steps:
            result[f"reload_{name}_mode_and_held_bytes"] = llm.collective_rpc(
                _stream_reload, args=(path,)
            )
            got = llm.collective_rpc(_checksums)
            result[f"reload_{name}_differs"] = list(map(_diff, expected, got))
            moved = llm.collective_rpc(_ptrs)
            result[f"reload_{name}_moved"] = list(map(_diff, ptrs, moved))

    # one assert, so a failure reports everything
    workers = len(cold_a)
    assert result == {
        f"reload_{name}_{key}": [value] * workers
        for name in ("b", "a")
        for key, value in (
            ("mode_and_held_bytes", ("direct", 0)),
            ("differs", {}),
            ("moved", {}),
        )
    }
