# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit equals fresh for direct weight reload (`VLLM_RELOAD_MODE=direct`).

An engine that cold-started on checkpoint A, with CUDA graphs captured,
reloads B and then A again the way NCCL/IPC weight transfer does (device
tensors in batches through the model's own `load_weights`, between
`start_reload` and `finish_reload`), so every reload writes new values. After
reloading B every tensor the model holds must equal a fresh engine on B, and
after reloading A its own cold start, on every rank. After each reload no
tensor may have moved, appeared or disappeared (captured graphs read the old
addresses), and no PyTorch-allocated device memory may be left held.

Unless a case names B, B is A with every tensor rolled by one along dim 0
(constant float tensors get one added), so every tensor a reload must refresh
changes; tensors left unchanged must be on a short allow-list.

"Every tensor" is the registered parameters and buffers plus the tensors
reachable from module attributes (quant methods, kernels and the modules they
hold, quant configs), and host copies of scales (`*_scale_float`). Checksums,
not logits, are compared: logits of separately started engines can differ by
kernel selection, weights cannot.
"""

import gc
import glob
import hashlib
import os
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import vllm.envs as envs
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import AsyncEngineArgs
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
from vllm.v1.engine.async_llm import AsyncLLM

from .test_reload import _fp8_reload_unsupported

BATCH_SIZE = 16  # tensors per `load_weights` call, as one transfer chunk
MAX_DEPTH = 8  # attribute hops from a registered module
# runtime state, not weights: KV/Mamba caches, EPLB load statistics
RUNTIME_STATE = ("kv_cache", "expert_load_view", "num_unpadded_tokens")


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
            if not any(s in path for s in RUNTIME_STATE):
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


def _snapshot(model: str) -> str:
    if os.path.isdir(model):
        return model
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"]
    return hf_api().snapshot_download(model, allow_patterns=patterns)


def _safetensors(folder: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    return filter_duplicate_safetensors_files(
        files, folder, "model.safetensors.index.json"
    )


def _stream_reload(worker, model_path: str) -> tuple[str, int]:
    """Reload `model_path` in device-tensor batches; return the worker's reload
    mode and the bytes of device memory still held afterwards."""
    folder = _snapshot(model_path)
    weights = safetensors_weights_iterator(_safetensors(folder), use_tqdm_on_load=False)
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


def _rolled(model: str, out: Path) -> str:
    """Write `model` with every tensor rolled by one along dim 0 to `out`;
    float tensors a roll leaves unchanged get one added."""
    folder = _snapshot(model)
    out.mkdir()
    for name in os.listdir(folder):
        src = os.path.join(folder, name)
        if os.path.isfile(src) and not name.endswith(".safetensors"):
            shutil.copy(src, out / name)
    for path in _safetensors(folder):
        with safe_open(path, "pt") as f:
            tensors = {n: f.get_tensor(n) for n in f.keys()}  # noqa: SIM118
            metadata = f.metadata()
        rolled = {n: _roll(t) for n, t in tensors.items()}
        save_file(rolled, out / os.path.basename(path), metadata=metadata)
    return str(out)


def _roll(t: torch.Tensor) -> torch.Tensor:
    b = t.roll(1, 0) if t.dim() and t.shape[0] > 1 else t
    if t.dtype in (torch.bfloat16, torch.float16, torch.float32) and torch.equal(b, t):
        b = t + 1  # constant, e.g. the norms of the DeepSeek-V3 debug checkpoints
    return b


def _unchanged(model_a: str, model_b: str) -> list[str]:
    """Checkpoint tensors equal in A and B, except per-tensor scalars and
    integer metadata (e.g. `weight_shape`), which a roll can't change."""

    def raw(t: torch.Tensor) -> torch.Tensor:
        return t.contiguous().reshape(-1).view(torch.uint8)

    b_files = _safetensors(_snapshot(model_b))
    b_index = {}
    for path in b_files:
        with safe_open(path, "pt") as f:
            b_index.update(dict.fromkeys(f.keys(), path))
    unchanged = []
    for path in _safetensors(_snapshot(model_a)):
        with safe_open(path, "pt") as f:
            for n in f.keys():  # noqa: SIM118
                a = f.get_tensor(n)
                if a.numel() == 1 or not (a.is_floating_point() or a.dim() > 1):
                    continue
                if n not in b_index:
                    continue  # the engines report it as missing
                with safe_open(b_index[n], "pt") as g:
                    if torch.equal(raw(a), raw(g.get_tensor(n))):
                        unchanged.append(n)
    return unchanged


ENGINE_KWARGS = dict(
    enable_prefix_caching=False,
    max_model_len=256,
    max_num_seqs=4,
    gpu_memory_utilization=0.4,
    # autotuning off keeps kernel choice stable across engines
    kernel_config={"enable_flashinfer_autotune": False},
)


@asynccontextmanager
async def _engine(model: str, parallel: dict):
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(model=model, **ENGINE_KWARGS, **parallel)
    )
    try:
        yield engine
    finally:
        engine.shutdown()


async def _rpc(engine: AsyncLLM, fn, *args) -> list:
    """Run `fn(worker, *args)` on every worker of every DP engine; one result
    per rank."""
    per_engine = await engine.engine_core.call_utility_all_async(
        "collective_rpc", fn, None, args, None
    )
    return [result for results in per_engine for result in results]


@dataclass(frozen=True)
class RefitCase:
    """Checkpoint A, the engine's parallel layout, and B if not derived."""

    family: str  # selects cases in CI (`-k`)
    quant: str
    model: str
    parallel: dict = field(default_factory=dict)
    model_b: str | None = None

    @property
    def num_gpus(self) -> int:
        p = self.parallel
        return p.get("tensor_parallel_size", 1) * p.get("data_parallel_size", 1)

    @property
    def id(self) -> str:
        p = self.parallel
        tp, dp = p.get("tensor_parallel_size", 1), p.get("data_parallel_size", 1)
        layout = [f"tp{tp}"] if tp > 1 or dp == 1 else []
        layout += [f"dp{dp}"] if dp > 1 else []
        layout += ["ep"] if p.get("enable_expert_parallel") else []
        layout += ["eplb"] if p.get("enable_eplb") else []
        return "-".join([self.family, self.quant, *layout])


TP2 = dict(tensor_parallel_size=2)
TP2_EP = dict(TP2, enable_expert_parallel=True)
TP2_EP_EPLB = dict(TP2_EP, enable_eplb=True)  # no rebalance runs
DP2_EP = dict(data_parallel_size=2, enable_expert_parallel=True)

DEEPSEEK_V3_FP8 = "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC"
QWEN3_MOE_FP8 = "Qwen/Qwen3-30B-A3B-FP8"

REFIT_CASES = [
    # one GPU
    RefitCase("qwen3", "fp8-block", "inference-optimization/Qwen3-0.6B-FP8_BLOCK"),
    RefitCase("qwen3", "w4a16", "inference-optimization/Qwen3-0.6B-W4A16-G128"),
    RefitCase("deepseek-v3", "fp8", DEEPSEEK_V3_FP8),
    RefitCase(
        "deepseek-v3",
        "nvfp4a16",
        "inference-optimization/DeepSeek-V3-debug-empty-NVFP4A16",
    ),
    RefitCase("nemotron-h", "bf16", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"),
    # two GPUs
    *(
        RefitCase(family, quant, model, parallel)
        for family, quant, model in (
            ("deepseek-v3", "fp8", DEEPSEEK_V3_FP8),
            ("qwen3-moe", "fp8-block", QWEN3_MOE_FP8),
        )
        for parallel in (TP2, TP2_EP, TP2_EP_EPLB, DP2_EP)
    ),
]


@pytest.fixture(scope="module")
def checkpoint_b(tmp_path_factory):
    """B for a case, and the tensors it leaves unchanged; derived B is
    written once per model."""
    cache: dict[str, tuple[str, list[str]]] = {}

    def get(case: RefitCase) -> tuple[str, list[str]]:
        if case.model not in cache:
            b = case.model_b or _rolled(
                case.model, tmp_path_factory.mktemp("rolled") / "b"
            )
            cache[case.model] = (b, _unchanged(case.model, b))
        return cache[case.model]

    return get


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA graphs and device tensors"
)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in REFIT_CASES])
@pytest.mark.asyncio
async def test_refit_equals_fresh(case: RefitCase, checkpoint_b, monkeypatch):
    if current_platform.device_count() < case.num_gpus:
        pytest.skip(f"needs {case.num_gpus} GPUs")
    if case.quant.startswith("fp8") and _fp8_reload_unsupported():
        pytest.skip("Requires FP8 support")
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model_a = case.model
    model_b, unchanged = checkpoint_b(case)

    async with _engine(model_b, case.parallel) as fresh:
        fresh_b = await _rpc(fresh, _checksums)

    result: dict[str, object] = {"unchanged_in_b": unchanged}
    async with _engine(model_a, case.parallel) as engine:
        ptrs = await _rpc(engine, _ptrs)
        cold_a = await _rpc(engine, _checksums)
        # each reload changes every tensor that differs between A and B, and
        # the second runs on what the first left behind
        steps = (("b", model_b, fresh_b), ("a", model_a, cold_a))
        for name, path, expected in steps:
            held = await _rpc(engine, _stream_reload, path)
            result[f"reload_{name}_mode_and_held_bytes"] = held
            got = await _rpc(engine, _checksums)
            result[f"reload_{name}_differs"] = list(map(_diff, expected, got))
            moved = await _rpc(engine, _ptrs)
            result[f"reload_{name}_moved"] = list(map(_diff, ptrs, moved))

    # one assert, so a failure reports everything
    ranks = len(cold_a)
    assert ranks == case.num_gpus
    assert result == {
        "unchanged_in_b": [],
        **{
            f"reload_{name}_{key}": [value] * ranks
            for name in ("b", "a")
            for key, value in (
                ("mode_and_held_bytes", ("direct", 0)),
                ("differs", {}),
                ("moved", {}),
            )
        },
    }
