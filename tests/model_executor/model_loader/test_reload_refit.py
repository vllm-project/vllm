# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit equals fresh for direct weight reload (`VLLM_RELOAD_MODE=direct`).

An engine that cold-started on checkpoint A, with CUDA graphs captured,
reloads B and then A through the worker's weight-update session, in
device-tensor chunks, optionally inside a level-2 sleep. After each reload,
on every rank:
- every tensor and scalar reachable from the model equals the reference
  (B: a fresh engine started in layerwise mode; A: its own cold start);
- no tensor moved; cold-start storages are kept alive, so a free and
  reallocation shows as a move;
- no allocator block appeared or disappeared, and the peak stays bounded;
- reloaded A decodes exactly as cold A, reloaded B close to fresh B.

Unless a case names B, B is A perturbed by `_perturb`. A `planned` case
reloads as a load-plan engine does, in-process: `plan_loads` at the first
update, then `load_planned` for every chunk, so the same checks hold for the
plan path; its fallbacks must match the case's allowlist.
"""

import asyncio
import enum
import gc
import glob
import hashlib
import itertools
import json
import os
import shutil
import time
import uuid
from collections import deque
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest
import regex as re
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedConfig

from tests.models.utils import check_logprobs_close
from tests.utils import wait_for_gpu_memory_to_clear
from vllm import SamplingParams
from vllm.config import set_current_vllm_config
from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.model_loader.reload import load_planned, plan_loads
from vllm.model_executor.model_loader.reload.direct import direct_reload_active
from vllm.model_executor.model_loader.weight_checksum import _tensors
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.engine.async_llm import AsyncLLM

BATCH_SIZE = 16  # tensors per `load_weights` call, as one transfer chunk
MAX_DEPTH = 8  # attribute hops from a registered module
# compared by address only: runtime state (KV/Mamba caches, EPLB statistics),
# and what differs between engines by construction (direct-mode bookkeeping,
# compile state and paths, a memo filled by cold-start backend selection)
NOT_COMPARED = (
    ".kv_cache",
    ".expert_load_view",
    ".num_unpadded_tokens_tensors",
    "._ckpt_",
    "._direct_reload",
    "name_or_path",
    "._aot_",
    ".compiled",
    ".was_aot_compile_fn_loaded_from_disk",
    "._ocp_mx_scheme",
)
SCALARS = (bool, int, float, str, type(None), enum.Enum)
# engine and model configs (below, also any PreTrainedConfig): same in both
# engines but for paths and cache keys
NOT_WALKED = ("torch", "builtins", "vllm.config", "transformers")
FLOATS = (torch.bfloat16, torch.float16, torch.float32)
CONFIG_FILES = ["*.json", "*.txt", "*.model", "*.jinja", "*.py", "*.tiktoken"]
PROMPTS = ["The capital of France is", "def fibonacci(n):"]
TIMEOUT_S = 300  # per engine call, below the per-test --timeout


def _model_state(model: torch.nn.Module) -> tuple[dict[str, object], int]:
    """Registered parameters and buffers, then the tensors and scalars
    reachable from other module attributes, keyed by name or by the shortest
    attribute path (breadth first); and how many entries are registered."""
    state: dict[str, object] = {}
    seen = {id(m) for m in model.modules()}
    for name, t in itertools.chain(model.named_parameters(), model.named_buffers()):
        state.update(_tensors(name, t))
        seen.add(id(t))
    registered = len(state)
    queue = deque(
        (value, f"{name}.{attr}", 0)
        for name, module in model.named_modules()
        for attr, value in vars(module).items()
        if attr not in ("_parameters", "_buffers", "_modules")
    )
    while queue:
        obj, path, depth = queue.popleft()
        if depth > MAX_DEPTH:
            continue
        if isinstance(obj, SCALARS):
            state[path] = obj
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            state.update(_tensors(path, obj))
            continue
        if isinstance(obj, (list, tuple)):
            children = ((f"{path}[{i}]", v) for i, v in enumerate(obj))
        elif isinstance(obj, dict):
            children = ((f"{path}[{k!r}]", v) for k, v in obj.items())
        elif (
            hasattr(obj, "__dict__")
            and not isinstance(obj, PreTrainedConfig)
            and (
                # modules outside the tree, e.g. experts held by a MoE kernel
                isinstance(obj, torch.nn.Module)
                or not type(obj).__module__.startswith(NOT_WALKED)
            )
        ):
            children = ((f"{path}.{k}", v) for k, v in vars(obj).items())
        else:
            continue
        queue.extend((v, p, depth + 1) for p, v in children)
    return state, registered


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _digest(v) -> str:
    if not isinstance(v, torch.Tensor):
        return repr(v)
    if v.is_meta:
        return f"meta{tuple(v.shape)}"
    digest = hashlib.sha256(_raw(v).cpu().numpy()).hexdigest()
    return f"{v.device.type}:{v.dtype}{tuple(v.shape)}:{digest}"


def _inspect(worker, keep_storages: bool = False) -> tuple[dict, dict, bool]:
    """Digests of the compared model state, the address of every device
    tensor, and whether the walk found a compared tensor off-registry."""
    state, registered = _model_state(worker.model_runner.get_model())
    tensors = {n: t for n, t in state.items() if isinstance(t, torch.Tensor)}
    if keep_storages:
        # a tensor freed and reallocated would otherwise get its block back
        worker._refit_storages = [
            t.untyped_storage() for t in tensors.values() if t.is_cuda
        ]
    digests = {
        n: _digest(v) for n, v in state.items() if not any(s in n for s in NOT_COMPARED)
    }
    ptrs = {
        n: (t.data_ptr(), t.stride(), t.storage_offset())
        for n, t in tensors.items()
        if t.is_cuda
    }
    off_registry = list(state)[registered:]
    return digests, ptrs, any(n in tensors and n in digests for n in off_registry)


def _safetensors(folder: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    return filter_duplicate_safetensors_files(
        files, folder, "model.safetensors.index.json"
    )


def _active_blocks() -> set[tuple[int, int]]:
    return {
        (block["address"], block["size"])
        for segment in torch.cuda.memory_snapshot()
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    }


def _metadata(folder: str) -> list[ParamMeta]:
    """What a trainer declares at init: every checkpoint tensor's name, dtype
    and shape, read from the file headers."""
    metas = []
    for path in _safetensors(folder):
        with safe_open(path, "pt") as f:
            for n in f.keys():  # noqa: SIM118
                s = f.get_slice(n)
                metas.append(
                    ParamMeta(n, _ST_DTYPES[s.get_dtype()], tuple(s.get_shape()))
                )
    return metas


_ST_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def _plan_report(plan) -> dict:
    """Per-rank M0 numbers: names and bytes by kind, fallbacks with reasons."""
    from vllm.model_executor.model_loader.reload.plan import _nbytes

    kinds: dict[str, list[int]] = {}
    for e in plan.entries.values():
        kind = "fallback" if e.writes is None else "direct" if e.boxes else "unneeded"
        n, b = kinds.setdefault(kind, [0, 0])
        kinds[kind] = [n + 1, b + _nbytes(e)]
    fallbacks = {n: e.reason for n, e in plan.entries.items() if e.writes is None}
    return {"kinds": kinds, "fallbacks": fallbacks}


def _stream_reload(worker, folder: str, planned: bool = False) -> dict:
    """Reload `folder` through the worker's weight-update session; return the
    mode seen inside it, the allocator blocks it added or freed, and whether
    its peak stayed within a chunk plus twice the largest transformed
    parameter (embeddings aren't). `planned`: as a load-plan engine."""
    weights = safetensors_weights_iterator(_safetensors(folder), use_tqdm_on_load=False)
    model = worker.model_runner.get_model()
    largest_param = max(
        p.nbytes
        for n, p in model.named_parameters()
        if "embed" not in n and "lm_head" not in n
    )
    largest_batch = 0
    gc.collect()
    torch.accelerator.synchronize()
    before = _active_blocks()
    base = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()

    worker.start_weight_update()
    mode = "direct" if direct_reload_active(model) else "layerwise"
    with set_current_vllm_config(worker.vllm_config):
        load = model.load_weights
        if planned:
            if not hasattr(worker, "_refit_plan"):
                t0 = time.perf_counter()
                worker._refit_plan = plan_loads(model, _metadata(folder))
                worker._refit_plan_seconds = time.perf_counter() - t0
            load = partial(load_planned, model, worker._refit_plan)
        while batch := [
            (n, t.to(worker.device)) for n, t in itertools.islice(weights, BATCH_SIZE)
        ]:
            largest_batch = max(largest_batch, sum(t.nbytes for _, t in batch))
            # one call per transfer chunk, so no call sees the whole checkpoint
            load(batch)
            torch.accelerator.synchronize()
            del batch  # one chunk in flight
    worker.finish_weight_update()

    peak = torch.accelerator.max_memory_allocated() - base
    gc.collect()
    torch.accelerator.synchronize()
    after = _active_blocks()
    return {
        "mode": mode,
        "new_blocks": sorted(after - before),
        "freed_blocks": sorted(before - after),
        "peak_bounded": peak <= largest_batch + 2 * largest_param,
    }


def _plan_summary(worker) -> dict:
    report = _plan_report(worker._refit_plan)
    report["bake_s"] = round(worker._refit_plan_seconds, 3)
    return report


def _diff(before: dict, after: dict) -> dict:
    """Entries that differ, as (before, after); None where one side lacks it."""
    return {
        k: (before.get(k), after.get(k))
        for k in sorted(before.keys() | after.keys())
        if before.get(k) != after.get(k)
    }


def _download(model: str) -> str:
    if os.path.isdir(model):
        return model
    folder = download_weights_from_hf(model, None, ["*.safetensors"])
    hf_api().snapshot_download(model, allow_patterns=CONFIG_FILES)
    return folder


_LAYER = re.compile(r"\.layers\.(\d+)\.")
_EXPERT = re.compile(r"\.experts\.(\d+)\.")
_EXPERT_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
)
_NOT_DECODER = ("vision", "visual", "audio", "mtp", "quantization")


def _is_pattern(v, n: int) -> bool:
    """A per-layer pattern string, e.g. Nemotron-H's `MEMEM*E...`."""
    return isinstance(v, str) and len(v) == n and len(set(v)) <= 4


def _cut_layers(d: dict, n: int, layers: int) -> None:
    """Truncate per-layer lists and patterns, and lists of layer indices, to
    `layers`; entries and indices of MTP layers (past `n`) stay, renumbered."""

    def index(i: int) -> int | None:
        return i if i < layers else i - n + layers if i >= n else None

    for k, v in d.items():
        if isinstance(v, list) and n <= len(v) <= n + 3 or _is_pattern(v, n):
            d[k] = v[:layers] + v[n:]  # MTP layers' entries stay
        elif "layer" in k and isinstance(v, list) and all(type(i) is int for i in v):
            d[k] = [j for i in v if (j := index(i)) is not None]
        elif "layer" in k and isinstance(v, str) and re.fullmatch(r"[\d,]+", v):
            kept = (index(int(i)) for i in v.split(","))
            d[k] = ",".join(str(j) for j in kept if j is not None)
        elif isinstance(v, dict) and not any(s in k for s in _NOT_DECODER):
            _cut_layers(v, n, layers)  # e.g. linear_attn_config


def _reduce_config(cfg: dict, layers: int, experts: int | None) -> int:
    """Cut the decoder's config to `layers` layers and `experts` routed
    experts in place; return the original expert count."""
    if "text_config" in cfg:
        return _reduce_config(cfg["text_config"], layers, experts)
    _cut_layers(cfg, cfg["num_hidden_layers"], layers)
    cfg["num_hidden_layers"] = layers
    num_experts = 0
    for k in _EXPERT_KEYS:
        if experts is not None and isinstance(cfg.get(k), int):
            num_experts, cfg[k] = cfg[k], experts
    if experts is not None and not num_experts:
        raise ValueError(f"no routed-expert count among {_EXPERT_KEYS}")
    return num_experts


def _not_config(folder: str, names: list[str]) -> list[str]:
    """For `copytree`: weights, caches and files a partial snapshot lacks."""
    skip = set(
        shutil.ignore_patterns("*.safetensors", "*.index.json", ".cache")(folder, names)
    )
    return [
        n for n in names if n in skip or not os.path.exists(os.path.join(folder, n))
    ]


def _reduced(model: str, layers: int, experts: int | None, out: Path) -> str:
    """A checkpoint of `model` cut to its first `layers` decoder layers and
    first `experts` routed experts, with real names, dtypes, quantization and
    values; only the shards holding kept tensors are downloaded."""
    local = os.path.isdir(model)
    src = (
        model
        if local
        else hf_api().snapshot_download(model, allow_patterns=CONFIG_FILES)
    )
    cfg = json.loads(Path(src, "config.json").read_text())
    num_experts = _reduce_config(cfg, layers, experts)

    def keep(name: str) -> bool:
        m = _LAYER.search(name)
        decoder = not any(v in name for v in _NOT_DECODER)
        if m and decoder and int(m.group(1)) >= layers:
            return False
        m = _EXPERT.search(name)
        return experts is None or not m or int(m.group(1)) < experts

    index = Path(src, "model.safetensors.index.json")
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted({f for n, f in weight_map.items() if keep(n)})
    else:
        files = ["*.safetensors"]
    if not local:
        src = hf_api().snapshot_download(model, allow_patterns=[*CONFIG_FILES, *files])
    shutil.copytree(
        src,
        out,
        ignore=_not_config,
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    tensors = {}
    paths = [os.path.join(src, f) for f in files] if index.exists() else None
    for path in paths or _safetensors(src):
        with safe_open(path, "pt") as f:
            for n in f.keys():  # noqa: SIM118
                if not keep(n):
                    continue
                sl = f.get_slice(n)
                shape = sl.get_shape()
                # stacked experts, router rows and biases: the first `experts`
                if (
                    experts
                    and shape
                    and shape[0] == num_experts
                    and not _EXPERT.search(n)
                    and re.search(r"(experts|moe|mlp|gate|router)[._]", n)
                ):
                    tensors[n] = sl[:experts].contiguous()
                else:
                    tensors[n] = f.get_tensor(n)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    return str(out)


def _counts(t: torch.Tensor) -> bool:
    """Whether a checkpoint tensor must differ between A and B; integer
    scalars and vectors (e.g. `weight_shape`) can't."""
    return t.is_floating_point() or t.dim() > 1


def _perturb(t: torch.Tensor, offsets: dict) -> torch.Tensor:
    """Roll along dim 0; scale float tensors by 3, which changes every
    maximum; give each constant float tensor an offset no other constant of
    its dtype has, so tensors equal in A differ in B."""
    b = t.roll(1, 0) if t.dim() and t.shape[0] > 1 else t
    if t.dtype in FLOATS:
        b = b * 3
        if bool((t == t.reshape(-1)[0]).all()):
            used = offsets.setdefault(t.dtype, set())
            step = itertools.count(len(used))
            offset = 0.0
            while offset == 0.0 or offset in used:
                offset = torch.tensor(1 + next(step) / 64, dtype=t.dtype).item()
            used.add(offset)
            b = b + offset
    return b


def _derive_b(folder: str, out: Path) -> list[str]:
    """Write A with every tensor perturbed to `out`; return the checkpoint
    tensors left unchanged."""
    shutil.copytree(folder, out, ignore=shutil.ignore_patterns("*.safetensors"))
    unchanged: list[str] = []
    offsets: dict = {}
    for path in _safetensors(folder):
        with safe_open(path, "pt") as f:
            a = {n: f.get_tensor(n) for n in f.keys()}  # noqa: SIM118
            metadata = f.metadata()
        b = {n: _perturb(t, offsets) for n, t in a.items()}
        unchanged += [
            n for n, t in a.items() if _counts(t) and torch.equal(_raw(t), _raw(b[n]))
        ]
        save_file(b, out / os.path.basename(path), metadata=metadata)
    return unchanged


def _unchanged(folder_a: str, folder_b: str) -> list[str]:
    """Checkpoint tensors equal in A and B, for a case that names B."""
    unchanged = []
    with ExitStack() as stack:
        files_b = [
            stack.enter_context(safe_open(p, "pt")) for p in _safetensors(folder_b)
        ]
        index_b = {n: f for f in files_b for n in f.keys()}  # noqa: SIM118
        for path in _safetensors(folder_a):
            with safe_open(path, "pt") as f:
                for n in f.keys():  # noqa: SIM118
                    a = f.get_tensor(n)
                    if n in index_b and _counts(a):
                        b = index_b[n].get_tensor(n)
                        if torch.equal(_raw(a), _raw(b)):
                            unchanged.append(n)
    return unchanged


KERNEL_CONFIG = {"enable_flashinfer_autotune": False}
COMPILATION_CONFIG = {"inductor_compile_config": {"deterministic": True}}
ENGINE_KWARGS = dict(
    enable_prefix_caching=False,
    max_model_len=256,
    max_num_seqs=4,
    gpu_memory_utilization=0.4,
    # autotuning off keeps kernel choice stable across engines; the fresh and
    # refit engines compile separately (their model paths differ), so
    # Inductor's reduction configs are fixed too
    kernel_config=KERNEL_CONFIG,
    compilation_config=COMPILATION_CONFIG,
    # the worker's weight-update session needs a transfer backend
    weight_transfer_config={"backend": "ipc"},
)


@asynccontextmanager
async def _engine(model: str, case: "RefitCase", mode: str, monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=model,
            enable_sleep_mode=case.sleep,
            **{**ENGINE_KWARGS, **case.engine, **case.parallel},
        )
    )
    try:
        yield engine
    finally:
        engine.shutdown()
        wait_for_gpu_memory_to_clear(
            devices=list(range(case.num_gpus)), threshold_ratio=0.1
        )


async def _rpc(engine: AsyncLLM, fn, *args) -> list:
    """Run `fn(worker, *args)` on every worker of every DP engine; one result
    per rank."""
    per_engine = await asyncio.wait_for(
        engine.engine_core.call_utility_all_async(
            "collective_rpc", fn, None, args, None
        ),
        TIMEOUT_S,
    )
    return [result for results in per_engine for result in results]


async def _last(stream):
    outs = [out async for out in stream]
    assert outs, "the request produced no output"
    return outs[-1]


async def _generate(engine: AsyncLLM, case: "RefitCase") -> list:
    """Greedy tokens and top logprobs for every prompt on every DP rank,
    one request at a time."""
    dp = case.dp
    params = SamplingParams(max_tokens=8, temperature=0, logprobs=5)
    outputs = []
    for rank, prompt in itertools.product(range(dp), PROMPTS):
        stream = engine.generate(
            prompt,
            params,
            request_id=uuid.uuid4().hex,
            data_parallel_rank=rank if dp > 1 else None,
        )
        out = await asyncio.wait_for(_last(stream), TIMEOUT_S)
        c = out.outputs[0]
        assert c.logprobs is not None
        logprobs = [{t: lp.logprob for t, lp in d.items()} for d in c.logprobs]
        outputs.append((list(c.token_ids), c.text, logprobs))
    return outputs


@dataclass(frozen=True)
class RefitCase:
    """Checkpoint A, the engine's parallel layout, and B if not derived."""

    family: str  # selects cases in CI (`-k`)
    quant: str
    model: str
    parallel: dict = field(default_factory=dict)
    model_b: str | None = None
    engine: dict = field(default_factory=dict)  # other engine arguments
    marks: tuple = ()
    # level-2 sleep before each reload, waking weights before and the KV
    # cache after it, as RL colocated with training does
    sleep: bool = False
    # A cut to (decoder layers, routed experts) at test time, see `_reduced`
    reduce: tuple[int, int | None] | None = None
    # reload through a load plan; fallback names must match `fallbacks`
    planned: bool = False
    fallbacks: tuple[str, ...] = ()  # regexes

    @property
    def tp(self) -> int:
        return self.parallel.get("tensor_parallel_size", 1)

    @property
    def dp(self) -> int:
        return self.parallel.get("data_parallel_size", 1)

    @property
    def num_gpus(self) -> int:
        return self.tp * self.dp

    @property
    def reference_key(self) -> tuple:
        """Cases that differ only in sleep share a fresh-B reference."""
        return (
            self.model,
            self.model_b,
            self.reduce,
            repr(self.parallel),
            repr(self.engine),
        )

    @property
    def id(self) -> str:
        p = self.parallel
        layout = [f"tp{self.tp}"] if self.tp > 1 or self.dp == 1 else []
        layout += [f"dp{self.dp}"] if self.dp > 1 else []
        layout += ["ep"] if p.get("enable_expert_parallel") else []
        layout += ["eplb"] if p.get("enable_eplb") else []
        layout += ["sleep"] if self.sleep else []
        layout += ["planned"] if self.planned else []
        return "-".join([self.family, self.quant, *layout])


TP2 = dict(tensor_parallel_size=2)
TP2_EP = dict(TP2, enable_expert_parallel=True)
# replicas, but no rebalance runs
TP2_EP_EPLB = dict(TP2_EP, enable_eplb=True, eplb_config={"num_redundant_experts": 2})
DP2_EP = dict(data_parallel_size=2, enable_expert_parallel=True)

DEEPSEEK_V3 = "inference-optimization/DeepSeek-V3-debug-empty"
DEEPSEEK_V3_FP8 = f"{DEEPSEEK_V3}-FP8_DYNAMIC"

NIGHTLY = (pytest.mark.slow_test,)  # CI runs these in the nightly step only
GPT_OSS = "openai/gpt-oss-20b"
# pinned: the only TRT-LLM MXFP4 reload coverage, not left to the oracle
TRTLLM_MOE = dict(kernel_config={**KERNEL_CONFIG, "moe_backend": "flashinfer_trtllm"})

QWEN3_MOE = "Qwen/Qwen3-30B-A3B"
PLAN_QWEN3_MOE = dict(planned=True, reduce=(2, 16))  # top-8 routing
PLAN_GPT_OSS = dict(planned=True, reduce=(2, 8))  # sliding and full attention

REFIT_CASES = [
    # one GPU
    RefitCase("qwen3", "bf16", "Qwen/Qwen3-0.6B", marks=NIGHTLY),
    RefitCase("qwen3", "fp8-block", "inference-optimization/Qwen3-0.6B-FP8_BLOCK"),
    RefitCase(
        "qwen3", "w4a16", "inference-optimization/Qwen3-0.6B-W4A16-G128", marks=NIGHTLY
    ),
    RefitCase("deepseek-v3", "bf16", DEEPSEEK_V3, marks=NIGHTLY),
    RefitCase("deepseek-v3", "fp8", DEEPSEEK_V3_FP8),
    RefitCase("deepseek-v3", "nvfp4a16", f"{DEEPSEEK_V3}-NVFP4A16", marks=NIGHTLY),
    RefitCase("gpt-oss", "mxfp4", GPT_OSS, engine=TRTLLM_MOE),  # padded MoE
    # level 2 must keep the state post-load processing reruns from (e.g. the
    # TRT-LLM MXFP4 permute indices)
    RefitCase("gpt-oss", "mxfp4", GPT_OSS, engine=TRTLLM_MOE, sleep=True),
    RefitCase("nemotron-h", "bf16", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"),
    # two GPUs
    *(
        RefitCase("deepseek-v3", "fp8", DEEPSEEK_V3_FP8, parallel)
        for parallel in (TP2, TP2_EP, TP2_EP_EPLB, DP2_EP)
    ),
    RefitCase("qwen3-moe", "fp8-block", "Qwen/Qwen3-30B-A3B-FP8", TP2_EP),
    # load plan (one case per RL family), on checkpoints cut at test time
    *(
        RefitCase(family, quant, model, parallel, engine=engine, **plan)
        for family, quant, model, engine, plan in (
            ("qwen3-moe", "bf16", QWEN3_MOE, {}, dict(PLAN_QWEN3_MOE, marks=NIGHTLY)),
            ("qwen3-moe", "fp8-block", f"{QWEN3_MOE}-FP8", {}, PLAN_QWEN3_MOE),
            ("gpt-oss", "mxfp4", GPT_OSS, TRTLLM_MOE, PLAN_GPT_OSS),
        )
        for parallel in ({}, TP2_EP)
    ),
    # Gated DeltaNet and full attention, FP8
    RefitCase(
        "qwen3.5-moe",
        "fp8",
        "Qwen/Qwen3.5-35B-A3B-FP8",
        engine={"limit_mm_per_prompt": {"image": 0, "video": 0}},
        marks=NIGHTLY,
        planned=True,
        reduce=(4, 32),
        fallbacks=(r".*\.linear_attn\.conv1d\.weight",),  # a `view` the plan rejects
    ),
]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """A's and B's local folders for a case, and the checkpoint tensors equal
    in both; derived B is written once per model."""
    cache: dict[tuple, tuple[str, str, list[str]]] = {}

    def get(case: RefitCase) -> tuple[str, str, list[str]]:
        key = (case.model, case.model_b, case.reduce)
        if key not in cache:
            if case.reduce is None:
                a = _download(case.model)
            else:
                a = _reduced(
                    case.model, *case.reduce, tmp_path_factory.mktemp("reduced") / "a"
                )
            if case.model_b is None:
                b = str(tmp_path_factory.mktemp("derived") / "b")
                unchanged = _derive_b(a, Path(b))
            else:
                b = _download(case.model_b)
                unchanged = _unchanged(a, b)
            cache[key] = (a, b, unchanged)
        return cache[key]

    return get


@pytest.fixture(scope="module")
def references() -> dict:
    """Fresh-B digests and tokens by `RefitCase.reference_key`."""
    return {}


def _gpu_marks(num_gpus: int) -> list:
    return [pytest.mark.distributed(num_gpus=num_gpus)] if num_gpus > 1 else []


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA graphs and device tensors"
)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=c.id, marks=[*c.marks, *_gpu_marks(c.num_gpus)])
        for c in REFIT_CASES
    ],
)
@pytest.mark.asyncio
async def test_refit_equals_fresh(
    case: RefitCase, checkpoints, references, monkeypatch
):
    if current_platform.device_count() < case.num_gpus:
        message = f"needs {case.num_gpus} GPUs"
        if os.environ.get("BUILDKITE"):
            pytest.fail(message)  # a CI step with too few GPUs must not pass empty
        pytest.skip(message)
    if case.quant.startswith("fp8") and not current_platform.supports_fp8():
        pytest.skip("Requires FP8 support")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    folder_a, folder_b, unchanged = checkpoints(case)

    if case.reference_key not in references:
        async with _engine(folder_b, case, "layerwise", monkeypatch) as fresh:
            # generate first: lazily cached attributes are then set everywhere
            fresh_tokens = await _generate(fresh, case)
            fresh_digests, _, _ = zip(*await _rpc(fresh, _inspect))
            references[case.reference_key] = (fresh_digests, fresh_tokens)
    fresh_b, fresh_b_tokens = references[case.reference_key]

    result: dict[str, object] = {"unchanged_in_b": unchanged}
    # the default mode: direct with weight updates
    async with _engine(folder_a, case, "auto", monkeypatch) as engine:
        cold_a_tokens = await _generate(engine, case)
        cold_a, ptrs, walked = zip(*await _rpc(engine, _inspect, True))
        result["walk_found_off_registry"] = list(walked)
        # each reload changes every tensor that differs between A and B, and
        # the second runs on what the first left behind
        tokens: dict[str, list] = {}
        differs: dict[str, list[dict]] = {}
        steps = (("b", folder_b, fresh_b), ("a", folder_a, cold_a))
        for name, folder, expected in steps:
            if case.sleep:
                await asyncio.wait_for(engine.sleep(level=2), TIMEOUT_S)
                result[f"reload_{name}_slept"] = await engine.is_sleeping()
                await asyncio.wait_for(engine.wake_up(tags=["weights"]), TIMEOUT_S)
            result[f"reload_{name}"] = await _rpc(
                engine, _stream_reload, folder, case.planned
            )
            if case.sleep:
                result[f"reload_{name}_woke"] = await asyncio.wait_for(
                    engine.wake_up(tags=["kv_cache"]), TIMEOUT_S
                )
            tokens[name] = await _generate(engine, case)
            digests, moved, _ = zip(*await _rpc(engine, _inspect))
            differs[name] = list(map(_diff, expected, digests))
            result[f"reload_{name}_moved"] = list(map(_diff, ptrs, moved))

        if case.planned:
            plans = await _rpc(engine, _plan_summary)
    for name, per_rank in differs.items():
        result[f"reload_{name}_differs"] = per_rank
    if case.planned:
        for rank, report in enumerate(plans):
            print(f"M0 {case.id} rank {rank}: {json.dumps(report, default=str)}")
        allowed = re.compile("|".join(case.fallbacks) or "(?!)")
        result["unexpected_fallbacks"] = [
            {n: r for n, r in p["fallbacks"].items() if not allowed.fullmatch(n)}
            for p in plans
        ]
    # one assert, so a failure reports everything
    ranks = case.num_gpus
    reload = dict(mode="direct", new_blocks=[], freed_blocks=[], peak_bounded=True)
    sleep = {"_slept": True, "_woke": True} if case.sleep else {}
    assert result == {
        "unchanged_in_b": [],
        "walk_found_off_registry": [True] * ranks,
        **{
            f"reload_{name}{key}": [value] * ranks
            for name in ("b", "a")
            for key, value in (("", reload), ("_differs", {}), ("_moved", {}))
        },
        **{f"reload_{name}{key}": v for name in ("b", "a") for key, v in sleep.items()},
        **({"unexpected_fallbacks": [{}] * ranks} if case.planned else {}),
    }
    # same engine, same graphs: reloading A must reproduce its cold outputs
    assert tokens["a"] == cold_a_tokens
    check_logprobs_close(
        outputs_0_lst=fresh_b_tokens,
        outputs_1_lst=tokens["b"],
        name_0="fresh B",
        name_1="reloaded B",
    )
