# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit equals fresh for direct weight reload (`VLLM_RELOAD_MODE=direct`).

An engine that cold-started on checkpoint A, with CUDA graphs captured,
reloads B and then A again through the worker's weight-update session
(`start_weight_update`, the model's own `load_weights` in device-tensor
batches, `finish_weight_update`), so every reload writes new values. After
each reload, on every rank:
- every tensor and host scalar the model holds equals a fresh engine on that
  checkpoint (B: a fresh engine started in layerwise mode, so the reference
  doesn't share direct-mode cold-start code; A: its own cold start);
- no tensor moved (captured graphs read the old addresses); the cold-start
  storages are kept alive, so a free and reallocation shows as a move;
- no allocator block appeared or disappeared, and the peak stays bounded;
- greedy logprobs match the reference engine, so values frozen into captured
  graphs or kept where the walk can't see show up.

Unless a case names B, B is A with every tensor rolled by one along dim 0 and
float tensors also scaled by 3 (constant ones get a distinct offset), so
per-tensor and per-row maxima change too; checkpoint tensors left unchanged
must be on a short allow-list.

"Every tensor" is the registered parameters and buffers plus every tensor
and scalar reachable from module attributes (quant methods, kernels and the
modules they hold, quant configs), breadth first up to MAX_DEPTH hops.
"""

import asyncio
import enum
import gc
import glob
import hashlib
import itertools
import os
import shutil
import uuid
import zlib
from collections import deque
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.models.utils import check_logprobs_close
from tests.utils import wait_for_gpu_memory_to_clear
from vllm import SamplingParams
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
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
# runtime state, not weights: compared by address only
RUNTIME_STATE = (".kv_cache", ".expert_load_view", ".num_unpadded_tokens_tensors")
# differs between engines by construction: direct-mode bookkeeping (absent
# from the layerwise reference), compile-cache paths keyed by model path, and
# a memo that cold-start backend selection fills on the first quant config
ENGINE_STATE = ("._ckpt_", "._direct_reload", "._aot_", "._ocp_mx_scheme")
SCALARS = (bool, int, float, str, type(None), enum.Enum)
# engine and model configs: same in both engines but for paths and cache keys
NOT_WALKED = ("torch", "builtins", "vllm.config", "transformers")
FLOATS = (torch.bfloat16, torch.float16, torch.float32)
CONFIG_FILES = ["*.json", "*.txt", "*.model", "*.jinja", "*.py", "*.tiktoken"]
PROMPTS = ["The capital of France is", "def fibonacci(n):"]
RPC_TIMEOUT_S = 900


def _model_state(model: torch.nn.Module) -> dict[str, object]:
    """Registered parameters and buffers, then the tensors and scalars
    reachable from other module attributes, keyed by name or by the shortest
    attribute path (breadth first)."""
    state: dict[str, object] = {}
    seen = {id(m) for m in model.modules()}
    for name, t in itertools.chain(model.named_parameters(), model.named_buffers()):
        state.update(_tensors(name, t))
        seen.add(id(t))
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
        elif hasattr(obj, "__dict__") and (
            # modules outside the tree, e.g. experts held by a MoE kernel
            isinstance(obj, torch.nn.Module)
            or not type(obj).__module__.startswith(NOT_WALKED)
        ):
            children = ((f"{path}.{k}", v) for k, v in vars(obj).items())
        else:
            continue
        queue.extend((v, p, depth + 1) for p, v in children)
    return state


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _digest(v) -> str:
    if not isinstance(v, torch.Tensor):
        return repr(v)
    if v.is_meta:
        return f"meta{tuple(v.shape)}"
    digest = hashlib.sha256(_raw(v).cpu().numpy()).hexdigest()
    return f"{v.device.type}:{v.dtype}{tuple(v.shape)}:{digest}"


def _inspect(worker, keep_storages: bool = False) -> tuple[dict, dict, int]:
    """Digests of the model state (runtime state excluded), the address of
    every device tensor, and how many tensors the walk found off-registry."""
    model = worker.model_runner.get_model()
    state = _model_state(model)
    tensors = {n: t for n, t in state.items() if isinstance(t, torch.Tensor)}
    if keep_storages:
        # a tensor freed and reallocated would otherwise get its block back
        worker._refit_storages = [
            t.untyped_storage() for t in tensors.values() if t.is_cuda
        ]
    digests = {
        n: _digest(v)
        for n, v in state.items()
        if not any(s in n for s in RUNTIME_STATE + ENGINE_STATE)
    }
    ptrs = {
        n: (t.data_ptr(), t.stride(), t.storage_offset())
        for n, t in tensors.items()
        if t.is_cuda
    }
    registered = sum(1 for _ in itertools.chain(model.parameters(), model.buffers()))
    return digests, ptrs, len(tensors) - registered


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


def _stream_reload(worker, folder: str) -> dict:
    """Reload the checkpoint in `folder` through the worker's weight-update
    session, in device-tensor batches. Returns the reload mode seen inside the
    session, the allocator blocks it left behind or freed, and whether its
    peak stayed within a transfer batch plus twice the largest parameter."""
    weights = safetensors_weights_iterator(_safetensors(folder), use_tqdm_on_load=False)
    model = worker.model_runner.get_model()
    largest_param = max(p.nbytes for p in model.parameters())
    largest_batch = 0
    gc.collect()
    torch.accelerator.synchronize()
    before = _active_blocks()
    base = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()

    worker.start_weight_update()
    mode = "direct" if direct_reload_active(model) else "layerwise"
    while batch := [
        (n, t.to(worker.device)) for n, t in itertools.islice(weights, BATCH_SIZE)
    ]:
        largest_batch = max(largest_batch, sum(t.nbytes for _, t in batch))
        # one call per transfer chunk, so no call sees the whole checkpoint
        with (
            set_current_vllm_config(worker.vllm_config),
            disable_mtp_completeness_check(),
        ):
            model.load_weights(iter(batch))
        torch.accelerator.synchronize()
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


def _counts(t: torch.Tensor) -> bool:
    """Whether a checkpoint tensor must differ between A and B; integer
    scalars and vectors (e.g. `weight_shape`) can't."""
    return t.is_floating_point() or t.dim() > 1


def _perturb(name: str, t: torch.Tensor) -> torch.Tensor:
    b = t.roll(1, 0) if t.dim() and t.shape[0] > 1 else t
    if t.dtype in FLOATS:
        # a factor that isn't a power of two changes every maximum; a small
        # one keeps the model's activations in range
        b = b * 1.25
        if bool((t == t.reshape(-1)[0]).all()):
            # constant: an offset of its own keeps it distinct from the others
            b = b + 1 + zlib.crc32(name.encode()) % 1000 / 1000
    return b


def _derive_b(folder: str, out: Path) -> list[str]:
    """Write A with every tensor perturbed to `out`; return the checkpoint
    tensors left unchanged."""
    shutil.copytree(folder, out, ignore=shutil.ignore_patterns("*.safetensors"))
    unchanged = []
    for path in _safetensors(folder):
        with safe_open(path, "pt") as f:
            a = {n: f.get_tensor(n) for n in f.keys()}  # noqa: SIM118
            metadata = f.metadata()
        b = {n: _perturb(n, t) for n, t in a.items()}
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


ENGINE_KWARGS = dict(
    enable_prefix_caching=False,
    max_model_len=256,
    max_num_seqs=4,
    gpu_memory_utilization=0.4,
    # autotuning off keeps kernel choice stable across engines
    kernel_config={"enable_flashinfer_autotune": False},
    # the worker's weight-update session needs a transfer backend
    weight_transfer_config={"backend": "ipc"},
)


@asynccontextmanager
async def _engine(model: str, case: "RefitCase", mode: str, monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(model=model, **ENGINE_KWARGS, **case.engine, **case.parallel)
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
        RPC_TIMEOUT_S,
    )
    return [result for results in per_engine for result in results]


async def _generate(engine: AsyncLLM, case: "RefitCase") -> list:
    """Greedy tokens and top logprobs for every prompt on every DP rank,
    one request at a time."""
    dp = case.parallel.get("data_parallel_size", 1)
    params = SamplingParams(max_tokens=8, temperature=0, logprobs=5)
    outputs = []
    for rank, prompt in itertools.product(range(dp), PROMPTS):
        async for out in engine.generate(
            prompt,
            params,
            request_id=uuid.uuid4().hex,
            data_parallel_rank=rank if dp > 1 else None,
        ):
            pass
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
# replicas, but no rebalance runs
TP2_EP_EPLB = dict(TP2_EP, enable_eplb=True, eplb_config={"num_redundant_experts": 2})
DP2_EP = dict(data_parallel_size=2, enable_expert_parallel=True)

DEEPSEEK_V3 = "inference-optimization/DeepSeek-V3-debug-empty"
DEEPSEEK_V3_FP8 = f"{DEEPSEEK_V3}-FP8_DYNAMIC"
QWEN3_MOE_FP8 = "Qwen/Qwen3-30B-A3B-FP8"

REFIT_CASES = [
    # one GPU
    RefitCase("qwen3", "bf16", "Qwen/Qwen3-0.6B"),
    RefitCase("qwen3", "fp8-block", "inference-optimization/Qwen3-0.6B-FP8_BLOCK"),
    RefitCase("qwen3", "w4a16", "inference-optimization/Qwen3-0.6B-W4A16-G128"),
    # checkpoint k/v scales; no in-scope family has a small one
    RefitCase(
        "llama",
        "fp8-kv",
        "nm-testing/Llama-3.2-1B-Instruct-FP8-KV",
        engine=dict(kv_cache_dtype="fp8"),
        marks=(
            pytest.mark.xfail(
                strict=True,
                reason="FlashInfer keeps the bmm1/bmm2 scales of the first "
                "forward after k/v scales change (vllm#61287 fixes eager and "
                "piecewise decode; FULL-graph decode still captures them)",
            ),
        ),
    ),
    RefitCase("deepseek-v3", "bf16", DEEPSEEK_V3),
    RefitCase("deepseek-v3", "fp8", DEEPSEEK_V3_FP8),
    RefitCase("deepseek-v3", "nvfp4a16", f"{DEEPSEEK_V3}-NVFP4A16"),
    RefitCase("gpt-oss", "mxfp4", "openai/gpt-oss-20b"),  # padded MoE
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
def checkpoints(tmp_path_factory):
    """A's and B's local folders for a case, and the checkpoint tensors equal
    in both; derived B is written once per model."""
    cache: dict[tuple[str, str | None], tuple[str, str, list[str]]] = {}

    def get(case: RefitCase) -> tuple[str, str, list[str]]:
        key = (case.model, case.model_b)
        if key not in cache:
            a = _download(case.model)
            if case.model_b is None:
                b = str(tmp_path_factory.mktemp("derived") / "b")
                unchanged = _derive_b(a, Path(b))
            else:
                b = _download(case.model_b)
                unchanged = _unchanged(a, b)
            cache[key] = (a, b, unchanged)
        return cache[key]

    return get


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA graphs and device tensors"
)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            id=c.id,
            marks=[
                *c.marks,
                *(
                    [pytest.mark.distributed(num_gpus=c.num_gpus)]
                    if c.num_gpus > 1
                    else []
                ),
            ],
        )
        for c in REFIT_CASES
    ],
)
@pytest.mark.asyncio
async def test_refit_equals_fresh(case: RefitCase, checkpoints, monkeypatch):
    if current_platform.device_count() < case.num_gpus:
        message = f"needs {case.num_gpus} GPUs"
        if os.environ.get("BUILDKITE"):
            pytest.fail(message)  # a CI step with too few GPUs must not pass empty
        pytest.skip(message)
    if case.quant.startswith("fp8") and not current_platform.supports_fp8():
        pytest.skip("Requires FP8 support")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    folder_a, folder_b, unchanged = checkpoints(case)

    async with _engine(folder_b, case, "layerwise", monkeypatch) as fresh:
        # generate first: lazily cached attributes are then set everywhere
        fresh_b_tokens = await _generate(fresh, case)
        fresh_b = [digests for digests, _, _ in await _rpc(fresh, _inspect)]

    result: dict[str, object] = {"unchanged_in_b": unchanged}
    async with _engine(folder_a, case, "direct", monkeypatch) as engine:
        cold_a_tokens = await _generate(engine, case)
        cold = await _rpc(engine, _inspect, True)
        cold_a = [digests for digests, _, _ in cold]
        ptrs = [p for _, p, _ in cold]
        result["walk_found_off_registry"] = [n > 0 for _, _, n in cold]
        # each reload changes every tensor that differs between A and B, and
        # the second runs on what the first left behind
        tokens = {}
        steps = (("b", folder_b, fresh_b), ("a", folder_a, cold_a))
        for name, folder, expected in steps:
            result[f"reload_{name}"] = await _rpc(engine, _stream_reload, folder)
            tokens[name] = await _generate(engine, case)
            got = await _rpc(engine, _inspect)
            result[f"reload_{name}_differs"] = [
                _diff(e, d) for e, (d, _, _) in zip(expected, got)
            ]
            result[f"reload_{name}_moved"] = [
                _diff(p, m) for p, (_, m, _) in zip(ptrs, got)
            ]

    # one assert, so a failure reports everything
    ranks = case.num_gpus
    reload = dict(mode="direct", new_blocks=[], freed_blocks=[], peak_bounded=True)
    assert result == {
        "unchanged_in_b": [],
        "walk_found_off_registry": [True] * ranks,
        **{
            f"reload_{name}{key}": [value] * ranks
            for name in ("b", "a")
            for key, value in (("", reload), ("_differs", {}), ("_moved", {}))
        },
    }
    mismatches = []
    for reference, name in ((fresh_b_tokens, "B"), (cold_a_tokens, "A")):
        try:
            check_logprobs_close(
                outputs_0_lst=reference,
                outputs_1_lst=tokens[name.lower()],
                name_0=f"fresh {name}",
                name_1=f"reloaded {name}",
            )
        except AssertionError as e:
            mismatches.append(str(e))
    assert not mismatches, "\n".join(mismatches)
