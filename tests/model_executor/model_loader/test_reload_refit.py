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
from collections import deque
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PreTrainedConfig

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
# from the layerwise reference), compile state and cache paths keyed by model
# path, and a memo that cold-start backend selection fills
ENGINE_STATE = (
    "._ckpt_",
    "._direct_reload",
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
    registered = {
        n
        for name, t in itertools.chain(model.named_parameters(), model.named_buffers())
        for n, _ in _tensors(name, t)
    }
    off_registry = [n for n in tensors if n in digests and n not in registered]
    return digests, ptrs, len(off_registry)


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
    peak stayed within a transfer batch plus twice the largest parameter
    (embeddings aside, which post-load processing doesn't transform)."""
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
    while True:
        batch = [
            (n, t.to(worker.device)) for n, t in itertools.islice(weights, BATCH_SIZE)
        ]
        if not batch:
            break
        largest_batch = max(largest_batch, sum(t.nbytes for _, t in batch))
        # one call per transfer chunk, so no call sees the whole checkpoint
        with (
            set_current_vllm_config(worker.vllm_config),
            disable_mtp_completeness_check(),
        ):
            model.load_weights(iter(batch))
        torch.accelerator.synchronize()
        del batch  # only one chunk is in flight
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


ENGINE_KWARGS = dict(
    enable_prefix_caching=False,
    max_model_len=256,
    max_num_seqs=4,
    gpu_memory_utilization=0.4,
    # autotuning off keeps kernel choice stable across engines; the fresh and
    # refit engines compile separately (their model paths differ), so
    # Inductor's reduction configs are fixed too
    kernel_config={"enable_flashinfer_autotune": False},
    compilation_config={"inductor_compile_config": {"deterministic": True}},
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
        TIMEOUT_S,
    )
    return [result for results in per_engine for result in results]


async def _last(stream):
    async for out in stream:
        pass
    return out


async def _generate(engine: AsyncLLM, case: "RefitCase") -> list:
    """Greedy tokens and top logprobs for every prompt on every DP rank,
    one request at a time."""
    dp = case.parallel.get("data_parallel_size", 1)
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
    # a bug on main: reference entries (path substrings) known to differ
    # after reload B; the case asserts everything else, then xfails
    known_bug: tuple[str, ...] = ()
    known_bug_reason: str = ""

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
        layout += ["sleep"] if self.sleep else []
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

REFIT_CASES = [
    # one GPU
    RefitCase("qwen3", "bf16", "Qwen/Qwen3-0.6B", marks=NIGHTLY),
    RefitCase("qwen3", "fp8-block", "inference-optimization/Qwen3-0.6B-FP8_BLOCK"),
    RefitCase(
        "qwen3", "w4a16", "inference-optimization/Qwen3-0.6B-W4A16-G128", marks=NIGHTLY
    ),
    # checkpoint k/v scales; no in-scope family has a small one
    RefitCase(
        "llama",
        "fp8-kv",
        "nm-testing/Llama-3.2-1B-Instruct-FP8-KV",
        engine=dict(kv_cache_dtype="fp8"),
        known_bug=(".impl.bmm1_scale", ".impl.bmm2_scale"),
        known_bug_reason="FlashInfer keeps the bmm1/bmm2 scales of the first "
        "forward after k/v scales change (vllm#65353 fixes eager and piecewise "
        "decode; FULL-graph decode still captures them)",
    ),
    RefitCase("deepseek-v3", "bf16", DEEPSEEK_V3, marks=NIGHTLY),
    RefitCase("deepseek-v3", "fp8", DEEPSEEK_V3_FP8),
    RefitCase("deepseek-v3", "nvfp4a16", f"{DEEPSEEK_V3}-NVFP4A16", marks=NIGHTLY),
    RefitCase("gpt-oss", "mxfp4", GPT_OSS),  # padded MoE
    # level 2 must keep the state post-load processing reruns from (e.g. the
    # TRT-LLM MXFP4 permute indices)
    RefitCase(
        "gpt-oss", "mxfp4", GPT_OSS, engine=dict(enable_sleep_mode=True), sleep=True
    ),
    RefitCase("nemotron-h", "bf16", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"),
    # two GPUs
    *(
        RefitCase("deepseek-v3", "fp8", DEEPSEEK_V3_FP8, parallel)
        for parallel in (TP2, TP2_EP, TP2_EP_EPLB, DP2_EP)
    ),
    RefitCase("qwen3-moe", "fp8-block", "Qwen/Qwen3-30B-A3B-FP8", TP2_EP),
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


@pytest.fixture(scope="module")
def references() -> dict:
    """Fresh-B digests and tokens, shared by cases that differ only in sleep
    mode, which doesn't change what a cold start computes."""
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

    ref_key = (
        case.model,
        case.model_b,
        repr(case.parallel),
        repr({k: v for k, v in case.engine.items() if k != "enable_sleep_mode"}),
    )
    if ref_key not in references:
        async with _engine(folder_b, case, "layerwise", monkeypatch) as fresh:
            # generate first: lazily cached attributes are then set everywhere
            fresh_tokens = await _generate(fresh, case)
            references[ref_key] = (
                [digests for digests, _, _ in await _rpc(fresh, _inspect)],
                fresh_tokens,
            )
    fresh_b, fresh_b_tokens = references[ref_key]

    result: dict[str, object] = {"unchanged_in_b": unchanged}
    async with _engine(folder_a, case, "direct", monkeypatch) as engine:
        cold_a_tokens = await _generate(engine, case)
        cold = await _rpc(engine, _inspect, True)
        cold_a = [digests for digests, _, _ in cold]
        ptrs = [p for _, p, _ in cold]
        result["walk_found_off_registry"] = [n > 0 for _, _, n in cold]
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
            result[f"reload_{name}"] = await _rpc(engine, _stream_reload, folder)
            if case.sleep:
                result[f"reload_{name}_woke"] = await asyncio.wait_for(
                    engine.wake_up(tags=["kv_cache"]), TIMEOUT_S
                )
            tokens[name] = await _generate(engine, case)
            got = await _rpc(engine, _inspect)
            differs[name] = [_diff(e, d) for e, (d, _, _) in zip(expected, got)]
            result[f"reload_{name}_moved"] = [
                _diff(p, m) for p, (_, m, _) in zip(ptrs, got)
            ]

    known = [
        {k: v for k, v in d.items() if any(s in k for s in case.known_bug)}
        for d in differs["b"]
    ]
    for name, per_rank in differs.items():
        result[f"reload_{name}_differs"] = [
            {k: v for k, v in d.items() if name == "a" or k not in kn}
            for d, kn in zip(per_rank, known)
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
    }
    # same engine, same graphs: reloading A must reproduce its cold outputs
    assert tokens["a"] == cold_a_tokens
    if any(known):
        pytest.xfail(case.known_bug_reason)  # reloaded B then decodes differently
    check_logprobs_close(
        outputs_0_lst=fresh_b_tokens,
        outputs_1_lst=tokens["b"],
        name_0="fresh B",
        name_1="reloaded B",
    )
