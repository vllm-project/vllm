# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot of post-processed model state.

The first start saves every CUDA tensor and Python scalar reachable from the
model modules after process_weights_after_loading. Later starts build the model
with dummy weights, which gives the same structure, and copy the saved bytes
back, skipping checkpoint reads and weight conversion.
"""

import json
import mmap
import os
import time
from collections.abc import Iterator
from pathlib import Path

import regex as re
import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 4096
_CHUNK = 256 << 20
_MAGIC = "vllm-weight-snapshot-v4"
_SCALARS = (bool, int, float)
_DEPTH = 6


def snapshot_path(root: str, config_hash: str, rank: int) -> Path:
    return Path(root) / f"{config_hash}-rank{rank}.bin"


def _checkpoint_id(path) -> object:
    """Location-independent identity of a checkpoint: HF repo@commit when the path
    is an HF cache snapshot, plus name/size of weight files and config.json bytes."""
    import hashlib

    if path is None or not os.path.isdir(str(path)):
        return path
    p = Path(path).resolve()
    ident = p.name
    if p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
        ident = (
            p.parent.parent.name[len("models--") :].replace("--", "/") + "@" + p.name
        )
    files = sorted(
        (f.name, f.stat().st_size)
        for f in p.iterdir()
        if f.suffix in (".safetensors", ".bin", ".pt", ".gguf")
    )
    cfg = p / "config.json"
    digest = hashlib.sha256(cfg.read_bytes()).hexdigest()[:16] if cfg.exists() else None
    return [ident, files, digest]


def weights_key(vllm_config) -> str:
    """Hash of the settings that shape post-processed weights.

    compute_hash() is not stable across processes for some configs (seen with
    dflash), so the key is built from model, quantization and parallel layout.
    It is independent of where the checkpoint lives, so snapshots can be copied
    between servers with the same GPU and software stack."""
    import hashlib

    import vllm

    m, p, s = vllm_config.model_config, vllm_config.parallel_config, None
    spec = vllm_config.speculative_config
    if spec is not None:
        s = [
            getattr(spec, "method", None),
            getattr(spec, "num_speculative_tokens", None),
        ]
        draft = getattr(spec, "draft_model_config", None)
        s.append(_checkpoint_id(draft.model) if draft is not None else spec.model)
    overrides = (
        m.hf_overrides if not callable(m.hf_overrides) else repr(type(m.hf_overrides))
    )
    parts = [
        _checkpoint_id(m.model),
        m.revision,
        str(m.dtype),
        m.quantization,
        overrides,
        getattr(vllm_config.cache_config, "cache_dtype", None),
        p.tensor_parallel_size,
        p.pipeline_parallel_size,
        getattr(p, "enable_expert_parallel", None),
        s,
        getattr(m.multimodal_config, "language_model_only", None),
        torch.__version__,
        torch.version.cuda,
        vllm.__version__,
        torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        _MAGIC,
    ]
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:10]


def _round_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _walk(value, key: str, depth: int, seen: set[int]) -> Iterator[tuple[str, object]]:
    """Yield (key, tensor or scalar) found in an attribute value.

    Non-module objects are followed several levels deep (e.g. MoE quant descriptors
    hold per-expert scales); each object is visited once to cut cycles."""
    if isinstance(value, (torch.Tensor, *_SCALARS)):
        yield key, value
        return
    if depth == 0 or isinstance(value, (nn.Module, type, str, bytes)):
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk(v, f"{key}[{i}]", depth - 1, seen)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, (str, int)):
                yield from _walk(v, f"{key}[{k!r}]", depth - 1, seen)
    elif hasattr(value, "__dict__"):
        for k, v in vars(value).items():
            yield from _walk(v, f"{key}.{k}", depth - 1, seen)


# Other keys under which each collected tensor was reached (same storage view).
aliases: dict[str, list[str]] = {}


def collect_state(
    roots: dict[str, nn.Module],
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, object]]:
    """CUDA and pinned host tensors (deduplicated by view) and scalars, in a stable
    order keyed by module path and attribute name."""
    tensors: list[tuple[str, torch.Tensor]] = []
    scalars: dict[str, object] = {}
    seen: dict[tuple, str] = {}
    visited: set[int] = set()
    aliases.clear()
    for root_name, root in roots.items():
        for mod_name, module in root.named_modules():
            for attr, value in vars(module).items():
                if attr in ("_modules", "training") or attr.startswith("__"):
                    continue
                base = f"{root_name}/{mod_name}/{attr}"
                for key, v in _walk(value, base, _DEPTH, visited):
                    if isinstance(v, torch.Tensor):
                        if not (v.is_cuda or v.is_pinned()) or v.numel() == 0:
                            continue
                        ident = (v.data_ptr(), v.nbytes, tuple(v.shape), v.dtype)
                        if ident in seen:
                            aliases.setdefault(seen[ident], []).append(key)
                            continue
                        seen[ident] = key
                        tensors.append((key, v))
                    else:
                        scalars[key] = v
    return tensors, scalars


def _describe(tensors) -> list[dict]:
    return [
        {
            "key": k,
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "nbytes": t.nbytes,
            "aliases": aliases.get(k, []),
        }
        for k, t in tensors
    ]


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _pinned() -> torch.Tensor:
    return torch.empty(_CHUNK, dtype=torch.uint8, pin_memory=True)


def save(roots: dict[str, nn.Module], path: Path) -> None:
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    tensors, scalars = collect_state(roots)
    header = json.dumps(
        {"magic": _MAGIC, "tensors": _describe(tensors), "scalars": scalars}
    ).encode()
    data_off = _round_up(len(header) + 8)
    tmp = path.with_suffix(f".tmp{os.getpid()}")  # concurrent cold starts on one node
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = _pinned()
    with open(tmp, "wb") as f:
        f.write(len(header).to_bytes(8, "little") + header)
        f.write(b"\0" * (data_off - 8 - len(header)))
        for _, t in tensors:
            src = _as_bytes(t)
            for off in range(0, src.numel(), _CHUNK):
                n = min(_CHUNK, src.numel() - off)
                buf[:n].copy_(src[off : off + n])
                f.write(memoryview(buf.numpy())[:n])
            f.write(b"\0" * (_round_up(src.numel()) - src.numel()))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    total = sum(t.nbytes for _, t in tensors)
    logger.info(
        "Weight snapshot saved: %d tensors, %d scalars, %.2f GiB in %.1f s to %s",
        len(tensors),
        len(scalars),
        total / 2**30,
        time.perf_counter() - t0,
        path,
    )


def _read_header(fd: int) -> tuple[dict, int]:
    head = mmap.mmap(-1, _ALIGN)
    os.preadv(fd, [head], 0)
    hlen = int.from_bytes(head[:8], "little")
    hbuf = mmap.mmap(-1, _round_up(hlen + 8))
    os.preadv(fd, [hbuf], 0)
    return json.loads(bytes(hbuf[8 : 8 + hlen])), _round_up(hlen + 8)


def restore(roots: dict[str, nn.Module], path: Path) -> None:
    """Copy the snapshot into the dummy-initialized model state."""
    tensors, scalars = collect_state(roots)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    try:
        header, offset = _read_header(fd)
        if header.get("magic") != _MAGIC:
            raise RuntimeError(f"weight snapshot {path} has an old format; delete it")
        plan = _plan(roots, header.get("tensors", []), tensors, offset, path)
        # Two pinned buffers: the disk read of one overlaps the H2D copy of the other.
        bufs = [_pinned(), _pinned()]
        views = [memoryview(b.numpy()) for b in bufs]
        events = [torch.cuda.Event(), torch.cuda.Event()]
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        i = 0
        for t, offset in plan:
            dst = t.detach()
            if dst.is_contiguous():
                target = dst.reshape(-1).view(torch.uint8)
            else:
                target = torch.empty(t.nbytes, dtype=torch.uint8, device=t.device)
            for off in range(0, t.nbytes, _CHUNK):
                n = min(_CHUNK, t.nbytes - off)
                events[i].synchronize()
                os.preadv(fd, [views[i][: _round_up(n)]], offset + off)
                target[off : off + n].copy_(bufs[i][:n], non_blocking=True)
                events[i].record()
                i ^= 1
            if not dst.is_contiguous():
                dst.copy_(target.view(t.dtype).view(t.shape))
        torch.accelerator.synchronize()
    finally:
        os.close(fd)
    _restore_scalars(roots, header.get("scalars", {}), scalars)
    if os.environ.get("VLLM_WEIGHT_SNAPSHOT_AUDIT"):
        _audit(roots, tensors)
    total = sum(t.nbytes for _, t in tensors)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Weight snapshot restored: %.2f GiB in %.1f s (%.2f GB/s) from %s",
        total / 2**30,
        elapsed,
        total / elapsed / 1e9,
        path,
    )


def _plan(roots, saved: list[dict], tensors, offset: int, path) -> list:
    """Match saved entries to live tensors by key or alias; return (tensor, offset).

    Attributes the real load created but the dummy build did not (e.g. fused
    drafter weights) are allocated, or bound to the live tensor they aliased."""
    now = dict(tensors)
    for k, others in list(aliases.items()):
        for o in others:
            now.setdefault(o, now.get(k))
    live_keys = set(now)
    plan, problems = [], []
    for e in saved:
        keys = [e["key"], *e.get("aliases", [])]
        found = [now.pop(k) for k in keys if now.get(k) is not None]
        t = found[0] if found else _create(roots, e["key"], e)
        at = offset
        offset += _round_up(e["nbytes"])
        if t is None or str(t.dtype) != e["dtype"] or list(t.shape) != e["shape"]:
            problems.append(e["key"])
            continue
        for k in keys:
            if k not in live_keys and not _bind(roots, k, t):
                problems.append(f"{k} (cannot bind alias)")
        plan.append((t, at))
    live = {id(t) for t, _ in plan}
    problems += [f"{k} (not in snapshot)" for k, t in now.items() if id(t) not in live]
    if problems:
        raise RuntimeError(
            f"weight snapshot {path} does not match the model: {problems[:6]} "
            f"({len(problems)} total); delete it to rebuild"
        )
    return plan


def _bind(roots, key: str, t) -> bool:
    """Point the attribute path `key` at tensor t if it is missing or None."""
    root_name, mod_name, path = key.split("/", 2)
    obj = roots[root_name].get_submodule(mod_name)
    # A plain view: binding a Parameter would register it on the module.
    if isinstance(t, nn.Parameter) and not path.startswith("_parameters"):
        t = t.detach()
    try:
        _set_path(obj, path, t, create=True)
        return True
    except (AttributeError, IndexError, KeyError, TypeError):
        return False


def _create(roots, key: str, e: dict):
    root_name, mod_name, path = key.split("/", 2)
    module = roots[root_name].get_submodule(mod_name)
    device = next(module.parameters(), torch.empty(0, device="cuda")).device
    dtype = getattr(torch, e["dtype"].removeprefix("torch."))
    t = torch.empty(e["shape"], dtype=dtype, device=device)
    return t if _bind(roots, key, t) else None


def _audit(roots, tensors) -> None:
    """Log live CUDA tensors whose storage the snapshot does not cover."""
    import gc

    covered = {t.untyped_storage().data_ptr() for _, t in tensors}
    seen, missed = set(), []
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.numel():
            ptr = obj.untyped_storage().data_ptr()
            if ptr in covered or ptr in seen:
                continue
            seen.add(ptr)
            owners = [type(r).__name__ for r in gc.get_referrers(obj)][:4]
            missed.append(
                (obj.untyped_storage().nbytes(), obj.dtype, tuple(obj.shape), owners)
            )
    missed.sort(key=lambda m: -m[0])
    logger.warning(
        "Weight snapshot audit: %d uncovered CUDA storages, %.2f GiB",
        len(missed),
        sum(m[0] for m in missed) / 2**30,
    )
    for m in missed[:40]:
        logger.warning("  uncovered %.1f MiB %s %s via %s", m[0] / 2**20, *m[1:])


_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+|'[^']*')\]")


def _set_path(obj, path: str, value, create: bool = False) -> None:
    """Set a value at an attribute/index path such as a.b[0]['k'].

    With create, missing attributes and list slots on the way are added."""
    steps = [(m[1], m[2]) for m in _TOKEN.finditer("." + path)]
    for i, (attr, idx) in enumerate(steps):
        last = i == len(steps) - 1
        nxt_is_index = not last and steps[i + 1][0] is None
        if attr is not None:
            if last:
                setattr(obj, attr, value)
            else:
                if create and getattr(obj, attr, None) is None:
                    setattr(obj, attr, [] if nxt_is_index else None)
                obj = getattr(obj, attr)
        else:
            k = idx[1:-1] if idx.startswith("'") else int(idx)
            if create and isinstance(obj, list) and isinstance(k, int):
                obj.extend([None] * (k + 1 - len(obj)))
            if last:
                obj[k] = value
            else:
                obj = obj[k]


def _restore_scalars(roots, saved: dict, now: dict) -> None:
    """Data-dependent scalars (e.g. global scales) come from the real weights."""
    changed = [k for k, v in saved.items() if k in now and now[k] != v]
    for key in changed:
        root_name, mod_name, path = key.split("/", 2)
        try:
            _set_path(roots[root_name].get_submodule(mod_name), path, saved[key])
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            logger.warning("Weight snapshot: scalar %s not restored (%s)", key, e)
    if changed:
        logger.info("Weight snapshot: restored %d data-dependent scalars", len(changed))


def _roots(model_runner) -> dict[str, nn.Module]:
    """The model plus any drafter modules (field names differ across versions)."""
    roots = {"model": model_runner.model}
    for holder in ("speculator", "drafter"):
        obj = getattr(model_runner, holder, None)
        for name, value in vars(obj).items() if obj is not None else ():
            if isinstance(value, nn.Module):
                roots[f"{holder}.{name}"] = value
    return roots


def _plausible_dummy_weights():
    """Fill dummy floats and packed bytes with 0x3C: finite and positive in fp32,
    bf16, fp16 and fp8, so quantization checks on the dummy build pass."""
    try:
        from vllm.model_executor.model_loader import dummy_loader
    except ImportError:
        return None
    orig = dummy_loader.initialize_single_dummy_weight

    def fill(param, *args, **kwargs):
        plausible = param.is_floating_point() or param.dtype == torch.uint8
        if plausible and param.is_contiguous():
            with torch.no_grad():
                param.reshape(-1).view(torch.uint8).fill_(0x3C)
        else:
            orig(param, *args, **kwargs)

    dummy_loader.initialize_single_dummy_weight = fill
    return lambda: setattr(dummy_loader, "initialize_single_dummy_weight", orig)


def _profile_build():
    """Time the dummy build phases and cProfile process_weights_after_loading."""
    import cProfile
    import io
    import pstats

    from vllm.model_executor.model_loader import base_loader, dummy_loader

    undo = []

    def timed(mod, name, label, profile=False):
        orig = getattr(mod, name)

        def wrapper(*args, **kwargs):
            torch.accelerator.synchronize()
            prof = cProfile.Profile() if profile else None
            t0 = time.perf_counter()
            out = prof.runcall(orig, *args, **kwargs) if prof else orig(*args, **kwargs)
            torch.accelerator.synchronize()
            logger.info(
                "Snapshot build: %s took %.2f s", label, time.perf_counter() - t0
            )
            if prof:
                buf = io.StringIO()
                pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(25)
                logger.info("Snapshot build profile:\n%s", buf.getvalue())
            return out

        setattr(mod, name, wrapper)
        undo.append(lambda: setattr(mod, name, orig))

    timed(dummy_loader.DummyModelLoader, "load_weights", "dummy fill")
    timed(base_loader, "process_weights_after_loading", "process_weights", True)
    return lambda: [u() for u in undo]


def _fast_profile_run(runner, tokens: int):
    """With an explicit KV size, profile_run only warms up: use fewer tokens."""
    orig = runner.profile_run

    def profile_run(*args, **kwargs):
        saved = runner.max_num_tokens
        runner.max_num_tokens = min(saved, tokens)
        try:
            return orig(*args, **kwargs)
        finally:
            runner.max_num_tokens = saved

    runner.profile_run = profile_run


def install(worker_cls) -> None:
    """Wrap worker_cls.load_model for vLLM builds without the built-in hook."""
    import inspect

    orig = worker_cls.load_model
    takes_dummy = "load_dummy_weights" in inspect.signature(orig).parameters

    def load_model(self, *args, **kwargs):
        root = os.environ.get("VLLM_WEIGHT_SNAPSHOT_DIR")
        if not root or kwargs.get("load_dummy_weights"):
            return orig(self, *args, **kwargs)
        path = snapshot_path(root, weights_key(self.vllm_config), self.rank)
        restore = path.exists()
        if restore and takes_dummy:
            kwargs["load_dummy_weights"] = True
        elif restore:
            self.vllm_config.load_config.load_format = "dummy"
        # Lets model-specific checkpoint validation skip the dummy load.
        os.environ["VLLM_WEIGHT_SNAPSHOT_RESTORING"] = "1" if restore else ""
        undo = _plausible_dummy_weights() if restore else None
        prof = os.environ.get("VLLM_WEIGHT_SNAPSHOT_PROFILE") and restore
        undo_prof = _profile_build() if prof else None
        try:
            out = orig(self, *args, **kwargs)
        finally:
            os.environ.pop("VLLM_WEIGHT_SNAPSHOT_RESTORING", None)
            if undo is not None:
                undo()
            if undo_prof is not None:
                undo_prof()
        fast = int(os.environ.get("VLLM_WEIGHT_SNAPSHOT_FAST_PROFILE", "0"))
        if restore and fast and self.cache_config.kv_cache_memory_bytes:
            _fast_profile_run(self.model_runner, fast)
        roots = _roots(self.model_runner)
        (restore_fn if restore else save)(roots, path)
        return out

    worker_cls.load_model = load_model


restore_fn = restore
