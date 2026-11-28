# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot of post-processed model state.

The first start saves every CUDA tensor and Python scalar reachable from the
model modules after process_weights_after_loading. Later starts build the model
with dummy weights, which gives the same structure, and copy the saved bytes
back, skipping checkpoint reads and weight conversion.

Enabled with VLLM_WEIGHT_SNAPSHOT_DIR; see load_model().
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
_MAGIC = "vllm-weight-snapshot-v5"
_SCALARS = (bool, int, float)
_DEPTH = 6


def snapshot_path(root: str, config_hash: str, rank: int) -> Path:
    return Path(root) / f"{config_hash}-rank{rank}.bin"


def resolve(root: str, config_hash: str, rank: int, clear: bool, rebuild: bool):
    """Snapshot path to use, after the one-shot CLEAR / REBUILD requests.

    CLEAR deletes every snapshot file of this rank in `root` (all keys) and
    returns None, so the model loads normally without saving. REBUILD deletes
    this rank's file for the current key and returns its path, so the model
    loads normally and saves a fresh snapshot. Both flags apply on every start
    while set."""
    if clear and rebuild:
        raise ValueError(
            "VLLM_WEIGHT_SNAPSHOT_CLEAR and VLLM_WEIGHT_SNAPSHOT_REBUILD are "
            "mutually exclusive"
        )
    path = snapshot_path(root, config_hash, rank)
    if clear:
        stale = [
            *Path(root).glob(f"*-rank{rank}.bin"),
            *Path(root).glob(f"*-rank{rank}.tmp*"),
        ]
        for f in stale:
            f.unlink(missing_ok=True)
        logger.info(
            "Weight snapshot: cleared %d file(s) of rank %d in %s",
            len(stale),
            rank,
            root,
        )
        return None
    if rebuild:
        path.unlink(missing_ok=True)
        logger.info("Weight snapshot: rebuilding %s", path)
    return path


def _checkpoint_id(path) -> object:
    """Location-independent identity of a checkpoint.

    For an HF cache snapshot, repo@commit already pins the files, so the weights
    need not be present when restoring; elsewhere use weight file names/sizes."""
    import hashlib

    if path is None or not os.path.isdir(str(path)):
        return path
    p = Path(path).resolve()
    cfg = p / "config.json"
    digest = hashlib.sha256(cfg.read_bytes()).hexdigest()[:16] if cfg.exists() else None
    if p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
        repo = p.parent.parent.name[len("models--") :].replace("--", "/")
        return [f"{repo}@{p.name}", digest]
    files = sorted(
        (f.name, f.stat().st_size)
        for f in p.iterdir()
        if f.suffix in (".safetensors", ".bin", ".pt", ".gguf")
    )
    return [p.name, files, digest]


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
    eplb = getattr(p, "eplb_config", None)
    parts = [
        _checkpoint_id(m.model),
        m.revision,
        str(m.dtype),
        m.quantization,
        overrides,
        getattr(vllm_config.cache_config, "cache_dtype", None),
        p.tensor_parallel_size,
        p.pipeline_parallel_size,
        # Which experts each rank holds: EP size and placement (same shapes,
        # different experts for a different placement), redundant experts.
        getattr(p, "data_parallel_size", None),
        getattr(p, "prefill_context_parallel_size", None),
        getattr(p, "enable_expert_parallel", None),
        getattr(p, "expert_placement_strategy", None),
        getattr(p, "enable_eplb", None),
        getattr(eplb, "num_redundant_experts", None),
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


def collect_state(
    roots: dict[str, nn.Module],
) -> tuple[
    list[tuple[str, torch.Tensor]], dict[str, object], dict[str, list[str]], list[str]
]:
    """CUDA tensors (deduplicated by view), scalars, the other keys each tensor was
    reached under, in a stable order keyed by module path and attribute name, and
    the keys of state this snapshot cannot hold (pinned host tensors)."""
    tensors: list[tuple[str, torch.Tensor]] = []
    scalars: dict[str, object] = {}
    aliases: dict[str, list[str]] = {}
    unsupported: list[str] = []
    seen: dict[tuple, str] = {}
    visited: set[int] = set()
    for root_name, root in roots.items():
        for mod_name, module in root.named_modules():
            for attr, value in vars(module).items():
                if attr in ("_modules", "training") or attr.startswith("__"):
                    continue
                base = f"{root_name}/{mod_name}/{attr}"
                for key, v in _walk(value, base, _DEPTH, visited):
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 0:
                            continue
                        if not v.is_cuda:
                            if v.is_pinned():
                                unsupported.append(key)
                            continue
                        ident = (v.data_ptr(), v.nbytes, tuple(v.shape), v.dtype)
                        if ident in seen:
                            aliases.setdefault(seen[ident], []).append(key)
                            continue
                        seen[ident] = key
                        tensors.append((key, v))
                    else:
                        scalars[key] = v
    return tensors, scalars, aliases, unsupported


def _describe(tensors, aliases) -> list[dict]:
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
    tensors, scalars, aliases, unsupported = collect_state(roots)
    if unsupported:
        # A partial snapshot would restore wrong weights without any error.
        logger.warning(
            "Weight snapshot not saved: the model keeps state the snapshot cannot "
            "hold (pinned host tensors, e.g. %s)",
            unsupported[:3],
        )
        return
    header = json.dumps(
        {"magic": _MAGIC, "tensors": _describe(tensors, aliases), "scalars": scalars}
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


def _pread_exact(fd: int, buf, pos: int, need: int) -> None:
    """Read at least `need` bytes into `buf` at `pos`; short reads raise."""
    view = memoryview(buf)
    got = 0
    while got < need:
        n = os.preadv(fd, [view[got:]], pos + got)
        if n <= 0:
            raise RuntimeError(
                f"weight snapshot is truncated: needed {need} bytes at offset "
                f"{pos}, read {got}; delete it to rebuild"
            )
        got += n


def _read_header(fd: int) -> tuple[dict, int]:
    head = mmap.mmap(-1, _ALIGN)
    _pread_exact(fd, head, 0, 8)
    hlen = int.from_bytes(head[:8], "little")
    hbuf = mmap.mmap(-1, _round_up(hlen + 8))
    _pread_exact(fd, hbuf, 0, hlen + 8)
    return json.loads(bytes(hbuf[8 : 8 + hlen])), _round_up(hlen + 8)


def _check_size(fd: int, header: dict, data_off: int, path: Path) -> None:
    """Reject a file shorter than its header says, before touching the model."""
    end = data_off + sum(_round_up(e["nbytes"]) for e in header.get("tensors", []))
    size = os.fstat(fd).st_size
    if size < end:
        raise RuntimeError(
            f"weight snapshot {path} is truncated ({size} of {end} bytes); "
            "delete it to rebuild"
        )


def _open_direct(path: Path) -> int:
    """O_DIRECT bypasses the page cache; tmpfs and some overlay setups reject it."""
    try:
        return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    except OSError:
        return os.open(path, os.O_RDONLY)


def restore(roots: dict[str, nn.Module], path: Path) -> None:
    """Copy the snapshot into the dummy-initialized model state."""
    tensors, scalars, aliases, _ = collect_state(roots)
    fd = _open_direct(path)
    try:
        header, offset = _read_header(fd)
        if header.get("magic") != _MAGIC:
            raise RuntimeError(f"weight snapshot {path} has an old format; delete it")
        _check_size(fd, header, offset, path)
        plan = _plan(header.get("tensors", []), tensors, aliases, offset, path)
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
                _pread_exact(fd, views[i][: _round_up(n)], offset + off, n)
                target[off : off + n].copy_(bufs[i][:n], non_blocking=True)
                events[i].record()
                i ^= 1
            if not dst.is_contiguous():
                dst.copy_(target.view(t.dtype).view(t.shape))
        torch.accelerator.synchronize()
    finally:
        os.close(fd)
    _restore_scalars(roots, header.get("scalars", {}), scalars)
    total = sum(t.nbytes for _, t in tensors)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Weight snapshot restored: %.2f GiB in %.1f s (%.2f GB/s) from %s",
        total / 2**30,
        elapsed,
        total / elapsed / 1e9,
        path,
    )


def _plan(saved: list[dict], tensors, aliases, offset: int, path) -> list:
    """Match saved entries to live tensors by key or alias; return (tensor, offset).

    Any entry without a live tensor of the same dtype and shape, and any live
    tensor the snapshot does not cover, is an error: never restore partially."""
    now = dict(tensors)
    for k, others in list(aliases.items()):
        for o in others:
            now.setdefault(o, now.get(k))
    plan, problems = [], []
    for e in saved:
        keys = [e["key"], *e.get("aliases", [])]
        found = [now.pop(k) for k in keys if now.get(k) is not None]
        t = found[0] if found else None
        at = offset
        offset += _round_up(e["nbytes"])
        if t is None or str(t.dtype) != e["dtype"] or list(t.shape) != e["shape"]:
            problems.append(e["key"])
            continue
        plan.append((t, at))
    live = {id(t) for t, _ in plan}
    problems += [f"{k} (not in snapshot)" for k, t in now.items() if id(t) not in live]
    if problems:
        raise RuntimeError(
            f"weight snapshot {path} does not match the model: {problems[:6]} "
            f"({len(problems)} total); delete it to rebuild"
        )
    return plan


_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+|'[^']*')\]")


def _set_path(obj, path: str, value) -> None:
    """Set a value at an attribute/index path such as `a.b[0]['k']`."""
    steps = [(m[1], m[2]) for m in _TOKEN.finditer("." + path)]
    for i, (attr, idx) in enumerate(steps):
        last = i == len(steps) - 1
        if attr is not None:
            if last:
                setattr(obj, attr, value)
            else:
                obj = getattr(obj, attr)
        else:
            k = idx[1:-1] if idx.startswith("'") else int(idx)
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


def load_model(model_runner, path: Path) -> None:
    """Load the model through the snapshot at `path`.

    If it exists, build the model with dummy weights and restore the saved state
    into it; otherwise load normally and save the post-processed state."""
    restore_ = path.exists()
    model_runner.load_model(load_dummy_weights=restore_)
    roots = _roots(model_runner)
    (restore if restore_ else save)(roots, path)
