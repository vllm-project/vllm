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

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 4096
_CHUNK = 256 << 20
_MAGIC = "vllm-weight-snapshot-v2"
_SCALARS = (bool, int, float)


def snapshot_path(root: str, config_hash: str, rank: int) -> Path:
    return Path(root) / f"{config_hash}-rank{rank}.bin"


def _round_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _walk(value, key: str, depth: int) -> Iterator[tuple[str, object]]:
    """Yield (key, tensor or scalar) found in an attribute value."""
    if isinstance(value, (torch.Tensor, *_SCALARS)):
        yield key, value
    elif depth == 0:
        return
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk(v, f"{key}[{i}]", depth - 1)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, (str, int)):
                yield from _walk(v, f"{key}[{k!r}]", depth - 1)
    elif hasattr(value, "__dict__") and not isinstance(value, (nn.Module, type)):
        for k, v in vars(value).items():
            yield from _walk(v, f"{key}.{k}", depth - 1)


def collect_state(
    roots: dict[str, nn.Module],
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, object]]:
    """CUDA tensors (deduplicated by storage view) and scalars, in a stable
    order keyed by module path and attribute name."""
    tensors: list[tuple[str, torch.Tensor]] = []
    scalars: dict[str, object] = {}
    seen: set[tuple] = set()
    for root_name, root in roots.items():
        for mod_name, module in root.named_modules():
            for attr, value in vars(module).items():
                if attr in ("_modules", "training") or attr.startswith("__"):
                    continue
                base = f"{root_name}/{mod_name}/{attr}"
                for key, v in _walk(value, base, depth=2):
                    if isinstance(v, torch.Tensor):
                        if not v.is_cuda or v.numel() == 0:
                            continue
                        ident = (v.data_ptr(), v.nbytes, tuple(v.shape), v.dtype)
                        if ident in seen:
                            continue
                        seen.add(ident)
                        tensors.append((key, v))
                    else:
                        scalars[key] = v
    return tensors, scalars


def _describe(tensors) -> list[dict]:
    return [
        {"key": k, "dtype": str(t.dtype), "shape": list(t.shape), "nbytes": t.nbytes}
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
    tmp = path.with_suffix(".tmp")
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
        saved, now = header.get("tensors", []), _describe(tensors)
        if header.get("magic") != _MAGIC or saved != now:
            diff = [(a, b) for a, b in zip(saved, now) if a != b][:4]
            raise RuntimeError(
                f"weight snapshot {path} does not match the model "
                f"({len(saved)} saved vs {len(now)} tensors, first differences "
                f"{diff}); delete it to rebuild"
            )
        buf = _pinned()
        view = memoryview(buf.numpy())
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _, t in tensors:
            dst = t.detach()
            if dst.is_contiguous():
                target = dst.reshape(-1).view(torch.uint8)
            else:
                target = torch.empty(t.nbytes, dtype=torch.uint8, device=t.device)
            for off in range(0, t.nbytes, _CHUNK):
                n = min(_CHUNK, t.nbytes - off)
                os.preadv(fd, [view[: _round_up(n)]], offset + off)
                target[off : off + n].copy_(buf[:n])
            if not dst.is_contiguous():
                dst.copy_(target.view(t.dtype).view(t.shape))
            offset += _round_up(t.nbytes)
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


def _restore_scalars(roots, saved: dict, now: dict) -> None:
    """Data-dependent scalars (e.g. global scales) come from the real weights."""
    changed = [k for k, v in saved.items() if k in now and now[k] != v]
    for key in changed:
        root_name, mod_name, attr = key.split("/", 2)
        if "." in attr or "[" in attr:
            logger.warning("Weight snapshot: nested scalar %s not restored", key)
            continue
        setattr(roots[root_name].get_submodule(mod_name), attr, saved[key])
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


def install(worker_cls) -> None:
    """Wrap worker_cls.load_model for vLLM builds without the built-in hook."""
    import inspect

    orig = worker_cls.load_model
    takes_dummy = "load_dummy_weights" in inspect.signature(orig).parameters

    def load_model(self, *args, **kwargs):
        root = os.environ.get("VLLM_WEIGHT_SNAPSHOT_DIR")
        if not root or kwargs.get("load_dummy_weights"):
            return orig(self, *args, **kwargs)
        path = snapshot_path(root, self.vllm_config.compute_hash(), self.rank)
        restore = path.exists()
        if restore and takes_dummy:
            kwargs["load_dummy_weights"] = True
        elif restore:
            self.vllm_config.load_config.load_format = "dummy"
        out = orig(self, *args, **kwargs)
        roots = _roots(self.model_runner)
        (restore_fn if restore else save)(roots, path)
        return out

    worker_cls.load_model = load_model


restore_fn = restore
