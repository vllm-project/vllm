# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verification harness for streaming weight reload (modulewise reload).

A worker extension that streams a safetensors checkpoint into a running engine
the way NCCL/IPC weight transfer does (device tensors, routed through the
model's own ``load_weights``), with a configurable arrival order, and reports:

- peak device memory over the resident model during the update,
- which incoming tensors were retained after their batch (weakrefs),
- ``data_ptr`` of every registered tensor and of every tensor reachable from
  module attributes that are not registered (the "off-registry census"), so a
  driver can diff them across a reload.

Use with ``worker_extension_cls="tests.model_executor.model_loader.reload_harness.
ReloadHarnessExtension"`` and ``PYTHONPATH`` pointing at the repo root.
"""

import gc
import glob
import json
import os
import random
import time
import weakref
from collections.abc import Iterator
from typing import Any

import regex as re
import torch

_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def _checkpoint_files(path: str) -> list[str]:
    if not os.path.isdir(path):
        from vllm.transformers_utils.repo_utils import hf_api

        path = hf_api().snapshot_download(
            path, allow_patterns=["*.safetensors", "*.json"]
        )
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(path, f) for f in files]
    return sorted(glob.glob(os.path.join(path, "*.safetensors")))


def _read_names(files: list[str]) -> list[tuple[str, str]]:
    from safetensors import safe_open

    out: list[tuple[str, str]] = []
    for fn in files:
        with safe_open(fn, "pt") as f:
            out.extend((name, fn) for name in f.keys())  # noqa: SIM118
    return out


def _stream_load(model, weights):
    """`load_weights` the way the weight transfer engines call it: each call is
    one streamed batch, so whole-checkpoint MTP completeness checks are off."""
    try:
        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )
    except ImportError:  # a baseline without the check
        from contextlib import nullcontext as disable_mtp_completeness_check
    with disable_mtp_completeness_check():
        return model.load_weights(weights)


def _layer_index(name: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def order_names(names: list[tuple[str, str]], order: str, seed: int = 0):
    """Arrival orders a trainer might use."""
    if order == "natural":
        return list(names)
    if order == "reverse":
        return list(reversed(names))
    if order == "shuffle":
        out = list(names)
        random.Random(seed).shuffle(out)
        return out
    if order == "interleave":
        # round-robin across layers: every layer is open at once
        by_layer: dict[int, list] = {}
        for n in names:
            by_layer.setdefault(_layer_index(n[0]), []).append(n)
        buckets = list(by_layer.values())
        out = []
        while any(buckets):
            for b in buckets:
                if b:
                    out.append(b.pop(0))
        return out
    if order == "nonexpert_first":
        experts = [n for n in names if _EXPERT_RE.search(n[0])]
        others = [n for n in names if not _EXPERT_RE.search(n[0])]
        return others + experts
    if order == "experts_worst":
        # all w1 (gate), then all w3 (up), then all w2 (down), per layer
        def key(n):
            proj = (
                0
                if "gate_proj" in n[0] or ".w1." in n[0]
                else (1 if "up_proj" in n[0] or ".w3." in n[0] else 2)
            )
            return (_layer_index(n[0]), proj if _EXPERT_RE.search(n[0]) else -1)

        return sorted(names, key=key)
    raise ValueError(order)


def _tensor_digest(t: torch.Tensor) -> str:
    import hashlib

    t = t.detach()
    if t.is_meta:
        return f"meta{tuple(t.shape)}"
    if t.numel() == 0:
        return f"empty{tuple(t.shape)}"
    b = t.contiguous().view(-1).view(torch.uint8) if t.element_size() else t
    return hashlib.md5(b.cpu().numpy().tobytes()).hexdigest()


def _collect_off_registry(
    model: torch.nn.Module, max_depth: int = 5, tensors: bool = False
):
    """Tensors reachable from module attributes (quant methods, kernels, quant
    configs, ...) that are not registered params/buffers. Keyed by a path."""
    registered = {id(t) for t in model.parameters()} | {id(t) for t in model.buffers()}
    out: dict[str, tuple[int, tuple[int, ...], str]] = {}
    seen: set[int] = set()

    def visit(obj, path, depth):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if (
                id(obj) not in registered
                and obj.device.type == "cuda"
                and "kv_cache" not in path  # runtime KV / Mamba state
            ):
                try:  # noqa: SIM105
                    out[path] = (
                        obj
                        if tensors
                        else (obj.data_ptr(), tuple(obj.shape), str(obj.dtype))
                    )
                except Exception:  # noqa: BLE001
                    pass
            return
        if isinstance(obj, torch.nn.Module):
            return  # modules are visited from the top-level loop
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]", depth + 1)
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:64]:
                visit(v, f"{path}[{k!r}]", depth + 1)
            return
        d = getattr(obj, "__dict__", None)
        if d is None or type(obj).__module__.startswith(("torch", "builtins")):
            return
        for k, v in list(d.items()):
            visit(v, f"{path}.{k}", depth + 1)

    for mname, module in model.named_modules():
        for k, v in module.__dict__.items():
            if k in ("_parameters", "_buffers", "_modules"):
                continue
            visit(v, f"{mname}.{k}", 0)
    return out


class ReloadHarnessExtension:
    """Methods callable through ``LLM.collective_rpc``."""

    # provided by the Worker this extension is mixed into
    model_runner: Any
    vllm_config: Any
    model_config: Any
    weight_transfer_engine: Any
    rank: int
    start_weight_update: Any
    finish_weight_update: Any

    def _mw_model(self):
        if getattr(self, "_mw_target", "target") == "draft":
            return self.model_runner.get_draft_model()
        return self.model_runner.get_model()

    def mw_set_target(self, which: str) -> None:
        """Point reload / checksum / pointer calls at "target" or "draft"."""
        assert which in ("target", "draft")
        self._mw_target = which

    def mw_ptr_snapshot(self) -> dict:
        model = self._mw_model()
        registered = {}
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            if t.device.type == "cuda":
                registered[name] = (t.data_ptr(), tuple(t.shape), str(t.dtype))
        return {"registered": registered, "off_registry": _collect_off_registry(model)}

    def mw_checksums(self) -> dict:
        """md5 of every registered tensor and every off-registry tensor."""
        model = self._mw_model()
        torch.accelerator.synchronize()
        registered = {}
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            registered[name] = _tensor_digest(t)
        off = {}
        for path, t in _collect_off_registry(model, tensors=True).items():
            try:
                off[path] = _tensor_digest(t)
            except Exception as e:  # noqa: BLE001
                off[path] = f"error:{type(e).__name__}"
        return {"registered": registered, "off_registry": off}

    def mw_memory(self) -> dict:
        return {
            "allocated": torch.accelerator.memory_allocated(),
            "reserved": torch.accelerator.memory_reserved(),
        }

    def mw_reload(
        self,
        path: str,
        order: str = "natural",
        batch_size: int = 16,
        seed: int = 0,
        skip_regex: str | None = None,
        fail_after: int | None = None,
        on_cpu: bool = False,
        perturb: float | None = None,
        use_public_api: bool = True,
        force_rebuild: bool = False,
    ) -> dict:
        """Stream ``path`` into the live model. Returns memory/retention stats."""
        return self._mw_reload(
            path,
            order,
            batch_size,
            seed,
            skip_regex,
            fail_after,
            on_cpu,
            perturb,
            use_public_api,
            force_rebuild,
        )

    def _mw_reload(
        self,
        path,
        order,
        batch_size,
        seed,
        skip_regex,
        fail_after,
        on_cpu,
        perturb,
        use_public_api,
        force_rebuild,
    ) -> dict:
        from safetensors import safe_open

        from vllm.config import set_current_vllm_config

        model = self._mw_model()
        if force_rebuild:
            # emulate kernels that are not built once: every MoE kernel is rebuilt
            for m in model.modules():
                qm = getattr(m, "quant_method", None)
                if qm is not None and hasattr(qm, "reload_safe"):
                    qm.reload_safe = False
        files = _checkpoint_files(path)
        names = order_names(_read_names(files), order, seed)
        if skip_regex:
            names = [n for n in names if not re.search(skip_regex, n[0])]

        handles = {fn: safe_open(fn, "pt", device="cpu") for fn in files}
        device = torch.device("cuda", torch.accelerator.current_device_index())

        from vllm.model_executor.model_loader import reload as reload_api

        start = getattr(reload_api, "start_reload", None) if use_public_api else None
        finish = getattr(reload_api, "finish_reload", None) if use_public_api else None
        start = start or reload_api.initialize_layerwise_reload
        finish = finish or reload_api.finalize_layerwise_reload

        gc.collect()
        torch.accelerator.synchronize()
        base = torch.accelerator.memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        retained: list[str] = []
        max_batch_bytes = 0
        max_held_after_batch = 0
        held_series: list[int] = []
        t0 = time.perf_counter()
        sent = 0
        with set_current_vllm_config(self.vllm_config):
            start(model)
            try:
                for i in range(0, len(names), batch_size):
                    batch = []
                    for name, fn in names[i : i + batch_size]:
                        t = handles[fn].get_tensor(name)
                        if perturb is not None and t.is_floating_point():
                            t = t * (1 + perturb)
                        if not on_cpu:
                            t = t.to(device)
                        batch.append((name, t))
                    del t
                    batch_bytes = sum(t.nbytes for _, t in batch)
                    max_batch_bytes = max(max_batch_bytes, batch_bytes)
                    refs = [(n, weakref.ref(t)) for n, t in batch]

                    def gen(batch=batch) -> Iterator[tuple[str, torch.Tensor]]:
                        while batch:
                            yield batch.pop(0)

                    _stream_load(model, gen())
                    del batch
                    sent += len(refs)
                    if fail_after is not None and sent >= fail_after:
                        raise RuntimeError("harness: injected failure")
                    gc.collect()  # loader generator chains can hold cycles
                    alive = [n for n, r in refs if r() is not None]
                    retained.extend(alive)
                    held = torch.accelerator.memory_allocated() - base
                    max_held_after_batch = max(max_held_after_batch, held)
                    held_series.append(held)
                finish(model, self.model_config)
            except BaseException:
                abort = getattr(reload_api, "abort_reload", None)
                if abort is not None and use_public_api:
                    abort(model)
                raise
        torch.accelerator.synchronize()
        dt = time.perf_counter() - t0
        stats = {
            "num_tensors": len(names),
            "peak_over_base": torch.accelerator.max_memory_allocated() - base,
            "max_batch_bytes": max_batch_bytes,
            "max_held_after_batch": max_held_after_batch,
            "end_over_base": torch.accelerator.memory_allocated() - base,
            "retained": retained[:50],
            "num_retained": len(retained),
            "held_series": held_series,
            "batch_size": batch_size,
            "order": order,
            "seconds": dt,
        }
        return stats

    def mw_call(self, fn_path: str, *args, **kwargs):
        """Call ``module:function(model, *args)`` in the worker."""
        import importlib

        mod, fn = fn_path.split(":")
        return getattr(importlib.import_module(mod), fn)(
            self._mw_model(), *args, **kwargs
        )

    def mw_load_only(
        self, path: str, skip_regex: str | None = None, batch_size: int = 16
    ) -> None:
        """Stream ``path`` into the weight transfer engine's model, between a
        caller's ``start_weight_update`` and ``finish_weight_update`` RPCs
        (stands in for ``update_weights`` without a trainer)."""
        from safetensors import safe_open

        from vllm.config import set_current_vllm_config

        files = _checkpoint_files(path)
        names = order_names(_read_names(files), "natural")
        if skip_regex:
            names = [n for n in names if not re.search(skip_regex, n[0])]
        handles = {fn: safe_open(fn, "pt", device="cpu") for fn in files}
        device = torch.device("cuda", torch.accelerator.current_device_index())
        model = self.weight_transfer_engine.model
        with set_current_vllm_config(self.vllm_config):
            for i in range(0, len(names), batch_size):
                batch = [
                    (n, handles[fn].get_tensor(n).to(device))
                    for n, fn in names[i : i + batch_size]
                ]
                _stream_load(model, iter(batch))
