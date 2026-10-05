# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install-time and per-step safety guards for the GLM-5.2 MonoKernel.

Install time (``check_before_install`` / ``check_after_install``):
  * RoPE table: built for ``cfg.max_model_len`` rows, raised to vLLM's max_model_len
    (positions past the table would read garbage).
  * KV cache size: the kernel addresses each layer's [slots, 576] bf16 cache with 32-bit
    byte offsets, so every mono layer's cache must be < 4 GiB.
  * Graph widths: under FULL cudagraphs the capture sizes must equal the kernel widths
    (a capture size without a width would bake a fallback).
Per step (``PollErrorWatch``): a non-blocking, one-step-delayed fail-stop on expired
kernel polls, so the engine fails instead of serving wrong tokens.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.mono.envs import failstop_mode

logger = init_logger(__name__)

KV_LIMIT_BYTES = 1 << 32


def _vllm_config(vllm_config=None):
    if vllm_config is not None:
        return vllm_config
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except Exception:  # noqa: BLE001
        return None


def check_before_install(model, cfg, vllm_config=None, check_kv: bool = True) -> dict:
    vc = _vllm_config(vllm_config)
    info: dict[str, Any] = {}
    mml = getattr(getattr(vc, "model_config", None), "max_model_len", None)
    if mml:
        if cfg.max_model_len < mml:
            logger.warning(
                "mono guards: raising RoPE table length %d -> vLLM max_model_len %d",
                cfg.max_model_len,
                mml,
            )
            cfg.max_model_len = int(mml)
        info["vllm_max_model_len"] = int(mml)
    rows = model.model.layers[cfg.layers[0]].self_attn.rotary_emb.cos_sin_cache.shape[0]
    if rows < cfg.max_model_len:
        raise RuntimeError(
            f"mono guards: vLLM cos_sin_cache has {rows} rows < max_model_len "
            f"{cfg.max_model_len}"
        )
    worst = 0
    # check_kv=False: at dispatch creation the caches are not bound yet
    for L in cfg.layers if check_kv else ():
        kv = model.model.layers[L].self_attn.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[0]
        if kv is None or kv.numel() == 0:
            raise RuntimeError(
                f"mono guards: layer {L} has no KV cache bound at install time"
            )
        nbytes = kv.numel() * kv.element_size()
        worst = max(worst, nbytes)
        if nbytes >= KV_LIMIT_BYTES:
            raise RuntimeError(
                f"mono guards: layer {L} KV cache is {nbytes / 2**30:.2f} GiB >= 4 "
                "GiB; the kernel's 32-bit buffer offsets would wrap. Bound it (e.g. "
                "--num-gpu-blocks-override) or do not install mono."
            )
    info["max_layer_kv_gib"] = worst / 2**30
    if check_kv and getattr(cfg, "early_cache_checks", False):
        info["cache_contract"] = check_cache_contract(model, cfg, vc)
    cc = getattr(vc, "compilation_config", None)
    if cc is not None:
        why = graph_width_mismatch(cfg.sizes, vc)
        if why is not None:
            raise RuntimeError(f"mono guards: {why}")
        info["cudagraph"] = dict(
            mode=str(cc.cudagraph_mode),
            capture_sizes=sorted(cc.cudagraph_capture_sizes or []),
        )
    logger.info("mono guards: pre-install OK %s", info)
    return info


SUPPORTED_WIDTHS = (1, 2, 4, 5, 6, 8, 10, 12)


def full_cudagraphs(vc) -> bool:
    """True when vLLM runs FULL (whole-forward) cudagraphs: replayed steps run no
    Python, so the per-step dispatch decision and every flag it reads are baked in at
    capture."""
    mode = getattr(getattr(vc, "compilation_config", None), "cudagraph_mode", None)
    return bool(
        mode is not None
        and hasattr(mode, "has_full_cudagraphs")
        and mode.has_full_cudagraphs()
    )


def piecewise_graphs_only(vc) -> bool:
    """True when decode steps are captured as breakable PIECEWISE graphs only: those
    captures carry no attention metadata, so every captured step falls back."""
    mode = getattr(getattr(vc, "compilation_config", None), "cudagraph_mode", None)
    if mode is None or full_cudagraphs(vc) or getattr(mode, "name", "") == "NONE":
        return False
    from vllm import envs

    return bool(envs.VLLM_USE_BREAKABLE_CUDAGRAPH)


def graph_width_mismatch(widths, vc) -> str | None:
    """None when the kernel widths fit vLLM's FULL-cudagraph capture sizes, else why not
    plus the configuration that would fit. Every capture size must be a kernel width
    (the captured dispatch pads T to a width; a capture size with no width would bake a
    fallback, a width with no capture size is dead weight) and none may exceed the
    widest."""
    cc = getattr(vc, "compilation_config", None)
    if cc is None or not full_cudagraphs(vc):
        return None
    sizes = sorted(cc.cudagraph_capture_sizes or [])
    widths = sorted(widths)
    if not sizes:
        return None
    cap = [s for s in sizes if s <= max(widths)]
    if cap == widths and max(sizes) <= max(widths):
        return None
    fix = (
        f"run with cudagraph_capture_sizes={widths} (e.g. --compilation-config "
        f"'{{\"cudagraph_capture_sizes\": {widths}}}') and max_num_seqs <= "
        f"{max(widths)}"
    )
    if set(sizes) <= set(SUPPORTED_WIDTHS):
        fix += (
            f', or set "sizes": {sizes} in the MonoKernel config to match the capture '
            "sizes"
        )
    return (
        f"FULL cudagraph capture sizes {sizes} must equal the kernel widths {widths}: "
        f"{fix}"
    )


def check_cache_contract(model, cfg, vc=None) -> str:
    """The MLA-cache properties live._begin_step asserts on the first mono step,
    checked at install. Skipped for non-tensor stand-ins (CPU tests)."""
    # the block dim is checked against the metadata at the first step: vLLM may split
    # a KV block into smaller kernel blocks
    ptrs: dict[int, int] = {}
    for L in cfg.layers:
        kv = model.model.layers[L].self_attn.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[0]
        if not isinstance(kv, torch.Tensor):
            return "skipped (non-tensor caches)"
        why = []
        if kv.dtype is not torch.bfloat16:
            why.append(f"dtype {kv.dtype} (mono needs kv_cache_dtype=auto -> bf16)")
        if not kv.is_contiguous():
            why.append("not contiguous")
        if kv.dim() < 2 or kv.shape[-1] != 576:
            why.append(f"row width {tuple(kv.shape)} != 576 (kv_lora 512 + rope 64)")
        if why:
            raise RuntimeError(f"mono guards: layer {L} MLA cache: " + "; ".join(why))
        if kv.data_ptr() in ptrs:
            raise RuntimeError(
                f"mono guards: layers {ptrs[kv.data_ptr()]} and {L} share one MLA "
                "cache tensor"
            )
        ptrs[kv.data_ptr()] = L
    return f"ok ({len(ptrs)} layers)"


def check_after_install(lv, vllm_config=None):
    vc = _vllm_config(vllm_config)
    mml = (
        getattr(getattr(vc, "model_config", None), "max_model_len", None)
        or lv.cfg.max_model_len
    )
    if lv.cos.shape[0] < mml or lv.sin.shape[0] < mml:
        raise RuntimeError(
            f"mono guards: RoPE table {tuple(lv.cos.shape)} shorter than max_model_len "
            f"{mml}"
        )


def _fail_hard(msg: str):
    """Terminate this worker: a non-output TP rank's exception never reaches the
    engine, so raising would let the other ranks keep emitting tokens."""
    logger.critical(
        "%s -> terminating this worker (non-output rank: an exception would only be "
        "logged)",
        msg,
    )
    os._exit(70)


# The output rank's first fail-stop message (None until one fires).
_FAILED: dict[str, str | None] = dict(msg=None)
# Seconds the output rank lives on after its fail-stop: time for the raised error to
# reach the engine through the worker's RPC reply.
FAILSTOP_EXIT_GRACE_S = 5.0


def _arm_exit_watchdog(grace_s: float = FAILSTOP_EXIT_GRACE_S):
    """Exit this worker grace_s seconds from now, whatever its threads are doing.

    After a fail-stop the peers have exited (_fail_hard), but the engine has usually
    queued the next step already: its collectives wait forever on them, the main thread
    spins in a device sync (signals are not handled there), and the worker would
    outlive the engine holding stdout open. faulthandler's watchdog is a C thread that
    needs no GIL: it dumps every thread's stack to stderr, then _exit(1)s."""
    import faulthandler

    try:
        faulthandler.dump_traceback_later(grace_s, exit=True, file=2)
    except Exception:  # noqa: BLE001
        import threading
        import time

        def _exit():
            time.sleep(grace_s)
            os._exit(70)

        threading.Thread(target=_exit, daemon=True, name="MonoFailStopExit").start()


def failstop_error() -> str | None:
    """The output rank's latched fail-stop message, or None."""
    return _FAILED["msg"]


def output_rank_failstop(msg: str) -> RuntimeError:
    """Latch an output-rank fail-stop and return the exception to raise.

    The latch lets a later call refuse to run: under async scheduling vLLM consumes
    sample_tokens' reply, not execute_model's, so an exception raised in the forward
    alone can be lost. The first latch also arms the exit watchdog."""
    if _FAILED["msg"] is None:
        _FAILED["msg"] = msg
        logger.error(
            "%s -> the output rank raises; this worker exits in %.0f s (the other "
            "ranks have exited, so it cannot run another step)",
            msg,
            FAILSTOP_EXIT_GRACE_S,
        )
        _arm_exit_watchdog()
    return RuntimeError(msg)


def fail_stop(msg: str, rank: int):
    """Raise on the output rank; terminate the worker on the others."""
    if rank != 0:
        _fail_hard(msg)
    raise output_rank_failstop(f"{msg} -> fail-stop")


class PollErrorWatch:
    """One-step-delayed, non-blocking fail-stop on expired kernel polls.

    mode "raise": the output rank raises, the others exit (_fail_hard); "warn": log,
    count, clear the device words and continue. One async D2H copy + event per step."""

    def __init__(self, lv, mode: str | None = None):
        from vllm.models.deepseek_v32.amd.mono.kernel.glm.layout import POLL_STAGES

        self.lv = lv
        self.mode = mode or failstop_mode()
        self.rank = getattr(lv, "rank", 0)
        self.stages = POLL_STAGES
        self.views = {}
        self.aborts = {}
        self.steps = {}
        self.xviews = {}
        for S in lv.sizes:
            # every runtime of the width (the fused indexer adds one)
            owners = (
                lv.runtime_ops(S)
                if hasattr(lv, "runtime_ops")
                else [lv.ops[(lv.first, S)]]
            )
            for i, op in enumerate(owners):
                off = op.scr_layout["poll_err"]
                self.views[(S, i)] = op.scratch[off : off + 4 * len(POLL_STAGES)].view(
                    torch.int32
                )
                if "poll_abort" in op.scr_layout:
                    oa = op.scr_layout["poll_abort"]
                    self.aborts[(S, i)] = op.scratch[oa : oa + 4].view(torch.int32)
                if i == 0 and hasattr(op, "step"):
                    self.steps[S] = op.step
                if getattr(op, "poll_xrank", None) is not None:
                    self.xviews[(S, i)] = op.poll_xrank
        n = len(POLL_STAGES)
        self.host = torch.zeros(len(self.views), n, dtype=torch.int32, pin_memory=True)
        # rank 0: ranks that flagged an expired wait into this rank's buffer
        npes = max((v.numel() for v in self.xviews.values()), default=0)
        self.host_x = (
            torch.zeros(len(self.xviews), npes, dtype=torch.int32, pin_memory=True)
            if npes
            else None
        )
        # optional device non-finite counter (device_nonfinite + failstop_nonfinite)
        nf = getattr(lv, "dev_nonfinite", None)
        self.nonfinite = (
            nf
            if (nf is not None and getattr(lv.cfg, "failstop_nonfinite", False))
            else None
        )
        self.host_nf = (
            torch.zeros(1, dtype=torch.int32, pin_memory=True)
            if self.nonfinite is not None
            else None
        )
        self.event: torch.cuda.Event | None = None
        self.checks = 0
        self.tripped = None
        self.incidents: list[
            dict
        ] = []  # warn mode: [{check, what, abort_marks, steps}] (first 64)
        self.n_incidents = 0

    def describe(self, bad) -> list:
        sizes = list(self.views)
        out = set()
        for i, j in bad:
            S, rt = sizes[i]
            out.add(f"S{S}{'' if rt == 0 else f'/rt{rt}'}:{self.stages[j]}")
        return sorted(out)

    def describe_x(self, bad) -> list:
        sizes = list(self.xviews)
        out = set()
        for i, j in bad:
            S, rt = sizes[i]
            out.add(f"rank{j}@S{S}{'' if rt == 0 else f'/rt{rt}'}")
        return sorted(out)

    def _incident(self, what, kind="expired kernel polls"):
        """Slow path only (an expiry happened): read the abort marks / step counters
        synchronously."""
        marks = {
            f"S{k[0]}/{k[1]}": int(v[0]) - 1
            for k, v in self.aborts.items()
            if int(v[0]) != 0
        }
        steps = {S: int(t[0]) for S, t in self.steps.items()}
        msg = (
            f"mono live rank {self.rank}: {kind} {what} (expired at kernel step "
            f"{marks}; step counters now {steps}; check {self.checks}) -- outputs of "
            "those steps are wrong"
        )
        if self.mode == "warn":
            self.n_incidents += 1
            if len(self.incidents) < 64:
                self.incidents.append(
                    dict(check=self.checks, what=what, abort_steps=marks, steps=steps)
                )
            logger.error(
                "%s [MONO_LIVE_FAILSTOP=warn: continuing, incident %d]",
                msg,
                self.n_incidents,
            )
            for v in self.views.values():
                v.zero_()
            for v in list(self.aborts.values()) + list(self.xviews.values()):
                v.zero_()
            self.host.zero_()
            if self.host_x is not None:
                self.host_x.zero_()
            return
        self.tripped = what
        if self.rank != 0:
            _fail_hard(msg)
        raise output_rank_failstop(
            f"{msg} -> fail-stop (outputs since the previous check may be wrong)"
        )

    def after_step(self):
        if self.event is not None and self.event.query():
            self.checks += 1
            bad = self.host.nonzero().tolist()
            if bad:
                self._incident(self.describe(bad))
            if self.host_x is not None:
                bad_x = self.host_x.nonzero().tolist()
                if bad_x:
                    self._incident(
                        self.describe_x(bad_x), "expired kernel polls on peer"
                    )
            if self.host_nf is not None and int(self.host_nf[0]) > 0:
                n = int(self.host_nf[0])
                if self.mode == "warn" and self.nonfinite is not None:
                    self.nonfinite.zero_()
                    self.host_nf.zero_()
                self._incident(
                    ["nonfinite"], f"{n} mono step(s) with non-finite hidden states"
                )
            self.event = None
        if self.event is None:
            for i, v in enumerate(self.views.values()):
                self.host[i].copy_(v, non_blocking=True)
            if self.host_x is not None:
                for i, v in enumerate(self.xviews.values()):
                    self.host_x[i].copy_(v, non_blocking=True)
            if self.host_nf is not None:
                self.host_nf.copy_(self.nonfinite, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self.event = event
