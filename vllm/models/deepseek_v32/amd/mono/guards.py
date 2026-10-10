# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install-time and per-step safety guards for the GLM-5.2 MonoKernel.

Install time: the RoPE table covers vLLM's max_model_len; every mono layer's [slots,
576] bf16 KV cache is < 4 GiB (32-bit byte offsets in the kernel); under FULL cudagraphs
the capture sizes equal the kernel widths (a capture size without a width would bake a
fallback). Per step (``PollErrorWatch``): a non-blocking, one-step-delayed fail-stop on
expired kernel polls, so the engine fails instead of serving wrong tokens."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.mono.envs import failstop_mode
from vllm.models.deepseek_v32.amd.mono.spec import KERNEL_WIDTHS

logger = init_logger(__name__)

KV_LIMIT_BYTES = 1 << 32


def check_before_install(model, cfg, vc, check_kv: bool = True) -> None:
    """RoPE length and graph widths; with check_kv (caches bound) also the KV size and
    the MLA cache contract."""
    mml = vc.model_config.max_model_len
    if cfg.max_model_len < mml:
        logger.warning(
            "mono guards: raising RoPE table length %d -> vLLM max_model_len %d",
            cfg.max_model_len,
            mml,
        )
        cfg.max_model_len = int(mml)
    rows = model.model.layers[cfg.layers[0]].self_attn.rotary_emb.cos_sin_cache.shape[0]
    if rows < cfg.max_model_len:
        raise RuntimeError(
            f"mono guards: vLLM cos_sin_cache has {rows} rows < max_model_len "
            f"{cfg.max_model_len}"
        )
    for L in cfg.layers if check_kv else ():
        kv = model.model.layers[L].self_attn.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[0]
        if kv is None or kv.numel() == 0:
            raise RuntimeError(
                f"mono guards: layer {L} has no KV cache bound at install time"
            )
        nbytes = kv.numel() * kv.element_size()
        if nbytes >= KV_LIMIT_BYTES:
            raise RuntimeError(
                f"mono guards: layer {L} KV cache is {nbytes / 2**30:.2f} GiB >= 4 "
                "GiB; the kernel's 32-bit buffer offsets would wrap. Bound it (e.g. "
                "--num-gpu-blocks-override) or do not install mono."
            )
    if check_kv and cfg.early_cache_checks:
        check_cache_contract(model, cfg)
    why = graph_width_mismatch(cfg.sizes, vc)
    if why is not None:
        raise RuntimeError(f"mono guards: {why}")


def full_cudagraphs(vc) -> bool:
    """FULL (whole-forward) cudagraphs: replays run no Python, so the per-step dispatch
    decision and every flag it reads are baked in at capture."""
    mode = getattr(getattr(vc, "compilation_config", None), "cudagraph_mode", None)
    return mode is not None and bool(mode.has_full_cudagraphs())


def piecewise_graphs_only(vc) -> bool:
    """Decode steps captured as breakable PIECEWISE graphs only: those captures carry no
    attention metadata, so every captured step falls back."""
    mode = getattr(getattr(vc, "compilation_config", None), "cudagraph_mode", None)
    if mode is None or full_cudagraphs(vc) or getattr(mode, "name", "") == "NONE":
        return False
    from vllm import envs

    return bool(envs.VLLM_USE_BREAKABLE_CUDAGRAPH)


def graph_width_mismatch(widths, vc) -> str | None:
    """None when the FULL-cudagraph capture sizes equal the kernel widths (none above
    the widest), else why not and the configuration that would fit."""
    if not full_cudagraphs(vc):
        return None
    sizes = sorted(vc.compilation_config.cudagraph_capture_sizes or [])
    widths = sorted(widths)
    if not sizes or (
        [s for s in sizes if s <= widths[-1]] == widths and sizes[-1] <= widths[-1]
    ):
        return None
    fix = (
        f"run with cudagraph_capture_sizes={widths} (e.g. --compilation-config "
        f"'{{\"cudagraph_capture_sizes\": {widths}}}') and max_num_seqs <= "
        f"{widths[-1]}"
    )
    if set(sizes) <= set(KERNEL_WIDTHS):
        fix += (
            f', or set "sizes": {sizes} in the MonoKernel config to match the capture '
            "sizes"
        )
    return (
        f"FULL cudagraph capture sizes {sizes} must equal the kernel widths {widths}: "
        f"{fix}"
    )


def check_cache_contract(model, cfg) -> None:
    """The MLA-cache properties live._bind_caches asserts, checked at install (the block
    dim is checked against the step metadata: vLLM may split a KV block). Skipped for
    non-tensor stand-ins."""
    ptrs: dict[int, int] = {}
    for L in cfg.layers:
        kv = model.model.layers[L].self_attn.kv_cache
        if isinstance(kv, (list, tuple)):
            kv = kv[0]
        if not isinstance(kv, torch.Tensor):
            return
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


def check_after_install(lv, vc) -> None:
    mml = vc.model_config.max_model_len or lv.cfg.max_model_len
    if lv.cos.shape[0] < mml or lv.sin.shape[0] < mml:
        raise RuntimeError(
            f"mono guards: RoPE table {tuple(lv.cos.shape)} shorter than max_model_len "
            f"{mml}"
        )


# the output rank's first fail-stop message (None until one fires)
_FAILED: dict[str, str | None] = dict(msg=None)
# seconds the output rank lives on after its fail-stop, for the raised error to reach
# the engine through the worker's RPC reply
FAILSTOP_EXIT_GRACE_S = 5.0


def _fail_hard(msg: str):
    """Terminate this worker: a non-output TP rank's exception never reaches the
    engine, so raising would let the other ranks keep emitting tokens."""
    logger.critical("%s -> terminating this worker (non-output rank)", msg)
    os._exit(70)


def _arm_exit_watchdog(grace_s: float = FAILSTOP_EXIT_GRACE_S):
    """Exit this worker grace_s seconds from now, whatever its threads do: after a
    fail-stop the engine has usually queued the next step, whose collectives wait on
    the exited peers while the main thread spins in a device sync (no signal handling).
    faulthandler's watchdog is a C thread (no GIL): it dumps every stack, then
    _exit(1)s."""
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


def fail_stop(msg: str, rank: int):
    """Terminate the worker on non-output ranks; on the output rank latch the first
    fail-stop (arming the exit watchdog once) and raise."""
    if rank != 0:
        _fail_hard(msg)
    msg = f"{msg} -> fail-stop"
    if _FAILED["msg"] is None:
        _FAILED["msg"] = msg
        logger.error(
            "%s -> the output rank raises; this worker exits in %.0f s (the other "
            "ranks have exited, so it cannot run another step)",
            msg,
            FAILSTOP_EXIT_GRACE_S,
        )
        _arm_exit_watchdog()
    raise RuntimeError(msg)


class PollErrorWatch:
    """One-step-delayed, non-blocking fail-stop on expired kernel polls (one async D2H
    copy + event per step). Mode "raise": the output rank raises, the others exit;
    "warn": log, count, clear the device words and continue."""

    def __init__(self, lv):
        from vllm.models.deepseek_v32.amd.mono.kernel.glm.layout import POLL_STAGES

        self.mode = failstop_mode()
        self.rank = lv.rank
        self.stages = POLL_STAGES
        n = len(POLL_STAGES)
        # (S, runtime index) -> device words; the fused indexer adds a runtime
        self.views, self.aborts, self.xviews = {}, {}, {}
        self.steps = {}
        for S in lv.sizes:
            for i, op in enumerate(lv.runtime_ops(S)):
                off, oa = op.scr_layout["poll_err"], op.scr_layout["poll_abort"]
                self.views[(S, i)] = op.scratch[off : off + 4 * n].view(torch.int32)
                self.aborts[(S, i)] = op.scratch[oa : oa + 4].view(torch.int32)
                # rank 0: the ranks that flagged an expired wait into its buffer
                self.xviews[(S, i)] = op.poll_xrank
                if i == 0:
                    self.steps[S] = op.step
        npes = max(v.numel() for v in self.xviews.values())
        self.host = torch.zeros(len(self.views), n, dtype=torch.int32, pin_memory=True)
        self.host_x = torch.zeros(
            len(self.xviews), npes, dtype=torch.int32, pin_memory=True
        )
        # optional device non-finite counter (device_nonfinite + failstop_nonfinite)
        nf = lv.dev_nonfinite if lv.cfg.failstop_nonfinite else None
        self.nonfinite = nf
        self.host_nf = (
            None if nf is None else torch.zeros(1, dtype=torch.int32, pin_memory=True)
        )
        self.event: torch.cuda.Event | None = None
        self.checks = 0
        self.n_incidents = 0  # warn mode

    @staticmethod
    def _describe(keys, bad, fmt) -> list:
        out = set()
        for i, j in bad:
            S, rt = keys[i]
            out.add(fmt(f"S{S}/rt{rt}" if rt else f"S{S}", j))
        return sorted(out)

    def _incident(self, what, kind="expired kernel polls"):
        """Slow path only (an expiry happened): reads the abort marks / step counters
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
        if self.mode != "warn":
            fail_stop(
                f"{msg} (outputs since the previous check may be wrong)", self.rank
            )
        self.n_incidents += 1
        logger.error(
            "%s [MONO_LIVE_FAILSTOP=warn: continuing, incident %d]",
            msg,
            self.n_incidents,
        )
        for v in (*self.views.values(), *self.aborts.values(), *self.xviews.values()):
            v.zero_()
        self.host.zero_()
        self.host_x.zero_()

    def after_step(self):
        if self.event is not None and self.event.query():
            self.checks += 1
            bad = self.host.nonzero().tolist()
            if bad:
                stage = lambda w, j: f"{w}:{self.stages[j]}"  # noqa: E731
                self._incident(self._describe(list(self.views), bad, stage))
            bad = self.host_x.nonzero().tolist()
            if bad:
                peer = lambda w, j: f"rank{j}@{w}"  # noqa: E731
                self._incident(
                    self._describe(list(self.xviews), bad, peer),
                    "expired kernel polls on peer",
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
            for i, v in enumerate(self.xviews.values()):
                self.host_x[i].copy_(v, non_blocking=True)
            if self.host_nf is not None:
                self.host_nf.copy_(self.nonfinite, non_blocking=True)
            self.event = torch.cuda.Event()
            self.event.record()
