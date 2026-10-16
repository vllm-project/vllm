# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build-time checks and the poll-error fail-stop of the GLM-5.2 specialized decode
(the config-only checks are ``runtime.unsupported``, the KV-bind checks
``MonoLayers.bind_caches``).

Build (``SpecializedDecode.__init__``): vLLM's RoPE cache covers max_model_len. Per step
(``PollErrorWatch``): a non-blocking, one-step-delayed fail-stop on expired kernel
polls, so the engine fails instead of serving wrong tokens."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def check_rope_table(backbone, cfg) -> None:
    """VLLM's cos_sin_cache covers the kernel's RoPE table (cfg.max_model_len >= vLLM's
    max_model_len, see ``runtime.decode_config``)."""
    rows = backbone.layers[cfg.layers[0]].self_attn.rotary_emb.cos_sin_cache.shape[0]
    if rows < cfg.max_model_len:
        raise RuntimeError(
            f"GLM-5.2 specialized decode: vLLM's cos_sin_cache has {rows} rows < "
            f"max_model_len {cfg.max_model_len}"
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
    except Exception:
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

    def __init__(self, mono):
        from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.layout import (
            POLL_STAGES,
        )

        self.mode = mono.cfg.failstop
        self.rank = mono.rank
        self.stages = POLL_STAGES
        n = len(POLL_STAGES)
        # (S, runtime index) -> device words
        self.views, self.aborts, self.xviews = {}, {}, {}
        self.steps = {}
        for S in mono.batch_sizes:
            for i, op in enumerate(mono.runtime_ops(S)):
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
        nf = mono.dev_nonfinite if mono.cfg.failstop_nonfinite else None
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
            f"GLM-5.2 specialized decode rank {self.rank}: {kind} {what} (expired at "
            f"kernel step {marks}; step counters now {steps}; check {self.checks}) -- "
            "outputs of those steps are wrong"
        )
        if self.mode != "warn":
            fail_stop(
                f"{msg} (outputs since the previous check may be wrong)", self.rank
            )
        self.n_incidents += 1
        logger.error(
            "%s [failstop=warn: continuing, incident %d]",
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
                stage = lambda w, j: f"{w}:{self.stages[j]}"
                self._incident(self._describe(list(self.views), bad, stage))
            bad = self.host_x.nonzero().tolist()
            if bad:
                peer = lambda w, j: f"rank{j}@{w}"
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
                    ["nonfinite"], f"{n} step(s) with non-finite hidden states"
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
