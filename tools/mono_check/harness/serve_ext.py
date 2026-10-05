# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving-path worker extension for the live GLM-5.2 MonoKernel (``vllm serve``).

Runtime mechanism (no vLLM file modified, nothing installed into site-packages):

    MONO_LIVE_PREINSTALL='<LiveConfig JSON>' VLLM_SERVER_DEV_MODE=1 \
    vllm serve <ckpt> ... --worker-extension-cls \
        tools.mono_check.harness.serve_ext.MonoServeWorkerExtension

* ``--worker-extension-cls`` is resolved by ``WorkerWrapperBase.init_worker`` inside
  every multiproc-executor worker process (the API-server and engine-core processes
  never import it). Importing this module imports ``live_worker_ext``, whose import-time
  hook wraps ``Worker.compile_or_warm_up_model`` so that each worker installs live mode
  (weights, ops, warm-up launch + barrier) right after model load and BEFORE vLLM's
  warm-up / HIP-graph capture. Env vars reach the workers because they are spawned from
  the engine core, which is spawned from the API server.
* ``VLLM_SERVER_DEV_MODE=1`` exposes ``POST /collective_rpc``; ``{"method":
  "mono_health"}`` returns per-rank health (see ``mono_health``) so a client can watch
  poll errors, the kernel's device step counters and the per-rank dispatch history while
  serving.

Rank-uniformity instrumentation:
``GPUModelRunner._determine_batch_execution_and_padding`` (called once per scheduled
step and per dummy/capture run, on every rank) is wrapped to fold (cudagraph mode,
padded tokens, real tokens, reqs, uniform) into a per-rank rolling hash and counters.
Equal hashes across ranks == every rank took the same execution-mode / padding sequence.
Together with equal kernel step counters (``op.step``, the mailbox epoch source, one per
kernel width) this checks the tag protocol end to end, also under graph replay (the
counters are device tensors advanced inside the captured graphs). """

from __future__ import annotations

import contextlib
import hashlib
import struct

from tools.mono_check.harness import (
    live_worker_ext as _lwe,  # noqa: F401 (import-time hook)
)
from tools.mono_check.harness.live_worker_ext import MonoLiveWorkerExtension

_DISPATCH = dict(calls=0, by_mode={}, hash="0" * 16, eager_decode_only=0)


def _record(mode, desc, num_tokens, num_reqs, max_q):
    d = _DISPATCH
    name = getattr(mode, "name", str(mode))
    padded = int(getattr(desc, "num_tokens", -1) or -1)
    uniform = bool(getattr(desc, "uniform", False))
    d["calls"] += 1
    key = f"{name}:{'U' if uniform else 'M'}"
    d["by_mode"][key] = d["by_mode"].get(key, 0) + 1
    h = hashlib.blake2b(bytes.fromhex(d["hash"]), digest_size=8)
    h.update(name.encode())
    h.update(
        struct.pack(
            "<qqqq?", padded, int(num_tokens), int(num_reqs), int(max_q), uniform
        )
    )
    d["hash"] = h.hexdigest()


def _install_dispatch_recorder():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig = GPUModelRunner._determine_batch_execution_and_padding
    if getattr(orig, "_mono_wrapped", False):
        return

    def wrapped(
        self,
        num_tokens,
        num_reqs,
        num_scheduled_tokens_np,
        max_num_scheduled_tokens,
        use_cascade_attn,
        *args,
        **kwargs,
    ):
        out = orig(
            self,
            num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            *args,
            **kwargs,
        )
        # instrumentation must never break serving
        with contextlib.suppress(Exception):
            _record(out[0], out[1], num_tokens, num_reqs, max_num_scheduled_tokens)
        return out

    wrapped._mono_wrapped = True
    GPUModelRunner._determine_batch_execution_and_padding = wrapped


_install_dispatch_recorder()

# The poll-error watch hooks live in live_worker_ext (every live install), shared here
# for mono_health.
from tools.mono_check.harness.live_worker_ext import _WATCH  # noqa: E402


class MonoServeWorkerExtension(MonoLiveWorkerExtension):
    _mono_poll_counts: dict | None = None

    def mono_health(self) -> dict:
        """Cheap per-rank health snapshot (synchronizes the device once)."""
        import torch

        from tools.mono_check.harness.hook import get_live

        vc = self.vllm_config
        sc, cc = vc.scheduler_config, vc.compilation_config
        lv = get_live()
        out = dict(
            rank=getattr(self, "rank", None),
            sched=dict(
                async_scheduling=sc.async_scheduling,
                chunked_prefill=getattr(sc, "enable_chunked_prefill", None),
                max_num_batched_tokens=sc.max_num_batched_tokens,
                max_num_seqs=sc.max_num_seqs,
                cudagraph_mode=str(cc.cudagraph_mode),
                capture_sizes=list(cc.cudagraph_capture_sizes or []),
                prefix_caching=vc.cache_config.enable_prefix_caching,
            ),
            dispatch=dict(_DISPATCH, by_mode=dict(_DISPATCH["by_mode"])),
            installed=lv is not None,
        )
        if lv is None:
            return out
        torch.accelerator.synchronize()
        if self._mono_poll_counts is None:
            self._mono_poll_counts = {}
        cnt = self._mono_poll_counts
        steps = {}
        for S in lv.sizes:
            owners = (
                lv.runtime_ops(S)
                if hasattr(lv, "runtime_ops")
                else [lv.ops[(lv.first, S)]]
            )
            # the fused indexer adds a second runtime per width
            for i, op in enumerate(owners):
                # Report only (sticky words, never cleared here): the fail-stop watch
                # must still see them.
                for stage in op.poll_error(clear=False):
                    cnt[f"S{S}{'' if i == 0 else f'/rt{i}'}:{stage}"] = 1
            steps[S] = int(owners[0].step.item())
        st = lv.stats
        out.update(
            enabled=lv.enabled,
            disabled_reason=lv.disabled_reason,
            poll_error_counts=dict(cnt),
            poll_error_total=sum(cnt.values()),
            kernel_step_counters=steps,
            dev_mono_tokens=int(lv.dev_mono_tokens.item()),
            dev_mono_steps=int(lv.dev_mono_steps.item()),
            dev_nonfinite_steps=(
                None
                if getattr(lv, "dev_nonfinite", None) is None
                else int(lv.dev_nonfinite.item())
            ),
            host=dict(
                steps_mono=st["steps_mono"],
                steps_fallback_decode=st["steps_fallback_decode"],
                decode_tokens=st["decode_tokens"],
                mono_tokens=st["mono_tokens"],
                fallback_reasons=dict(st["fallback_reasons"]),
                mono_steps_by_S=dict(st["mono_steps_by_S"]),
                expired=st["expired"][:5],
                nonfinite_steps=st["nonfinite_steps"],
            ),
            mem_allocated_gib=torch.accelerator.memory_allocated() / 2**30,
            failstop_checks=None if _WATCH["obj"] is None else _WATCH["obj"].checks,
            failstop_mode=None if _WATCH["obj"] is None else _WATCH["obj"].mode,
            poll_incidents=None if _WATCH["obj"] is None else _WATCH["obj"].n_incidents,
            poll_incident_log=None
            if _WATCH["obj"] is None
            else list(_WATCH["obj"].incidents),
            inband_trips=_WATCH.get("inband_trips", 0),
        )
        return out
