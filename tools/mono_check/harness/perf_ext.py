# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Perf A/B worker extension: per-step GPU timestamps (CUDA events) for decode-latency
measurement.

``Worker.execute_model`` is wrapped (when MONO_PERF_STEPS=1) to record one event on the
current stream after each scheduled step's forward has been enqueued, tagged with the
step's shape (total tokens, number of requests, any new/prefill requests). The time
between consecutive events of two back-to-back pure-decode steps is that decode step's
GPU wall time (forward + the previous step's sampling + any idle gap), independent of
async-scheduling CPU overlap. Inherits the serving extension (mono_health, dispatch
recorder, preinstall hook)."""

from __future__ import annotations

import os
import time

from tools.mono_check.harness.serve_ext import MonoServeWorkerExtension

_STEPS = dict(on=False, recs=[])


def _install_step_timer():
    if os.environ.get("MONO_PERF_STEPS", "0") != "1":
        return
    from vllm.v1.worker.gpu_worker import Worker

    orig = Worker.execute_model
    if getattr(orig, "_mono_perf_wrapped", False):
        return

    def execute_model(self, scheduler_output):
        t_in = time.time_ns() if _STEPS["on"] else 0
        out = orig(self, scheduler_output)
        if _STEPS["on"]:
            import torch

            so = scheduler_output
            n_tok = so.total_num_scheduled_tokens
            n_req = len(so.num_scheduled_tokens)
            new = len(so.scheduled_new_reqs)
            # pure decode: every request runs exactly its next token plus its MTP drafts
            # (verify step)
            spec = so.scheduled_spec_decode_tokens or {}
            is_dec = [
                n == 1 + len(spec.get(r, ()))
                for r, n in so.num_scheduled_tokens.items()
            ]
            pure = n_tok > 0 and new == 0 and all(is_dec)
            # decode / MTP-verify rows of this step, mixed steps included (prefill
            # chunks excluded)
            dec_rows = sum(
                n for (r, n), d in zip(so.num_scheduled_tokens.items(), is_dec) if d
            )
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            # host wall-clock (CLOCK_REALTIME, comparable across the ranks
            # of one host) at entry / after the step's launches were enqueued -- a rank
            # whose host enters late shows up as the max-min skew
            _STEPS["recs"].append(
                (ev, n_tok, n_req, new, pure, dec_rows, t_in, time.time_ns())
            )
        return out

    execute_model._mono_perf_wrapped = True
    Worker.execute_model = execute_model


_install_step_timer()


class MonoPerfWorkerExtension(MonoServeWorkerExtension):
    def mono_perf_start(self) -> bool:
        _STEPS["recs"] = []
        _STEPS["on"] = True
        return True

    def mono_perf_stop(self) -> dict:
        import torch

        _STEPS["on"] = False
        torch.accelerator.synchronize()
        recs = _STEPS["recs"]
        _STEPS["recs"] = []
        out = []
        for i in range(1, len(recs)):
            e0, e1 = recs[i - 1][0], recs[i][0]
            _, n_tok, n_req, new, pure, dec_rows, t_in, t_out = recs[i]
            prev_pure = recs[i - 1][4]
            out.append(
                dict(
                    ms=e0.elapsed_time(e1),
                    n_tok=n_tok,
                    n_req=n_req,
                    new=new,
                    pure=pure,
                    prev_pure=prev_pure,
                    dec_rows=dec_rows,
                    t_in=t_in,
                    t_out=t_out,
                )
            )
        return dict(rank=getattr(self, "rank", None), steps=out)
