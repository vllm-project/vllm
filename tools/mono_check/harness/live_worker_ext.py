# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension that installs the live MonoKernel dispatch inside each TP worker.

Use: ``LLM(...,
worker_extension_cls="tools.mono_check.harness.live_worker_ext.MonoLiveWorkerExtension")``
then ``llm.collective_rpc("mono_live_install", args=(cfg_dict,))``."""

from __future__ import annotations

from tools.mono_check.harness import live_selftest


class MonoLiveWorkerExtension:
    def mono_live_install(self, cfg: dict) -> dict:
        from tools.mono_check.harness.hook import install
        from vllm.models.deepseek_v32.amd.mono.live import LiveConfig

        cfg = dict(cfg)
        if "sizes" in cfg:
            cfg["sizes"] = tuple(cfg["sizes"])
        lv = install(self.model_runner.model, LiveConfig(**cfg), self.vllm_config)
        live_selftest.attach(lv)
        return dict(
            rank=lv.rank,
            layers=[lv.first, lv.last],
            sizes=lv.sizes,
            mem=lv.mem,
            indexer_layers=[L for L, v in lv.has_indexer.items() if v],
        )

    def mono_live_enable(self, on: bool) -> bool:
        """True when applied; False when nothing is installed or when refused: under
        FULL cudagraphs a state change is refused (captured graphs keep their decision;
        restart or re-capture) -- see MonoLive.set_enabled."""
        from tools.mono_check.harness.hook import get_live

        lv = get_live()
        return False if lv is None else lv.set_enabled(on)

    def mono_live_stats(self) -> dict:
        from tools.mono_check.harness.hook import get_live

        lv = get_live()
        return {} if lv is None else lv.get_stats()

    def mono_live_reset_stats(self) -> bool:
        from tools.mono_check.harness.hook import get_live

        lv = get_live()
        if lv is not None:
            lv.reset_stats()
        return True

    def mono_live_poll_errors(self) -> dict:
        from tools.mono_check.harness.hook import get_live

        lv = get_live()
        if lv is None:
            return {}
        # Report only (clear=False, as mono_health): the poll_err words are sticky so
        # the fail-stop watch (guards.PollErrorWatch, one step behind) still sees them;
        # clearing here could hide an expiry from it.
        return {
            S: [e for op in lv.runtime_ops(S) for e in op.poll_error(clear=False)]
            for S in lv.sizes
        }

    def mono_mem(self) -> dict:
        import torch

        free, total = torch.accelerator.get_memory_info()
        return dict(
            allocated_gib=torch.accelerator.memory_allocated() / 2**30,
            max_allocated_gib=torch.accelerator.max_memory_allocated() / 2**30,
            device_used_gib=(total - free) / 2**30,
        )


def _maybe_preinstall_hook():
    """HIP-graph mode: the kernel ops must exist before vLLM warms up / captures graphs,
    so when MONO_LIVE_PREINSTALL (a JSON LiveConfig) is set, wrap
    Worker.compile_or_warm_up_model to install live mode first (this module is imported
    in every worker at init)."""
    import json
    import os

    spec = os.environ.get("MONO_LIVE_PREINSTALL")
    if not spec:
        return
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker.compile_or_warm_up_model, "_mono_wrapped", False):
        return
    orig = Worker.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *args, **kwargs):
        from tools.mono_check.harness.hook import get_live, install
        from vllm.models.deepseek_v32.amd.mono.live import LiveConfig

        if get_live() is None:
            cfg = json.loads(spec)
            if "sizes" in cfg:
                cfg["sizes"] = tuple(cfg["sizes"])
            live_selftest.attach(
                install(self.model_runner.model, LiveConfig(**cfg), self.vllm_config)
            )
        return orig(self, *args, **kwargs)

    compile_or_warm_up_model._mono_wrapped = True
    Worker.compile_or_warm_up_model = compile_or_warm_up_model


_maybe_preinstall_hook()


_WATCH = dict(obj=None, inband=None, inband_trips=0)


def _live_watch():
    """The rank's PollErrorWatch for the installed live MonoLive (None when nothing is
    installed or the fail-stop is off). Created lazily on the first executed step after
    install."""
    from tools.mono_check.harness.hook import get_live
    from vllm.models.deepseek_v32.amd.mono.guards import PollErrorWatch, failstop_mode

    lv = get_live()
    if lv is None or failstop_mode() == "off":
        return None
    if _WATCH["obj"] is None or _WATCH["obj"].lv is not lv:
        _WATCH["obj"] = PollErrorWatch(lv)
    return _WATCH["obj"]


def _install_poll_failstop():
    """Poll-error fail-stop for every live install: a one-step-delayed, non-blocking
    watch after every execute_model on every rank (guards.PollErrorWatch), plus an
    in-band check on the output rank: its poll_err words (and the peers' expiry flags)
    are copied on the async output-copy stream right after the sampled tokens and
    checked in get_output before the tokens are returned, so an expiry never emits that
    step's tokens. MONO_LIVE_FAILSTOP=0 turns
    both off, =warn logs and continues (guards.failstop_mode)."""
    import os

    from vllm.models.deepseek_v32.amd.mono.guards import failstop_mode

    mode = failstop_mode()
    if mode == "off":
        if os.environ.get("MONO_LIVE_PREINSTALL"):
            import logging

            logging.getLogger(__name__).warning(
                "mono live: MONO_LIVE_FAILSTOP=0 -> expired kernel polls are NOT "
                "checked (wrong tokens possible)"
            )
        return
    from vllm.models.deepseek_v32.amd.mono.guards import failstop_error
    from vllm.v1.worker.gpu_worker import Worker

    if not getattr(Worker, "_mono_failstop_installed", False):
        orig, orig_sample = Worker.execute_model, Worker.sample_tokens

        def _refuse_after_failstop(what):
            # Output rank only (it alone latches): under async scheduling the engine
            # reads sample_tokens' reply, so an execute_model exception would be lost
            # and the engine would get an empty output (KeyError in the scheduler).
            err = failstop_error()
            if err is not None:
                raise RuntimeError(f"{err} [{what} refused after the fail-stop]")

        def execute_model(self, scheduler_output):
            # no further steps: the peers have exited, their collectives never finish
            _refuse_after_failstop("execute_model")
            out = orig(self, scheduler_output)
            w = _live_watch()
            if w is not None:
                w.after_step()
            return out

        def sample_tokens(self, grammar_output):
            _refuse_after_failstop("sample_tokens")
            return orig_sample(self, grammar_output)

        Worker.execute_model = execute_model
        Worker.sample_tokens = sample_tokens
        Worker._mono_failstop_installed = True
    _install_inband_check()


def _install_inband_check():
    import torch

    from vllm.v1.worker import gpu_model_runner as gmr

    cls = getattr(gmr, "AsyncGPUModelRunnerOutput", None)
    if cls is None or getattr(cls, "_mono_inband", False):
        return
    orig_init, orig_get = cls.__init__, cls.get_output

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._mono_poll = None
        w = _live_watch()
        if w is None or not w.views:
            return
        stream = (
            kwargs.get("async_output_copy_stream")
            if "async_output_copy_stream" in kwargs
            else args[4]
        )
        buf, bx = _WATCH.get("inband"), _WATCH.get("inband_x")
        if buf is None or buf.shape[0] != len(w.views):
            buf = _WATCH["inband"] = torch.zeros(
                len(w.views), len(w.stages), dtype=torch.int32, pin_memory=True
            )
            bx = _WATCH["inband_x"] = (
                None if w.host_x is None else torch.zeros_like(w.host_x).pin_memory()
            )
        # ordered after the forward: orig_init made the stream wait for it
        with torch.cuda.stream(stream):
            for i, v in enumerate(w.views.values()):
                buf[i].copy_(v, non_blocking=True)
            # peers' expiries, flagged into rank 0 by the kernel
            for i, v in enumerate(w.xviews.values()):
                bx[i].copy_(v, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        self._mono_poll = (buf, bx, ev, w)

    def get_output(self):
        out = orig_get(self)
        mp = getattr(self, "_mono_poll", None)
        if mp is not None:
            buf, bx, ev, w = mp
            # same stream as the token copy orig_get already waited for: ~free
            ev.synchronize()
            bad = buf.nonzero().tolist()
            bad_x = [] if bx is None else bx.nonzero().tolist()
            if bad or bad_x:
                _WATCH["inband_trips"] += 1
                what = w.describe(bad) + w.describe_x(bad_x)
                if w.mode == "warn":
                    import logging

                    logging.getLogger(__name__).error(
                        "mono live (in-band, output rank): expired kernel polls %s in "
                        "the step being returned "
                        "[warn: tokens emitted anyway]",
                        what,
                    )
                else:
                    from vllm.models.deepseek_v32.amd.mono.guards import (
                        output_rank_failstop,
                    )

                    raise output_rank_failstop(
                        f"mono live: expired kernel polls {what} in this step -> its "
                        "tokens are withheld (fail-stop)"
                    )
        return out

    cls.__init__ = __init__
    cls.get_output = get_output
    cls._mono_inband = True


_install_poll_failstop()
