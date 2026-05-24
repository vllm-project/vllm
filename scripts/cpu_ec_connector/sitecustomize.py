# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sitecustomize for the CPU EC connector e2e test.

Auto-imported by every Python interpreter that starts with this dir on
PYTHONPATH (the test driver sets it for the spawned `vllm serve`
processes). With `EC_TEST_ROLE` unset this module is a no-op, so an
accidental PYTHONPATH leak into an unrelated shell does nothing.

The patches install INFO log lines around the EC connector and the
multimodal encoder forward so the driver can assert on log slices.
"""

import logging
import os
import sys


def _install_ec_test_patches(role, log):
    from vllm.distributed.ec_transfer.ec_connector.cpu import (
        scheduler as sched_mod,
    )
    from vllm.distributed.ec_transfer.ec_connector.cpu import (
        worker as worker_mod,
    )
    from vllm.model_executor.models import qwen2_5_vl as qwen_mod
    from vllm.v1.worker import gpu_model_runner as gmr_mod
    from vllm.v1.worker.gpu.mm import encoder_runner as enc_mod

    def _wrap(cls, attr, fn):
        orig = getattr(cls, attr)

        def wrapper(self, *args, **kwargs):
            return fn(orig, self, *args, **kwargs)

        setattr(cls, attr, wrapper)

    # Producer: each XferReq, each posted NIXL WRITE, each save.
    # A WRITE was posted iff _in_flight grew during the call (NACKs return
    # early without touching _in_flight; sweep_completions runs on the same
    # router thread, so the size comparison is race-free).
    def _on_handle(orig, self, identity, req):
        log.info("producer XferReq mm_hash=%s", req.mm_hash)
        before = len(self._in_flight)
        orig(self, identity, req)
        if len(self._in_flight) > before:
            log.info("producer NIXL WRITE mm_hash=%s", req.mm_hash)

    _wrap(sched_mod.ECCPUScheduler, "_handle_xfer_req", _on_handle)

    def _on_save(orig, self, encoder_cache, mm_hash, **kwargs):
        meta = kwargs.get("connector_metadata")
        n = len(meta.saves[mm_hash]) if (meta and mm_hash in meta.saves) else "?"
        log.info("producer save mm_hash=%s n_blocks=%s", mm_hash, n)
        return orig(self, encoder_cache, mm_hash, **kwargs)

    _wrap(worker_mod.ECCPUWorker, "save_caches", _on_save)

    # Consumer: each ack accepted, ok-or-not. ok=True adds mm_hash to
    # `_ready`; ok=False replaces the indices with a `None` tombstone in
    # `_remote_encodings`. Track the live (non-None, not-popped) set across
    # the call to catch both transitions.
    def _on_drain(orig, self):
        before_ready = set(self._ready)
        before_live = {h for h, v in self._remote_encodings.items() if v is not None}
        orig(self)
        after_live = {h for h, v in self._remote_encodings.items() if v is not None}
        for mm_hash in self._ready - before_ready:
            log.info("consumer XferAck ok=True mm_hash=%s", mm_hash)
        for mm_hash in (before_live - after_live) - self._ready:
            log.info("consumer XferAck ok=False mm_hash=%s", mm_hash)

    _wrap(sched_mod.ECCPUScheduler, "_drain_acks", _on_drain)

    def _on_load(orig, self, encoder_cache, **kwargs):
        meta = kwargs.get("connector_metadata")
        if meta is not None and meta.loads:
            log.info("consumer load mm_hashes=%s", list(meta.loads))
        return orig(self, encoder_cache, **kwargs)

    _wrap(worker_mod.ECCPUWorker, "start_load_caches", _on_load)

    # Both roles: encoder forward. Three call sites converge on
    # `embed_multimodal`; each is invoked every scheduler step but
    # short-circuits to an empty result on cache hit, so log only when the
    # result is non-empty. The negative assertion ("encoder did NOT run on
    # consumer") relies on this — logging on every entry would produce
    # false positives whenever the EC-served cache is queried.
    def _make_enc_hook(label):
        def hook(orig, self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            if result:
                log.info(
                    "%s ENCODER FORWARD via %s n_outputs=%d", role, label, len(result)
                )
            return result

        return hook

    _wrap(
        gmr_mod.GPUModelRunner,
        "_execute_mm_encoder",
        _make_enc_hook("_execute_mm_encoder"),
    )
    _wrap(
        enc_mod.EncoderRunner,
        "execute_mm_encoder",
        _make_enc_hook("execute_mm_encoder"),
    )
    _wrap(
        qwen_mod.Qwen2_5_VLForConditionalGeneration,
        "embed_multimodal",
        _make_enc_hook("embed_multimodal"),
    )


_ROLE = os.environ.get("EC_TEST_ROLE")
if _ROLE:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s ec-test [pid=%(process)d] %(message)s"
        )
    )
    _ec_log = logging.getLogger("ec-test")
    _ec_log.addHandler(_h)
    _ec_log.setLevel(logging.INFO)
    _ec_log.propagate = False
    try:
        _install_ec_test_patches(_ROLE, _ec_log)
        _ec_log.info("%s patches installed", _ROLE)
    except ImportError as e:
        # Probe interpreters that don't load vllm get here; nothing to do.
        _ec_log.info("%s sitecustomize: skipping (%s)", _ROLE, e)
