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

    # ------------------------------------------------------------------
    # Modular-scheduler patches — only available when scheduler/ package
    # is importable (it shadows scheduler.py only when the modular split
    # is in effect; on the monolithic branch scheduler.py wins and these
    # imports fail, so we skip gracefully rather than aborting all patches).
    # ------------------------------------------------------------------
    try:
        from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
            zmq_transport as _zmq_t,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.consumer import (
            ECCPUConsumer,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.producer import (
            ECCPUProducer,
        )

        ZmqConsumerTransport = _zmq_t.ZmqConsumerTransport
        ZmqProducerTransport = _zmq_t.ZmqProducerTransport

        # Producer: each XferReq, each posted NIXL WRITE.
        def _on_handle(orig, self, identity, req):
            log.info("producer XferReq mm_hash=%s", req.mm_hash)
            before = len(self._in_flight)
            orig(self, identity, req)
            if len(self._in_flight) > before:
                log.info("producer NIXL WRITE mm_hash=%s", req.mm_hash)

        _wrap(ECCPUProducer, "handle_xfer_req", _on_handle)

        # Consumer: each ack accepted, ok-or-not.
        def _on_drain(orig, self):
            before_ready = set(self._ready)
            before_live = {
                h for h, v in self._remote_encodings.items() if v is not None
            }
            orig(self)
            after_live = {h for h, v in self._remote_encodings.items() if v is not None}
            for mm_hash in self._ready - before_ready:
                log.info("consumer XferAck ok=True mm_hash=%s", mm_hash)
            for mm_hash in (before_live - after_live) - self._ready:
                log.info("consumer XferAck ok=False mm_hash=%s", mm_hash)

        _wrap(ECCPUConsumer, "drain_acks", _on_drain)

        def _on_get_or_add_peer(orig, self, addr, metadata):
            host, port = addr
            existed = addr in self._peer_pool
            existing = self._peer_pool.get(addr)
            stale = existing is not None and existing.nixl_metadata_bytes != metadata
            peer = orig(self, addr, metadata)
            if not existed:
                log.info(
                    "consumer peer_pool ADD host=%s port=%s agent=%s",
                    host,
                    port,
                    peer.nixl_agent_name,
                )
            elif stale:
                log.info(
                    "consumer peer_pool REPLACE host=%s port=%s agent=%s",
                    host,
                    port,
                    peer.nixl_agent_name,
                )
            return peer

        _wrap(ZmqConsumerTransport, "get_or_create_peer", _on_get_or_add_peer)

        def _on_send_xfer_acks_mod(orig, self, routes):
            for _, mm_hash, ok in routes:
                log.info("producer XferAck SEND ok=%s mm_hash=%s", ok, mm_hash)
            return orig(self, routes)

        _wrap(ZmqProducerTransport, "_send_xfer_acks", _on_send_xfer_acks_mod)

    except ImportError as e:
        log.info(
            "%s modular EC patches skipped; using monolithic fallbacks (%s)", role, e
        )

        # Monolithic ECCPUScheduler has these as private methods with the
        # same semantics as the modular split, but different names/signatures.

        def _on_handle_mono(orig, self, identity, req):
            log.info("producer XferReq mm_hash=%s", req.mm_hash)
            before = len(self._in_flight)
            orig(self, identity, req)
            if len(self._in_flight) > before:
                log.info("producer NIXL WRITE mm_hash=%s", req.mm_hash)

        _wrap(sched_mod.ECCPUScheduler, "_handle_xfer_req", _on_handle_mono)

        def _on_drain_mono(orig, self):
            before_ready = set(self._ready)
            before_live = {
                h for h, v in self._remote_encodings.items() if v is not None
            }
            orig(self)
            after_live = {h for h, v in self._remote_encodings.items() if v is not None}
            for mm_hash in self._ready - before_ready:
                log.info("consumer XferAck ok=True mm_hash=%s", mm_hash)
            for mm_hash in (before_live - after_live) - self._ready:
                log.info("consumer XferAck ok=False mm_hash=%s", mm_hash)

        _wrap(sched_mod.ECCPUScheduler, "_drain_acks", _on_drain_mono)

        def _on_get_or_add_peer_mono(orig, self, info):
            host = info["peer_host"]
            port = int(info["peer_port"])
            key = (host, port)
            existed = key in self._peer_pool
            old_agent = self._peer_pool[key].nixl_agent_name if existed else None
            peer = orig(self, info)
            if not existed:
                log.info(
                    "consumer peer_pool ADD host=%s port=%s agent=%s",
                    host,
                    port,
                    peer.nixl_agent_name,
                )
            elif old_agent != peer.nixl_agent_name:
                log.info(
                    "consumer peer_pool REPLACE host=%s port=%s agent=%s",
                    host,
                    port,
                    peer.nixl_agent_name,
                )
            return peer

        _wrap(sched_mod.ECCPUScheduler, "_get_or_add_peer", _on_get_or_add_peer_mono)

        def _on_sweep_completions_mono(orig, self):
            with self._lock:
                before_map = {xid: mh for xid, (_, mh, _) in self._in_flight.items()}
            orig(self)
            with self._lock:
                after_ids = set(self._in_flight.keys())
            for xid in set(before_map) - after_ids:
                log.info(
                    "producer sweep NIXL done xfer_id=%s mm_hash=%s",
                    xid,
                    before_map[xid],
                )

        _wrap(
            sched_mod.ECCPUScheduler, "_sweep_completions", _on_sweep_completions_mono
        )

        def _on_send_xfer_acks_mono(orig, self, routes, ok):
            for _, mm_hash in routes:
                log.info("producer XferAck SEND ok=%s mm_hash=%s", ok, mm_hash)
            return orig(self, routes, ok)

        _wrap(sched_mod.ECCPUScheduler, "_send_xfer_acks", _on_send_xfer_acks_mono)

    # ------------------------------------------------------------------
    # Worker patches (always installed)
    # ------------------------------------------------------------------

    def _on_save(orig, self, encoder_cache, mm_hash, **kwargs):
        meta = kwargs.get("connector_metadata")
        n = len(meta.saves[mm_hash]) if (meta and mm_hash in meta.saves) else "?"
        log.info("producer save mm_hash=%s n_blocks=%s", mm_hash, n)
        return orig(self, encoder_cache, mm_hash, **kwargs)

    _wrap(worker_mod.ECCPUWorker, "save_caches", _on_save)

    def _on_load(orig, self, encoder_cache, **kwargs):
        meta = kwargs.get("connector_metadata")
        if meta is not None and meta.loads:
            log.info("consumer load mm_hashes=%s", list(meta.loads))
        return orig(self, encoder_cache, **kwargs)

    _wrap(worker_mod.ECCPUWorker, "start_load_caches", _on_load)

    # ------------------------------------------------------------------
    # Scheduler patches (always installed — ECCPUScheduler is in both the
    # monolithic scheduler.py and the modular scheduler/__init__.py)
    # ------------------------------------------------------------------

    def _on_request_finished(orig, self, request):
        skip, params = orig(self, request)
        if params:
            first = next(iter(params.values()))
            log.info(
                "producer request_finished peer_host=%s peer_port=%s mm_hashes=%s",
                first.get("peer_host"),
                first.get("peer_port"),
                list(params.keys()),
            )
        return skip, params

    _wrap(sched_mod.ECCPUScheduler, "request_finished", _on_request_finished)

    def _on_ensure_cache(orig, self, request, num_computed_tokens):
        params = getattr(request, "ec_transfer_params", None) or {}
        if params:
            first = next(iter(params.values()))
            log.info(
                "consumer ensure_cache peer_host=%s peer_port=%s mm_hashes=%s",
                first.get("peer_host"),
                first.get("peer_port"),
                list(params.keys()),
            )
        return orig(self, request, num_computed_tokens)

    _wrap(sched_mod.ECCPUScheduler, "ensure_cache_available", _on_ensure_cache)

    # ------------------------------------------------------------------
    # Encoder forward patches (always installed)
    # ------------------------------------------------------------------

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
