# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sitecustomize for the CPU EC connector e2e test.

Auto-imported by every Python interpreter that starts with this dir on
PYTHONPATH (the test driver sets it for the spawned `vllm serve`
processes). With `EC_TEST_ROLE` unset this module is a no-op, so an
accidental PYTHONPATH leak into an unrelated shell does nothing.

The patches install INFO log lines around the EC connector and the
multimodal encoder forward. Those records go to two sinks: stderr (for a
human reading the pod log) and, when ``EC_TEST_EVENT_FILE`` is set, a JSONL
file the driver pulls synchronously. The driver asserts on the JSONL file —
never on the streamed log — so assertions don't race a log transport.
"""

import json
import logging
import os
import sys


class _JsonlEventHandler(logging.Handler):
    """Append one JSON object per record to a local event file.

    One ``O_APPEND`` write per record: writes this small are atomic on Linux,
    so vLLM's several processes (API server, EngineCore, workers) can share a
    single file without locking or interleaved lines.
    """

    def __init__(self, path):
        super().__init__()
        self._path = path

    def emit(self, record):
        try:
            line = (
                json.dumps(
                    {
                        "ts": record.created,
                        "pid": record.process,
                        "msg": record.getMessage(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)
        except Exception:
            self.handleError(record)


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
    # Session-layer patches (producer XferReq/grant, consumer read ok/fail,
    # peer-pool changes).  These target the session objects that carry the
    # NIXL peer-to-peer transfer state.
    # ------------------------------------------------------------------
    try:
        from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
            XferStatus as _XferStatus,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.session import (
            ConsumerSession as _ConsumerSession,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.session import (
            ProducerSession as _ProducerSession,
        )

        # Producer: log each inbound XferReq and whether it was granted.
        def _on_grant_or_nack(orig, self, req):
            log.info("producer XferReq mm_hash=%s", req.mm_hash)
            ack = orig(self, req)
            if ack.status == _XferStatus.OK:
                log.info("producer read granted mm_hash=%s", req.mm_hash)
            return ack

        _wrap(_ProducerSession, "_grant_or_nack", _on_grant_or_nack)

        # Consumer: log each completed or failed read as results are drained.
        def _on_take_results(orig, self):
            results = orig(self)
            for mm_hash in results.completed:
                log.info("consumer read ok mm_hash=%s", mm_hash)
            for mm_hash in results.tombstoned:
                log.info("consumer read failed mm_hash=%s", mm_hash)
            return results

        _wrap(_ConsumerSession, "take_results", _on_take_results)

        # Consumer: log ADD (first registration) or REPLACE (metadata changed
        # after producer restart) so tests can assert on peer-pool events.
        def _on_ensure_registered(orig, self, metadata, mem_descriptor):
            was_registered = self._nixl_agent_name is not None
            was_same = (
                self._nixl_metadata_bytes == metadata if was_registered else False
            )
            result = orig(self, metadata, mem_descriptor)
            host, port = self._addr
            if not was_registered:
                log.info(
                    "consumer peer_pool ADD host=%s port=%s agent=%s",
                    host,
                    port,
                    result,
                )
            elif not was_same:
                log.info(
                    "consumer peer_pool REPLACE host=%s port=%s agent=%s",
                    host,
                    port,
                    result,
                )
            return result

        _wrap(_ConsumerSession, "_ensure_registered", _on_ensure_registered)

    except (ImportError, AttributeError) as e:
        log.info("%s session-layer EC patches skipped (%s)", role, e)

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
    # Scheduler patches (always installed)
    # ------------------------------------------------------------------

    def _on_request_finished(orig, self, request):
        log.info(
            "producer request_finished DIAGNOSTIC req_id=%s mm_features=%s",
            request.request_id,
            [f.identifier for f in (request.mm_features or [])],
        )
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
    _EVENT_FILE = os.environ.get("EC_TEST_EVENT_FILE")
    if _EVENT_FILE:
        _ec_log.addHandler(_JsonlEventHandler(_EVENT_FILE))
    _ec_log.setLevel(logging.INFO)
    _ec_log.propagate = False
    try:
        _install_ec_test_patches(_ROLE, _ec_log)
        _ec_log.info("%s patches installed", _ROLE)
    except ImportError as e:
        # Probe interpreters that don't load vllm get here; nothing to do.
        _ec_log.info("%s sitecustomize: skipping (%s)", _ROLE, e)
