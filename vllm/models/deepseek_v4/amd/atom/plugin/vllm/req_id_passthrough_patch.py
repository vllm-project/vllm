"""Expose the current step's request ids (CPU, batch-ordered) to ATOM builders.

The DeepSeek-V4 proxy metadata build needs a stable per-request key to assign a
state slot (its SWA ring + compressor state). Previously it derived that key
from ``block_table_tensor[:, 0]`` with a ``.cpu()`` copy, which forces a host<->
device sync and leaves a large bubble on the decode stream even though the copy
itself is tiny.

vLLM already has the canonical, host-resident key: ``input_batch.req_ids``. By
the time attention metadata is built it has been reordered together with the
block table / seq_lens rows (``InputBatch.swap_states``), so ``req_ids[i]``
lines up with row ``i`` of every per-request tensor.

This patch wraps ``GPUModelRunner._build_attention_metadata`` -- the method that
constructs ``CommonAttentionMetadata`` *and* drives ``builder.build()`` in one
synchronous, single-threaded call -- to snapshot ``req_ids`` into a thread-local
for the duration of that call. ATOM's V4 metadata builder reads it via
``get_current_req_ids()`` and keys slot allocation on it, with no D2H. All of
this lives in ATOM; no vLLM source is modified.
"""

from __future__ import annotations

import functools
import logging
import threading

logger = logging.getLogger("atom")

_req_id_local = threading.local()


def get_current_req_ids() -> list[str] | None:
    """Return the current step's batch-ordered request ids, or None.

    Valid only while ``GPUModelRunner._build_attention_metadata`` is on the
    stack (i.e. inside an attention metadata builder's ``build()``). Returns
    None otherwise, or if the pass-through patch was not applied -- callers must
    treat None as "fall back to the device-side key".
    """
    return getattr(_req_id_local, "req_ids", None)


def apply_vllm_req_id_passthrough_patch() -> bool:
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # pragma: no cover - import guard
        logger.debug(
            "ATOM vLLM req_id passthrough patch: GPUModelRunner unavailable (%s), "
            "skip",
            e,
        )
        return False

    original = getattr(GPUModelRunner, "_build_attention_metadata", None)
    if original is None or getattr(original, "_atom_req_id_passthrough_patched", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        prev = getattr(_req_id_local, "req_ids", None)
        try:
            # Snapshot now: req_ids is already batch-reordered (swap_states ran
            # in _prepare_inputs) so it aligns with the block-table rows the
            # builder sees. A copy keeps it stable even if the batch mutates
            # later in the step.
            _req_id_local.req_ids = list(self.input_batch.req_ids)
        except Exception:
            _req_id_local.req_ids = None
        try:
            return original(self, *args, **kwargs)
        finally:
            _req_id_local.req_ids = prev

    wrapped._atom_req_id_passthrough_patched = True  # type: ignore[attr-defined]
    GPUModelRunner._build_attention_metadata = wrapped
    logger.info(
        "ATOM plugin: patched vLLM GPUModelRunner._build_attention_metadata to "
        "expose batch-ordered req_ids to ATOM metadata builders (removes the "
        "block-table D2H in DeepSeek-V4 slot assignment)"
    )
    return True
