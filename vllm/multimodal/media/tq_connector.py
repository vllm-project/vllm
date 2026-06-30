# SPDX-License-Identifier: Apache-2.0
"""
TransferQueue MediaConnector for vLLM.

Registers a "tq" connector into vLLM's MEDIA_CONNECTOR_REGISTRY so that
vLLM can resolve ``tq://<partition_id>/<batch_key>/<index>`` URLs
transparently, just like ``http://`` or ``data:`` URLs.

Usage:
    1. Set env var: ``VLLM_MEDIA_CONNECTOR=tq``
    2. Pass image URLs like ``tq://mm_images/<batch_key>/0`` in chat
       completion requests or directly in ``image_data`` of ``generate()``
       calls.
    3. vLLM will fetch the image bytes from TransferQueue and decode them.

The TQ client is lazily initialised on the first ``tq://`` request inside
the **rank-0 vLLM server process** (the Ray actor that owns the engine).
Connection details are read from environment variables injected by the
verl trainer at launch time.

Storage format
--------------
Images are stored in **batched** form by ``verl.utils.tq_multimodal``:

*  A single TQ KV entry per batch, keyed by ``batch_key`` (a uuid4 hex).
*  Fields inside the entry:

   - ``pixel_flat``  – ``[1, total_elements]`` contiguous uint8 tensor of
     all images' pixels concatenated.
   - ``shapes``      – ``[1, N, 3]`` int64 tensor of ``[H, W, C]`` per image.
   - ``offsets``     – ``[1, N]`` int64 tensor of byte offsets into
     ``pixel_flat`` for each image.

The ``<index>`` in the URL selects which image within the batch to decode.

Performance
-----------
*  **No PNG round-trip** — pixel data from TQ is converted directly to a
   PIL Image via ``Image.fromarray`` and wrapped in ``MediaWithBytes``
   without an intermediate PNG/BMP encode + decode cycle.  The
   ``original_bytes`` field of ``MediaWithBytes`` is populated with a
   cheap uncompressed BMP encoding (just a header + memcpy) instead of
   a CPU-intensive PNG compression.

*  **Per-instance batch cache** — when multiple ``tq://`` URLs in the
   same request share the same ``batch_key`` (the common case), the TQ
   entry is fetched once and reused.  ``AsyncMultiModalContentParser``
   creates one connector per request, so the cache is naturally scoped.

*  **Deadlock-free sync path** — the synchronous ``load_from_url`` runs
   the async logic on a dedicated daemon thread with its own event loop,
   avoiding deadlocks when called from within an already-running loop.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import os
import pickle
import threading
from io import BytesIO
from typing import Any, TypeVar

import numpy as np
from PIL import Image

from .base import MediaIO, MediaWithBytes
from .connector import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from .image import ImageMediaIO

logger = logging.getLogger(__name__)

_M = TypeVar("_M")

# ---------------------------------------------------------------------------
# Module-level TQ client singleton (one per process)
# ---------------------------------------------------------------------------
_tq_client = None
_tq_client_lock = threading.Lock()

_ENV_TQ_CONTROLLER_INFO = "VERL_TQ_CONTROLLER_INFO"
_ENV_TQ_STORAGE_UNIT_INFOS = "VERL_TQ_STORAGE_UNIT_INFOS"
_ENV_TQ_STORAGE_BACKEND = "VERL_TQ_STORAGE_BACKEND"


@dataclasses.dataclass(frozen=True)
class _TQStorageConfig:
    """Typed configuration object for ``initialize_storage_manager``."""

    storage_backend: str
    controller_info: Any
    storage_unit_infos: Any | None = None


# ---------------------------------------------------------------------------
# Serialisation / deserialisation
# ---------------------------------------------------------------------------


def _deserialize_env_var(b64_str: str) -> Any:
    """Deserialize a base64-encoded environment variable value.

    Prefers JSON for security (avoids arbitrary code execution).  Falls
    back to pickle for backward compatibility with older verl versions,
    but logs a warning so operators know they should upgrade.
    """
    raw = base64.b64decode(b64_str)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(
            "TQMediaConnector: env var uses pickle serialization. "
            "This is insecure — please upgrade verl to use JSON-based "
            "serialize_tq_info(). Falling back to pickle.loads()."
        )
        return pickle.loads(raw)  # noqa: S301


# ---------------------------------------------------------------------------
# TQ client lifecycle
# ---------------------------------------------------------------------------


def _init_tq_client():
    """Initialise the per-process TQ client singleton (thread-safe).

    Called lazily on the first ``tq://`` URL resolution.  All configuration
    is read from environment variables that the verl trainer injects into
    the Ray actor's ``runtime_env``.
    """
    global _tq_client
    if _tq_client is not None:
        return _tq_client

    with _tq_client_lock:
        if _tq_client is not None:
            return _tq_client

        controller_info_b64 = os.environ.get(_ENV_TQ_CONTROLLER_INFO)
        if controller_info_b64 is None:
            raise RuntimeError(
                f"TQMediaConnector requires the {_ENV_TQ_CONTROLLER_INFO} "
                "environment variable.  Make sure TransferQueue is initialised "
                "and the verl trainer injects the variable before launching "
                "the vLLM server."
            )

        from transfer_queue import AsyncTransferQueueClient

        controller_info = _deserialize_env_var(controller_info_b64)
        storage_backend = os.environ.get(
            _ENV_TQ_STORAGE_BACKEND, "AsyncSimpleStorageManager"
        )

        storage_unit_infos = None
        storage_unit_infos_b64 = os.environ.get(_ENV_TQ_STORAGE_UNIT_INFOS)
        if storage_unit_infos_b64:
            storage_unit_infos = _deserialize_env_var(storage_unit_infos_b64)

        client_id = f"vllm_tq_media_{os.getpid()}"
        client = AsyncTransferQueueClient(client_id, controller_info)

        tq_config = _TQStorageConfig(
            storage_backend=storage_backend,
            controller_info=controller_info,
            storage_unit_infos=storage_unit_infos,
        )
        try:
            client.initialize_storage_manager(
                manager_type=storage_backend,
                config=tq_config,
            )
        except Exception:
            logger.exception(
                "TQMediaConnector: failed to initialise TQ storage manager"
            )
            raise

        _tq_client = client
        logger.info("TQMediaConnector: TQ client initialised (pid=%s)", os.getpid())

    return _tq_client


# ---------------------------------------------------------------------------
# URL parsing & tensor helpers
# ---------------------------------------------------------------------------


def _parse_tq_url(url: str) -> tuple[str, str, int]:
    """Parse ``tq://<partition_id>/<batch_key>/<index>`` via simple split.

    Returns:
        (partition_id, batch_key, index)
    """
    if not url.startswith("tq://"):
        raise ValueError(
            f"Invalid tq:// URL: {url!r}.  "
            "Expected format: tq://<partition_id>/<batch_key>/<index>"
        )
    parts = url[5:].split("/")  # strip "tq://"
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid tq:// URL: {url!r}.  "
            "Expected format: tq://<partition_id>/<batch_key>/<index>"
        )
    return parts[0], parts[1], int(parts[2])


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Safely convert a tensor (possibly on GPU) to a numpy array."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.detach().cpu()
    return tensor.numpy()


def _extract_image_from_batch(td: Any, index: int, url: str) -> tuple[np.ndarray, str]:
    """Extract a single image's numpy pixels from a batched TQ entry.

    Returns:
        (pixel_np, mode) — numpy array in (H, W, C) or (H, W) layout,
        and the PIL mode string ("RGB", "RGBA", or "L").
    """
    for field in ("pixel_flat", "shapes", "offsets"):
        if field not in td:
            available = list(td.keys()) if hasattr(td, "keys") else "N/A"
            raise ValueError(
                f"TQ data for {url!r} does not contain '{field}' field.  "
                f"Available fields: {available}"
            )

    pixel_flat = td["pixel_flat"][0]  # [total_elements]
    shapes = td["shapes"][0]  # [N, 3]
    offsets_t = td["offsets"][0]  # [N]

    num_images = shapes.shape[0]
    if index < 0 or index >= num_images:
        raise IndexError(
            f"Image index {index} out of range for batch with "
            f"{num_images} images (URL: {url!r})"
        )

    h, w, c = (int(v) for v in shapes[index].tolist())
    offset = int(offsets_t[index].item())
    num_elements = h * w * c
    pixel_data = pixel_flat[offset : offset + num_elements].reshape(h, w, c)
    pixel_np = _tensor_to_numpy(pixel_data)

    if c == 1:
        pixel_np = pixel_np.squeeze(2)
    mode = "RGB" if c == 3 else ("RGBA" if c == 4 else "L")

    return pixel_np, mode


# ---------------------------------------------------------------------------
# Connector implementation
# ---------------------------------------------------------------------------


@MEDIA_CONNECTOR_REGISTRY.register("tq")
class TQMediaConnector(MediaConnector):
    """MediaConnector that supports the ``tq://`` URL scheme.

    For ``http://``, ``data:`` and ``file:`` URLs the standard
    :class:`MediaConnector` logic is used unchanged.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Per-instance cache: (partition_id, batch_key) → TensorDict.
        # vLLM creates one connector per request (via MEDIA_CONNECTOR_REGISTRY
        # .load in MultiModalContentParser.__init__), so the cache is
        # naturally request-scoped and needs no eviction policy.
        self._batch_cache: dict[tuple[str, str], Any] = {}

    # -- URL routing -------------------------------------------------------

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:
        if url.startswith("tq://"):
            return self._load_from_tq_sync(url, media_io)
        return super().load_from_url(url, media_io, fetch_timeout=fetch_timeout)

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:
        if url.startswith("tq://"):
            return await self._load_from_tq_async(url, media_io)
        return await super().load_from_url_async(
            url, media_io, fetch_timeout=fetch_timeout
        )

    # -- Batch cache -------------------------------------------------------

    async def _fetch_batch(self, partition_id: str, batch_key: str) -> Any:
        """Fetch a TQ batch entry, with per-instance caching (P1 fix).

        Multiple ``tq://`` URLs in the same request typically share one
        ``batch_key``.  This ensures the TQ round-trip happens only once.
        """
        cache_key = (partition_id, batch_key)
        cached = self._batch_cache.get(cache_key)
        if cached is not None:
            return cached

        client = _init_tq_client()
        metadata = await client.async_kv_retrieve_meta(
            keys=[batch_key],
            partition_id=partition_id,
            create=False,
        )
        td = await client.async_get_data(metadata)
        self._batch_cache[cache_key] = td
        return td

    # -- Sync path (P2 fix: deadlock-free) ---------------------------------

    def _load_from_tq_sync(self, url: str, media_io: MediaIO[_M]) -> _M:
        """Synchronous fetch via a dedicated daemon thread.

        Running the async coroutine on a **separate** thread with its own
        event loop avoids the deadlock that occurs when
        ``run_coroutine_threadsafe`` + ``future.result()`` is called from
        the same thread whose loop must execute the coroutine.
        """
        result_box: list = []
        exc_box: list = []

        def _worker() -> None:
            loop = asyncio.new_event_loop()
            try:
                result_box.append(
                    loop.run_until_complete(self._load_from_tq_async(url, media_io))
                )
            except BaseException as e:
                exc_box.append(e)
            finally:
                loop.close()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join()

        if exc_box:
            raise exc_box[0]
        return result_box[0]

    # -- Async path (P0 fix: no PNG round-trip) ----------------------------

    async def _load_from_tq_async(self, url: str, media_io: MediaIO[_M]) -> _M:
        """Fetch one image from TQ and return it via *media_io*.

        **Conversion pipeline (optimised for ImageMediaIO)**::

            TQ pixel_flat[offset:offset+n]    # zero-copy slice
                → .reshape(H, W, C)           # zero-copy view
                → _tensor_to_numpy()          # .detach().cpu().numpy()
                → Image.fromarray(np, mode)   # 1 memcpy (PIL owns buffer)
                → _convert_image_mode()       # mode conversion if needed
                → MediaWithBytes(img, bmp)    # BMP header + 1 memcpy

        Total: **2 memcpy** (numpy→PIL, PIL→BMP bytes).
        Old path had 4+ copies via PNG encode/decode.

        For non-image ``MediaIO`` types the fallback encodes to PNG bytes
        and calls ``media_io.load_bytes()``.
        """
        partition_id, batch_key, index = _parse_tq_url(url)

        # P1: batch cache — one TQ round-trip per batch_key per request.
        td = await self._fetch_batch(partition_id, batch_key)
        pixel_np, mode = _extract_image_from_batch(td, index, url)

        image = Image.fromarray(pixel_np, mode=mode)

        # ── Fast path for ImageMediaIO (the common case) ──────────────
        # Skip PNG encode/decode entirely.  Construct MediaWithBytes
        # directly with cheap BMP bytes for the original_bytes field.
        if isinstance(media_io, ImageMediaIO):
            converted = media_io._convert_image_mode(image)
            # BMP is uncompressed: just a ~54-byte header + raw pixels.
            # ~0 CPU cost vs PNG's ~50-200ms compression per image.
            buf = BytesIO()
            converted.save(buf, format="BMP")
            return MediaWithBytes(converted, buf.getvalue())  # type: ignore[return-value]

        # ── Fallback for other MediaIO types ──────────────────────────
        buf = BytesIO()
        image.save(buf, format="PNG")
        return media_io.load_bytes(buf.getvalue())
