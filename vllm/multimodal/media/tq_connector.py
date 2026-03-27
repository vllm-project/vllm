# SPDX-License-Identifier: Apache-2.0
"""
TransferQueue MediaConnector for vLLM.

Registers a "tq" connector into vLLM's MEDIA_CONNECTOR_REGISTRY so that
vLLM can resolve ``tq://<partition_id>/<global_index>`` URLs transparently,
just like ``http://`` or ``data:`` URLs.

Usage:
    1. Set env var: ``VLLM_MEDIA_CONNECTOR=tq``
    2. Pass image URLs like ``tq://mm_images/42`` in chat completion requests
       or directly in ``image_data`` of ``generate()`` calls.
    3. vLLM will fetch the image bytes from TransferQueue and decode them.

The TQ client is lazily initialised on the first ``tq://`` request inside
the **rank-0 vLLM server process** (the Ray actor that owns the engine).
Connection details are read from environment variables injected by the
verl trainer at launch time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, TypeVar

import numpy as np
from PIL import Image
from urllib3.util import parse_url

from .connector import MEDIA_CONNECTOR_REGISTRY, MediaConnector
from .base import MediaIO

logger = logging.getLogger(__name__)

_M = TypeVar("_M")

# ---------------------------------------------------------------------------
# Module-level TQ client singleton (one per process)
# ---------------------------------------------------------------------------
_tq_client = None
_tq_client_lock = threading.Lock()

# Environment variable names used to pass TQ connection info from the trainer
# process into the vLLM server Ray actor process.
_ENV_TQ_CONTROLLER_INFO = "VERL_TQ_CONTROLLER_INFO"
_ENV_TQ_STORAGE_UNIT_INFOS = "VERL_TQ_STORAGE_UNIT_INFOS"
_ENV_TQ_STORAGE_BACKEND = "VERL_TQ_STORAGE_BACKEND"


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
        # Double-check after acquiring the lock.
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

        import base64
        import pickle

        from transfer_queue import AsyncTransferQueueClient

        controller_info = pickle.loads(base64.b64decode(controller_info_b64))
        storage_backend = os.environ.get(_ENV_TQ_STORAGE_BACKEND, "AsyncSimpleStorageManager")

        storage_unit_infos = None
        storage_unit_infos_b64 = os.environ.get(_ENV_TQ_STORAGE_UNIT_INFOS)
        if storage_unit_infos_b64:
            storage_unit_infos = pickle.loads(base64.b64decode(storage_unit_infos_b64))

        client_id = f"vllm_tq_media_{os.getpid()}"
        client = AsyncTransferQueueClient(client_id, controller_info)

        # ``initialize_storage_manager`` expects a config-like object with
        # attributes ``controller_info``, ``storage_unit_infos``, etc.
        from types import SimpleNamespace

        tq_config = SimpleNamespace(
            storage_backend=storage_backend,
            controller_info=controller_info,
            storage_unit_infos=storage_unit_infos,
        )
        client.initialize_storage_manager(
            manager_type=storage_backend,
            config=tq_config,
        )

        _tq_client = client
        logger.info("TQMediaConnector: TQ client initialised (pid=%s)", os.getpid())

    return _tq_client


def _parse_tq_url(url: str) -> tuple[str, int]:
    """Parse ``tq://<partition_id>/<global_index>`` into components.

    Returns:
        (partition_id, global_index)
    """
    url_spec = parse_url(url)
    # url_spec.host = partition_id, url_spec.path = /<global_index>
    partition_id = url_spec.host or ""
    path = (url_spec.path or "").lstrip("/")
    if not partition_id or not path:
        raise ValueError(
            f"Invalid tq:// URL: {url!r}.  "
            "Expected format: tq://<partition_id>/<global_index>"
        )
    global_index = int(path)
    return partition_id, global_index


# ---------------------------------------------------------------------------
# Connector implementation
# ---------------------------------------------------------------------------


@MEDIA_CONNECTOR_REGISTRY.register("tq")
class TQMediaConnector(MediaConnector):
    """MediaConnector that additionally supports the ``tq://`` URL scheme.

    For ``http://``, ``data:`` and ``file:`` URLs the standard
    :class:`MediaConnector` logic is used unchanged.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # -- sync path (used by MultiModalContentParser) -----------------------

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:
        url_spec = parse_url(url)
        if url_spec.scheme == "tq":
            return self._load_from_tq_sync(url, media_io)
        return super().load_from_url(url, media_io, fetch_timeout=fetch_timeout)

    # -- async path (used by AsyncMultiModalContentParser) -----------------

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:
        url_spec = parse_url(url)
        if url_spec.scheme == "tq":
            return await self._load_from_tq_async(url, media_io)
        return await super().load_from_url_async(
            url, media_io, fetch_timeout=fetch_timeout
        )

    # -- TQ-specific helpers -----------------------------------------------

    def _load_from_tq_sync(self, url: str, media_io: MediaIO[_M]) -> _M:
        """Synchronous fetch from TransferQueue."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._load_from_tq_async(url, media_io))
        finally:
            loop.close()

    async def _load_from_tq_async(self, url: str, media_io: MediaIO[_M]) -> _M:
        """Asynchronous fetch from TransferQueue.

        The image is stored in TQ as a ``pixel_data`` field — a uint8 tensor
        of shape ``[H, W, C]``.  We reconstruct it as a PIL Image and pass it
        through ``media_io`` for mode conversion (e.g. RGBA → RGB).
        """
        partition_id, global_index = _parse_tq_url(url)
        client = _init_tq_client()

        # Retrieve the metadata for this single image sample, then fetch data.
        metadata = await client.async_kv_retrieve_meta(
            keys=[str(global_index)],
            partition_id=partition_id,
            create=False,
        )
        td = await client.async_get_data(metadata)

        # Expect a ``pixel_data`` field: uint8 tensor [H, W, C]
        pixel_tensor = td["pixel_data"][0]  # first (and only) sample
        pixel_np = pixel_tensor.numpy()

        # Retrieve image dimensions from metadata (stored as custom_meta)
        height, width = pixel_np.shape[0], pixel_np.shape[1]
        channels = pixel_np.shape[2] if pixel_np.ndim == 3 else 1
        mode = "RGB" if channels == 3 else ("RGBA" if channels == 4 else "L")

        image = Image.fromarray(pixel_np, mode=mode)

        # Run through media_io for mode conversion / wrapping with MediaWithBytes
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        return media_io.load_bytes(image_bytes)
