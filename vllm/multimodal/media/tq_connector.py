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
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import dataclasses
import json
import logging
import os
import pickle
import threading
from io import BytesIO
from typing import Any, TypeVar

from PIL import Image
from urllib3.util import parse_url

from .base import MediaIO
from .connector import MEDIA_CONNECTOR_REGISTRY, MediaConnector

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


@dataclasses.dataclass(frozen=True)
class _TQStorageConfig:
    """Typed configuration object for ``initialize_storage_manager``."""

    storage_backend: str
    controller_info: Any
    storage_unit_infos: Any | None = None


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


def _init_tq_client():
    """Initialise the per-process TQ client singleton (thread-safe).

    Called lazily on the first ``tq://`` URL resolution.  All configuration
    is read from environment variables that the verl trainer injects into
    the Ray actor's ``runtime_env``.

    Raises:
        RuntimeError: If required environment variables are missing or if
            TQ client initialisation fails.
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
        logger.info(
            "TQMediaConnector: TQ client initialised (pid=%s)", os.getpid()
        )

    return _tq_client


def _parse_tq_url(url: str) -> tuple[str, str, int]:
    """Parse ``tq://<partition_id>/<batch_key>/<index>`` into components.

    Returns:
        (partition_id, batch_key, index)
    """
    url_spec = parse_url(url)
    # url_spec.host = partition_id, url_spec.path = /<batch_key>/<index>
    partition_id = url_spec.host or ""
    path = (url_spec.path or "").lstrip("/")
    parts = path.split("/") if path else []
    if not partition_id or len(parts) != 2:
        raise ValueError(
            f"Invalid tq:// URL: {url!r}.  "
            "Expected format: tq://<partition_id>/<batch_key>/<index>"
        )
    batch_key = parts[0]
    index = int(parts[1])
    return partition_id, batch_key, index


def _tensor_to_numpy(tensor):
    """Safely convert a tensor (possibly on GPU) to a numpy array."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.detach().cpu()
    return tensor.numpy()


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
        """Synchronous fetch from TransferQueue.

        Reuses an existing event loop if one is running in the current
        thread (e.g. Ray's asyncio loop); otherwise creates a temporary
        loop.  This avoids the pitfall of ``asyncio.new_event_loop()``
        conflicting with TQ client internals bound to another loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside an async context (e.g. Ray actor).  Schedule the
            # coroutine on the *existing* loop via a concurrent.futures.Future
            # so that TQ client coroutines run on the same loop.
            future: concurrent.futures.Future = asyncio.run_coroutine_threadsafe(
                self._load_from_tq_async(url, media_io), loop
            )
            return future.result()
        else:
            # No running loop — safe to create a temporary one.
            tmp_loop = asyncio.new_event_loop()
            try:
                return tmp_loop.run_until_complete(
                    self._load_from_tq_async(url, media_io)
                )
            finally:
                tmp_loop.close()

    async def _load_from_tq_async(self, url: str, media_io: MediaIO[_M]) -> _M:
        """Asynchronous fetch from TransferQueue.

        Images are stored in batched form (see module docstring).  This
        method retrieves the batch entry and extracts the single image at
        the requested ``index``.
        """
        partition_id, batch_key, index = _parse_tq_url(url)
        client = _init_tq_client()

        # Retrieve the batch entry by its explicit KV key.
        metadata = await client.async_kv_retrieve_meta(
            keys=[batch_key],
            partition_id=partition_id,
            create=False,
        )
        td = await client.async_get_data(metadata)

        # Validate expected fields.
        for field in ("pixel_flat", "shapes", "offsets"):
            if field not in td:
                available = list(td.keys()) if hasattr(td, "keys") else "N/A"
                raise ValueError(
                    f"TQ data for {url!r} does not contain '{field}' field.  "
                    f"Available fields: {available}"
                )

        pixel_flat = td["pixel_flat"][0]   # [total_elements]
        shapes = td["shapes"][0]           # [N, 3]
        offsets_t = td["offsets"][0]        # [N]

        # Bounds check on the requested index.
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
            pixel_np = pixel_np.squeeze(2)  # restore (H, W) for grayscale
        mode = "RGB" if c == 3 else ("RGBA" if c == 4 else "L")

        image = Image.fromarray(pixel_np, mode=mode)

        # Run through media_io for mode conversion / wrapping with MediaWithBytes
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        return media_io.load_bytes(image_bytes)
