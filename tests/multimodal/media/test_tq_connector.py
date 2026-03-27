# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TQMediaConnector (TransferQueue media connector).

These tests mock the ``transfer_queue`` dependency so they can run
without a real TQ cluster.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_image(width: int, height: int, mode: str = "RGB") -> Image.Image:
    """Create a deterministic test image."""
    if mode == "L":
        arr = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    elif mode == "RGBA":
        arr = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
    else:
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def _encode_images_as_tq_batch(images: list[Image.Image]) -> dict[str, torch.Tensor]:
    """Encode images in the batched TQ format (pixel_flat/shapes/offsets).

    Mirrors the logic in ``verl.utils.tq_multimodal.store_images_to_tq``.
    """
    flat_parts: list[torch.Tensor] = []
    shapes: list[list[int]] = []
    offsets: list[int] = []
    current_offset = 0

    for img in images:
        pixel_np = np.asarray(img)
        if pixel_np.ndim == 2:
            pixel_np = pixel_np[:, :, np.newaxis]
        t = torch.from_numpy(pixel_np.copy())
        flat = t.reshape(-1)
        flat_parts.append(flat)
        shapes.append(list(pixel_np.shape))
        offsets.append(current_offset)
        current_offset += flat.numel()

    pixel_flat = torch.cat(flat_parts)
    shapes_tensor = torch.tensor(shapes, dtype=torch.int64)
    offsets_tensor = torch.tensor(offsets, dtype=torch.int64)

    return {
        "pixel_flat": pixel_flat.unsqueeze(0),
        "shapes": shapes_tensor.unsqueeze(0),
        "offsets": offsets_tensor.unsqueeze(0),
    }


def _build_mock_tq_client(td: dict[str, torch.Tensor]) -> MagicMock:
    """Build a mock TQ client that returns *td* for any retrieve+get_data."""
    client = MagicMock()
    client.async_kv_retrieve_meta = AsyncMock(return_value="mock_meta")
    client.async_get_data = AsyncMock(return_value=td)
    return client


# ---------------------------------------------------------------------------
# _parse_tq_url tests
# ---------------------------------------------------------------------------


class TestParseTqUrl:
    """Tests for ``_parse_tq_url`` helper."""

    def test_valid_url(self):
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        partition, batch_key, index = _parse_tq_url(
            "tq://mm_images/abc123def/2"
        )
        assert partition == "mm_images"
        assert batch_key == "abc123def"
        assert index == 2

    def test_valid_url_zero_index(self):
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        partition, batch_key, index = _parse_tq_url(
            "tq://my_partition/batchkey/0"
        )
        assert partition == "my_partition"
        assert batch_key == "batchkey"
        assert index == 0

    def test_missing_partition(self):
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        with pytest.raises(ValueError, match="Invalid tq:// URL"):
            _parse_tq_url("tq:///batch_key/0")

    def test_missing_index(self):
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        with pytest.raises(ValueError, match="Invalid tq:// URL"):
            _parse_tq_url("tq://partition/batch_key")

    def test_too_many_segments(self):
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        with pytest.raises(ValueError, match="Invalid tq:// URL"):
            _parse_tq_url("tq://partition/batch_key/0/extra")

    def test_old_two_segment_format_rejected(self):
        """The old ``tq://partition/123`` 2-segment format must be rejected."""
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        with pytest.raises(ValueError, match="Invalid tq:// URL"):
            _parse_tq_url("tq://mm_images/42")

    def test_non_integer_index(self):
        from vllm.multimodal.media.tq_connector import _parse_tq_url

        with pytest.raises(ValueError):
            _parse_tq_url("tq://partition/batch_key/abc")


# ---------------------------------------------------------------------------
# _tensor_to_numpy tests
# ---------------------------------------------------------------------------


class TestTensorToNumpy:
    """Tests for ``_tensor_to_numpy`` helper."""

    def test_cpu_tensor(self):
        from vllm.multimodal.media.tq_connector import _tensor_to_numpy

        t = torch.tensor([1, 2, 3], dtype=torch.uint8)
        result = _tensor_to_numpy(t)
        np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.uint8))

    def test_tensor_with_grad(self):
        from vllm.multimodal.media.tq_connector import _tensor_to_numpy

        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = _tensor_to_numpy(t)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_plain_numpy_array(self):
        """Non-tensor objects with no .cpu() should still work."""
        from vllm.multimodal.media.tq_connector import _tensor_to_numpy

        arr = np.array([4, 5, 6])
        # numpy arrays have no .cpu, so the function should call .numpy() directly
        # This will fail because numpy arrays don't have .numpy()
        # but the function checks hasattr(tensor, "cpu") first
        result = _tensor_to_numpy(arr)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# _TQStorageConfig tests
# ---------------------------------------------------------------------------


class TestTQStorageConfig:
    """Ensure the dataclass behaves correctly."""

    def test_frozen(self):
        from vllm.multimodal.media.tq_connector import _TQStorageConfig

        cfg = _TQStorageConfig(
            storage_backend="AsyncSimpleStorageManager",
            controller_info={"host": "localhost"},
        )
        with pytest.raises(AttributeError):
            cfg.storage_backend = "other"  # type: ignore[misc]

    def test_optional_storage_unit_infos(self):
        from vllm.multimodal.media.tq_connector import _TQStorageConfig

        cfg = _TQStorageConfig(
            storage_backend="backend",
            controller_info={},
        )
        assert cfg.storage_unit_infos is None

        cfg2 = _TQStorageConfig(
            storage_backend="backend",
            controller_info={},
            storage_unit_infos={"unit": "info"},
        )
        assert cfg2.storage_unit_infos == {"unit": "info"}


# ---------------------------------------------------------------------------
# TQMediaConnector._load_from_tq_async tests (mocked TQ)
# ---------------------------------------------------------------------------


class TestLoadFromTqAsync:
    """Test the async TQ loading path with mocked TQ client."""

    @pytest.mark.asyncio
    async def test_single_rgb_image(self):
        """Store one RGB image, retrieve it, verify pixel equality."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(64, 48, "RGB")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()
        url = "tq://mm_images/batchkey123/0"

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")
            result = await connector._load_from_tq_async(url, media_io)

        # P0 fix: result should be MediaWithBytes, not raw PIL Image
        from vllm.multimodal.media.base import MediaWithBytes as MWB
        assert isinstance(result, MWB)
        assert result.media.mode == "RGB"
        assert result.media.size == (64, 48)
        # Pixel values must match exactly (no lossy encode/decode)
        np.testing.assert_array_equal(np.asarray(result.media), np.asarray(img))

    @pytest.mark.asyncio
    async def test_batch_of_images(self):
        """Store 3 images in one batch, retrieve each by index."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        images = [
            _make_test_image(32, 32, "RGB"),
            _make_test_image(64, 48, "RGB"),
            _make_test_image(16, 16, "RGB"),
        ]
        td = _encode_images_as_tq_batch(images)
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")

            for i, original in enumerate(images):
                url = f"tq://mm_images/batch42/{i}"
                result = await connector._load_from_tq_async(url, media_io)
                assert result.size == original.size
                np.testing.assert_array_equal(
                    np.asarray(result), np.asarray(original)
                )

    @pytest.mark.asyncio
    async def test_grayscale_image(self):
        """Grayscale images (mode L) should round-trip correctly."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(32, 32, "L")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="L")
            result = await connector._load_from_tq_async(
                "tq://part/key/0", media_io
            )

        assert result.mode == "L"
        assert result.size == (32, 32)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(img))

    @pytest.mark.asyncio
    async def test_rgba_image(self):
        """RGBA images should round-trip correctly."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(24, 24, "RGBA")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGBA")
            result = await connector._load_from_tq_async(
                "tq://part/key/0", media_io
            )

        assert result.mode == "RGBA"
        np.testing.assert_array_equal(np.asarray(result), np.asarray(img))

    @pytest.mark.asyncio
    async def test_index_out_of_range(self):
        """Requesting an index beyond the batch size should raise IndexError."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(8, 8, "RGB")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")

            with pytest.raises(IndexError, match="out of range"):
                await connector._load_from_tq_async(
                    "tq://part/key/5", media_io
                )

    @pytest.mark.asyncio
    async def test_missing_field_raises_value_error(self):
        """If the TQ entry lacks expected fields, raise ValueError."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        # Return a TensorDict-like dict missing the required fields
        td_missing = {"some_other_field": torch.zeros(1)}
        mock_client = _build_mock_tq_client(td_missing)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")

            with pytest.raises(ValueError, match="does not contain"):
                await connector._load_from_tq_async(
                    "tq://part/key/0", media_io
                )

    @pytest.mark.asyncio
    async def test_batch_cache_avoids_duplicate_fetches(self):
        """P1 fix: multiple URLs with same batch_key should fetch TQ only once."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        images = [
            _make_test_image(16, 16, "RGB"),
            _make_test_image(32, 32, "RGB"),
            _make_test_image(24, 24, "RGB"),
        ]
        td = _encode_images_as_tq_batch(images)
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")

            # Fetch all 3 images from the same batch.
            for i in range(3):
                await connector._load_from_tq_async(
                    f"tq://mm_images/samebatch/{i}", media_io
                )

        # TQ should have been called only once (meta + data).
        assert mock_client.async_kv_retrieve_meta.call_count == 1
        assert mock_client.async_get_data.call_count == 1

    @pytest.mark.asyncio
    async def test_different_batch_keys_fetched_separately(self):
        """Different batch_keys should each trigger a TQ fetch."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(8, 8, "RGB")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")

            await connector._load_from_tq_async(
                "tq://part/batch_a/0", media_io
            )
            await connector._load_from_tq_async(
                "tq://part/batch_b/0", media_io
            )

        # Two different batch_keys → two TQ fetches.
        assert mock_client.async_kv_retrieve_meta.call_count == 2
        assert mock_client.async_get_data.call_count == 2


# ---------------------------------------------------------------------------
# TQMediaConnector URL routing tests (tq:// vs http/data/file)
# ---------------------------------------------------------------------------


class TestUrlRouting:
    """Verify that tq:// URLs go through TQ, others fall back to parent."""

    @pytest.mark.asyncio
    async def test_tq_url_routed_to_tq_path(self):
        """A tq:// URL should invoke the TQ-specific async path."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        connector = TQMediaConnector()

        with patch.object(
            connector, "_load_from_tq_async", new_callable=AsyncMock
        ) as mock_tq:
            mock_tq.return_value = _make_test_image(8, 8)
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")
            await connector.load_from_url_async(
                "tq://partition/batchkey/0", media_io
            )
            mock_tq.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_url_falls_back_to_parent(self):
        """A data: URL should use parent MediaConnector logic."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        connector = TQMediaConnector()

        img = _make_test_image(4, 4, "RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        import pybase64 as base64

        b64 = base64.b64encode(buf.getvalue()).decode()
        data_url = f"data:image/png;base64,{b64}"

        from vllm.multimodal.media.image import ImageMediaIO

        media_io = ImageMediaIO(image_mode="RGB")
        result = await connector.load_from_url_async(data_url, media_io)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# _init_tq_client tests
# ---------------------------------------------------------------------------


class TestInitTqClient:
    """Tests for lazy TQ client initialisation."""

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Missing VERL_TQ_CONTROLLER_INFO should raise RuntimeError."""
        import vllm.multimodal.media.tq_connector as mod

        # Reset the module-level singleton
        monkeypatch.setattr(mod, "_tq_client", None)
        monkeypatch.delenv("VERL_TQ_CONTROLLER_INFO", raising=False)

        with pytest.raises(RuntimeError, match="VERL_TQ_CONTROLLER_INFO"):
            mod._init_tq_client()

    def test_returns_cached_client(self, monkeypatch: pytest.MonkeyPatch):
        """Once initialised, subsequent calls should return the cached client."""
        import vllm.multimodal.media.tq_connector as mod

        sentinel = MagicMock()
        monkeypatch.setattr(mod, "_tq_client", sentinel)

        assert mod._init_tq_client() is sentinel

    def test_init_failure_does_not_cache(self, monkeypatch: pytest.MonkeyPatch):
        """If initialize_storage_manager fails, _tq_client stays None."""
        import base64
        import json

        import vllm.multimodal.media.tq_connector as mod

        monkeypatch.setattr(mod, "_tq_client", None)
        controller_info = {"host": "localhost", "port": 12345}
        b64 = base64.b64encode(json.dumps(controller_info).encode()).decode()
        monkeypatch.setenv("VERL_TQ_CONTROLLER_INFO", b64)
        monkeypatch.delenv("VERL_TQ_STORAGE_UNIT_INFOS", raising=False)

        mock_client_cls = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.initialize_storage_manager.side_effect = (
            ConnectionError("TQ unavailable")
        )
        mock_client_cls.return_value = mock_client_instance

        with patch.dict(
            "sys.modules",
            {"transfer_queue": MagicMock(AsyncTransferQueueClient=mock_client_cls)},
        ):
            with pytest.raises(ConnectionError, match="TQ unavailable"):
                mod._init_tq_client()

        # Client should NOT have been cached
        assert mod._tq_client is None


# ---------------------------------------------------------------------------
# _load_from_tq_sync tests
# ---------------------------------------------------------------------------


class TestLoadFromTqSync:
    """Test the synchronous wrapper."""

    def test_sync_path_no_running_loop(self):
        """When no event loop is running, sync path creates a temp loop."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(16, 16, "RGB")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")
            result = connector._load_from_tq_sync(
                "tq://part/batch/0", media_io
            )

        assert isinstance(result, Image.Image)
        assert result.size == (16, 16)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(img))

    def test_sync_load_from_url(self):
        """The public load_from_url dispatches tq:// to the sync TQ path."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(8, 8, "RGB")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")
            result = connector.load_from_url(
                "tq://part/batch/0", media_io
            )

        assert isinstance(result, Image.Image)

    @pytest.mark.asyncio
    async def test_sync_from_async_context_no_deadlock(self):
        """P2 fix: sync path must not deadlock when called inside an async ctx."""
        from vllm.multimodal.media.tq_connector import TQMediaConnector

        img = _make_test_image(8, 8, "RGB")
        td = _encode_images_as_tq_batch([img])
        mock_client = _build_mock_tq_client(td)

        connector = TQMediaConnector()

        with patch(
            "vllm.multimodal.media.tq_connector._init_tq_client",
            return_value=mock_client,
        ):
            from vllm.multimodal.media.image import ImageMediaIO

            media_io = ImageMediaIO(image_mode="RGB")
            # Call the sync path from within a running async event loop.
            # The old implementation would deadlock here.
            result = connector._load_from_tq_sync(
                "tq://part/batch/0", media_io
            )

        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# _deserialize_env_var tests
# ---------------------------------------------------------------------------


class TestDeserializeEnvVar:
    """Tests for the JSON-first, pickle-fallback deserialization."""

    def test_json_encoded_value(self):
        """JSON-encoded base64 should be deserialized without pickle."""
        import base64
        import json

        from vllm.multimodal.media.tq_connector import _deserialize_env_var

        data = {"host": "localhost", "port": 5555, "nodes": [1, 2, 3]}
        b64 = base64.b64encode(json.dumps(data).encode()).decode()
        result = _deserialize_env_var(b64)
        assert result == data

    def test_pickle_fallback_with_warning(self):
        """Pickle-encoded base64 should work but log a warning."""
        import base64
        import pickle

        from vllm.multimodal.media.tq_connector import _deserialize_env_var

        data = {"host": "localhost", "port": 5555}
        b64 = base64.b64encode(pickle.dumps(data)).decode()

        with patch(
            "vllm.multimodal.media.tq_connector.logger"
        ) as mock_logger:
            result = _deserialize_env_var(b64)
            mock_logger.warning.assert_called_once()
            assert "pickle" in mock_logger.warning.call_args[0][0].lower()

        assert result == data

    def test_invalid_base64_raises(self):
        """Completely invalid data should raise an error."""
        from vllm.multimodal.media.tq_connector import _deserialize_env_var

        with pytest.raises(Exception):
            _deserialize_env_var("not-valid-base64!!!")
