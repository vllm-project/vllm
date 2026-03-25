# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for video streaming via the realtime WebSocket API.

These are unit tests for the protocol and connection logic. They do NOT
require a running vLLM server or GPU — they validate serialization, frame
decoding, and event routing in isolation.
"""

import asyncio
import io
import json

import numpy as np
import pybase64 as base64
import pytest

from vllm.entrypoints.openai.realtime.protocol import (
    InputVideoFrameAppend,
    InputVideoFrameCommit,
    VideoChatDelta,
    VideoChatDone,
)


# ---------------------------------------------------------------------------
# Protocol serialization tests
# ---------------------------------------------------------------------------

class TestVideoProtocol:
    """Test video protocol event serialization/deserialization."""

    def test_input_video_frame_append_roundtrip(self):
        """InputVideoFrameAppend serializes and deserializes correctly."""
        # Create a small 2x2 red JPEG
        from PIL import Image

        img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        event = InputVideoFrameAppend(image=b64, timestamp=1.5)
        data = json.loads(event.model_dump_json())

        assert data["type"] == "input_video_frame.append"
        assert data["image"] == b64
        assert data["timestamp"] == 1.5

        # Reconstruct
        reconstructed = InputVideoFrameAppend(**data)
        assert reconstructed.image == b64
        assert reconstructed.timestamp == 1.5

    def test_input_video_frame_append_no_timestamp(self):
        """Timestamp is optional."""
        event = InputVideoFrameAppend(image="abc123")
        data = json.loads(event.model_dump_json())
        assert data["timestamp"] is None

    def test_input_video_frame_commit_with_query(self):
        """Commit event carries query and final flag."""
        event = InputVideoFrameCommit(query="What is happening?", final=True)
        data = json.loads(event.model_dump_json())

        assert data["type"] == "input_video_frame.commit"
        assert data["query"] == "What is happening?"
        assert data["final"] is True

    def test_input_video_frame_commit_defaults(self):
        """Default commit has no query and final=False."""
        event = InputVideoFrameCommit()
        data = json.loads(event.model_dump_json())

        assert data["query"] is None
        assert data["final"] is False

    def test_video_chat_delta_serialization(self):
        """VideoChatDelta serializes correctly."""
        event = VideoChatDelta(delta="A person is")
        data = json.loads(event.model_dump_json())

        assert data["type"] == "video_chat.delta"
        assert data["delta"] == "A person is"

    def test_video_chat_done_serialization(self):
        """VideoChatDone includes text and optional usage."""
        from vllm.entrypoints.openai.engine.protocol import UsageInfo

        usage = UsageInfo(prompt_tokens=100, completion_tokens=20,
                          total_tokens=120)
        event = VideoChatDone(text="A person is walking.", usage=usage)
        data = json.loads(event.model_dump_json())

        assert data["type"] == "video_chat.done"
        assert data["text"] == "A person is walking."
        assert data["usage"]["prompt_tokens"] == 100
        assert data["usage"]["completion_tokens"] == 20

    def test_video_chat_done_no_usage(self):
        """VideoChatDone works without usage stats."""
        event = VideoChatDone(text="Done")
        data = json.loads(event.model_dump_json())
        assert data["usage"] is None


# ---------------------------------------------------------------------------
# Frame decoding tests
# ---------------------------------------------------------------------------

class TestFrameDecoding:
    """Test that frames are correctly decoded from base64 to numpy arrays."""

    def _encode_frame(self, width: int, height: int,
                      fmt: str = "JPEG") -> str:
        """Create a test frame and encode it to base64."""
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(0, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def test_jpeg_decode(self):
        """JPEG frames decode to correct shape and dtype."""
        from PIL import Image

        b64 = self._encode_frame(64, 48, "JPEG")
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)

        assert arr.shape == (48, 64, 3)
        assert arr.dtype == np.uint8

    def test_png_decode(self):
        """PNG frames decode to correct shape and dtype."""
        from PIL import Image

        b64 = self._encode_frame(32, 32, "PNG")
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)

        assert arr.shape == (32, 32, 3)
        assert arr.dtype == np.uint8

    def test_large_frame(self):
        """A 1080p frame encodes and decodes without error."""
        from PIL import Image

        b64 = self._encode_frame(1920, 1080, "JPEG")
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(img, dtype=np.uint8)

        assert arr.shape == (1080, 1920, 3)


# ---------------------------------------------------------------------------
# Video queue tests (unit-level, no server needed)
# ---------------------------------------------------------------------------

class TestVideoQueue:
    """Test the video frame queue mechanics."""

    @pytest.mark.asyncio
    async def test_queue_put_get(self):
        """Frames flow through asyncio.Queue correctly."""
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        queue.put_nowait(frame)
        queue.put_nowait(None)  # sentinel

        got = await queue.get()
        assert got is not None
        assert got.shape == (48, 64, 3)

        sentinel = await queue.get()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_stream_generator_pattern(self):
        """The async generator pattern yields frames then stops on None."""
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        frames = [
            np.zeros((48, 64, 3), dtype=np.uint8),
            np.ones((48, 64, 3), dtype=np.uint8) * 128,
            np.ones((48, 64, 3), dtype=np.uint8) * 255,
        ]
        for f in frames:
            queue.put_nowait(f)
        queue.put_nowait(None)

        async def video_stream():
            while True:
                frame = await queue.get()
                if frame is None:
                    break
                yield frame

        collected = []
        async for frame in video_stream():
            collected.append(frame)

        assert len(collected) == 3
        assert collected[0].mean() == 0
        assert collected[2].mean() == 255
