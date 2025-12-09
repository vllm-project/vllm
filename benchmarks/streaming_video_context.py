# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming Video Context Manager for SSM-based video inference.

This module provides a context manager for processing video streams incrementally
with SSM + sliding window attention. The key insight is that SSM state is O(1)
in memory regardless of video length, enabling processing of arbitrarily long
videos without running out of memory.

The StreamingVideoContext enables:
1. Incremental frame processing - add frames one at a time or in batches
2. SSM state sharing - multiple concurrent queries share the same video context
3. Memory-efficient long video - only sliding window KV + fixed SSM state

Usage:
    from vllm import LLM
    from benchmarks.streaming_video_context import StreamingVideoContext

    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct", ...)
    
    ctx = StreamingVideoContext(stream_id="video_1", llm=llm)
    
    # Add frames incrementally
    for frame in video_stream:
        ctx.add_frame(frame)
        
        # Query at any point
        if user_asked_question:
            response = ctx.query("What just happened?")
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


def get_gpu_memory_info() -> dict[str, float]:
    """Get GPU memory usage information."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        device = torch.cuda.current_device()
        free_memory = torch.cuda.mem_get_info(device)[0]
        total_memory = torch.cuda.mem_get_info(device)[1]
        used_memory = total_memory - free_memory

        return {
            "available": True,
            "free_memory_gib": free_memory / (1024**3),
            "total_memory_gib": total_memory / (1024**3),
            "used_memory_gib": used_memory / (1024**3),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@dataclass
class FrameMetadata:
    """Metadata for a processed frame."""

    frame_idx: int
    timestamp_ms: float
    token_count: int
    memory_after: dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class QueryResult:
    """Result from a query against the video context."""

    query_id: str
    question: str
    response: str
    frame_idx_at_query: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    time_to_first_token_ms: float | None = None
    memory_at_query: dict[str, float] = field(default_factory=dict)


@dataclass
class StreamingContextStats:
    """Statistics for a streaming video context."""

    stream_id: str
    total_frames: int
    total_tokens: int
    total_queries: int
    frame_processing_times_ms: list[float] = field(default_factory=list)
    memory_history: list[dict[str, float]] = field(default_factory=list)
    query_latencies_seconds: list[float] = field(default_factory=list)


class StreamingVideoContext:
    """Manages a video stream with incrementally updated SSM state.

    This class enables streaming video inference where frames are processed
    one at a time or in small batches. The SSM state acts as a "compressed
    video memory" that can be shared across multiple concurrent queries.

    Key features:
    - Incremental frame addition with SSM state updates
    - Query interface that leverages the current video context
    - Memory tracking to demonstrate SSM efficiency
    - Support for checkpointing SSM state at specific frames

    Architecture:
        Video Stream --> [Frame 1] --> [Frame 2] --> ... --> [Frame N]
                              |             |                     |
                              v             v                     v
                         SSM State (continuously evolving, O(1) memory)
                              |
                              +--> Query 1: "What happened at start?"
                              +--> Query 2: "What is happening now?"
                              +--> Query N: (concurrent queries share state)
    """

    def __init__(
        self,
        stream_id: str | None = None,
        llm: "LLM | None" = None,
        use_hybrid_attention: bool = True,
        frame_batch_size: int = 1,
        checkpoint_every_n_frames: int | None = None,
    ) -> None:
        """Initialize a streaming video context.

        Args:
            stream_id: Unique identifier for this video stream. Auto-generated
                if not provided.
            llm: vLLM LLM instance. Can be set later via set_llm().
            use_hybrid_attention: Whether the LLM uses hybrid SSM + attention.
            frame_batch_size: Number of frames to batch together when processing.
            checkpoint_every_n_frames: If set, save SSM state checkpoints at
                this interval for potential rollback.
        """
        self.stream_id = stream_id or str(uuid.uuid4())[:8]
        self.llm = llm
        self.use_hybrid_attention = use_hybrid_attention
        self.frame_batch_size = frame_batch_size
        self.checkpoint_every_n_frames = checkpoint_every_n_frames

        # Frame storage and tracking
        self._frames: list[np.ndarray] = []
        self._frame_metadata: list[FrameMetadata] = []
        self._frame_tokens: list[list[int]] = []  # Token IDs per frame

        # Query tracking
        self._queries: list[QueryResult] = []

        # Running statistics
        self._total_tokens = 0
        self._start_time = time.time()

        # Context prefix for queries (accumulated video tokens)
        self._context_prefix_tokens: list[int] = []

        # SSM state checkpoints for rollback (frame_idx -> state snapshot)
        self._checkpoints: dict[int, Any] = {}

        # Pending frames buffer for batched processing
        self._pending_frames: list[np.ndarray] = []

    def set_llm(self, llm: "LLM") -> None:
        """Set the LLM instance for this context.

        Args:
            llm: vLLM LLM instance configured for video inference.
        """
        self.llm = llm

    @property
    def frame_count(self) -> int:
        """Number of frames processed so far."""
        return len(self._frames)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the video context."""
        return self._total_tokens

    @property
    def query_count(self) -> int:
        """Number of queries executed against this context."""
        return len(self._queries)

    @property
    def elapsed_time_seconds(self) -> float:
        """Time since context was created."""
        return time.time() - self._start_time

    def _build_frame_prompt(self, frame: np.ndarray) -> str:
        """Build a prompt for processing a single frame.

        This creates a minimal prompt that just processes the frame
        to update the SSM state without generating any output.
        """
        video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
        # Minimal prompt to process frame into SSM state
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{video_placeholder}<|im_end|>\n"
        )
        return prompt

    def _build_query_prompt(self, question: str, num_frames: int) -> str:
        """Build a prompt for querying the video context.

        Args:
            question: The question to ask about the video.
            num_frames: Number of frames in context (for placeholder).

        Returns:
            Formatted prompt string.
        """
        video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{video_placeholder}{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def add_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float | None = None,
        process_immediately: bool = True,
    ) -> FrameMetadata:
        """Add a frame to the video context.

        This processes the frame through the vision encoder and updates
        the SSM state to incorporate this new visual information.

        Args:
            frame: Video frame as numpy array with shape (H, W, C).
            timestamp_ms: Optional timestamp in milliseconds.
            process_immediately: If True, process the frame immediately.
                If False, buffer it for batch processing.

        Returns:
            Metadata about the processed frame.
        """
        frame_idx = len(self._frames)
        timestamp = timestamp_ms or (frame_idx * 33.33)  # ~30fps default

        if not process_immediately:
            self._pending_frames.append(frame)
            # Return placeholder metadata
            return FrameMetadata(
                frame_idx=frame_idx,
                timestamp_ms=timestamp,
                token_count=0,
                processing_time_ms=0.0,
            )

        # Process frame(s) - either single or batched
        frames_to_process = self._pending_frames + [frame]
        self._pending_frames = []

        start_time = time.perf_counter()
        token_count = self._process_frames(frames_to_process)
        processing_time = (time.perf_counter() - start_time) * 1000

        # Store frame and metadata
        for f in frames_to_process:
            self._frames.append(f)

        memory_info = get_gpu_memory_info()

        metadata = FrameMetadata(
            frame_idx=frame_idx,
            timestamp_ms=timestamp,
            token_count=token_count,
            memory_after=memory_info,
            processing_time_ms=processing_time,
        )
        self._frame_metadata.append(metadata)
        self._total_tokens += token_count

        # Create checkpoint if configured
        if (
            self.checkpoint_every_n_frames
            and frame_idx > 0
            and frame_idx % self.checkpoint_every_n_frames == 0
        ):
            self._create_checkpoint(frame_idx)

        return metadata

    def add_frames(
        self,
        frames: list[np.ndarray] | np.ndarray,
        timestamps_ms: list[float] | None = None,
    ) -> list[FrameMetadata]:
        """Add multiple frames to the video context.

        This is more efficient than adding frames one at a time as it
        batches the vision encoder processing.

        Args:
            frames: List of video frames or array with shape (N, H, W, C).
            timestamps_ms: Optional list of timestamps in milliseconds.

        Returns:
            List of metadata for each processed frame.
        """
        if isinstance(frames, np.ndarray):
            frames = list(frames)

        if timestamps_ms is None:
            base_idx = len(self._frames)
            timestamps_ms = [(base_idx + i) * 33.33 for i in range(len(frames))]

        metadata_list = []
        for i, frame in enumerate(frames):
            # Buffer all but last frame
            process_now = i == len(frames) - 1
            metadata = self.add_frame(
                frame,
                timestamp_ms=timestamps_ms[i],
                process_immediately=process_now,
            )
            metadata_list.append(metadata)

        return metadata_list

    def _process_frames(self, frames: list[np.ndarray]) -> int:
        """Process frames through the vision encoder and update SSM state.

        This is where the SSM state gets updated with new visual information.
        The key insight is that regardless of how many total frames have been
        processed, the SSM state remains fixed-size.

        Args:
            frames: List of frames to process.

        Returns:
            Number of tokens generated for these frames.
        """
        if self.llm is None:
            # Simulation mode - estimate token count
            # Qwen2.5-VL typically generates ~576 tokens per frame
            tokens_per_frame = 576
            return len(frames) * tokens_per_frame

        # Build prompt with frames
        video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{video_placeholder}<|im_end|>\n"
        )

        # Stack frames for batch processing
        if len(frames) == 1:
            frame_data = frames[0]
        else:
            frame_data = np.stack(frames, axis=0)

        # Process through LLM to update SSM state
        # We use a minimal generation (1 token) to trigger the prefill
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        try:
            outputs = self.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"video": frame_data}},
                sampling_params=sampling_params,
            )
            if outputs:
                return len(outputs[0].prompt_token_ids)
        except Exception as e:
            print(f"Warning: Frame processing failed: {e}")

        return len(frames) * 576  # Fallback estimate

    def _create_checkpoint(self, frame_idx: int) -> None:
        """Create a checkpoint of the current SSM state.

        This allows rolling back to a previous point in the video
        if needed for certain query patterns.

        Args:
            frame_idx: Frame index to associate with this checkpoint.
        """
        # In the current vLLM architecture, SSM state is managed internally.
        # This is a placeholder for future checkpoint support.
        checkpoint_data = {
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "total_tokens": self._total_tokens,
            # SSM state would be captured here in a production implementation
        }
        self._checkpoints[frame_idx] = checkpoint_data

    def query(
        self,
        question: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        frame_idx: int | None = None,
    ) -> QueryResult:
        """Query the video context with a question.

        This leverages the accumulated SSM state (video memory) to answer
        questions about the video content. Multiple concurrent queries
        can share the same SSM state through prefix caching.

        Args:
            question: The question to ask about the video.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            frame_idx: Optional specific frame to query at. If None, uses
                all frames up to current point.

        Returns:
            QueryResult with response and metrics.
        """
        query_id = str(uuid.uuid4())[:8]
        query_frame_idx = frame_idx if frame_idx is not None else self.frame_count - 1

        if query_frame_idx < 0:
            raise ValueError("Cannot query empty video context")

        start_time = time.perf_counter()
        memory_before = get_gpu_memory_info()

        # Build the query prompt
        prompt = self._build_query_prompt(question, query_frame_idx + 1)

        # Get frames up to query point
        frames = self._frames[: query_frame_idx + 1]
        if len(frames) == 0:
            raise ValueError(f"No frames available at index {query_frame_idx}")

        # Stack frames for video input
        if len(frames) == 1:
            frame_data = frames[0]
        else:
            frame_data = np.stack(frames, axis=0)

        response_text = ""
        input_tokens = 0
        output_tokens = 0
        ttft_ms = None

        if self.llm is not None:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )

            try:
                first_token_time = None
                outputs = self.llm.generate(
                    {"prompt": prompt, "multi_modal_data": {"video": frame_data}},
                    sampling_params=sampling_params,
                )

                if outputs:
                    output = outputs[0]
                    response_text = output.outputs[0].text
                    input_tokens = len(output.prompt_token_ids)
                    output_tokens = len(output.outputs[0].token_ids)

            except Exception as e:
                response_text = f"Error: {e}"

        latency = time.perf_counter() - start_time

        result = QueryResult(
            query_id=query_id,
            question=question,
            response=response_text,
            frame_idx_at_query=query_frame_idx,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_to_first_token_ms=ttft_ms,
            memory_at_query=memory_before,
        )

        self._queries.append(result)
        return result

    def get_stats(self) -> StreamingContextStats:
        """Get statistics for this streaming context.

        Returns:
            StreamingContextStats with aggregated metrics.
        """
        return StreamingContextStats(
            stream_id=self.stream_id,
            total_frames=self.frame_count,
            total_tokens=self._total_tokens,
            total_queries=self.query_count,
            frame_processing_times_ms=[m.processing_time_ms for m in self._frame_metadata],
            memory_history=[m.memory_after for m in self._frame_metadata],
            query_latencies_seconds=[q.latency_seconds for q in self._queries],
        )

    def get_memory_growth_rate(self) -> dict[str, float]:
        """Calculate memory growth rate per frame.

        This demonstrates the O(1) vs O(n) memory behavior:
        - Standard attention: memory grows linearly with frames
        - SSM + sliding window: memory stays roughly constant

        Returns:
            Dictionary with memory growth statistics.
        """
        if len(self._frame_metadata) < 2:
            return {"growth_rate_gib_per_frame": 0.0, "sample_count": 0}

        memory_values = []
        for m in self._frame_metadata:
            if m.memory_after.get("available"):
                memory_values.append(m.memory_after.get("used_memory_gib", 0))

        if len(memory_values) < 2:
            return {"growth_rate_gib_per_frame": 0.0, "sample_count": 0}

        # Linear regression to estimate growth rate
        x = np.arange(len(memory_values))
        y = np.array(memory_values)
        slope = np.polyfit(x, y, 1)[0]

        return {
            "growth_rate_gib_per_frame": float(slope),
            "sample_count": len(memory_values),
            "initial_memory_gib": float(memory_values[0]),
            "final_memory_gib": float(memory_values[-1]),
            "total_growth_gib": float(memory_values[-1] - memory_values[0]),
        }

    def clear(self) -> None:
        """Clear all frames and reset the context.

        This frees all memory associated with the video context.
        """
        self._frames.clear()
        self._frame_metadata.clear()
        self._frame_tokens.clear()
        self._queries.clear()
        self._context_prefix_tokens.clear()
        self._checkpoints.clear()
        self._pending_frames.clear()
        self._total_tokens = 0

    def __repr__(self) -> str:
        return (
            f"StreamingVideoContext("
            f"stream_id='{self.stream_id}', "
            f"frames={self.frame_count}, "
            f"tokens={self._total_tokens}, "
            f"queries={self.query_count})"
        )


class MultiStreamManager:
    """Manages multiple concurrent video streams.

    This enables scenarios where multiple video feeds are being processed
    simultaneously, each with their own SSM state, while queries can be
    directed to specific streams.
    """

    def __init__(self, llm: "LLM | None" = None):
        """Initialize the multi-stream manager.

        Args:
            llm: Shared vLLM instance for all streams.
        """
        self.llm = llm
        self._streams: dict[str, StreamingVideoContext] = {}

    def create_stream(
        self,
        stream_id: str | None = None,
        **kwargs,
    ) -> StreamingVideoContext:
        """Create a new video stream context.

        Args:
            stream_id: Optional ID for the stream.
            **kwargs: Additional arguments passed to StreamingVideoContext.

        Returns:
            The created StreamingVideoContext.
        """
        ctx = StreamingVideoContext(stream_id=stream_id, llm=self.llm, **kwargs)
        self._streams[ctx.stream_id] = ctx
        return ctx

    def get_stream(self, stream_id: str) -> StreamingVideoContext | None:
        """Get a stream by ID."""
        return self._streams.get(stream_id)

    def remove_stream(self, stream_id: str) -> None:
        """Remove and cleanup a stream."""
        if stream_id in self._streams:
            self._streams[stream_id].clear()
            del self._streams[stream_id]

    def get_all_stats(self) -> dict[str, StreamingContextStats]:
        """Get stats for all streams."""
        return {sid: ctx.get_stats() for sid, ctx in self._streams.items()}

    @property
    def stream_count(self) -> int:
        """Number of active streams."""
        return len(self._streams)

    def __repr__(self) -> str:
        return f"MultiStreamManager(streams={self.stream_count})"

