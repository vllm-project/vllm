# SPDX-License-Identifier: Apache-2.0
"""
Public API for Batched Stream, a wrapper of Request Streams that are batched
together.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import time

# First Party
from lmcache.cli.metrics import Metrics, StreamHandler, get_formatter
from lmcache.sdk.context import ModifyFnType
import lmcache.sdk.request as lmc_request


class LMCacheBatchedStreamError(RuntimeError):
    """Raised when a BatchedStream operation fails."""


class LMCacheBatchedStream:
    """A collection of LMCacheRequestStreams run together as one batch."""

    def __init__(self) -> None:
        """Create a batch.

        Attributes:
            request_streams: Member streams keyed by request stream id.
            perf_metrics: Latest StreamPerfMetrics per request stream id, overwritten
                on each call to run_request_streams.
        """
        self.request_streams: dict[str, lmc_request.LMCacheRequestStream] = {}
        self.perf_metrics: dict[str, lmc_request.StreamPerfMetrics] = {}

    def add(self, request_stream: lmc_request.LMCacheRequestStream) -> None:
        """Add a request to the batch, keyed by its request stream id.

        Args:
            request_stream: The request to add.

        Raises:
            LMCacheBatchedStreamError: If a stream with the same id is
                in the batch.
        """
        if request_stream.request_stream_id in self.request_streams:
            raise LMCacheBatchedStreamError(
                f"RequestStream {request_stream.request_stream_id} already exists."
            )
        self.request_streams[request_stream.request_stream_id] = request_stream

    def get_request_stream(self, stream_id: str) -> lmc_request.LMCacheRequestStream:
        """Return the stream registered under stream_id.

        Args:
            stream_id: The id of the stream to fetch.

        Returns:
            The matching LMCacheRequestStream.

        Raises:
            LMCacheBatchedStreamError: If no stream has that id.
        """
        if stream_id not in self.request_streams:
            raise LMCacheBatchedStreamError(
                f"RequestStream with id {stream_id} does not exist."
            )
        return self.request_streams[stream_id]

    def run_request_streams(
        self, sampling_params: dict[str, Any], stream_ids: list[str] | None = None
    ) -> float:
        """Decode every selected stream concurrently and store their metrics.

        Submits stream.generate(sampling_params) on a thread pool (one worker
        per stream) and records each StreamPerfMetrics in perf_metrics.

        Args:
            sampling_params: Engine sampling params (e.g. max_tokens,
                temperature) passed to every stream's generate().
            stream_ids: Streams to run; None means all streams in the batch.

        Returns:
            Wall-clock duration of the batch, in seconds (0.0 if no streams).

        Raises:
            LMCacheBatchedStreamError: If any requested stream id is unknown.
        """
        if stream_ids is None:
            stream_ids = list(self.request_streams.keys())

        for stream_id in stream_ids:
            if stream_id not in self.request_streams:
                raise LMCacheBatchedStreamError(
                    f"RequestStream {stream_id} does not exist."
                )

        if not stream_ids:
            return 0.0

        self.perf_metrics.clear()
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(stream_ids)) as executor:
            futures = {}
            for stream_id in stream_ids:
                request_stream = self.request_streams[stream_id]
                future = executor.submit(
                    request_stream.generate, sampling_params=sampling_params
                )
                futures[future] = stream_id
            # collect metrics
            for future in as_completed(futures):
                stream_id = futures[future]
                self.perf_metrics[stream_id] = future.result()

        return time.perf_counter() - start_time  # in seconds

    def modify_request_streams(
        self,
        fn: ModifyFnType,
        stream_ids: list[str] | None = None,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
    ) -> float:
        """Apply KV edit to every selected stream concurrently.

        Submits stream.modify_kv(fn) on a thread pool (one worker per stream).

        Args:
            fn: KV editor applied to each stream. See LMCacheRequestStream.modify_kv().
            stream_ids: Streams to modify. None means all streams in the batch.
            timeout: Max seconds to wait for the cached KV to appear.
            poll_interval: Seconds between retrieve attempts.

        Returns:
            Wall-clock duration of the batch, in seconds (0.0 if no streams).

        Raises:
            LMCacheBatchedStreamError: If any requested stream id is unknown.
        """
        if stream_ids is None:
            stream_ids = list(self.request_streams.keys())

        for stream_id in stream_ids:
            if stream_id not in self.request_streams:
                raise LMCacheBatchedStreamError(
                    f"RequestStream {stream_id} does not exist."
                )

        if not stream_ids:
            return 0.0

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=len(stream_ids)) as executor:
            futures = {}
            for stream_id in stream_ids:
                request_stream = self.request_streams[stream_id]
                future = executor.submit(
                    request_stream.modify_kv,
                    fn,
                    timeout=timeout,
                    poll_interval=poll_interval,
                )
                futures[future] = stream_id
            for future in as_completed(futures):
                stream_id = futures[future]
                future.result()

        return time.perf_counter() - start_time  # in seconds

    def get_perf_metrics(
        self,
        duration: float,
        fmt: str,
        width: int,
        mode: str,
        stream_ids: list[str] | None = None,
    ) -> Metrics:
        """Aggregate the stored per-stream metrics into a Metrics report.

        Args:
            duration: Batch wall-clock duration in seconds, for throughput.
            fmt: Formatter name for the output handler (e.g. "terminal").
            width: Output width passed to the formatter.
            mode: "prefill" (input tokens + TTFT) or "decode" (output tokens +
                TPOT + decoding speed).
            stream_ids: Streams to include. None means all streams in batch.

        Returns:
            The aggregated Metrics report.

        Raises:
            LMCacheBatchedStreamError: If mode is not "prefill" or "decode".
        """
        if stream_ids is None:
            stream_ids = list(self.request_streams.keys())

        if mode not in ["prefill", "decode"]:
            raise LMCacheBatchedStreamError(f"Invalid mode {mode}.")

        metrics = Metrics(title=f"Batched Stream Metrics ({mode})")
        metrics.add_handler(StreamHandler(get_formatter(fmt, width=width)))

        cfg_section = metrics.add_section("config", "Configuration")
        cfg_section.add(
            "num_request_streams",
            "Number of Request Streams",
            len(self.request_streams),
        )

        result_section = metrics.add_section("results", "Results")
        result_section.add("duration", "Total Duration (s)", duration)

        input_tokens = 0
        output_tokens = 0
        for stream_id, perf_metrics in self.perf_metrics.items():
            if stream_id in stream_ids:
                input_tokens += perf_metrics.input_tokens
                output_tokens += perf_metrics.output_tokens

        if mode == "prefill":
            result_section.add("input_tokens", "Total Input Tokens", input_tokens)
            result_section.add(
                "input_tput",
                "Input Throughput (tokens/s)",
                input_tokens / duration if duration > 0 else 0.0,
            )

        if mode == "decode":
            result_section.add("output_tokens", "Total Output Tokens", output_tokens)
            result_section.add(
                "output_tput",
                "Decode Throughput (tokens/s)",
                output_tokens / duration if duration > 0 else 0.0,
            )

        return metrics

    def prefill(
        self,
        sampling_params: dict[str, Any],
        fmt: str = "terminal",
        width: int = 80,
        stream_ids: list[str] | None = None,
    ) -> Metrics:
        """Prefill every stream once (max_tokens forced to 1) and report.

        Args:
            sampling_params: Engine sampling params, max_tokens is set 1.
            fmt: Formatter name for the output handler.
            width: Output width passed to the formatter.
            stream_ids: Streams to run; None means all streams in the batch.

        Returns:
            A prefill-mode Metrics report (input tokens, throughput, TTFT).
        """
        sampling_params = {**sampling_params, "max_tokens": 1}
        duration = self.run_request_streams(
            sampling_params=sampling_params, stream_ids=stream_ids
        )
        return self.get_perf_metrics(
            duration=duration,
            fmt=fmt,
            width=width,
            mode="prefill",
            stream_ids=stream_ids,
        )

    def modify(
        self,
        fn: ModifyFnType,
        fmt: str = "terminal",
        width: int = 80,
        timeout: float = 30.0,
        poll_interval: float = 0.2,
        stream_ids: list[str] | None = None,
    ) -> Metrics:
        """Apply a KV edit to every stream and report the time taken.

        Args:
            fn: KV editor applied to each stream (see modify_request_streams).
            fmt: Formatter name for the output handler.
            width: Output width passed to the formatter.
            stream_ids: Streams to modify. None means all streams in the batch.

        Returns:
            A Metrics report containing the total modify duration.
        """
        duration = self.modify_request_streams(
            fn, stream_ids=stream_ids, timeout=timeout, poll_interval=poll_interval
        )

        metrics = Metrics(title="Batched Stream Modify Metrics")
        metrics.add_handler(StreamHandler(get_formatter(fmt, width=width)))
        result_section = metrics.add_section("results", "Results")
        result_section.add("duration", "Total Duration (s)", duration)

        return metrics

    def decode(
        self,
        sampling_params: dict[str, Any],
        fmt: str = "terminal",
        width: int = 80,
        stream_ids: list[str] | None = None,
    ) -> Metrics:
        """Decode every stream and report decode metrics.

        Args:
            sampling_params: Engine sampling params (e.g. max_tokens,
                temperature) passed to every stream's generate().
            fmt: Formatter name for the output handler.
            width: Output width passed to the formatter.
            stream_ids: Streams to run. None means all streams in the batch.

        Returns:
            A decode-mode Metrics report (output tokens, throughput, TPOT,
            decoding speed).
        """
        duration = self.run_request_streams(
            sampling_params=sampling_params, stream_ids=stream_ids
        )
        return self.get_perf_metrics(
            duration=duration,
            fmt=fmt,
            width=width,
            mode="decode",
            stream_ids=stream_ids,
        )
