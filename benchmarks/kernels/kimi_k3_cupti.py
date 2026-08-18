# SPDX-License-Identifier: Apache-2.0
"""Minimal copied CUPTI Activity timer from the TRT-LLM replay benchmark."""
import atexit
import ctypes
import multiprocessing as mp
import os
import queue
import threading
import time
from multiprocessing import shared_memory
import torch

_LIBCUPTI_CANDIDATES = (
    os.environ.get("CUPTI_LIBRARY_PATH"),
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13",
    "libcupti.so.13",
    "libcupti.so",
)
_CUPTI_SUCCESS = 0
_CUPTI_ERROR_MAX_LIMIT_REACHED = 12
_CUPTI_ERROR_INVALID_KIND = 21
_CUPTI_ACTIVITY_KIND_KERNEL = 3
_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
_CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER = 5
_CUPTI_HOST_BUFFER_BYTES = 1024 * 1024
_CUPTI_HOST_BUFFER_COUNT = 16

# Multiprocessing start method for compile-warmup + CUPTI parser children.
# Set in __main__ from --mp-start-method.  "spawn" (default) is robust; each
# child re-imports torch/triton/etc (~15s).  "forkserver" preloads once and
# forks cheaply (~1s/child) — see __main__ block for the preload setup.
_MP_START_METHOD = "spawn"
_DEFAULT_CUDA_GRAPH_GROUP_ITERS_PURE = 1
_DEFAULT_CUDA_GRAPH_GROUP_ITERS_MIX = 4


def _load_libcupti() -> ctypes.CDLL:
    errors = []
    for candidate in _LIBCUPTI_CANDIDATES:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise ImportError("Unable to load libcupti: " + "; ".join(errors))


class _CuptiActivityKernel11Prefix(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_int),
        ("cache_config", ctypes.c_uint8),
        ("shared_memory_config", ctypes.c_uint8),
        ("registers_per_thread", ctypes.c_uint16),
        ("partitioned_global_cache_requested", ctypes.c_int),
        ("partitioned_global_cache_executed", ctypes.c_int),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("device_id", ctypes.c_uint32),
        ("context_id", ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("grid_x", ctypes.c_int32),
        ("grid_y", ctypes.c_int32),
        ("grid_z", ctypes.c_int32),
        ("block_x", ctypes.c_int32),
        ("block_y", ctypes.c_int32),
        ("block_z", ctypes.c_int32),
        ("static_shared_memory", ctypes.c_int32),
        ("dynamic_shared_memory", ctypes.c_int32),
        ("local_memory_per_thread", ctypes.c_uint32),
        ("local_memory_total", ctypes.c_uint32),
        ("correlation_id", ctypes.c_uint32),
        ("grid_id", ctypes.c_int64),
        ("name", ctypes.c_void_p),
        ("reserved0", ctypes.c_void_p),
        ("queued", ctypes.c_uint64),
        ("submitted", ctypes.c_uint64),
        ("launch_type", ctypes.c_uint8),
        ("is_shared_memory_carveout_requested", ctypes.c_uint8),
        ("shared_memory_carveout_requested", ctypes.c_uint8),
        ("padding", ctypes.c_uint8),
        ("shared_memory_executed", ctypes.c_uint32),
        ("graph_node_id", ctypes.c_uint64),
    ]


def _configure_cupti_get_next_record(libcupti) -> None:
    libcupti.cuptiActivityGetNextRecord.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    libcupti.cuptiActivityGetNextRecord.restype = ctypes.c_int


def _parse_cupti_buffer_ptr(libcupti, buffer_ptr: int, valid_size: int, *, include_names: bool):
    records = []
    zero_ts_count = 0
    zero_ts_names: dict[str, int] = {}
    record_ptr = ctypes.c_void_p(None)
    while True:
        result = libcupti.cuptiActivityGetNextRecord(
            ctypes.c_void_p(buffer_ptr),
            valid_size,
            ctypes.byref(record_ptr),
        )
        if result == _CUPTI_SUCCESS:
            kind = ctypes.cast(record_ptr, ctypes.POINTER(ctypes.c_int)).contents.value
            if kind not in (_CUPTI_ACTIVITY_KIND_KERNEL, _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL):
                continue
            kernel = ctypes.cast(record_ptr, ctypes.POINTER(_CuptiActivityKernel11Prefix)).contents
            name = None
            if include_names:
                if kernel.name:
                    name = ctypes.string_at(kernel.name).decode("utf-8", errors="replace")
                else:
                    name = "?"
            if kernel.start == 0 or kernel.end == 0:
                zero_ts_count += 1
                if name is not None:
                    zero_ts_names[name] = zero_ts_names.get(name, 0) + 1
                continue
            if include_names:
                records.append(
                    (
                        name,
                        int(kernel.start),
                        int(kernel.end),
                        int(kernel.correlation_id),
                        0,
                        int(kernel.graph_node_id),
                        int(kernel.stream_id),
                    )
                )
            else:
                records.append(
                    (
                        int(kernel.start),
                        int(kernel.end),
                        int(kernel.correlation_id),
                        int(kernel.graph_node_id),
                        int(kernel.stream_id),
                    )
                )
        elif result == _CUPTI_ERROR_MAX_LIMIT_REACHED:
            break
        elif result == _CUPTI_ERROR_INVALID_KIND:
            break
        else:
            raise RuntimeError(f"cuptiActivityGetNextRecord failed with CUptiResult={result}")
    return records, zero_ts_count, zero_ts_names


def _apply_cupti_filter_plan(numeric_records, filter_plan):
    if not filter_plan:
        return [
            (None, start, end, corr, 0, graph_node_id, stream_id)
            for start, end, corr, graph_node_id, stream_id in sorted(numeric_records)
        ]

    filtered = []
    replay_idx = 0
    record_idx = 0
    for start, end, corr, graph_node_id, stream_id in sorted(numeric_records):
        if replay_idx >= len(filter_plan):
            break
        records_per_replay, ordinal_names = filter_plan[replay_idx]
        if record_idx < len(ordinal_names):
            name = ordinal_names[record_idx]
            if name is not None:
                filtered.append((name, start, end, corr, 0, graph_node_id, stream_id))
        record_idx += 1
        if record_idx >= records_per_replay:
            replay_idx += 1
            record_idx = 0
    return filtered


def _cupti_parser_worker(input_queue, output_queue, ready_event) -> None:
    libcupti = _load_libcupti()
    _configure_cupti_get_next_record(libcupti)
    shared_blocks: dict[str, shared_memory.SharedMemory] = {}
    records_by_generation: dict[int, list[tuple[int, int, int, int, int]]] = {}
    zero_ts_by_generation: dict[int, int] = {}
    ready_event.set()
    while True:
        item = input_queue.get()
        if item is None:
            break
        kind = item[0]
        if kind == "buffer":
            _, generation, buffer_id, name, valid_size = item
            shm = shared_blocks.get(name)
            if shm is None:
                shm = shared_memory.SharedMemory(name=name)
                shared_blocks[name] = shm
            shared_char = ctypes.c_char.from_buffer(shm.buf)
            try:
                parser_ptr = ctypes.addressof(shared_char)
                records, zero_ts_count, _ = _parse_cupti_buffer_ptr(
                    libcupti,
                    parser_ptr,
                    valid_size,
                    include_names=False,
                )
                records_by_generation.setdefault(generation, []).extend(records)
                zero_ts_by_generation[generation] = (
                    zero_ts_by_generation.get(generation, 0) + zero_ts_count
                )
                ctypes.memset(parser_ptr, 0, len(shm.buf))
            except Exception as exc:  # pragma: no cover - diagnostic worker path
                output_queue.put({"kind": "error", "generation": generation, "error": repr(exc)})
            finally:
                del shared_char
            output_queue.put(
                {"kind": "buffer_done", "generation": generation, "buffer_id": buffer_id}
            )
        elif kind == "finish":
            if len(item) == 4:
                _, generation, filter_plan, stats_request = item
            else:
                _, generation, filter_plan = item
                stats_request = None
            try:
                raw_records = records_by_generation.pop(generation, [])
                zero_ts_count = zero_ts_by_generation.pop(generation, 0)
                filtered_records = _apply_cupti_filter_plan(raw_records, filter_plan)
                stats = None
                parser_stats_ms = 0.0
                stats_ready = stats_request is not None
                if stats_request is not None:
                    stats_start_s = time.perf_counter()
                    stats = _stats_from_cupti_records(
                        filtered_records,
                        int(stats_request["warmup"]),
                        int(stats_request["iters"]),
                        str(stats_request["tag"]),
                        int(stats_request["expected_K"]),
                        zero_ts_count=zero_ts_count,
                        zero_ts_names={},
                        include_details=bool(stats_request.get("include_details", True)),
                    )
                    parser_stats_ms = 1000.0 * (time.perf_counter() - stats_start_s)
                    filtered_records = []
                output_queue.put(
                    {
                        "kind": "finish_done",
                        "generation": generation,
                        "records": filtered_records,
                        "zero_ts_count": zero_ts_count,
                        "zero_ts_names": {},
                        "raw_record_count": len(raw_records),
                        "stats": stats,
                        "stats_ready": stats_ready,
                        "parser_stats_ms": parser_stats_ms,
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic worker path
                output_queue.put({"kind": "error", "generation": generation, "error": repr(exc)})
        else:
            output_queue.put(
                {"kind": "error", "generation": -1, "error": f"unknown parser message {kind!r}"}
            )
    for shm in shared_blocks.values():
        shm.close()


class CuptiKernelTimer:
    """Raw CUPTI Activity timer with out-of-process parsing for timed runs.

    CUPTI's callback gives us raw activity buffers.  The callback only hands
    shared-memory buffer metadata to a parser process, so the main process
    avoids the cupti-python per-record object creation cost during the timed
    path.  A single local calibration replay may parse names in-process to
    build an ordinal filter plan for a just-captured CUDA graph.
    """

    _instance = None
    _import_error = None

    _request_callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    _complete_callback_type = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    )

    @classmethod
    def get(cls) -> "CuptiKernelTimer":
        if cls._instance is not None:
            return cls._instance
        if cls._import_error is not None:
            raise cls._import_error
        try:
            cls._instance = cls()
            return cls._instance
        except ImportError as exc:  # pragma: no cover - env-dependent
            cls._import_error = exc
            raise

    def __init__(self) -> None:
        self._libcupti = _load_libcupti()
        self._configure_functions()
        self._lock = threading.Lock()
        self._shared_buffers: dict[int, shared_memory.SharedMemory] = {}
        self._buffer_id_by_ptr: dict[int, int] = {}
        self._free_buffer_ids: list[int] = []
        self._local_completed: list[tuple[int, int]] = []
        self._mode = "drop"
        self._generation = 0
        self._finish_results: dict[int, dict] = {}
        self._parser_errors: list[str] = []
        self._filter_plan = ()
        self._last_start_timing: dict[str, float] = {}
        self._last_stop_timing: dict[str, float] = {}
        self._current_flush_period_ms = 0
        self._mp_ctx = mp.get_context(_MP_START_METHOD)
        # Retry parser-process spawn: concurrent bench instances on the same
        # node race on POSIX named semaphores in /dev/shm — child can die in
        # pickle.load with FileNotFoundError in SemLock._rebuild before
        # signalling ready_event.  Detect early-dead child via is_alive() so
        # we don't waste the full timeout, and retry up to 3x with jitter.
        last_err = None
        for _spawn_attempt in range(3):
            self._parse_input_queue = self._mp_ctx.Queue()
            self._parse_output_queue = self._mp_ctx.Queue()
            ready_event = self._mp_ctx.Event()
            self._parse_process = self._mp_ctx.Process(
                target=_cupti_parser_worker,
                args=(self._parse_input_queue, self._parse_output_queue, ready_event),
            )
            self._parse_process.start()
            deadline = time.time() + 30.0
            spawn_ok = False
            while time.time() < deadline:
                if ready_event.wait(timeout=0.5):
                    spawn_ok = True
                    break
                if not self._parse_process.is_alive():
                    break
            if spawn_ok:
                last_err = None
                break
            last_err = (
                f"attempt {_spawn_attempt + 1}: "
                f"alive={self._parse_process.is_alive()}, "
                f"exitcode={self._parse_process.exitcode}"
            )
            try:
                if self._parse_process.is_alive():
                    self._parse_process.terminate()
                self._parse_process.join(timeout=2.0)
            except Exception:
                pass
            time.sleep(0.5 + 0.5 * _spawn_attempt)
        if last_err is not None:
            raise RuntimeError(
                f"CUPTI parser process did not initialize after 3 attempts: {last_err}"
            )

        self._set_zeroed_host_buffer_attr()
        for _ in range(_CUPTI_HOST_BUFFER_COUNT):
            self._free_buffer_ids.append(self._allocate_shared_buffer())

        self._request_callback = self._request_callback_type(self._request_buffer)
        self._complete_callback = self._complete_callback_type(self._complete_buffer)
        self._check(
            self._libcupti.cuptiActivityRegisterCallbacks(
                self._request_callback,
                self._complete_callback,
            )
        )
        self._check(self._libcupti.cuptiActivityEnable(_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL))
        atexit.register(self.close)

    def _configure_functions(self) -> None:
        self._libcupti.cuptiActivityRegisterCallbacks.argtypes = [
            self._request_callback_type,
            self._complete_callback_type,
        ]
        self._libcupti.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        self._libcupti.cuptiActivityEnable.argtypes = [ctypes.c_int]
        self._libcupti.cuptiActivityEnable.restype = ctypes.c_int
        self._libcupti.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        self._libcupti.cuptiActivityFlushAll.restype = ctypes.c_int
        self._libcupti.cuptiActivityFlushPeriod.argtypes = [ctypes.c_uint32]
        self._libcupti.cuptiActivityFlushPeriod.restype = ctypes.c_int
        self._libcupti.cuptiActivitySetAttribute.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self._libcupti.cuptiActivitySetAttribute.restype = ctypes.c_int
        _configure_cupti_get_next_record(self._libcupti)

    def _set_zeroed_host_buffer_attr(self) -> None:
        value_obj = ctypes.c_uint8(1)
        size_obj = ctypes.c_size_t(ctypes.sizeof(value_obj))
        result = self._libcupti.cuptiActivitySetAttribute(
            _CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER,
            ctypes.byref(size_obj),
            ctypes.byref(value_obj),
        )
        if result != _CUPTI_SUCCESS:
            print(
                "[WARN] CUPTI zeroed host-buffer attribute failed; "
                f"continuing with default CUPTI buffer handling (CUptiResult={result}).",
                file=sys.stderr,
            )

    def _check(self, result: int) -> None:
        if result != _CUPTI_SUCCESS:
            raise RuntimeError(f"CUPTI call failed with CUptiResult={result}")

    def _allocate_shared_buffer(self) -> int:
        buffer_id = len(self._shared_buffers)
        shm = shared_memory.SharedMemory(create=True, size=_CUPTI_HOST_BUFFER_BYTES)
        shared_char = ctypes.c_char.from_buffer(shm.buf)
        try:
            ptr = ctypes.addressof(shared_char)
        finally:
            del shared_char
        if ptr % 8 != 0:
            shm.close()
            shm.unlink()
            raise RuntimeError("CUPTI shared-memory activity buffer was not 8-byte aligned")
        self._shared_buffers[buffer_id] = shm
        self._buffer_id_by_ptr[ptr] = buffer_id
        return buffer_id

    def _buffer_ptr(self, buffer_id: int) -> int:
        shm = self._shared_buffers[buffer_id]
        shared_char = ctypes.c_char.from_buffer(shm.buf)
        try:
            return ctypes.addressof(shared_char)
        finally:
            del shared_char

    def _request_buffer(self, buffer, size, max_num_records) -> None:
        with self._lock:
            if self._free_buffer_ids:
                buffer_id = self._free_buffer_ids.pop()
            else:
                buffer_id = self._allocate_shared_buffer()
            ptr = self._buffer_ptr(buffer_id)
        buffer[0] = ptr
        size[0] = _CUPTI_HOST_BUFFER_BYTES
        max_num_records[0] = 0

    def _complete_buffer(self, context, stream_id, buffer, size, valid_size) -> None:
        del context, stream_id, size
        buffer_ptr = int(buffer)
        valid_size_int = int(valid_size)
        with self._lock:
            mode = self._mode
            generation = self._generation
            buffer_id = self._buffer_id_by_ptr[buffer_ptr]
            if valid_size_int == 0 or mode == "drop":
                self._free_buffer_ids.append(buffer_id)
                return
            if mode == "local":
                self._local_completed.append((buffer_id, valid_size_int))
                return
            shm = self._shared_buffers[buffer_id]
        self._parse_input_queue.put(("buffer", generation, buffer_id, shm.name, valid_size_int))

    def _handle_parser_result(self, result: dict) -> None:
        kind = result.get("kind")
        if kind == "buffer_done":
            with self._lock:
                self._free_buffer_ids.append(int(result["buffer_id"]))
        elif kind == "finish_done":
            self._finish_results[int(result["generation"])] = result
        elif kind == "error":
            self._parser_errors.append(str(result.get("error")))

    def _drain_parser_results(self) -> None:
        while True:
            try:
                result = self._parse_output_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_parser_result(result)

    def is_generation_ready(self, generation: int) -> bool:
        self._drain_parser_results()
        return generation in self._finish_results or bool(self._parser_errors)

    def _flush(self, flag: int) -> None:
        self._check(self._libcupti.cuptiActivityFlushAll(flag))

    def _set_flush_period_ms(self, period_ms: int) -> None:
        if period_ms == self._current_flush_period_ms:
            return
        self._check(self._libcupti.cuptiActivityFlushPeriod(period_ms))
        self._current_flush_period_ms = period_ms

    def _begin(
        self,
        mode: str,
        filter_plan=(),
        flush_period_ms: int = 0,
        collect_timing: bool = False,
    ) -> int:
        start_timing: dict[str, float] = {}
        with self._lock:
            self._mode = "drop"
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._flush(1)
        if collect_timing:
            start_timing["forced_flush_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._drain_parser_results()
        if collect_timing:
            start_timing["drain_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._mode = mode
            self._local_completed = []
            self._filter_plan = filter_plan
        if flush_period_ms > 0:
            phase_start_s = time.perf_counter() if collect_timing else 0.0
            self._set_flush_period_ms(flush_period_ms)
            if collect_timing:
                start_timing["period_enable_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        self._last_start_timing = start_timing
        return generation

    def capture_names(self, replay_fn) -> tuple[list[tuple], int, dict]:
        """Run a small calibration replay and parse kernel names locally."""
        self._begin("local")
        replay_fn()
        torch.cuda.synchronize()
        self._flush(0)
        records: list[tuple] = []
        zero_ts_count = 0
        zero_ts_names: dict[str, int] = {}
        with self._lock:
            completed = list(self._local_completed)
            self._local_completed = []
            self._mode = "drop"
        for buffer_id, valid_size in completed:
            ptr = self._buffer_ptr(buffer_id)
            recs, zeros, zero_names = _parse_cupti_buffer_ptr(
                self._libcupti,
                ptr,
                valid_size,
                include_names=True,
            )
            records.extend(recs)
            zero_ts_count += zeros
            for name, count in zero_names.items():
                zero_ts_names[name] = zero_ts_names.get(name, 0) + count
            ctypes.memset(ptr, 0, _CUPTI_HOST_BUFFER_BYTES)
            with self._lock:
                self._free_buffer_ids.append(buffer_id)
        records.sort(key=lambda r: r[1])
        return records, zero_ts_count, zero_ts_names

    def start(
        self,
        filter_plan=(),
        flush_period_ms: int = 0,
        collect_timing: bool = False,
    ) -> None:
        self._begin("parser", filter_plan, flush_period_ms, collect_timing)

    def stop_async(
        self,
        collect_timing: bool = False,
        stats_request: dict | None = None,
    ) -> tuple[int, dict[str, float]]:
        stop_timing: dict[str, float] = {}
        generation = self._generation
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._set_flush_period_ms(0)
        if collect_timing:
            stop_timing["period_disable_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        self._flush(0)
        if collect_timing:
            stop_timing["flush_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
        with self._lock:
            self._mode = "drop"
            filter_plan = self._filter_plan
        self._parse_input_queue.put(("finish", generation, filter_plan, stats_request))
        self._last_stop_timing = stop_timing
        return generation, stop_timing

    def wait_for_generation_result(
        self,
        generation: int,
        stop_timing: dict[str, float] | None = None,
        collect_timing: bool = False,
    ) -> dict:
        if stop_timing is None:
            stop_timing = {}
        phase_start_s = time.perf_counter() if collect_timing else 0.0
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            result = self._finish_results.pop(generation, None)
            if result is not None:
                if collect_timing:
                    stop_timing["parser_wait_ms"] = 1000.0 * (time.perf_counter() - phase_start_s)
                    stop_timing["total_ms"] = (
                        stop_timing.get("period_disable_ms", 0.0)
                        + stop_timing.get("flush_ms", 0.0)
                        + stop_timing["parser_wait_ms"]
                    )
                self._last_stop_timing = stop_timing
                return result
            timeout_s = max(0.0, min(0.01, deadline - time.perf_counter()))
            try:
                parser_result = self._parse_output_queue.get(timeout=timeout_s)
            except queue.Empty:
                continue
            self._handle_parser_result(parser_result)
            if self._parser_errors:
                raise RuntimeError("CUPTI parser process failed: " + "; ".join(self._parser_errors))
        raise TimeoutError("Timed out waiting for CUPTI parser process")

    def wait_for_generation(
        self,
        generation: int,
        stop_timing: dict[str, float] | None = None,
        collect_timing: bool = False,
    ) -> tuple[list[tuple], int, dict, int]:
        result = self.wait_for_generation_result(generation, stop_timing, collect_timing)
        return (
            list(result["records"]),
            int(result["zero_ts_count"]),
            dict(result["zero_ts_names"]),
            int(result["raw_record_count"]),
        )

    def stop(self, collect_timing: bool = False) -> tuple[list[tuple], int, dict, int]:
        generation, stop_timing = self.stop_async(collect_timing)
        return self.wait_for_generation(generation, stop_timing, collect_timing)

    def last_start_timing(self) -> dict[str, float]:
        return dict(self._last_start_timing)

    def last_stop_timing(self) -> dict[str, float]:
        return dict(self._last_stop_timing)

    def close(self) -> None:
        parse_process = getattr(self, "_parse_process", None)
        if parse_process is not None and parse_process.is_alive():
            self._parse_input_queue.put(None)
            parse_process.join(timeout=5.0)
            if parse_process.is_alive():
                parse_process.terminate()
                parse_process.join(timeout=1.0)
        for shm in getattr(self, "_shared_buffers", {}).values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


# =============================================================================
# Timing helpers
# =============================================================================

