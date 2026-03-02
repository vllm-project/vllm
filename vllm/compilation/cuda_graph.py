# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream, weak_ref_tensors

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str


class CUDAGraphLogging:
    """Aggregate and log cudagraph metrics"""

    COLUMN_HEADERS = [
        "Unpadded Tokens",
        "Padded Tokens",
        "Num Paddings",
        "Runtime Mode",
        "Count",
    ]

    def __init__(
        self, cg_mode: CUDAGraphMode, cg_capture_sizes: list[int] | None
    ) -> None:
        self.reset()
        self.cg_mode = str(cg_mode)
        self.cg_capture_sizes = str(cg_capture_sizes or [])

        self.settings_header = (
            "**CUDAGraph Config Settings:**\n\n"
            f"- Mode: {self.cg_mode}\n"
            f"- Capture sizes: {self.cg_capture_sizes}\n\n"
            "**CUDAGraph Stats:**\n\n"
        )

    def reset(self) -> None:
        self.stats: list[CUDAGraphStat] = []

    def observe(self, cudagraph_stat: CUDAGraphStat) -> None:
        self.stats.append(cudagraph_stat)

    def generate_metric_table(self) -> str:
        stats_counts = Counter(self.stats)

        # Convert stats to rows of strings, in descending order of observed frequencies
        rows = []
        for stat, count in sorted(
            stats_counts.items(), key=lambda item: item[1], reverse=True
        ):
            rows.append(
                [
                    str(stat.num_unpadded_tokens),
                    str(stat.num_padded_tokens),
                    str(stat.num_paddings),
                    stat.runtime_mode,
                    str(count),
                ]
            )

        # Calculate column widths (max of header and data)
        col_widths = []
        for i, header_text in enumerate(self.COLUMN_HEADERS):
            max_width = len(header_text)
            for row in rows:
                max_width = max(max_width, len(row[i]))
            col_widths.append(max_width)

        table_header_list = [
            h.ljust(w) for h, w in zip(self.COLUMN_HEADERS, col_widths)
        ]
        table_header = "| " + " | ".join(table_header_list) + " |\n"

        table_separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n"

        # Create data rows with proper alignment
        data_rows = []
        for row in rows:
            formatted_row = [
                str(val).ljust(width) for val, width in zip(row, col_widths)
            ]
            data_rows.append("| " + " | ".join(formatted_row) + " |")

        return (
            self.settings_header
            + table_header
            + table_separator
            + "\n".join(data_rows)
            + "\n"
        )

    def log(self, log_fn: Callable[..., Any] = logger.info) -> None:
        if not self.stats:
            return
        log_fn(self.generate_metric_table())
        self.reset()


def _extract_tensor_addresses(
    obj: Any, prefix: str = "", visited: set | None = None
) -> dict[str, int]:
    """
    Recursively extract memory addresses of CUDA tensors from nested structures.
    Uses visited set to prevent infinite recursion, but never adds tensors to visited
    since views can recreate Python objects pointing to same memory.
    """
    if visited is None:
        visited = set()

    addresses: dict[str, int] = {}

    # Handle tensor before checking visited, since we don't add tensors to visited
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            addresses[prefix] = obj.data_ptr()
        return addresses

    # Prevent infinite recursion for container objects
    if id(obj) in visited:
        return addresses

    visited.add(id(obj))

    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else str(k)
                addresses.update(_extract_tensor_addresses(v, new_prefix, visited))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                addresses.update(_extract_tensor_addresses(v, new_prefix, visited))
        elif dataclasses.is_dataclass(obj):
            for field in dataclasses.fields(obj):
                new_prefix = f"{prefix}.{field.name}" if prefix else field.name
                try:
                    v = getattr(obj, field.name)
                    addresses.update(_extract_tensor_addresses(v, new_prefix, visited))
                except AttributeError:
                    pass
        elif hasattr(obj, "__dict__"):
            # Use vars(obj) instead of dir() to avoid triggering @property getters
            try:
                for k, v in vars(obj).items():
                    if not k.startswith("_") and not isinstance(
                        getattr(type(obj), k, None), property
                    ):
                        new_prefix = f"{prefix}.{k}" if prefix else k
                        addresses.update(
                            _extract_tensor_addresses(v, new_prefix, visited)
                        )
            except TypeError:
                pass
    finally:
        pass

    return addresses


@dataclasses.dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: dict[str, int] | None = None


@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: CUDAGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}

    def __getattr__(self, key: str) -> Any:
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def unwrap(self) -> Callable[..., Any]:
        # in case we need to access the original runnable.
        return self.runnable

    def _get_inputs_to_extract(
        self, forward_context: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            target_func = getattr(self.runnable, "forward", self.runnable)
            sig = inspect.signature(target_func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            inputs_to_extract = dict(bound_args.arguments)
        except (ValueError, TypeError):
            inputs_to_extract = {"args": args, "kwargs": kwargs}

        if self.runtime_mode == CUDAGraphMode.FULL:
            inputs_to_extract["attn_metadata"] = forward_context.attn_metadata

        return inputs_to_extract

    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            if self.is_debugging_mode:
                inputs_to_extract = self._get_inputs_to_extract(
                    forward_context, args, kwargs
                )
                entry.input_addresses = _extract_tensor_addresses(inputs_to_extract)

            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())

                # Sync offloader's copy stream before capture.
                # Ensure any pre-capture prefetches from offloader are complete.
                get_offloader().sync_prev_onload()

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(
                    cudagraph,
                    pool=self.graph_pool,
                    stream=current_stream(),
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    # Join offloader's copy stream after forward to avoid
                    # unjoined stream error. The last layer's start_prefetch
                    # forks copy_stream, but wait_prefetch only happens in
                    # the next forward pass.
                    get_offloader().join_after_forward()
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            inputs_to_extract = self._get_inputs_to_extract(
                forward_context, args, kwargs
            )
            new_input_addresses = _extract_tensor_addresses(inputs_to_extract)

            if new_input_addresses != entry.input_addresses:
                # Find the differences
                old_keys = set(
                    entry.input_addresses.keys() if entry.input_addresses else []
                )
                new_keys = set(new_input_addresses.keys())

                missing_keys = old_keys - new_keys
                added_keys = new_keys - old_keys

                mismatched_keys = []
                for k in old_keys.intersection(new_keys):
                    if (
                        entry.input_addresses is not None
                        and entry.input_addresses[k] != new_input_addresses[k]
                    ):
                        mismatched_keys.append(
                            (k, entry.input_addresses[k], new_input_addresses[k])
                        )

                error_msg = [
                    "Input addresses for cudagraphs are different during replay."
                ]
                if missing_keys:
                    error_msg.append(f"Missing keys: {missing_keys}")
                if added_keys:
                    error_msg.append(f"Added keys: {added_keys}")
                if mismatched_keys:
                    error_msg.append(
                        "Mismatched addresses (key, old_address, new_address):"
                    )
                    for k, old_addr, new_addr in mismatched_keys:
                        error_msg.append(f"  {k}: {old_addr} vs {new_addr}")

                raise AssertionError("\n".join(error_msg))

        # Sync offloader before replay - ensures any external dependencies
        # from pre-capture prefetches are satisfied.
        get_offloader().sync_prev_onload()
        entry.cudagraph.replay()
        return entry.output
