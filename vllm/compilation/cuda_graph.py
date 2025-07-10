# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import CUDAGraphRuntimeStyle, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import weak_ref_tensors

logger = init_logger(__name__)


@dataclasses.dataclass
class CUDAGraphEntry:
    runtime_shape: int
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


@dataclasses.dataclass
class CUDAGraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True
    usage_str: Optional[str] = None  # For debug logging only


class CUDAGraphWrapper:
    """
    This class simply wrap a runnable for cudagraph functionality,
    taking responsibility of capturing cudagraph and running the replay.
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_style: CUDAGraphRuntimeStyle,
                 graph_pool: Any = None,
                 cudagraph_options: Optional[CUDAGraphOptions] = None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.graph_pool = graph_pool
        self.runtime_style = runtime_style
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_style is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_style != CUDAGraphRuntimeStyle.NONE
        if self.graph_pool is None:
            self.graph_pool = current_platform.get_default_cudagraph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options

        self.cudagraph_capture_sizes: set[int] = set(
            self.compilation_config.cudagraph_capture_sizes)
        # the entries for different shapes that we need to capture cudagraph
        self.concrete_cudagraph_entries: dict[int, CUDAGraphEntry] = {}

        for shape in self.cudagraph_capture_sizes:
            self.concrete_cudagraph_entries[shape] = CUDAGraphEntry(
                runtime_shape=shape)

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        runtime_shape = forward_context.num_tokens
        cudagraph_runtime_style = forward_context.cudagraph_runtime_style

        if cudagraph_runtime_style == CUDAGraphRuntimeStyle.NONE or\
                                                    runtime_shape is None:
            # make sure it's on profile run, eager run, or warmup stage.
            return self.runnable(*args, **kwargs)
        if cudagraph_runtime_style != self.runtime_style:
            # Only triggers capture/replay if the runtime style matches,
            # otherwise, we fallback to the original runnable to handle
            # no match case. This is a hack to avoid double capturing
            # cudagraph and ensure extra safety in situations where we
            # have nested CUDAdGraphWrapper structure, e.g., we have
            # piecewise cudagraph for piecewise backend, which may be
            # further wrapped to obtain a full cudagraph. See #20059 for
            # more details.
            return self.runnable(*args, **kwargs)

        if runtime_shape not in self.concrete_cudagraph_entries:
            # we don't need to do anything for this shape.
            return self.runnable(*args, **kwargs)

        entry = self.concrete_cudagraph_entries[runtime_shape]

        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. We only log it in the debug mode.
                logger.debug("Capturing a cudagraph of %s usage for shape %s",
                             self.cudagraph_options.usage_str,
                             entry.runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
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
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last
                        # graph will not be used by any other cuda graph.
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
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs of "
                f"{self.cudagraph_options.usage_str} are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}")

        entry.cudagraph.replay()
        return entry.output
