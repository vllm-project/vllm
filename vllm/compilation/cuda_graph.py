# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig, CUDAGraphRuntimeStyle
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors

logger = init_logger(__name__)


@dataclasses.dataclass
class CUDAGraphEntry:
    runtime_shape: int
    num_finished_warmup: int = 0
    runnable: Callable = None  # type: ignore
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None

    usage_type: Optional[str] = None  # For debug logging only


class CUDAGraphWrapper:
    """
    This class simply wrap a runnable for cudagraph functionality,
    taking responsibility of capturing cudagraph and running the replay.
    """

    def __init__(self, runnable: Any, vllm_config: VllmConfig, graph_pool: Any,
                 runtime_style: CUDAGraphRuntimeStyle,
                 cudagraph_specific_config: dict[str, Any]={}):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.graph_pool = graph_pool
        self.runtime_style = runtime_style
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        
        assert self.runtime_style >= CUDAGraphRuntimeStyle.PIECEWISE
        assert graph_pool is not None
        self.debug_capturing = cudagraph_specific_config.get(
            "debug_capturing", True)
        self.gc_disable = cudagraph_specific_config.get(
            "gc_disable", False)
        self.weak_ref_output = cudagraph_specific_config.get(
            "weak_ref_output", True)
        usage_type = cudagraph_specific_config.get("usage_type", None)
        self.cudagraph_capture_sizes: set[int] = set(
            self.compilation_config.cudagraph_capture_sizes
        )
        # the entries for different shapes that we need to capture cudagraph
        self.concrete_cudagraph_entries: dict[int, CUDAGraphEntry] = {}

        for shape in self.cudagraph_capture_sizes:
            
            self.concrete_cudagraph_entries[shape] = CUDAGraphEntry(
                runtime_shape=shape,
                runnable=self.runnable,
                usage_type=usage_type,  # for debug logging only
            )
    
    def maybe_replace_runnable(self, shape: int, runnable: Callable):
        # this is a hack to replace a general shape runnable with a compiled
        # runnable of a specific shape.
        if shape not in self.concrete_cudagraph_entries:
            return
        entry = self.concrete_cudagraph_entries[shape]
        assert entry.cudagraph is None, "Cudagraph is already captured"
        entry.runnable = runnable
            
    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        runtime_shape = forward_context.num_tokens
        cudagraph_runtime_style = forward_context.cudagraph_runtime_style

        if cudagraph_runtime_style == CUDAGraphRuntimeStyle.NONE or\
                                                    runtime_shape is None:
            # TODO: make sure here is on profile running or eager running
            return self.runnable(*args, **kwargs)
        if cudagraph_runtime_style != self.runtime_style:
            # CUDAGraph runtime style don't match the current
            # configuration, so directly call runnable eagerly 
            # as it's always safe. 
            return self.runnable(*args, **kwargs)

        if runtime_shape not in self.concrete_cudagraph_entries:
            # we don't need to do anything for this shape.
            return self.runnable(*args, **kwargs)

        entry = self.concrete_cudagraph_entries[runtime_shape]
        

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.debug_capturing:
                    logger.debug(
                        "Warming up %s/%s of %s usage for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_config.cudagraph_num_of_warmups,
                        entry.usage_type, entry.runtime_shape)
                return entry.runnable(*args, **kwargs)

            if self.debug_capturing:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. We only log it in the debug mode.
                logger.debug(
                    "Capturing a cudagraph of %s usage for shape %s",
                    entry.usage_type, entry.runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.gc_disable:
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
                    output = entry.runnable(*args, **kwargs)
                    if self.weak_ref_output:
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
                "Input addresses for cudagraphs are different during "
                f"replay. Expected {entry.input_addresses}, got "
                f"{new_input_addresses}")

        entry.cudagraph.replay()
        return entry.output