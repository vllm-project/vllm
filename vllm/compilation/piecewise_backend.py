# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from typing import Any, Callable, Optional

import torch
import torch.fx as fx

import vllm.envs as envs
from vllm.compilation.backends import VllmBackend
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig, CUDAGraphRuntimeStyle
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import resolve_obj_by_qualname
logger = init_logger(__name__)


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in cudagraph_capture_sizes
    compiled: bool = False
    runnable: Callable = None  # type: ignore

    usage_type: Optional[str] = None  # For debug logging only


class PiecewiseBackend:

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig,
                 graph_pool: Any, piecewise_compile_index: int,
                 total_piecewise_compiles: int, sym_shape_indices: list[int],
                 compiled_graph_for_general_shape: Callable,
                 vllm_backend: VllmBackend):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, the static graph capturing (e.g. CUDA graph) is handled 
        by a separate static graph wrapper, which is expected to wrap the 
        compiled callable of the general shape.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.is_full_graph = total_piecewise_compiles == 1

        self.compile_sizes: set[int] = set(
            self.compilation_config.compile_sizes)
        

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()

        usage_type = "full/general" if self.is_full_graph else \
                                                    "piecewise/general"

        self.cudagraph_capture_sizes: set[int] = set()
        self.cudagraph_runable: Optional[CUDAGraphWrapper] = None
        if self.compilation_config.cudagraph_mode > 0:
            cudagraph_specific_config = {
                    "debug_capturing": self.is_first_graph,
                    "gc_disable": not self.is_first_graph,
                    "weak_ref_output": self.is_last_graph,
                    "usage_type" : usage_type }
            
            # Note: To easier distinguish whether it is under the 
            # piecewise backend, we always assume CUDAGraphRuntimeStyle.PIECEWISE 
            # here, no matter it is on a full fx graph or piecewise fx graph. 

            static_graph_wrapper_class = resolve_obj_by_qualname(
                current_platform.get_static_graph_wrapper_cls())
            self.cudagraph_runable = static_graph_wrapper_class(
                self.compiled_graph_for_general_shape,
                vllm_config,
                self.graph_pool,
                runtime_style = CUDAGraphRuntimeStyle.PIECEWISE,
                cudagraph_specific_config = cudagraph_specific_config)
            
            self.cudagraph_capture_sizes = (self.compilation_config.\
                                            cudagraph_capture_sizes)
        

        # We now only keep compilation management inside this class directly.
        # The cudagraph logic is delegated to the CUDAGraphWrapper class.
        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.cudagraph_capture_sizes,
                runnable=self.compiled_graph_for_general_shape,
                usage_type=usage_type,  # for debug logging only
            )


    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.vllm_backend.compiler_manager.save_to_file()
            end_monitoring_torch_compile(self.vllm_config)

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)
        
        runtime_shape = args[self.sym_shape_indices[0]]
        if self.is_debugging_mode:
            assert runtime_shape==get_forward_context().num_tokens

        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape)
            
            # replace the runnable with the compiled one for
            # cudagraph capturing
            if self.cudagraph_runable is not None:
                self.cudagraph_runable.maybe_replace_runnable(runtime_shape, 
                                                            entry.runnable)

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()
        
        if not entry.use_cudagraph:
            return entry.runnable(*args)
        
        # safety check to ensure the cudagraph runnable is not None
        assert self.cudagraph_runable is not None
        return self.cudagraph_runable(*args)

