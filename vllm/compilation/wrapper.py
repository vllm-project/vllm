# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from types import CodeType
from typing import Callable, Optional

import torch

import vllm.envs as envs
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class for torch.compile, with a custom dispatch logic.
    Subclasses should:
    1. Implement the forward method
    2. Implement the dispatch logic in the __call__ method
        It can use `self.compiled_codes` to access the compiled bytecode,
        and `with self.dispatch_to_code(index):` to dispatch to
        the compiled code.
    3. Implement the `__init__` method to determine how to call
        `torch.compile` over the forward method.
    """

    def __init__(self,
                 compiled_callable: Optional[Callable] = None,
                 compilation_level: int = 0):

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        if compiled_callable is None:
            # default compilation settings
            # compiling the forward method

            backend = vllm_config.compilation_config.init_backend(vllm_config)
            options = None
            if isinstance(backend, str) and backend == "inductor":
                options = get_current_vllm_config(
                ).compilation_config.inductor_compile_config

            compiled_callable = torch.compile(
                self.forward,
                fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                backend=backend,
                options=options)

        self.compiled_callable = compiled_callable
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: list[CodeType] = []
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

        # read the env var to determine whether to use the custom dispatcher
        # subclasses can use this to switch between the custom dispatcher
        # and the default Dynamo guard mechanism.
        self.use_custom_dispatcher: bool = \
            compilation_level >= CompilationLevel.DYNAMO_ONCE

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile level.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        return self.compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object:
            return
        # code borrowed from https://github.com/thuml/depyf/blob/f4ad79fadee27ea113b4c75202db1eb1a11c0dbc/depyf/explain/enable_debugging.py#L25
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == old_code

        if frame.f_locals["self"] is not self:
            return

        self.compiled_codes.append(new_code)
        debug_dump_dir = self.vllm_config.compilation_config.debug_dump_path
        if isinstance(debug_dump_dir, str) and debug_dump_dir != "":
            rank = self.vllm_config.parallel_config.rank
            decompiled_file = os.path.join(debug_dump_dir, f"rank_{rank}",
                                           "transformed_code.py")
            if not os.path.exists(decompiled_file):
                try:
                    # usually the decompilation will succeed for most models,
                    # as we guarantee a full-graph compilation in Dynamo.
                    # but there's no 100% guarantee, since decompliation is
                    # not a reversible process.
                    import depyf
                    src = depyf.decompile(new_code)

                    with open(decompiled_file, "w") as f:
                        f.write(src)

                    logger.debug("Dynamo transformed code saved to %s",
                                 decompiled_file)
                except Exception:
                    pass

        if self.vllm_config.compilation_config.use_cudagraph and \
            "update" in new_code.co_names:
            import depyf
            src = depyf.decompile(new_code)
            msg = "Assigning / modifying buffers of nn.Module during forward pass is not allowed when using cudagraph inside the compiler because it will cause silent errors. Please use eager mode or fix the code. The following code contains clues about which buffer is being modified (please search for the usage of the function `update`):\n" + src  # noqa
            raise RuntimeError(msg)

    @contextmanager
    def dispatch_to_code(self, index: int):
        """Context manager to dispatch to the compiled code.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """ # noqa
        self.__class__.forward.__code__ = self.compiled_codes[index]
        yield
        self.__class__.forward.__code__ = self.original_code_object


class CudaGraphWrapper:

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        # configs
        self.cudagraph_capture_sizes = set(
            self.compilation_config.cudagraph_capture_sizes)
        self.cudagraph_num_of_warmups = (
            self.compilation_config.cudagraph_num_of_warmups)
        assert self.compilation_config.simple_cuda_graph
        assert self.compilation_config.full_cuda_graph

        # states
        # batch size -> graph
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        # batch size -> hidden states
        self.hidden_states: dict[int, torch.Tensor] = {}
        # batch size -> number of warmups
        self.num_warmups: dict[int, int] = defaultdict(int)
        # Special flag to handle the first memory profiling run.
        self.first_run_finished = False

    def capture_graph(self, *args, **kwargs) -> None:
        batch_size = self._get_batch_size(*args, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool):
            hidden_states = self.forward(*args, **kwargs)
        self.hidden_states[batch_size] = hidden_states
        self.graphs[batch_size] = graph

    def forward_graph(self, *args, **kwargs) -> torch.Tensor:
        if not self.first_run_finished:
            # Memory profiling run.
            self.first_run_finished = True
            return self.forward(*args, **kwargs)

        batch_size = self._get_batch_size(*args, **kwargs)
        if batch_size not in self.cudagraph_capture_sizes:
            # Run in eager mode.
            return self.forward(*args, **kwargs)

        if self.num_warmups[batch_size] < self.cudagraph_num_of_warmups:
            # Warmup mode. Run in eager mode.
            self.num_warmups[batch_size] += 1
            return self.forward(*args, **kwargs)

        if batch_size not in self.graphs:
            # Capture the graph.
            self.capture_graph(*args, **kwargs)
            return self.hidden_states[batch_size]

        # Run the graph and return the hidden states.
        graph = self.graphs[batch_size]
        graph.replay()
        hidden_states = self.hidden_states[batch_size]
        return hidden_states

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def _get_batch_size(self, *args, **kwargs) -> int:
        # NOTE(woosuk): Ensure that the keyword arguments here match those
        # in the model's forward method signature.
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            return input_ids.shape[0]
        input_embeds = kwargs.get("inputs_embeds")
        if input_embeds is not None:
            return input_embeds.shape[0]
        intermediate_tensors = kwargs.get("intermediate_tensors")
        if intermediate_tensors is not None:
            return intermediate_tensors.shape[0]
        # NOTE(woosuk): We don't use the `positions` tensor for batch size
        # because its first dimension may not be the batch dimension for some
        # models such as Qwen2.5-VL.
        if len(args) > 0:
            # For LoRA models, kwargs could be empty.
            # FIXME(woosuk): This is a hack. We should find a more robust way
            # to get the batch size.
            return args[0].shape[0]
        raise ValueError("No batch size found in arguments")
