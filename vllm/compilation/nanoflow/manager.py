import contextlib
import copy
import torch
from typing import Callable, ContextManager, List, Optional

import torch.fx.graph_module

from vllm.compilation.nanoflow.split_utils import (
    analyze_graph,
    split_graph,
    NanoOpInfo,
    NanoSplitConfig,
    FakeModule,
    get_split_config,
    tag_graph,
)
from vllm.config import VllmConfig


class NanoSplitManager:
    def __init__(
        self, graph_module: torch.fx.GraphModule, vllm_config: VllmConfig,
    ) -> None:
        self.original_graph_module = graph_module
        self.original_graph = graph_module.graph

        # Nano split preparation
        self.min_nano_split_tokens = vllm_config.model_config.min_nano_split_tokens
        self.max_num_nano_batches = vllm_config.model_config.max_num_nano_batches
        # Initialize the base graph
        tag_graph(
            self.original_graph_module,
            {
                "vllm.unified_attention": "attention",
                "vllm.unified_attention_with_output": "attention",
                "vllm.all_reduce": "all_reduce",
            },
        )
        self.graph_modules = {1: self.original_graph_module}

        # Runtime preparation
        self.cached_config: Optional[NanoSplitConfig] = None
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()
        self.hook: Optional[Callable[[NanoOpInfo], ContextManager[None]]] = None
        self.get_bs_fn = "get_batch_size"
        self.split_fn = "split_input"
        self.wrapper_fn = "op_wrapper"
        setattr(self.original_graph_module, self.get_bs_fn, None)
        setattr(self.original_graph_module, self.split_fn, None)
        setattr(self.original_graph_module, self.wrapper_fn, None)

        splittable_inputs, base_graph = analyze_graph(self.original_graph)
        for num_splits in range(2, self.max_num_nano_batches + 1):
            new_graph = copy.deepcopy(base_graph)
            split_graph(
                self.original_graph,
                out=new_graph,
                splittable_inputs=splittable_inputs,
                num_splits=num_splits,
                get_bs_fn=self.get_bs_fn,
                split_fn=self.split_fn,
                wrapper_fn=self.wrapper_fn,
            )
            new_graph_module = torch.fx.GraphModule(self.original_graph_module, new_graph)
            for name, _ in self.original_graph_module.named_modules():
                if "." in name or name == "":
                    continue
                torch.fx.graph_module._copy_attr(self.original_graph_module, new_graph_module, name)
            self.graph_modules[num_splits] = new_graph_module


    @staticmethod
    def get_batch_size(idx: int, cached_config: NanoSplitConfig):
        return cached_config.num_tokens[idx]

    @staticmethod
    def split_input(x: torch.Tensor, idx: int, cached_config: NanoSplitConfig):
        return x[
            cached_config.split_indices[idx] : cached_config.split_indices[idx + 1]
        ]

    @staticmethod
    def op_wrapper(
        submod_name: str,
        idx: int,
        args: tuple,
        kwargs: dict,
        gm: torch.fx.GraphModule,
        hooks: List[Callable[[NanoOpInfo], ContextManager[None]]],
    ):
        module = getattr(gm, submod_name)
        tag = getattr(module, "tag", "")
        with contextlib.ExitStack() as stack:
            for hook in hooks:
                stack.enter_context(
                    hook(NanoOpInfo(submod_name, tag, idx, args, kwargs))
                )
            output = module(*args, **kwargs)
        return output

    def get_callable(self) -> Callable:
        def _forward(*args, **kwargs):
            if self.cached_config is None:
                return self.original_graph_module(*args, **kwargs)

            num_nano_batches = self.cached_config.num_nano_batches
            comm_finished = [None for _ in range(num_nano_batches)]
            comp_finished = [None for _ in range(num_nano_batches)]

            @contextlib.contextmanager
            def set_stream(op_info: NanoOpInfo):
                if op_info.tag == "all_reduce":
                    torch.cuda.set_stream(self.comm_stream)  # type: ignore
                    comm_finished[op_info.idx] = torch.cuda.Event()  # type: ignore
                    if comp_finished[op_info.idx] is not None:
                        comp_finished[op_info.idx].wait()  # type: ignore
                        comp_finished[op_info.idx] = None
                else:
                    torch.cuda.set_stream(self.comp_stream)  # type: ignore
                    comp_finished[op_info.idx] = torch.cuda.Event()  # type: ignore
                    if comm_finished[op_info.idx] is not None:
                        comm_finished[op_info.idx].wait()  # type: ignore
                        comm_finished[op_info.idx] = None
                try:
                    yield
                finally:
                    if op_info.tag == "all_reduce":
                        comm_finished[op_info.idx].record()  # type: ignore
                    else:
                        comp_finished[op_info.idx].record()  # type: ignore

            @contextlib.contextmanager
            def nvtx_mark(op_info: NanoOpInfo):
                try:
                    with torch.cuda.nvtx.range(
                        f"op_{op_info.submod_name}_{op_info.tag}_{op_info.idx}"
                    ):
                        yield
                except Exception as e:
                    print(f"Error in nvtx_mark: {e}")
                    raise e

            # Register fake modules
            assert self.hook is not None
            op_wrapper = FakeModule(
                NanoSplitManager.op_wrapper,
                gm=self.graph_modules[num_nano_batches],
                hooks=[
                    set_stream,
                    nvtx_mark,
                    self.hook,
                ],
            )
            get_batch_size = FakeModule(
                NanoSplitManager.get_batch_size,
                cached_config=self.cached_config,
            )
            split_input = FakeModule(
                NanoSplitManager.split_input,
                cached_config=self.cached_config,
            )
            setattr(self.graph_modules[num_nano_batches], self.wrapper_fn, op_wrapper)
            setattr(self.graph_modules[num_nano_batches], self.get_bs_fn, get_batch_size)
            setattr(self.graph_modules[num_nano_batches], self.split_fn, split_input)
            output = self.graph_modules[num_nano_batches](*args, **kwargs)
            return output

        return _forward

    def prepare(
        self,
        batch_size: int,
        num_tokens: List[int],
        cached_seqlens: List[int],
    ) -> NanoSplitConfig:
        self.cached_config = get_split_config(
            batch_size,
            num_tokens,
            cached_seqlens,
            self.max_num_nano_batches,
            self.min_nano_split_tokens,
        )
        return self.cached_config

    def set_hooks(self, op_hook: Callable[[NanoOpInfo], ContextManager[None]]):
        self.hook = op_hook


_split_manager = None


def get_callable(graph_module: torch.fx.GraphModule, vllm_config: VllmConfig) -> Callable:
    global _split_manager
    if _split_manager is None:
        _split_manager = NanoSplitManager(graph_module, vllm_config)
    return _split_manager.get_callable()


def prepare_nano_split(
    batch_size: int,
    num_tokens: List[int],
    cached_seqlens: List[int],
) -> NanoSplitConfig:
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    return _split_manager.prepare(batch_size, num_tokens, cached_seqlens)


def set_op_hook(op_hook: Callable[[NanoOpInfo], ContextManager[None]]):
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    _split_manager.set_hooks(op_hook)
