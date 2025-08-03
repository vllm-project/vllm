import contextlib
import copy
import torch
from typing import Callable, ContextManager, List, Optional

from vllm.compilation.nano_utils import (
    NanoOpInfo,
    NanoSplitConfig,
    FakeModule,
    display_graph,
    get_split_config,
    tag_graph,
)
from vllm.compilation.nano_split import (
    analyze_graph,
    concat_outputs,
    split_computations,
    split_inputs,
)


class NanoSplitManager:
    def __init__(
        self, graph_module: torch.fx.GraphModule, max_nano_splits: int = 2
    ) -> None:
        self.graph_module = graph_module
        self.original_graph = graph_module.graph

        # Nano split preparation
        self.max_nano_splits = max_nano_splits
        self.new_graphs = {1: self.original_graph}

        # Runtime preparation
        self.cached_config: Optional[NanoSplitConfig] = None
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()
        self.hook: Optional[Callable[[NanoOpInfo], ContextManager[None]]] = None

        # Initialize the base graph
        tag_graph(self.graph_module, {
            "vllm.unified_attention": "attention",
            "vllm.unified_attention_with_output": "attention",
            "vllm.all_reduce": "all_reduce",
        })
        splittable_inputs, base_graph = analyze_graph(self.original_graph)
        for num_splits in range(2, max_nano_splits + 1):
            new_graph = copy.deepcopy(base_graph)
            nano_batch_sizes = []
            for i in range(num_splits):
                nano_batch_sizes.append(
                    new_graph.call_module(
                        "get_batch_size",
                        args=(i,),
                        kwargs={},
                    )
                )
            mapping = split_inputs(
                new_graph, splittable_inputs, "split_input", num_splits
            )
            split_computations(
                self.original_graph,
                new_graph,
                mapping,
                nano_batch_sizes,
                "op_wrapper",
                num_splits,
            )
            concat_outputs(
                self.original_graph,
                new_graph,
                mapping,
            )
            self.new_graphs[num_splits] = new_graph
            print(new_graph)
            self.graph_module.graph = new_graph
            display_graph(self.graph_module, f"after nano split {num_splits}")

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
                self.graph_module.graph = self.original_graph
                return self.graph_module(*args, **kwargs)

            num_nano_batches = self.cached_config.num_nano_batches
            # NOTE(yi): This can be time consuming
            if self.graph_module.graph != self.new_graphs[num_nano_batches]:
                self.graph_module.graph = self.new_graphs[num_nano_batches]
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
                gm=self.graph_module,
                hooks=[
                    set_stream,
                    nvtx_mark,
                    self.hook,
                ],
            )
            setattr(self.graph_module, "op_wrapper", op_wrapper)
            get_batch_size = FakeModule(
                NanoSplitManager.get_batch_size,
                cached_config=self.cached_config,
            )
            setattr(self.graph_module, "get_batch_size", get_batch_size)
            split_input = FakeModule(
                NanoSplitManager.split_input,
                cached_config=self.cached_config,
            )
            setattr(self.graph_module, "split_input", split_input)
            output = self.graph_module(*args, **kwargs)
            delattr(self.graph_module, "op_wrapper")
            delattr(self.graph_module, "get_batch_size")
            delattr(self.graph_module, "split_input")
            return output

        return _forward

    def prepare(
        self,
        batch_size: int,
        num_tokens: List[int],
        cached_seqlens: List[int],
    ) -> NanoSplitConfig:
        self.cached_config = get_split_config(batch_size, num_tokens, cached_seqlens)
        return self.cached_config
    
    def set_hooks(self, op_hook: Callable[[NanoOpInfo], ContextManager[None]]):
        self.hook = op_hook


_split_manager = None


def get_callable(graph_module: torch.fx.GraphModule) -> Callable:
    global _split_manager
    if _split_manager is None:
        _split_manager = NanoSplitManager(graph_module)
    return _split_manager.get_callable()


def prepare_nano_split(
    batch_size: int,
    num_tokens: List[int],
    cached_seqlens: List[int],
) -> NanoSplitConfig:
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    return _split_manager.prepare(
        batch_size, num_tokens, cached_seqlens
    )

def set_op_hook(op_hook: Callable[[NanoOpInfo], ContextManager[None]]):
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    _split_manager.set_hooks(op_hook)
