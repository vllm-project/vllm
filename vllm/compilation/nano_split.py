import contextlib
import copy
from dataclasses import dataclass
import torch
from typing import Callable, ContextManager, Dict, List, Tuple, Optional, Set

@dataclass
class InputInfo:
    batch_size: int
    num_tokens: list[int]
    cached_seqlens: list[int]


@dataclass
class NanoBatchSplitConfig:
    split_indices: List[int]
    batch_sizes: List[int]


@dataclass
class NanoOpInfo:
    gm: torch.fx.GraphModule
    submod_name: str
    tag: str
    idx: int
    args: tuple
    kwargs: dict


class NanoOpWrapper(torch.nn.Module):
    def __init__(self, gm: torch.fx.GraphModule, hook: List[Callable[[NanoOpInfo], ContextManager[None]]]):
        super().__init__()
        self.gm = gm
        self.hook = hook

    def forward(self, submod_name: str, idx: int, args: tuple, kwargs: dict):
        module = getattr(self.gm, submod_name)
        tag = getattr(module, "tag", "")
        with contextlib.ExitStack() as stack:
            for hook in self.hook:
                stack.enter_context(hook(NanoOpInfo(self.gm, submod_name, tag, idx, args, kwargs)))
            output = module(*args, **kwargs)
        return output


class NanoSplitManager:
    def __init__(self, graph_module: torch.fx.GraphModule) -> None:
        self.graph_module = graph_module
        self.original_graph = graph_module.graph
        self.base_graph = torch.fx.Graph()

        # Initialize the base graph
        batch_size: Optional[torch.SymInt] = None
        weight_nodes = set()
        splittable_inputs = []
        base_graph = torch.fx.Graph()
        for node in self.original_graph.nodes:
            # Skip computation nodes
            if node.op != "placeholder":
                continue

            # We assume the batch size is the first argument
            if batch_size is None:
                arg = node.meta["example_value"]
                if not isinstance(arg, torch.SymInt):
                    raise ValueError("Batch size is not set")
                batch_size = arg
            elif isinstance(input_tensor := node.meta["example_value"], torch.Tensor):
                shape = input_tensor.shape
                if shape[0] == batch_size:
                    splittable_inputs.append(node)
                    print(f"Found splittable input: {node.name} with shape {shape}")
                else:
                    weight_nodes.add(node)
                    print(f"Found weight tensor: {node.name} with shape {shape}")
            # Copy all placeholder nodes to the new graph
            base_graph.node_copy(node, arg_transform=lambda n: n)
        self.base_graph = base_graph
        self.splittable_inputs: List[torch.fx.Node] = splittable_inputs
        self.weight_nodes: Set[torch.fx.Node] = weight_nodes
    
        # Nano split preparation
        self.new_graph: Optional[torch.fx.Graph] = None
        self.input_splits = {}
        self.node_splits = {}
        self.op_wrapper = NanoOpWrapper(self.graph_module, [])

        # Runtime preparation
        self.cached_config: Optional[NanoBatchSplitConfig] = None
        self.comm_stream = torch.cuda.Stream()
        self.comp_stream = torch.cuda.Stream()


    def get_callable(self) -> Callable:
        def _forward(*args, **kwargs):
            assert self.op_wrapper is not None
            setattr(self.graph_module, "op_wrapper", self.op_wrapper)
            output = self.graph_module(*args, **kwargs)
            delattr(self.graph_module, "op_wrapper")
            return output
        return _forward

    def _init_input_splits(self, split_indices: List[int]) -> None:
        num_splits = len(split_indices) - 1
        assert self.new_graph is not None
        for node in self.splittable_inputs:
            self.input_splits[node] = []
            for i in range(num_splits):
                start_idx = split_indices[i]
                end_idx = split_indices[i + 1]
                slice_node = self.new_graph.call_function(
                    lambda x, start, end: x[start:end],
                    args=(node, start_idx, end_idx),
                )
                self.input_splits[node].append(slice_node)

    def _replicate_computations(self, split_indices: List[int]) -> None:
        num_splits = len(split_indices) - 1
        assert self.new_graph is not None
        print(f"Replicating computations for {num_splits} splits")
        for node in self.original_graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            print(f"Processing node: {node.name}, op: {node.op}, args: {len(node.args) if node.args else 0}")
            splits = []
            for split_idx in range(num_splits):
                new_args = self._get_split_args(node.args, split_idx, split_indices)
                new_kwargs = self._get_split_kwargs(node.kwargs, split_idx)
                orig_vals = list(node.args) + list(node.kwargs.values())
                new_vals = list(new_args) + list(new_kwargs.values())
                orig_to_new = {o: n for o, n in zip(orig_vals, new_vals)}

                if node.op == "call_module":
                    new_node = self.new_graph.call_module(
                        "op_wrapper",
                        args=(str(node.target), split_idx, new_args, new_kwargs),
                    )
                else:
                    new_node = self.new_graph.node_copy(
                        node, arg_transform=lambda n: orig_to_new[n]
                    )
                splits.append(new_node)

            self.node_splits[node] = splits
            print(
                f"Replicated computation node {node.name} into {num_splits} parts"
            )

    def _handle_outputs(self) -> None:
        """Handle output nodes by concatenating split outputs and cleaning up original computations."""
        assert self.new_graph is not None
        output_nodes = [
            node for node in self.original_graph.nodes if node.op == "output"
        ]
        assert len(output_nodes) == 1, f"Expected 1 output node, found {len(output_nodes)}"
        output_node = output_nodes[0]

        # Find the original computation that feeds into this output
        if not output_node.args:
            raise ValueError("Output node has no arguments")
        original_outputs = output_node.args[0]
        is_tuple = isinstance(original_outputs, tuple)
        if not isinstance(original_outputs, tuple):
            original_outputs = (original_outputs,)
        new_outputs = []

        for original_output in original_outputs:
            if original_output in self.node_splits:
                # Get all split outputs
                split_outputs = self.node_splits[original_output]

                # Create concatenation node
                if len(split_outputs) == 1:
                    # If there's only one split, no need to concatenate
                    concat_node = split_outputs[0]
                else:
                    # Create concatenation node
                    concat_node = self.new_graph.call_function(
                        torch.cat,
                        args=(split_outputs, 0),  # Concatenate along first dimension
                    )

                new_outputs.append(concat_node)
                print(f"Concatenated {len(split_outputs)} output splits")
            else:
                raise ValueError(
                    f"Original output {original_output} not found in node_splits"
                )

        self.new_graph.output(tuple(new_outputs) if is_tuple else new_outputs[0])

    def _get_split_args(self, args: Tuple, split_idx: int, split_indices: List[int]) -> Tuple:
        """Get arguments for a specific split."""
        new_args = []

        for arg in args:
            if isinstance(arg, torch.fx.Node):
                if arg in self.input_splits:
                    new_args.append(self.input_splits[arg][split_idx])
                elif arg in self.node_splits:
                    new_args.append(self.node_splits[arg][split_idx])
                elif arg in self.weight_nodes:
                    # Weight tensors are shared across splits
                    new_args.append(arg)
                elif isinstance(arg.meta["example_value"], torch.SymInt):
                    new_args.append(split_indices[split_idx + 1] - split_indices[split_idx])
                else:
                    new_args.append(arg)
            else:
                new_args.append(arg)

        return tuple(new_args)

    def _get_split_kwargs(self, kwargs: Dict, split_idx: int) -> Dict:
        """Get keyword arguments for a specific split."""
        new_kwargs = {}

        for key, value in kwargs.items():
            if isinstance(value, torch.fx.Node):
                if value in self.input_splits:
                    new_kwargs[key] = self.input_splits[value][split_idx]
                elif value in self.node_splits:
                    new_kwargs[key] = self.node_splits[value][split_idx]
                elif value in self.weight_nodes:
                    # Weight tensors are shared across splits
                    new_kwargs[key] = value
                else:
                    new_kwargs[key] = value
            else:
                new_kwargs[key] = value

        return new_kwargs

    def prepare_split(
        self,
        input_info: InputInfo,
    ) -> list[int]:
        total_batch_size = input_info.batch_size
        if total_batch_size == 1:
            batch_sizes = [1]
            split_indices = [0, input_info.num_tokens[0]]
        else:
            batch_sizes = [total_batch_size // 2, total_batch_size - total_batch_size // 2]
            split_indices = [0, sum(input_info.num_tokens[:total_batch_size // 2]), sum(input_info.num_tokens)]
        assert self.base_graph is not None
        self.new_graph = copy.deepcopy(self.base_graph)
        self._init_input_splits(split_indices)
        self._replicate_computations(split_indices)
        self._handle_outputs()
        assert self.graph_module is not None
        self.graph_module.graph = self.new_graph
        self.cached_config = NanoBatchSplitConfig(split_indices, batch_sizes)
        from torch._dynamo.utils import lazy_format_graph_code # type: ignore
        print(lazy_format_graph_code("after nano split", self.graph_module))
        return batch_sizes
    
    def prepare_runtime(
        self,
        *,
        forward_hook: Optional[Callable] = None,
        op_hook: Optional[Callable[[NanoOpInfo], ContextManager[None]]] = None,
    ) -> None:
        assert self.cached_config is not None
        batch_sizes = self.cached_config.batch_sizes
        comm_finished = [None for _ in range(len(batch_sizes))]

        @contextlib.contextmanager
        def set_stream(op_info: NanoOpInfo):
            if op_info.tag == "vllm.all_reduce":
                torch.cuda.set_stream(self.comm_stream) # type: ignore
                comm_finished[op_info.idx] = torch.cuda.Event() # type: ignore
            else:
                torch.cuda.set_stream(self.comp_stream) # type: ignore
                if comm_finished[op_info.idx] is not None:
                    comm_finished[op_info.idx].wait() # type: ignore
                    comm_finished[op_info.idx] = None
            try:
                yield
            finally:
                if op_info.tag == "vllm.all_reduce":
                    comm_finished[op_info.idx].record() # type: ignore

        @contextlib.contextmanager
        def nvtx_mark(op_info: NanoOpInfo):
            try:
                with torch.cuda.nvtx.range(f"op_{op_info.submod_name}_{op_info.tag}_{op_info.idx}"):
                    yield
            except Exception as e:
                print(f"Error in nvtx_mark: {e}")
                raise e
            
        hooks = [] if op_hook is None else [op_hook]
        self.op_wrapper = NanoOpWrapper(self.graph_module, [set_stream, nvtx_mark] + hooks)
        if forward_hook is not None:
            self.graph_module.register_forward_hook(forward_hook)


_split_manager = None


def init_split_manager_and_get_callable(graph_module: torch.fx.GraphModule) -> Callable:
    global _split_manager
    if _split_manager is None:
        _split_manager = NanoSplitManager(graph_module)
    return _split_manager.get_callable()



def prepare_split(input_info: InputInfo) -> list[int]:
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    return _split_manager.prepare_split(input_info)


def prepare_runtime(
    *,
    forward_hook: Optional[Callable] = None,
    op_hook: Optional[Callable[[NanoOpInfo], ContextManager[None]]] = None,
) -> None:
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    _split_manager.prepare_runtime(
        forward_hook=forward_hook,
        op_hook=op_hook,
    )
