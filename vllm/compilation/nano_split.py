import copy
from dataclasses import dataclass
import torch
from typing import Callable, Dict, List, Tuple, Optional, Set

@dataclass
class InputInfo:
    batch_size: int
    num_tokens: list[int]
    cached_seqlens: list[int]


@dataclass
class NanoBatchSplitConfig:
    split_indices: List[int]
    batch_sizes: List[int]


class HookWrapper(torch.nn.Module):
    def __init__(self, hook: Callable):
        super().__init__()
        self.hook = hook
    
    def forward(self, *args, **kwargs):
        self.hook(*args, **kwargs)


class NanoBatchSplit:
    def __init__(self):
        self.input_splits: Dict[torch.fx.Node, List[torch.fx.Node]] = {}
        self.node_splits: Dict[torch.fx.Node, List[torch.fx.Node]] = {}
        self.weight_nodes: Set[torch.fx.Node] = set()
        self.splittable_inputs: List[torch.fx.Node] = []
        self.graph_module: Optional[torch.fx.GraphModule] = None
        self.original_graph: torch.fx.Graph
        self.base_graph: Optional[torch.fx.Graph] = None
        self.new_graph: Optional[torch.fx.Graph] = None

    def _init_placeholders(self) -> None:
        batch_size: Optional[torch.SymInt] = None
        assert self.base_graph is not None
        self.base_graph.call_module("pre_forward_hook", args=())
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
            else:
                shape = node.meta["example_value"].shape
                if shape[0] == batch_size:
                    self.splittable_inputs.append(node)
                    print(f"Found splittable input: {node.name} with shape {shape}")
                else:
                    self.weight_nodes.add(node)
                    print(f"Found weight tensor: {node.name} with shape {shape}")
            # Copy all placeholder nodes to the new graph
            self.base_graph.node_copy(node, arg_transform=lambda n: n)

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
                # Call pre_op_hook with proper arguments
                self.new_graph.call_module(
                    "pre_op_hook",
                    args=(node.name, split_idx, new_args, new_kwargs),
                )
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

    def auto_search_and_split(
        self,
        input_info: InputInfo,
    ) -> list[int]:
        total_batch_size = input_info.batch_size
        if total_batch_size == 1:
            batch_sizes = [1]
            split_indices = [0, input_info.num_tokens[0]]
        else:
            batch_sizes = [1, total_batch_size - 1]
            split_indices = [0, input_info.num_tokens[0], sum(input_info.num_tokens)]
        assert self.base_graph is not None
        self.new_graph = copy.deepcopy(self.base_graph)
        self._init_input_splits(split_indices)
        self._replicate_computations(split_indices)
        self._handle_outputs()
        assert self.graph_module is not None
        self.graph_module.graph = self.new_graph
        print(self.graph_module.code)
        setattr(self.graph_module, "cached_config", NanoBatchSplitConfig(split_indices, batch_sizes))
        return batch_sizes

    def init_callable(self, graph_module: torch.fx.GraphModule) -> Callable:
        self.base_graph = torch.fx.Graph()
        self.graph_module = graph_module
        self.original_graph = graph_module.graph
        self._init_placeholders()
        return self.graph_module


_split_manager = None


def init_split_manager_and_get_callable(graph_module: torch.fx.GraphModule) -> Callable:
    global _split_manager
    if _split_manager is None:
        _split_manager = NanoBatchSplit()
    return _split_manager.init_callable(graph_module)


def auto_search_and_split(input_info: InputInfo) -> list[int]:
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    return _split_manager.auto_search_and_split(input_info)


def set_forward_hook(
    pre_forward_hook: Callable[[], None],
    pre_op_hook: Callable[[str, int, Tuple, Dict], None]
) -> None:
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    setattr(_split_manager.graph_module, "pre_forward_hook", HookWrapper(pre_forward_hook))
    setattr(_split_manager.graph_module, "pre_op_hook", HookWrapper(pre_op_hook))
