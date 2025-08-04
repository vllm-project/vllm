import torch
from typing import Callable, List, Tuple, Union
import dataclasses


@dataclasses.dataclass
class NanoOpInfo:
    submod_name: str
    tag: str
    idx: int
    args: tuple
    kwargs: dict


@dataclasses.dataclass
class NanoSplitConfig:
    num_nano_batches: int
    # Request level information
    batch_sizes: List[int]
    batch_indices: List[int]
    # Token level information
    num_tokens: List[int]
    split_indices: List[int]  # start/end indices of each nano batch


class FakeModule(torch.nn.Module):
    def __init__(self, fn: Callable, **kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.fn(*args, **self.kwargs, **kwargs)


def get_split_config(
    batch_size: int,
    num_tokens: List[int],
    cached_seqlens: List[int],
) -> NanoSplitConfig:
    if batch_size == 1:
        nano_batch_sizes = [1]
        nano_batch_indices = [0, 1]
        nano_batch_num_tokens = num_tokens.copy()
        nano_batch_split_indices = [0, num_tokens[0]]
    else:
        nano_batch_sizes = [batch_size // 2, batch_size - batch_size // 2]
        nano_batch_indices = [0, batch_size // 2, batch_size]
        nano_batch_num_tokens = [
            sum(num_tokens[: batch_size // 2]),
            sum(num_tokens[batch_size // 2 :]),
        ]
        nano_batch_split_indices = [
            0,
            nano_batch_num_tokens[0],
            sum(nano_batch_num_tokens),
        ]

    return NanoSplitConfig(
        num_nano_batches=len(nano_batch_sizes),
        batch_sizes=nano_batch_sizes,
        batch_indices=nano_batch_indices,
        num_tokens=nano_batch_num_tokens,
        split_indices=nano_batch_split_indices,
    )


def display_graph(graph_module: torch.fx.GraphModule, name: str):
    from torch._dynamo.utils import lazy_format_graph_code  # type: ignore

    print(lazy_format_graph_code(name, graph_module))


def tag_graph(gm: torch.fx.GraphModule, op_tags: dict[str, str]):
    submodules = [
        (name, module)
        for (name, module) in gm.named_modules()
        if hasattr(module, "graph")
    ]
    for _, module in submodules:
        for node in module.graph.nodes:
            if (
                node.op == "call_function"
                and (tag := op_tags.get(str(node.target))) is not None
            ):
                assert (
                    getattr(module, "tag", None) is None
                    or getattr(module, "tag") == tag
                ), f"tag mismatch: {getattr(module, 'tag')} != {tag}"
                setattr(module, "tag", tag)


def analyze_graph(
    graph: torch.fx.Graph, batch_size: Union[int, torch.SymInt, None] = None
) -> Tuple[List[torch.fx.Node], torch.fx.Graph]:
    weight_nodes = set()
    splittable_inputs = []
    base_graph = torch.fx.Graph()
    for node in graph.nodes:
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
    return splittable_inputs, base_graph


def split_graph(
    graph: torch.fx.Graph,
    *,
    out: torch.fx.Graph,
    splittable_inputs: List[torch.fx.Node],
    num_splits: int,
    get_bs_fn: str,
    split_fn: str,
    wrapper_fn: str,
) -> torch.fx.Graph:
    mapping = {}
    nano_batch_sizes = []

    # Step 1: Get nano batch sizes and split inputs
    for i in range(num_splits):
        nano_batch_sizes.append(
            out.call_module(
                get_bs_fn,
                args=(i,),
            )
        )
    for node in splittable_inputs:
        mapping[node] = []
        for i in range(num_splits):
            slice_node = out.call_module(
                split_fn,
                args=(node, i),
            )
            mapping[node].append(slice_node)
    
    # Step 2: Split computation nodes
    def _transform(idx, n) -> torch.fx.Node:
        if n in mapping:
            return mapping[n][idx]
        if isinstance(getattr(n, "meta", {}).get("example_value", None), torch.SymInt):
            return nano_batch_sizes[idx]
        return n

    for node in graph.nodes:
        if node.op in ["placeholder", "output"]:
            continue
        splits = []
        for split_idx in range(num_splits):
            if node.op == "call_module":
                new_args = [_transform(split_idx, arg) for arg in node.args]
                new_kwargs = {
                    k: _transform(split_idx, v) for k, v in node.kwargs.items()
                }
                new_node = out.call_module(
                    wrapper_fn,
                    args=(str(node.target), split_idx, new_args, new_kwargs),
                )
            else:
                new_node = out.node_copy(
                    node, arg_transform=lambda n: _transform(split_idx, n)
                )
            splits.append(new_node)
        mapping[node] = splits

    # Step 3: Concatenate outputs
    output_nodes = [node for node in graph.nodes if node.op == "output"]
    assert len(output_nodes) == 1, f"Expected 1 output node, found {len(output_nodes)}"
    output_node = output_nodes[0]
    if not output_node.args:
        raise ValueError("Output node has no arguments")
    original_outputs = output_node.args[0]
    is_tuple = isinstance(original_outputs, tuple)
    if not isinstance(original_outputs, tuple):
        original_outputs = (original_outputs,)
    new_outputs = []

    for original_output in original_outputs:
        if original_output in mapping:
            # Get all split outputs
            split_outputs = mapping[original_output]

            # Create concatenation node
            if len(split_outputs) == 1:
                # If there's only one split, no need to concatenate
                concat_node = split_outputs[0]
            else:
                # Create concatenation node
                concat_node = out.call_function(
                    torch.cat,
                    args=(split_outputs, 0),  # Concatenate along first dimension
                )

            new_outputs.append(concat_node)
            print(f"Concatenated {len(split_outputs)} output splits")
        else:
            raise ValueError(
                f"Original output {original_output} not found in node_splits"
            )

    out.output(tuple(new_outputs) if is_tuple else new_outputs[0])
    return out
