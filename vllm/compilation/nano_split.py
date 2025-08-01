import torch
from typing import Any, Dict, List, Set, Tuple, Union


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


def split_inputs(
    graph: torch.fx.Graph,
    splittable_inputs: List[torch.fx.Node],
    split_module: str,
    num_splits: int,
) -> Dict[torch.fx.Node, List[torch.fx.Node]]:
    mapping = {}
    for node in splittable_inputs:
        mapping[node] = []
        for i in range(num_splits):
            slice_node = graph.call_module(
                split_module,
                args=(node, i),
            )
            mapping[node].append(slice_node)
    return mapping


def split_computations(
    org_graph: torch.fx.Graph,
    new_graph: torch.fx.Graph,
    mapping: Dict[torch.fx.Node, List[torch.fx.Node]],
    nano_batch_sizes: List[torch.fx.Node],
    wrapper_module: str,
    num_splits: int,
):
    def _transform(idx, n) -> torch.fx.Node:
        if n in mapping:
            return mapping[n][idx]
        if isinstance(getattr(n, "meta", {}).get("example_value", None), torch.SymInt):
            return nano_batch_sizes[idx]
        return n

    for node in org_graph.nodes:
        if node.op in ["placeholder", "output"]:
            continue
        splits = []
        for split_idx in range(num_splits):
            if node.op == "call_module":
                new_args = [_transform(split_idx, arg) for arg in node.args]
                new_kwargs = {
                    k: _transform(split_idx, v) for k, v in node.kwargs.items()
                }
                new_node = new_graph.call_module(
                    wrapper_module,
                    args=(str(node.target), split_idx, new_args, new_kwargs),
                )
            else:
                new_node = new_graph.node_copy(
                    node, arg_transform=lambda n: _transform(split_idx, n)
                )
            splits.append(new_node)
        mapping[node] = splits
    return mapping


def concat_outputs(
    org_graph: torch.fx.Graph,
    new_graph: torch.fx.Graph,
    mapping: Dict[torch.fx.Node, List[torch.fx.Node]],
):
    output_nodes = [node for node in org_graph.nodes if node.op == "output"]
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
                concat_node = new_graph.call_function(
                    torch.cat,
                    args=(split_outputs, 0),  # Concatenate along first dimension
                )

            new_outputs.append(concat_node)
            print(f"Concatenated {len(split_outputs)} output splits")
        else:
            raise ValueError(
                f"Original output {original_output} not found in node_splits"
            )

    new_graph.output(tuple(new_outputs) if is_tuple else new_outputs[0])
