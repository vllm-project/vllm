import dataclasses
from contextlib import contextmanager
from typing import Callable, ContextManager, List

import torch

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


def split_graph_with_tags(
    graph: torch.fx.GraphModule,
    split_ops: list[str],
    op_tags: dict[str, str],
) -> tuple[torch.fx.GraphModule]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    subgraph_to_tag = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in split_ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id
        if (tag := op_tags.get(node.op)) is not None:
            assert subgraph_to_tag[subgraph_id] is None or subgraph_to_tag[subgraph_id] == tag, \
                f"tag mismatch: {subgraph_to_tag[subgraph_id]} != {tag}"
            subgraph_to_tag[subgraph_id] = tag

    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True)

    names = [name for (name, _) in split_gm.named_modules()]
    for name in names:
        if "." in name or name == "":
            continue
        module = getattr(split_gm, name)
        graph_id = int(name.replace("submod_", ""))
        setattr(module, "tag", subgraph_to_tag.get(graph_id, ""))

    return split_gm

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
