import operator
import os
from typing import Dict, Iterable, Optional

import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import Match

from vllm.config import CompilationConfig
# yapf: disable
from vllm.distributed import get_tensor_model_parallel_rank as get_tp_rank
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tp_world_size)
from vllm.distributed import model_parallel_is_initialized as p_is_init
from vllm.logger import init_logger

# yapf: enable

logger = init_logger(__name__)

COUNTS: Dict[str, int] = {}


def find_fn(nodes: Iterable[fx.Node], op) -> Optional[fx.Node]:
    for node in nodes:
        if node.op == "call_function" and node.target == op:
            return node
    return None


def find_auto_fn(nodes: Iterable[fx.Node], op) -> Optional[fx.Node]:
    for node in nodes:
        if (node.op == "call_function" and node.target == auto_functionalized
                and node.args[0] == op):
            return node
    return None


def find_getitem(node: fx.Node, idx: int) -> Optional[fx.Node]:
    for user in node.users:
        if (user.op == "call_function" and user.target == operator.getitem
                and user.args[1] == idx):
            return user
    return None


def last_node_in_match(match: Match) -> fx.Node:
    if len(match.nodes) > 0:
        graph = match.nodes[0].graph
        for n in reversed(graph.nodes):
            if n in reversed(match.nodes):
                return n
    raise ValueError("No nodes in graph")


def dump_graph(config: CompilationConfig.PassConfig, graph: fx.Graph,
               name: str) -> None:
    global COUNTS
    count = COUNTS.get(name, 0)

    # Make sure filename includes rank in the distributed setting
    parallel = p_is_init() and get_tp_world_size() > 1
    rank = f"-{get_tp_rank()}" if parallel else ""
    filepath = config.dump_graph_dir / f"{name}{rank}-{count}.py"
    COUNTS[name] = count + 1

    os.makedirs(config.dump_graph_dir, exist_ok=True)
    logger.info("%s printing graph to %s", name, filepath)
    with open(filepath, "w") as f:
        src = graph.owning_module.print_readable(print_output=False)
        # Add imports so it's not full of errors
        print("import torch; from torch import device", file=f)
        print(src, file=f)
