import os
from pathlib import Path
from typing import Dict

import torch.fx as fx

# yapf: enable
from vllm.distributed import get_tensor_model_parallel_rank as get_tp_rank
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tp_world_size)
from vllm.distributed import model_parallel_is_initialized as p_is_init
from vllm.logger import init_logger

# yapf: disable


logger = init_logger(__name__)

COUNTS: Dict[str, int] = {}


def dump_graph(pass_config, graph: fx.Graph, name: str) -> None:
    global COUNTS
    count = COUNTS.get(name, 0)

    # Make sure filename includes rank in the distributed setting
    parallel = p_is_init() and get_tp_world_size() > 1
    rank = f"-{get_tp_rank()}" if parallel else ""
    filepath = Path(pass_config.dump_graph_dir) / f"{name}{rank}-{count}.py"
    COUNTS[name] = count + 1

    os.makedirs(pass_config.dump_graph_dir, exist_ok=True)
    logger.info("%s printing graph to %s", name, filepath)
    with open(filepath, "w") as f:
        src = graph.owning_module.print_readable(print_output=False)
        # Add imports so it's not full of errors
        print("import torch; from torch import device", file=f)
        print(src, file=f)
