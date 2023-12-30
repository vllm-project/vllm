import os
import itertools
import random
from re import M
import pytest
import torch

from vllm.model_executor.layers.moe import MoE
from vllm.model_executor.weight_utils import initialize_dummy_weights
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)


def test_moe():
    torch.distributed.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            init_method='file:///tmp/123'
        )
    initialize_model_parallel(1, 1)
    with torch.device("cuda"):
        moe = MoE(
            num_experts=4,
            top_k=2,
            hidden_size=8,
            intermediate_size=8,
            tp_size=1,
        )
    initialize_dummy_weights(moe)

    output = moe.forward(torch.randn(4, 1, 8, device="cuda"))