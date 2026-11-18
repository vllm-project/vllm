# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
import sys

import pytest
import torch
from torch import nn
from transformers import OPTConfig

from tests.v1.core.utils import full_layer, new_kv_cache_spec
from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.entrypoints.memory_estimate import (
    _device_tensor_bytes,
    _request_kv_cache_bytes,
)
from vllm.model_executor.model_loader.meta_loader import (
    _HOST_TENSORS,
    _meta_construction,
    mark_host_tensors,
)
from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="workers need a GPU"
)


def test_weights_count_device_tensors_only():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 8))
            self.workspace = torch.empty(16)  # a plain attribute, still on device
            pinned = torch.empty(32, 2, device="cpu", pin_memory=True)
            self.table = nn.Parameter(pinned, requires_grad=False)
            self.table_view = self.table.view(64)
            stand_in = torch.empty(100)
            mark_host_tensors(stand_in)  # a host tensor the model built on meta
            self.stand_in = stand_in
            self.index = torch.arange(3, device="cpu")

    with _meta_construction() as host:
        model = Model()
    _HOST_TENSORS[model] = host  # as MetaModelLoader.load_model does
    assert model.table.is_meta  # pinned tables are never allocated
    assert _device_tensor_bytes([model]) == (4 * 8 + 16) * 4


@pytest.mark.parametrize("tp,pp,dp", [(2, 2, 1), (4, 2, 2)])
def test_fake_backend_builds_the_real_group_layout(tp: int, pp: int, dp: int):
    world = tp * pp
    rank = world - 1  # the last TP rank of the last PP stage, in DP rank 0
    pp_rank, tp_rank = divmod(rank, tp)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            distributed_executor_backend="mp",
        )
    )
    with set_current_vllm_config(config):
        ps.init_distributed_environment(world, rank, "env://", 0, "fake")
        try:
            ps.initialize_model_parallel(tp, pp)
            assert ps.is_fake_distributed()
            assert ps.get_world_group().world_size == world * dp
            assert ps.get_tp_group().ranks == [pp_rank * tp + i for i in range(tp)]
            assert ps.get_pp_group().ranks == [i * tp + tp_rank for i in range(pp)]
            assert ps.get_dp_group().ranks == [rank + i * world for i in range(dp)]
            assert ps.get_ep_group().ranks == [
                d * world + pp_rank * tp + t for d in range(dp) for t in range(tp)
            ]
            assert ps.get_tp_group().device_communicator is None
            with pytest.raises(RuntimeError, match="collective"):
                torch.distributed.all_reduce(
                    torch.ones(1), group=ps.get_tp_group().device_group
                )
        finally:
            ps.destroy_model_parallel()
            ps.destroy_distributed_environment()


def test_request_kv_cache_bytes_per_stage():
    config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    config.cache_config.kv_cache_layout = "LBNHC"
    config.cache_config.prefix_cache_retention_interval = None
    layer = full_layer(new_kv_cache_spec())
    stages = [{"layer1": layer}, {"layer2": layer, "layer3": layer}]
    groups = get_global_kv_cache_groups(config, stages)
    page = new_kv_cache_spec().page_size_bytes
    # A 16-token request takes one block per layer, plus the pool's null block.
    assert [_request_kv_cache_bytes(config, groups, s) for s in stages] == [
        2 * page,
        4 * page,
    ]


def _tiny_opt(path) -> str:
    # Config only: the meta load format never reads a checkpoint.
    OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=128,
        ffn_dim=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=256,
        max_position_embeddings=256,
        word_embed_proj_dim=128,
    ).save_pretrained(path)
    return str(path)


@requires_gpu
def test_estimate_every_layout_on_one_gpu(tmp_path):
    cli = [sys.executable, "-m", "vllm.entrypoints.memory_estimate"]
    cli += ["--model", _tiny_opt(tmp_path), "--dtype", "float16"]
    cli += ["--skip-tokenizer-init", "--enforce-eager", "--sweep"]
    cli += ["--gpus-per-node", "2", "--jobs", "6", "--context-lengths", "16", "256"]
    out = subprocess.run(cli, capture_output=True, text=True, check=True).stdout
    records = [json.loads(line) for line in out.splitlines()]
    assert len(records) == 6  # TP x PP x DP in {1x1x2, 1x2x1, 2x1x1}, EP off/on
    stages = {}
    for record in records:
        if record["enable_expert_parallel"]:
            assert "experts" in record["error"]  # a dense model has no experts
        else:
            tp_pp = (record["tensor_parallel_size"], record["pipeline_parallel_size"])
            stages[tp_pp] = record["stages"]
    # Each DP replica holds the whole model; TP shards it and PP splits it.
    whole = stages[(1, 1)][0]["weight_bytes"]
    assert stages[(2, 1)][0]["weight_bytes"] < whole
    assert len(stages[(1, 2)]) == 2
    assert all(stage["weight_bytes"] < whole for stage in stages[(1, 2)])
    for stage in sum(stages.values(), []):
        assert 0 < stage["kv_cache_bytes"]["16"] < stage["kv_cache_bytes"]["256"]
