# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    _make_fused_conv1d_weight_loader,
)
from vllm.model_executor.models.olmo_hybrid import (
    OlmoHybridForCausalLM,
    OlmoHybridModel,
)


def test_olmo_hybrid_mapper_registered():
    """Verify OlmoHybridForCausalLM inherits OlmoHybridModel's mapper."""
    assert hasattr(OlmoHybridForCausalLM, "hf_to_vllm_mapper")
    assert OlmoHybridForCausalLM.hf_to_vllm_mapper is OlmoHybridModel.hf_to_vllm_mapper


def test_fused_conv1d_weight_loader_shard_id_param():
    """Test loading with explicit loaded_shard_id parameter."""
    dims = [64, 64, 128]
    tp_size = 2
    tp_rank = 1
    loader = _make_fused_conv1d_weight_loader(dims, tp_size, tp_rank)

    param = nn.Parameter(torch.zeros(sum(dims) // tp_size, 1, 4))
    loaded_weight = torch.ones(dims[1], 4)  # 2D, unsqueezed by loader
    loader(param, loaded_weight, loaded_shard_id=1)

    assert (param[:32] == 0).all()
    assert (param[32:64] == 1).all()
    assert (param[64:] == 0).all()


def test_fused_conv1d_weight_loader_shard_id_attr():
    """Test loading with shard_id attached to loaded_weight (AutoWeightsLoader)."""
    dims = [64, 64, 128]
    tp_size = 2
    tp_rank = 0
    loader = _make_fused_conv1d_weight_loader(dims, tp_size, tp_rank)

    param = nn.Parameter(torch.zeros(sum(dims) // tp_size, 1, 4))
    loaded_weight = torch.ones(dims[2], 1, 4)
    loaded_weight.shard_id = 2  # loaded_shard_id is None, extracted from attr
    loader(param, loaded_weight)

    # dim 2 offset is 32 + 32 = 64, shard_size is 64
    assert (param[:64] == 0).all()
    assert (param[64:128] == 1).all()


def test_fused_conv1d_weight_loader_already_fused():
    """Test loading weights already fused across sections."""
    dims = [64, 64, 128]
    tp_size = 2
    tp_rank = 0
    loader = _make_fused_conv1d_weight_loader(dims, tp_size, tp_rank)

    param = nn.Parameter(torch.zeros(sum(dims) // tp_size, 1, 4))
    full_fused = torch.arange(sum(dims) * 4).view(sum(dims), 1, 4).float()
    loader(param, full_fused)
    assert torch.equal(param.data, full_fused[:128])


def test_fused_conv1d_weight_loader_rejects_missing_shard():
    """Test error when shard_id is missing and shape does not match fused layout."""
    dims = [64, 64, 128]
    loader = _make_fused_conv1d_weight_loader(dims, tp_size=1, tp_rank=0)
    param = nn.Parameter(torch.zeros(sum(dims), 1, 4))
    unmatched_weight = torch.zeros(33, 1, 4)

    with pytest.raises(
        ValueError, match="Cannot load conv1d weight without loaded_shard_id"
    ):
        loader(param, unmatched_weight)


def test_fused_conv1d_weight_loader_rejects_invalid_shard_id():
    """Test error when loaded_shard_id is out of bounds."""
    dims = [64, 64, 128]
    loader = _make_fused_conv1d_weight_loader(dims, tp_size=1, tp_rank=0)
    param = nn.Parameter(torch.zeros(sum(dims), 1, 4))
    weight = torch.zeros(64, 1, 4)

    with pytest.raises(ValueError, match="Invalid loaded_shard_id"):
        loader(param, weight, loaded_shard_id=5)
