# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import CompilationConfig, ModelConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
from vllm.platforms import current_platform
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)


@pytest.fixture(autouse=True)
def mock_dist():
    with (
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.parallel_state.model_parallel_is_initialized",
            return_value=True,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tp_group",
            return_value=MagicMock(rank_in_group=0),
        ),
    ):
        yield


@pytest.fixture
def vllm_config():
    model_config = MagicMock(spec=ModelConfig)
    model_config.model = "test-model"
    model_config.tokenizer = "test-model"
    model_config.dtype = torch.float32
    model_config.seed = 0
    model_config.max_model_len = 1024
    model_config.skip_tokenizer_init = True
    model_config.tokenizer_mode = "auto"
    model_config.revision = None
    model_config.tokenizer_revision = None
    model_config.trust_remote_code = True
    model_config.quantization = None
    model_config.enforce_eager = False
    model_config.enable_return_routed_experts = False
    model_config.served_model_name = "test"
    model_config.pooler_config = None
    model_config.get_hidden_size.return_value = 16
    model_config.architecture = "ShortConv"
    model_config.is_hybrid = False
    model_config.convert_type = None
    model_config.config_updated = False
    model_config.attention_chunk_size = None

    compilation_config = CompilationConfig()
    config = VllmConfig(
        model_config=model_config,
        compilation_config=compilation_config,
    )
    return config


def test_short_conv_forward_native_prefill(vllm_config):
    prefix = "test_layer"
    config = SimpleNamespace(conv_L_cache=4, conv_bias=True)
    dim = 16

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        layer = ShortConv(config=config, dim=dim, layer_idx=0, prefix=prefix)

    layer.to("cpu")
    dispatch_cpu_unquantized_gemm(layer.in_proj, remove_weight=False)
    dispatch_cpu_unquantized_gemm(layer.out_proj, remove_weight=False)

    # Mock AttentionMetadata
    num_prefills = 1
    num_prefill_tokens = 5
    query_start_loc_p = torch.tensor([0, 5], dtype=torch.int32)
    state_indices_tensor_p = torch.tensor([0], dtype=torch.int32)

    # ShortConvAttentionMetadata
    attn_metadata = ShortConvAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=1,
        query_start_loc_p=query_start_loc_p,
        has_initial_states_p=torch.tensor([False]),
        state_indices_tensor_p=state_indices_tensor_p,
        state_indices_tensor_d=torch.empty((0, 1), dtype=torch.int32),
        num_accepted_tokens=None,
        query_start_loc_d=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([5]),
    )

    # Mock KV cache
    # conv_state shape (num_blocks, L_cache - 1, dim)
    conv_state = torch.zeros((1, config.conv_L_cache - 1, dim))
    layer.kv_cache = (conv_state,)

    hidden_states = torch.randn((num_prefill_tokens, dim))
    output = torch.zeros_like(hidden_states)

    attn_metadata_dict = {prefix: attn_metadata}
    with set_forward_context(attn_metadata=attn_metadata_dict, vllm_config=vllm_config):
        layer.forward_native(hidden_states, output)

    # Check if KV cache was updated
    assert not torch.allclose(conv_state, torch.zeros_like(conv_state))


def test_short_conv_forward_native_decode(vllm_config):
    prefix = "test_layer_decode"
    config = SimpleNamespace(conv_L_cache=4, conv_bias=True)
    dim = 16

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        layer = ShortConv(config=config, dim=dim, layer_idx=0, prefix=prefix)

    layer.to("cpu")
    dispatch_cpu_unquantized_gemm(layer.in_proj, remove_weight=False)
    dispatch_cpu_unquantized_gemm(layer.out_proj, remove_weight=False)

    # Mock AttentionMetadata for 2 decode requests
    num_decodes = 2
    state_indices_tensor_d = torch.tensor([0, 1], dtype=torch.int32)

    attn_metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_reqs=num_decodes,
        query_start_loc_p=None,
        has_initial_states_p=None,
        state_indices_tensor_p=torch.empty((0,), dtype=torch.int32),
        state_indices_tensor_d=state_indices_tensor_d,
        num_accepted_tokens=None,
        query_start_loc_d=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        num_computed_tokens_p=None,
        seq_lens=torch.tensor([1, 1]),
    )

    # Mock KV cache (2 blocks for 2 requests)
    conv_state = torch.randn((2, config.conv_L_cache - 1, dim))
    layer.kv_cache = (conv_state,)

    hidden_states = torch.randn((num_decodes, dim))
    output = torch.zeros_like(hidden_states)

    old_conv_state = conv_state.clone()

    attn_metadata_dict = {prefix: attn_metadata}
    with set_forward_context(attn_metadata=attn_metadata_dict, vllm_config=vllm_config):
        layer.forward_native(hidden_states, output)

    # Check if KV cache was updated
    assert not torch.allclose(conv_state, old_conv_state)


def test_dispatch_cpu_unquantized_gemm_conv_layer():
    # Create a mock layer with >2D weights (convolution-like)
    class MockConvLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 1, 4))
            self.bias = torch.nn.Parameter(torch.randn(16))

    layer = MockConvLayer()

    # This should set layer.cpu_linear to _gemm_not_supported
    dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    assert hasattr(layer, "cpu_linear")

    with pytest.raises(NotImplementedError, match="GEMM not supported for this layer"):
        layer.cpu_linear(torch.randn(1, 16), layer.weight, layer.bias)
