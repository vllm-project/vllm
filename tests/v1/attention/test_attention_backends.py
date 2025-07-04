# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 attention backends without GPUModelRunner dependency."""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from vllm.config import (CacheConfig, CompilationConfig, DeviceConfig,
                         LoadConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass
class ModelParams:
    """Model-specific parameters for attention testing."""
    block_size: int = 16
    num_kv_heads: int = 8
    head_size: int = 64
    dtype: torch.dtype = torch.float16
    use_mla: bool = False
    sliding_window: Optional[int] = None

    def __post_init__(self):
        # Validate that block_size is a power of 2 and within reasonable range
        assert self.block_size in [1, 2, 4, 8, 16, 32, 64, 128
                                   ], f"Invalid block_size: {self.block_size}"
        assert self.num_kv_heads > 0, (
            f"num_kv_heads must be positive: {self.num_kv_heads}")
        assert self.head_size > 0, (
            f"head_size must be positive: {self.head_size}")


@dataclass
class BatchSpec:
    """Specification for a batch configuration (workload shape only)."""
    name: str
    batch_size: int
    num_tokens: int
    seq_lens: list[int]
    query_lens: list[int]

    def __post_init__(self):
        assert len(self.seq_lens) == self.batch_size
        assert len(self.query_lens) == self.batch_size
        assert sum(self.query_lens) == self.num_tokens


@dataclass
class AttentionTestSpec:
    """
    Complete specification combining batch configuration and model parameters.
    """
    batch_spec: BatchSpec
    model_params: ModelParams


# Define common model parameter configurations
DEFAULT_MODEL_PARAMS = ModelParams()

MODEL_PARAM_VARIANTS = {
    "default": DEFAULT_MODEL_PARAMS,
    "large_block": ModelParams(block_size=32),
    "small_block": ModelParams(block_size=8),
    "multi_head": ModelParams(num_kv_heads=16),
    "small_head": ModelParams(num_kv_heads=4),
    "bfloat16": ModelParams(dtype=torch.bfloat16),
    "float32": ModelParams(dtype=torch.float32),
    "sliding_window": ModelParams(sliding_window=256),
    "mla": ModelParams(use_mla=True),
}

# Define common batch configurations
BATCH_SPECS = [
    BatchSpec("small_decode",
              batch_size=2,
              num_tokens=2,
              seq_lens=[32, 40],
              query_lens=[1, 1]),
    BatchSpec("small_prefill",
              batch_size=2,
              num_tokens=16,
              seq_lens=[32, 40],
              query_lens=[8, 8]),
    BatchSpec("mixed_small",
              batch_size=4,
              num_tokens=12,
              seq_lens=[32, 40, 48, 56],
              query_lens=[1, 1, 5, 5]),
    BatchSpec("medium_decode",
              batch_size=8,
              num_tokens=8,
              seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
              query_lens=[1, 1, 1, 1, 1, 1, 1, 1]),
    BatchSpec("medium_prefill",
              batch_size=4,
              num_tokens=64,
              seq_lens=[256, 512, 1024, 2048],
              query_lens=[16, 16, 16, 16]),
    BatchSpec("mixed_medium",
              batch_size=6,
              num_tokens=24,
              seq_lens=[512, 1024, 2048, 512, 1024, 2048],
              query_lens=[1, 1, 1, 7, 7, 7]),
    BatchSpec("large_decode",
              batch_size=32,
              num_tokens=32,
              seq_lens=[2048] * 32,
              query_lens=[1] * 32),
    BatchSpec("large_prefill",
              batch_size=8,
              num_tokens=256,
              seq_lens=[4096] * 8,
              query_lens=[32] * 8),
    BatchSpec("single_decode",
              batch_size=1,
              num_tokens=1,
              seq_lens=[1024],
              query_lens=[1]),
    BatchSpec("single_prefill",
              batch_size=1,
              num_tokens=64,
              seq_lens=[1024],
              query_lens=[64]),
]


# Create combined specs for legacy compatibility and specific test cases
def create_combined_test_specs():
    """Create combined test specifications by constructing AttentionTestSpec."""
    return [
        # Legacy specs with embedded model params for backward compatibility
        AttentionTestSpec(
            BatchSpec("small_decode",
                      batch_size=2,
                      num_tokens=2,
                      seq_lens=[32, 40],
                      query_lens=[1, 1]), DEFAULT_MODEL_PARAMS),
        AttentionTestSpec(
            BatchSpec("small_prefill",
                      batch_size=2,
                      num_tokens=16,
                      seq_lens=[32, 40],
                      query_lens=[8, 8]), DEFAULT_MODEL_PARAMS),
        AttentionTestSpec(
            BatchSpec("mixed_small",
                      batch_size=4,
                      num_tokens=12,
                      seq_lens=[32, 40, 48, 56],
                      query_lens=[1, 1, 5, 5]), DEFAULT_MODEL_PARAMS),

        # Different model configurations with same batch shape
        AttentionTestSpec(
            BatchSpec("small_decode",
                      batch_size=2,
                      num_tokens=2,
                      seq_lens=[32, 40],
                      query_lens=[1, 1]), MODEL_PARAM_VARIANTS["large_block"]),
        AttentionTestSpec(
            BatchSpec("small_decode",
                      batch_size=2,
                      num_tokens=2,
                      seq_lens=[32, 40],
                      query_lens=[1, 1]), MODEL_PARAM_VARIANTS["multi_head"]),
        AttentionTestSpec(
            BatchSpec("small_decode",
                      batch_size=2,
                      num_tokens=2,
                      seq_lens=[32, 40],
                      query_lens=[1, 1]), MODEL_PARAM_VARIANTS["bfloat16"]),
        AttentionTestSpec(
            BatchSpec("small_decode",
                      batch_size=2,
                      num_tokens=2,
                      seq_lens=[32, 40],
                      query_lens=[1, 1]),
            MODEL_PARAM_VARIANTS["sliding_window"]),
        AttentionTestSpec(
            BatchSpec("small_decode",
                      batch_size=2,
                      num_tokens=2,
                      seq_lens=[32, 40],
                      query_lens=[1, 1]), MODEL_PARAM_VARIANTS["mla"]),

        # Medium batch configurations
        AttentionTestSpec(
            BatchSpec("medium_decode",
                      batch_size=8,
                      num_tokens=8,
                      seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
                      query_lens=[1, 1, 1, 1, 1, 1, 1, 1]),
            DEFAULT_MODEL_PARAMS),
        AttentionTestSpec(
            BatchSpec("medium_prefill",
                      batch_size=4,
                      num_tokens=64,
                      seq_lens=[256, 512, 1024, 2048],
                      query_lens=[16, 16, 16, 16]), DEFAULT_MODEL_PARAMS),

        # Large batch configurations
        AttentionTestSpec(
            BatchSpec("large_decode",
                      batch_size=32,
                      num_tokens=32,
                      seq_lens=[2048] * 32,
                      query_lens=[1] * 32), DEFAULT_MODEL_PARAMS),
        AttentionTestSpec(
            BatchSpec("large_prefill",
                      batch_size=8,
                      num_tokens=256,
                      seq_lens=[4096] * 8,
                      query_lens=[32] * 8), DEFAULT_MODEL_PARAMS),
    ]


COMBINED_TEST_SPECS = create_combined_test_specs()


# Fixtures
@pytest.fixture
def device():
    """Create a CUDA device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        pytest.skip("CUDA not available")


@pytest.fixture
def vllm_config():
    """Create a minimal VllmConfig for testing."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        max_model_len=1024,
        dtype=torch.float16,
    )
    cache_config = CacheConfig(
        block_size=16,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    scheduler_config = SchedulerConfig(
        max_num_seqs=32,
        max_num_batched_tokens=8192,  # Must be >= max_model_len
    )
    device_config = DeviceConfig()
    load_config = LoadConfig()
    compilation_config = CompilationConfig()

    # Add mock methods to satisfy the FlashInfer backend's requirements.
    # This is a workaround because this test does not build a full, real model,
    # but FlashInfer expects to be able to query the model for layer-specific
    # parameters. We provide default values that are consistent with the
    # test environment.
    model_config.get_num_layers = lambda: 1
    model_config.get_sliding_window_for_layer = lambda i: None
    model_config.get_logits_soft_cap_for_layer = lambda i: 0.0
    # Default head size is 64 for these tests.
    model_config.get_sm_scale_for_layer = lambda i: 1.0 / 64**0.5

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        compilation_config=compilation_config,
    )


@pytest.fixture
def default_model_params():
    """Create default ModelParams for testing."""
    return DEFAULT_MODEL_PARAMS


@pytest.fixture
def kv_cache_spec(default_model_params):
    """Create a FullAttentionSpec for testing."""
    return create_kv_cache_spec_from_model_params(default_model_params)


@pytest.fixture
def common_attn_metadata(device, default_model_params):
    """Create CommonAttentionMetadata for testing."""
    batch_spec = BatchSpec("default",
                           batch_size=4,
                           num_tokens=32,
                           seq_lens=[64, 72, 80, 88],
                           query_lens=[8, 8, 8, 8])
    return create_common_attn_metadata(batch_spec, default_model_params,
                                       device)


# Helper functions
def create_kv_cache_spec(test_spec: AttentionTestSpec) -> FullAttentionSpec:
    """Create a FullAttentionSpec from a AttentionTestSpec."""
    return FullAttentionSpec(
        block_size=test_spec.model_params.block_size,
        num_kv_heads=test_spec.model_params.num_kv_heads,
        head_size=test_spec.model_params.head_size,
        dtype=test_spec.model_params.dtype,
        use_mla=test_spec.model_params.use_mla,
        sliding_window=test_spec.model_params.sliding_window,
    )


def create_kv_cache_spec_from_model_params(
        model_params: ModelParams) -> FullAttentionSpec:
    """Create a FullAttentionSpec from ModelParams only."""
    return FullAttentionSpec(
        block_size=model_params.block_size,
        num_kv_heads=model_params.num_kv_heads,
        head_size=model_params.head_size,
        dtype=model_params.dtype,
        use_mla=model_params.use_mla,
        sliding_window=model_params.sliding_window,
    )


def create_common_attn_metadata(
        batch_spec: BatchSpec, model_params: ModelParams,
        device: torch.device) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata from a BatchSpec and ModelParams."""
    # Create query start locations
    query_start_loc = torch.zeros(batch_spec.batch_size + 1,
                                  dtype=torch.int32,
                                  device=device)
    query_start_loc[1:] = torch.tensor(batch_spec.query_lens,
                                       dtype=torch.int32,
                                       device=device).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()

    # Create sequence lengths
    seq_lens = torch.tensor(batch_spec.seq_lens,
                            dtype=torch.int32,
                            device=device)
    seq_lens_cpu = seq_lens.cpu()

    # Create computed tokens (assume all tokens are computed for simplicity)
    num_computed_tokens_cpu = seq_lens_cpu.clone()

    # Create block table (random for testing)
    max_blocks = max(batch_spec.seq_lens) // model_params.block_size + 1
    block_table_tensor = torch.randint(0,
                                       1000,
                                       (batch_spec.batch_size, max_blocks),
                                       dtype=torch.int32,
                                       device=device)

    # Create slot mapping
    slot_mapping = torch.randint(0,
                                 1000, (batch_spec.num_tokens, ),
                                 dtype=torch.int64,
                                 device=device)
    slot_mapping_cpu = slot_mapping.cpu()

    # Calculate max query length
    max_query_len = max(batch_spec.query_lens)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=batch_spec.num_tokens,
        max_query_len=max_query_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        slot_mapping_cpu=slot_mapping_cpu,
    )


def create_common_attn_metadata_from_combined(
        test_spec: AttentionTestSpec,
        device: torch.device) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata from a AttentionTestSpec."""
    return create_common_attn_metadata(test_spec.batch_spec,
                                       test_spec.model_params, device)


def create_dummy_kv_cache(kv_cache_spec: FullAttentionSpec,
                          device: torch.device) -> torch.Tensor:
    """Create a dummy KV cache tensor for testing."""
    # Assume we have enough blocks for our test cases
    num_blocks = 100
    kv_cache = torch.randn(
        num_blocks,
        2,  # K and V
        kv_cache_spec.block_size,
        kv_cache_spec.num_kv_heads,
        kv_cache_spec.head_size,
        dtype=kv_cache_spec.dtype,
        device=device)
    return kv_cache


def get_attention_backend_classes(backend_name: str):
    """Get the attention backend classes for the given backend name."""
    backend_map = {
        "flash_attn":
        ("vllm.v1.attention.backends.flash_attn", "FlashAttentionBackend"),
        "flashinfer":
        ("vllm.v1.attention.backends.flashinfer", "FlashInferBackend"),
        "flex_attention":
        ("vllm.v1.attention.backends.flex_attention", "FlexAttentionBackend"),
    }

    if backend_name not in backend_map:
        raise ValueError(f"Unknown backend: {backend_name}")

    module_name, backend_class_name = backend_map[backend_name]

    try:
        import importlib
        module = importlib.import_module(module_name)
        backend_class = getattr(module, backend_class_name)
        return backend_class.get_builder_cls(), backend_class.get_impl_cls()
    except ImportError as e:
        pytest.skip(f"{backend_name} not available: {e}")


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self):
        self._q_scale = torch.tensor(1.0)
        self._k_scale = torch.tensor(1.0)
        self._v_scale = torch.tensor(1.0)


def run_attention_backend(backend_name: str, kv_cache_spec: FullAttentionSpec,
                          vllm_config, device: torch.device,
                          common_attn_metadata: CommonAttentionMetadata,
                          query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor,
                          kv_cache: torch.Tensor) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    builder_cls, impl_cls = get_attention_backend_classes(backend_name)

    # Build metadata
    builder = builder_cls(kv_cache_spec, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Instantiate implementation
    num_heads = kv_cache_spec.num_kv_heads
    head_size = kv_cache_spec.head_size
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer()
    output = torch.empty_like(query)

    # Run forward pass
    # NOTE: The query, key, and value are already shaped correctly
    # in the calling test function.
    output = impl.forward(mock_layer,
                          query,
                          key,
                          value,
                          kv_cache,
                          attn_metadata,
                          output=output)

    return output


@pytest.mark.parametrize(
    "test_spec",
    [
        # Use a subset of test specs for correctness testing
        AttentionTestSpec(
            BatchSpec("small_decode",
                      batch_size=2,
                      num_tokens=2,
                      seq_lens=[32, 40],
                      query_lens=[1, 1]), DEFAULT_MODEL_PARAMS),
        AttentionTestSpec(
            BatchSpec("small_prefill",
                      batch_size=2,
                      num_tokens=16,
                      seq_lens=[32, 40],
                      query_lens=[8, 8]), DEFAULT_MODEL_PARAMS),
        AttentionTestSpec(
            BatchSpec("mixed_small",
                      batch_size=4,
                      num_tokens=12,
                      seq_lens=[32, 40, 48, 56],
                      query_lens=[1, 1, 5, 5]), DEFAULT_MODEL_PARAMS),
    ],
    ids=lambda spec: f"correctness_{spec.batch_spec.name}")
def test_backend_correctness_against_flash_attention(
        test_spec: AttentionTestSpec, vllm_config, device):
    """
    Test that all backends produce similar outputs to a reference implementation
    using torch.nn.functional.scaled_dot_product_attention.

    This test works by:
    1. Generating a batch of sequences with specified context and query lengths.
    2. Computing a ground-truth attention output using torch.sdpa on
       contiguous Q, K, and V tensors.
    3. Simulating vLLM's paged KV cache: It takes the context portion of the
       K/V tensors and manually places them into a paged buffer according to
       the test's (randomly generated) block table.
    4. Running each vLLM attention backend with the new queries and the
       simulated paged KV cache.
    5. Comparing the vLLM backend's output to the ground-truth SDPA output.
    """
    kv_cache_spec = create_kv_cache_spec(test_spec)
    common_attn_metadata = create_common_attn_metadata_from_combined(
        test_spec, device)

    # 1. Setup
    batch_size = test_spec.batch_spec.batch_size
    seq_lens = test_spec.batch_spec.seq_lens
    query_lens = test_spec.batch_spec.query_lens
    num_q_heads = test_spec.model_params.num_kv_heads
    num_kv_heads = test_spec.model_params.num_kv_heads
    head_size = test_spec.model_params.head_size
    dtype = test_spec.model_params.dtype
    block_size = test_spec.model_params.block_size
    scale = 1.0 / (head_size**0.5)

    # 2. Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    all_sdpa_outputs = []
    all_k_context, all_v_context = [], []

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence to be used in SDPA
        q_for_sdpa = torch.randn(q_len,
                                 num_q_heads,
                                 head_size,
                                 dtype=dtype,
                                 device=device)
        k_full = torch.randn(s_len,
                             num_kv_heads,
                             head_size,
                             dtype=dtype,
                             device=device)
        v_full = torch.randn(s_len,
                             num_kv_heads,
                             head_size,
                             dtype=dtype,
                             device=device)

        # SDPA expects (N, H, L, D), so unsqueeze batch and permute
        q_sdpa_in = q_for_sdpa.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)

        # Create a causal mask that reflects that the query tokens are at the
        # end of the full sequence.
        attn_mask = torch.ones(q_len, s_len, dtype=torch.bool,
                               device=device).tril(diagonal=context_len)

        sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, scale=scale)
        # Convert back to (L, H, D)
        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))

        # Inputs for vLLM backends are just the new tokens
        all_q_vllm.append(q_for_sdpa)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        all_k_context.append(k_full[:context_len])
        all_v_context.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    block_table = common_attn_metadata.block_table_tensor
    num_blocks = int(block_table.max().item()) + 1
    kv_cache = torch.zeros(2,
                           num_blocks,
                           block_size,
                           num_kv_heads,
                           head_size,
                           dtype=dtype,
                           device=device)

    # Create a realistic slot mapping that corresponds to the block table
    slot_mapping_list = []
    query_start_locs = common_attn_metadata.query_start_loc_cpu.tolist()

    for i in range(batch_size):
        context_len = seq_lens[i] - query_lens[i]
        start_idx = query_start_locs[i]
        end_idx = query_start_locs[i + 1]

        for token_idx_in_query in range(end_idx - start_idx):
            token_seq_idx = context_len + token_idx_in_query
            logical_block_idx = token_seq_idx // block_size
            offset_in_block = token_seq_idx % block_size
            physical_block_num = int(block_table[i, logical_block_idx].item())
            slot = physical_block_num * block_size + offset_in_block
            slot_mapping_list.append(slot)

    common_attn_metadata.slot_mapping = torch.tensor(slot_mapping_list,
                                                     dtype=torch.long,
                                                     device=device)

    # Populate the cache with the context tokens
    for i in range(batch_size):
        k_context, v_context = all_k_context[i], all_v_context[i]
        context_len = k_context.shape[0]

        for token_idx in range(context_len):
            logical_block_idx = token_idx // block_size
            offset_in_block = token_idx % block_size
            phys_block_num = int(block_table[i, logical_block_idx].item())

            kv_cache[0, phys_block_num, offset_in_block] = k_context[token_idx]
            kv_cache[1, phys_block_num, offset_in_block] = v_context[token_idx]

    # 4. Run vLLM backends and compare
    backends_to_test = ["flash_attn", "flex_attention"]
    for backend_name in backends_to_test:
        try:
            backend_output = run_attention_backend(backend_name, kv_cache_spec,
                                                   vllm_config, device,
                                                   common_attn_metadata,
                                                   query_vllm, key_vllm,
                                                   value_vllm, kv_cache)

            # Check shape and dtype consistency
            assert backend_output.shape == sdpa_output.shape, (
                f"[{backend_name}] shape {backend_output.shape} != "
                f"SDPA shape {sdpa_output.shape}")
            assert backend_output.dtype == sdpa_output.dtype, (
                f"[{backend_name}] dtype {backend_output.dtype} != "
                f"SDPA dtype {sdpa_output.dtype}")

            assert torch.isfinite(backend_output).all(), (
                f"[{backend_name}] produced non-finite values")

            # Check numerical similarity
            rtol = 1e-5 if backend_output.dtype == torch.float32 else 1e-2
            atol = 1e-4 if backend_output.dtype == torch.float32 else 1e-3

            max_diff = torch.max(torch.abs(backend_output -
                                           sdpa_output)).item()
            assert torch.allclose(
                backend_output, sdpa_output, rtol=rtol, atol=atol), (
                    f"[{backend_name}] output differs from SDPA baseline. "
                    f"Max diff: {max_diff:.6f}")

        except Exception as e:
            if "not available" in str(e) or "not supported" in str(e).lower():
                pytest.skip(f"{backend_name} not available/supported: {e}")
            else:
                pytest.fail(f"[{backend_name}] failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
