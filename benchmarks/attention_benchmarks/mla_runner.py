# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MLA benchmark runner - shared utilities for MLA benchmarks.

This module provides helpers for running MLA backends without
needing full VllmConfig integration.
"""

import numpy as np
import torch
from batch_spec import parse_batch_spec
from common import (
    BenchmarkResult,
    MockHfConfig,
    MockIndexer,
    MockKVBProj,
    MockLayer,
    setup_mla_dims,
)

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)

# ============================================================================
# VllmConfig Creation
# ============================================================================


def _add_mock_methods_to_model_config(model_config: ModelConfig) -> None:
    """
    Add mock methods for layer-specific queries to ModelConfig.

    These methods are needed by metadata builders but aren't normally
    present on ModelConfig when used in benchmark contexts.
    """
    import types

    model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
    model_config.get_sliding_window_for_layer = types.MethodType(
        lambda self, _i: None, model_config
    )
    model_config.get_logits_soft_cap_for_layer = types.MethodType(
        lambda self, _i: None, model_config
    )
    model_config.get_sm_scale_for_layer = types.MethodType(
        lambda self, _i: 1.0 / model_config.get_head_size() ** 0.5, model_config
    )


def create_minimal_vllm_config(
    model_name: str = "deepseek-v3",
    block_size: int = 128,
    max_num_seqs: int = 256,
    mla_dims: dict | None = None,
    index_topk: int | None = None,
) -> VllmConfig:
    """
    Create minimal VllmConfig for MLA benchmarks.

    Args:
        model_name: Model name (deepseek-v2, deepseek-v3, etc.) - used if mla_dims not
                    provided
        block_size: KV cache block size
        max_num_seqs: Maximum number of sequences
        mla_dims: Optional custom MLA dimensions dict. If not provided, uses
                  setup_mla_dims(model_name)
        index_topk: Optional topk value for sparse MLA backends. If provided,
                    the config will include index_topk for sparse attention.

    Returns:
        VllmConfig for benchmarking
    """
    # Get MLA dimensions - use provided or load from model name
    if mla_dims is None:
        mla_dims = setup_mla_dims(model_name)

    # Create mock HF config first (avoids downloading from HuggingFace)
    mock_hf_config = MockHfConfig(mla_dims, index_topk=index_topk)

    # Create a temporary minimal config.json to avoid HF downloads
    # This ensures consistent ModelConfig construction without network access
    import json
    import os
    import shutil
    import tempfile

    minimal_config = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "num_attention_heads": mla_dims["num_q_heads"],
        "num_key_value_heads": mla_dims["num_kv_heads"],
        "hidden_size": mla_dims["head_dim"] * mla_dims["num_q_heads"],
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 163840,  # DeepSeek V3 default
        "rope_theta": 10000.0,
        "vocab_size": 128256,
    }

    # Create temporary directory with config.json
    temp_dir = tempfile.mkdtemp(prefix="vllm_bench_")
    config_path = os.path.join(temp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(minimal_config, f)

    try:
        # Create model config using local path - no HF downloads
        model_config = ModelConfig(
            model=temp_dir,  # Use local temp directory
            tokenizer=None,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="bfloat16",
            seed=0,
            max_model_len=32768,
            quantization=None,
            enforce_eager=False,
            max_logprobs=20,
            disable_sliding_window=False,
            skip_tokenizer_init=True,
            served_model_name=None,
            limit_mm_per_prompt=None,
            config_format="auto",
        )
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Override with our mock config
    model_config.hf_config = mock_hf_config
    model_config.hf_text_config = mock_hf_config

    # Add mock methods for layer-specific queries
    _add_mock_methods_to_model_config(model_config)

    # Create sub-configs
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        max_model_len=32768,
        is_encoder_decoder=False,
        enable_chunked_prefill=True,
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
    )

    compilation_config = CompilationConfig()

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        compilation_config=compilation_config,
    )


# ============================================================================
# Backend Configuration
# ============================================================================


# Backend-specific properties that can't be inferred from the backend class
# Keys are AttentionBackendEnum names (uppercase)
_BACKEND_PROPERTIES = {
    "FLASHMLA": {
        "query_format": "concat",  # Single concatenated tensor (vs tuple)
    },
    "FLASHMLA_SPARSE": {
        "query_format": "concat",  # Single concatenated tensor (vs tuple)
    },
}


def _get_backend_config(backend: str) -> dict:
    """
    Get backend configuration from AttentionBackendEnum.

    Uses the registry to get the backend class and extract configuration
    from its methods (get_impl_cls, get_builder_cls, is_sparse, etc.).

    Args:
        backend: Backend name matching AttentionBackendEnum exactly
        (e.g., "FLASHMLA_SPARSE")

    Returns:
        Dict with backend configuration
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    try:
        backend_enum = AttentionBackendEnum[backend]
        backend_class = backend_enum.get_class()
    except (KeyError, ValueError) as e:
        valid_backends = [e.name for e in AttentionBackendEnum if e.name != "CUSTOM"]
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Valid MLA backends: {[b for b in valid_backends if 'MLA' in b]}"
        ) from e

    # Get block size from backend class
    block_sizes = backend_class.get_supported_kernel_block_sizes()
    # Use first supported block size (backends typically support one for MLA)
    block_size = block_sizes[0] if block_sizes else None
    if hasattr(block_size, "value"):
        # Handle MultipleOf enum
        block_size = None

    # Check if sparse via class method if available
    is_sparse = getattr(backend_class, "is_sparse", lambda: False)()

    # Get properties that can't be inferred
    props = _BACKEND_PROPERTIES.get(backend, {})

    return {
        "backend_class": backend_class,
        "impl_class": backend_class.get_impl_cls(),
        "builder_class": backend_class.get_builder_cls(),
        "query_format": props.get("query_format", "tuple"),
        "block_size": block_size,
        "is_sparse": is_sparse,
    }


# ============================================================================
# Metadata Building Helpers
# ============================================================================


def _build_attention_metadata(
    requests: list,
    block_size: int,
    device: torch.device,
    builder_instance,
) -> tuple:
    """
    Build attention metadata from batch requests.

    Args:
        requests: List of BatchRequest objects
        block_size: KV cache block size
        device: Target device
        builder_instance: Metadata builder instance

    Returns:
        Tuple of (metadata, kv_cache_num_blocks)
    """
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)
    max_kv = max(kv_lens)

    # Build query start locations
    q_start_cpu = torch.tensor(
        [0] + [sum(q_lens[: i + 1]) for i in range(len(q_lens))],
        dtype=torch.int32,
    )
    q_start_gpu = q_start_cpu.to(device)

    # Build sequence lengths
    seq_lens_cpu = torch.tensor(kv_lens, dtype=torch.int32)
    seq_lens_gpu = seq_lens_cpu.to(device)

    # Build num_computed_tokens (context length for each request)
    context_lens = [kv_len - q_len for q_len, kv_len in zip(q_lens, kv_lens)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)

    # Build block table
    num_blocks_per_req = [(kv + block_size - 1) // block_size for kv in kv_lens]
    max_num_blocks = max(num_blocks_per_req)

    block_table_cpu = np.zeros((len(requests), max_num_blocks), dtype=np.int32)
    current_block = 0
    for i, num_blocks in enumerate(num_blocks_per_req):
        for j in range(num_blocks):
            block_table_cpu[i, j] = current_block
            current_block += 1

    block_table_gpu = torch.from_numpy(block_table_cpu).to(device)

    # Build slot mapping
    slot_mapping_list = []
    for i, (q_len, kv_len, num_blocks) in enumerate(
        zip(q_lens, kv_lens, num_blocks_per_req)
    ):
        context_len = kv_len - q_len
        for j in range(q_len):
            token_kv_idx = context_len + j
            block_idx = token_kv_idx // block_size
            offset_in_block = token_kv_idx % block_size
            global_block_id = block_table_cpu[i, block_idx]
            slot_id = global_block_id * block_size + offset_in_block
            slot_mapping_list.append(slot_id)

    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int64, device=device)

    # Create CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    common_attn_metadata = CommonAttentionMetadata(
        num_reqs=len(requests),
        max_query_len=max(q_lens),
        max_seq_len=max_kv,
        num_actual_tokens=total_q,
        query_start_loc=q_start_gpu,
        query_start_loc_cpu=q_start_cpu,
        seq_lens=seq_lens_gpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        slot_mapping=slot_mapping,
        block_table_tensor=block_table_gpu,
        dcp_local_seq_lens=None,
    )

    # Use the production build() method
    metadata = builder_instance.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        fast_build=False,
    )

    return metadata, current_block


def _create_input_tensors(
    total_q: int,
    mla_dims: dict,
    query_format: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Create input tensors for both decode and prefill modes.

    MLA requires different tensor formats for decode vs prefill:
    - Decode: Uses kv_lora_rank (512) dimension
    - Prefill: Uses qk_nope_head_dim (128) to stay under FlashAttention's 256 limit

    Args:
        total_q: Total number of query tokens
        mla_dims: MLA dimension configuration
        query_format: Either "tuple" or "concat"
        device: Target device
        dtype: Tensor dtype

    Returns:
        Tuple of (decode_inputs, prefill_inputs)
        - decode_inputs: Query tensor(s) for decode mode
        - prefill_inputs: Dict with 'q', 'k_c_normed', 'k_pe', 'k_scale' for prefill
    """
    if query_format == "tuple":
        # Decode mode format: (q_nope, q_pe) where q_nope has kv_lora_rank dim
        q_nope_decode = torch.randn(
            total_q,
            mla_dims["num_q_heads"],
            mla_dims["kv_lora_rank"],
            device=device,
            dtype=dtype,
        )
        q_pe = torch.randn(
            total_q,
            mla_dims["num_q_heads"],
            mla_dims["qk_rope_head_dim"],
            device=device,
            dtype=dtype,
        )
        decode_inputs = (q_nope_decode, q_pe)

        # For prefill, we need q with qk_nope_head_dim instead of kv_lora_rank
        q_nope_prefill = torch.randn(
            total_q,
            mla_dims["num_q_heads"],
            mla_dims["qk_nope_head_dim"],
            device=device,
            dtype=dtype,
        )
        prefill_q = torch.cat([q_nope_prefill, q_pe], dim=-1)
    else:  # concat
        decode_inputs = torch.randn(
            total_q,
            mla_dims["num_q_heads"],
            mla_dims["kv_lora_rank"] + mla_dims["qk_rope_head_dim"],
            device=device,
            dtype=dtype,
        )
        # For prefill with concat format
        prefill_q = torch.randn(
            total_q,
            mla_dims["num_q_heads"],
            mla_dims["qk_nope_head_dim"] + mla_dims["qk_rope_head_dim"],
            device=device,
            dtype=dtype,
        )

    # Create additional inputs needed for prefill forward
    k_c_normed = torch.randn(
        total_q,
        mla_dims["kv_lora_rank"],
        device=device,
        dtype=dtype,
    )
    k_pe = torch.randn(
        total_q,
        1,  # Single head for MLA
        mla_dims["qk_rope_head_dim"],
        device=device,
        dtype=dtype,
    )
    k_scale = torch.ones(1, device=device, dtype=torch.float32)

    output = torch.zeros(
        total_q,
        mla_dims["num_q_heads"] * mla_dims["v_head_dim"],
        device=device,
        dtype=dtype,
    )

    prefill_inputs = {
        "q": prefill_q,
        "k_c_normed": k_c_normed,
        "k_pe": k_pe,
        "k_scale": k_scale,
        "output": output,
    }

    return decode_inputs, prefill_inputs


# ============================================================================
# Backend Initialization
# ============================================================================


def _create_backend_impl(
    backend_cfg: dict,
    mla_dims: dict,
    vllm_config: VllmConfig,
    device: torch.device,
    max_num_tokens: int = 8192,
    index_topk: int | None = None,
):
    """
    Create backend implementation instance.

    Args:
        backend_cfg: Backend configuration dict from _get_backend_config()
        mla_dims: MLA dimension configuration
        vllm_config: VllmConfig instance
        device: Target device
        max_num_tokens: Maximum number of tokens for sparse indexer buffer
        index_topk: Topk value for sparse MLA backends

    Returns:
        Tuple of (impl, layer, builder_instance, indexer)
    """
    # Get classes from backend config (already resolved by _get_backend_config)
    impl_class = backend_cfg["impl_class"]
    builder_class = backend_cfg["builder_class"]

    # Calculate scale
    scale = 1.0 / np.sqrt(mla_dims["qk_nope_head_dim"] + mla_dims["qk_rope_head_dim"])

    # Create mock kv_b_proj layer for prefill mode
    mock_kv_b_proj = MockKVBProj(
        num_heads=mla_dims["num_q_heads"],
        qk_nope_head_dim=mla_dims["qk_nope_head_dim"],
        v_head_dim=mla_dims["v_head_dim"],
    )

    # Create indexer for sparse backends
    indexer = None
    if backend_cfg.get("is_sparse", False):
        if index_topk is None:
            index_topk = 2048  # Default topk for sparse MLA
        indexer = MockIndexer(
            max_num_tokens=max_num_tokens,
            topk_tokens=index_topk,
            device=device,
        )

    # Build impl kwargs
    impl_kwargs = {
        "num_heads": mla_dims["num_q_heads"],
        "head_size": mla_dims["head_dim"],
        "scale": scale,
        "num_kv_heads": mla_dims["num_kv_heads"],
        "alibi_slopes": None,
        "sliding_window": None,
        "kv_cache_dtype": "auto",
        "logits_soft_cap": None,
        "attn_type": "decoder",
        "kv_sharing_target_layer_name": None,
        "q_lora_rank": None,
        "kv_lora_rank": mla_dims["kv_lora_rank"],
        "qk_nope_head_dim": mla_dims["qk_nope_head_dim"],
        "qk_rope_head_dim": mla_dims["qk_rope_head_dim"],
        "qk_head_dim": mla_dims["qk_nope_head_dim"] + mla_dims["qk_rope_head_dim"],
        "v_head_dim": mla_dims["v_head_dim"],
        "kv_b_proj": mock_kv_b_proj,
    }

    # Add indexer for sparse backends
    if indexer is not None:
        impl_kwargs["indexer"] = indexer

    # Create impl
    impl = impl_class(**impl_kwargs)

    # Initialize DCP attributes
    if not hasattr(impl, "dcp_world_size") or impl.dcp_world_size in (None, -1):
        impl.dcp_world_size = 1
        impl.dcp_rank = 0

    # Create KV cache spec for MockLayer
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    kv_cache_spec = FullAttentionSpec(
        block_size=backend_cfg["block_size"] or vllm_config.cache_config.block_size,
        num_kv_heads=1,  # MLA uses 1 KV head
        head_size=576,  # MLA head dim
        dtype=torch.bfloat16,
    )

    # Create mock layer
    layer = MockLayer(device, impl=impl, kv_cache_spec=kv_cache_spec)

    # Create builder instance if needed
    builder_instance = None
    if builder_class:
        # Populate static_forward_context so builder can find the layer
        # MockLayer inherits from AttentionLayerBase, so isinstance checks pass
        vllm_config.compilation_config.static_forward_context = {"placeholder": layer}

        builder_instance = builder_class(
            kv_cache_spec=kv_cache_spec,
            layer_names=["placeholder"],
            vllm_config=vllm_config,
            device=device,
        )

    return impl, layer, builder_instance, indexer


# ============================================================================
# Config Helpers
# ============================================================================


def _extract_mla_dims_from_config(config) -> dict | None:
    """
    Extract MLA dimensions from BenchmarkConfig if all required fields are present.

    Args:
        config: BenchmarkConfig instance

    Returns:
        Dict with MLA dimensions if all fields are provided, None otherwise
    """
    # Check if all MLA-specific fields are provided
    if all(
        [
            config.kv_lora_rank is not None,
            config.qk_nope_head_dim is not None,
            config.qk_rope_head_dim is not None,
            config.v_head_dim is not None,
        ]
    ):
        return {
            "kv_lora_rank": config.kv_lora_rank,
            "qk_nope_head_dim": config.qk_nope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "v_head_dim": config.v_head_dim,
            "num_q_heads": config.num_q_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
        }
    # Fallback: if MLA fields not fully specified, try to construct from basic fields
    elif config.head_dim == 576:
        # This looks like a DeepSeek MLA config, use standard dimensions with custom
        # head count
        return {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_q_heads": config.num_q_heads,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
        }
    return None


# ============================================================================
# Benchmark Execution
# ============================================================================


def _run_single_benchmark(
    config,
    impl,
    layer,
    builder_instance,
    backend_cfg: dict,
    mla_dims: dict,
    device: torch.device,
    indexer=None,
) -> BenchmarkResult:
    """
    Run a single benchmark iteration.

    Args:
        config: BenchmarkConfig instance
        impl: Backend implementation instance
        layer: MockLayer instance
        builder_instance: Metadata builder instance
        backend_cfg: Backend configuration dict
        mla_dims: MLA dimension configuration
        device: Target device
        indexer: Optional MockIndexer for sparse backends

    Returns:
        BenchmarkResult with timing statistics
    """
    # Parse batch spec
    requests = parse_batch_spec(config.batch_spec)
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)
    max_kv_len = max(kv_lens)

    # Determine block size
    block_size = backend_cfg["block_size"] or config.block_size

    # Build metadata
    metadata, num_blocks = _build_attention_metadata(
        requests, block_size, device, builder_instance
    )

    # Create KV cache
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        mla_dims["kv_lora_rank"] + mla_dims["qk_rope_head_dim"],
        device=device,
        dtype=torch.bfloat16,
    )

    # Create input tensors for both decode and prefill modes
    decode_inputs, prefill_inputs = _create_input_tensors(
        total_q,
        mla_dims,
        backend_cfg["query_format"],
        device,
        torch.bfloat16,
    )

    # Fill indexer with random indices for sparse backends
    is_sparse = backend_cfg.get("is_sparse", False)
    if is_sparse and indexer is not None:
        indexer.fill_random_indices(total_q, max_kv_len)

    # Determine which forward method to use
    if is_sparse:
        # Sparse backends use forward_mqa
        forward_fn = lambda: impl.forward_mqa(decode_inputs, kv_cache, metadata, layer)
    elif metadata.decode is not None:
        forward_fn = lambda: impl._forward_decode(
            decode_inputs, kv_cache, metadata, layer
        )
    elif metadata.prefill is not None:
        forward_fn = lambda: impl._forward_prefill(
            prefill_inputs["q"],
            prefill_inputs["k_c_normed"],
            prefill_inputs["k_pe"],
            kv_cache,
            metadata,
            prefill_inputs["k_scale"],
            prefill_inputs["output"],
        )
    else:
        raise RuntimeError("Metadata has neither decode nor prefill metadata")

    # Warmup
    for _ in range(config.warmup_iters):
        forward_fn()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(config.num_layers):
            forward_fn()
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms / 1000.0 / config.num_layers)

    mean_time = float(np.mean(times))
    return BenchmarkResult(
        config=config,
        mean_time=mean_time,
        std_time=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        throughput_tokens_per_sec=total_q / mean_time if mean_time > 0 else 0,
    )


def _run_mla_benchmark_batched(
    backend: str,
    configs_with_params: list[tuple],  # [(config, threshold, num_splits), ...]
    index_topk: int = 2048,
) -> list[BenchmarkResult]:
    """
    Unified batched MLA benchmark runner for all backends.

    Works for: flashattn_mla, flashmla, flashinfer_mla, cutlass_mla,
               flashinfer_mla_sparse, flashmla_sparse

    This function reuses backend initialization across multiple benchmarks
    to avoid setup/teardown overhead.

    Args:
        backend: Backend name
        configs_with_params: List of (config, threshold, num_splits) tuples
            - threshold: reorder_batch_threshold (FlashAttn/FlashMLA only)
            - num_splits: num_kv_splits (CUTLASS only)
        index_topk: Topk value for sparse MLA backends (default 2048)

    Returns:
        List of BenchmarkResult objects
    """
    if not configs_with_params:
        return []

    backend_cfg = _get_backend_config(backend)
    device = torch.device(configs_with_params[0][0].device)
    torch.cuda.set_device(device)

    # Determine block size
    config_block_size = configs_with_params[0][0].block_size
    block_size = backend_cfg["block_size"] or config_block_size

    # Extract MLA dimensions from the first config
    first_config = configs_with_params[0][0]
    mla_dims = _extract_mla_dims_from_config(first_config)

    # If config didn't provide MLA dims, fall back to default model
    if mla_dims is None:
        mla_dims = setup_mla_dims("deepseek-v3")

    # Determine if this is a sparse backend
    is_sparse = backend_cfg.get("is_sparse", False)

    # Create and set vLLM config for MLA (reused across all benchmarks)
    vllm_config = create_minimal_vllm_config(
        model_name="deepseek-v3",  # Used only for model path
        block_size=block_size,
        mla_dims=mla_dims,  # Use custom dims from config or default
        index_topk=index_topk if is_sparse else None,
    )

    results = []

    with set_current_vllm_config(vllm_config):
        # Create backend impl, layer, builder, and indexer (reused across benchmarks)
        impl, layer, builder_instance, indexer = _create_backend_impl(
            backend_cfg,
            mla_dims,
            vllm_config,
            device,
            index_topk=index_topk if is_sparse else None,
        )

        # Run each benchmark with the shared impl
        for config, threshold, num_splits in configs_with_params:
            # Set threshold for this benchmark (FlashAttn/FlashMLA only)
            original_threshold = None
            if threshold is not None and builder_instance:
                original_threshold = builder_instance.reorder_batch_threshold
                builder_instance.reorder_batch_threshold = threshold

            # Set num_splits for CUTLASS
            original_num_splits = None
            if num_splits is not None and hasattr(impl, "_num_kv_splits"):
                original_num_splits = impl._num_kv_splits
                impl._num_kv_splits = num_splits

            try:
                result = _run_single_benchmark(
                    config,
                    impl,
                    layer,
                    builder_instance,
                    backend_cfg,
                    mla_dims,
                    device,
                    indexer=indexer,
                )
                results.append(result)

            finally:
                # Restore original threshold
                if original_threshold is not None:
                    builder_instance.reorder_batch_threshold = original_threshold

                # Restore original num_splits
                if original_num_splits is not None:
                    impl._num_kv_splits = original_num_splits

    return results


# ============================================================================
# Public API
# ============================================================================


def run_mla_benchmark(
    backend: str,
    config,
    reorder_batch_threshold: int | None = None,
    num_kv_splits: int | None = None,
    index_topk: int = 2048,
) -> BenchmarkResult | list[BenchmarkResult]:
    """
    Unified MLA benchmark runner for all backends.

    Works for: flashattn_mla, flashmla, flashinfer_mla, cutlass_mla,
               flashinfer_mla_sparse, flashmla_sparse

    Always uses batched execution internally for optimal performance.

    Args:
        backend: Backend name (flashattn_mla, flashmla, flashinfer_mla, cutlass_mla,
                 flashinfer_mla_sparse, flashmla_sparse)
        config: BenchmarkConfig or list of (BenchmarkConfig, param) tuples
        reorder_batch_threshold: Threshold override for FlashAttn/FlashMLA
                                 (single config mode only)
        num_kv_splits: Number of KV splits for CUTLASS (single config mode only)
        index_topk: Topk value for sparse MLA backends (default 2048)

    Returns:
        BenchmarkResult (single mode) or list of BenchmarkResult (batched mode)
    """
    # Normalize to batched mode: (config, threshold, num_splits)
    if isinstance(config, list):
        # Already in batched format
        if len(config) > 0 and isinstance(config[0], tuple):
            # Format: [(cfg, param), ...] where param is threshold or num_splits
            if backend in ("flashattn_mla", "flashmla", "flashmla_sparse"):
                configs_with_params = [(cfg, param, None) for cfg, param in config]
            else:  # cutlass_mla, flashinfer_mla, or sparse backends
                configs_with_params = [(cfg, None, param) for cfg, param in config]
        else:
            # Format: [cfg, ...] - just configs
            configs_with_params = [(cfg, None, None) for cfg in config]
        return_single = False
    else:
        # Single config: convert to batched format
        configs_with_params = [(config, reorder_batch_threshold, num_kv_splits)]
        return_single = True

    # Use unified batched execution
    results = _run_mla_benchmark_batched(backend, configs_with_params, index_topk)

    # Return single result or list based on input
    return results[0] if return_single else results
