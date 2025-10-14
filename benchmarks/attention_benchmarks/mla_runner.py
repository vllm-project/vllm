# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MLA benchmark runner - shared utilities for MLA benchmarks.

This module provides helpers for running MLA backends without
needing full VllmConfig integration.
"""

import importlib
from typing import Optional

import numpy as np
import torch
from batch_spec import parse_batch_spec
from common import MockHfConfig, MockLayer, setup_mla_dims

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
) -> VllmConfig:
    """
    Create minimal VllmConfig for MLA benchmarks.

    Args:
        model_name: Model name (deepseek-v2, deepseek-v3, etc.)
        block_size: KV cache block size
        max_num_seqs: Maximum number of sequences

    Returns:
        VllmConfig for benchmarking
    """
    # Get MLA dimensions
    mla_dims = setup_mla_dims(model_name)

    # Create model config
    model_config = ModelConfig(
        model=f"deepseek-ai/{model_name}",
        tokenizer=None,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=0,
        max_model_len=32768,
        quantization=None,
        quantization_param_path=None,
        enforce_eager=False,
        max_context_len_to_capture=None,
        max_seq_len_to_capture=8192,
        max_logprobs=20,
        disable_sliding_window=False,
        skip_tokenizer_init=True,
        served_model_name=None,
        limit_mm_per_prompt=None,
        use_async_output_proc=True,
        config_format="auto",
    )

    # Override head counts and dims for MLA
    model_config.hf_config = MockHfConfig(mla_dims)

    # Add mock methods for layer-specific queries
    _add_mock_methods_to_model_config(model_config)

    # Create sub-configs
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        num_gpu_blocks=None,
        num_cpu_blocks=None,
        sliding_window=None,
        enable_prefix_caching=False,
        cpu_offload_gb=0,
    )

    scheduler_config = SchedulerConfig(
        task="auto",
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=None,
        max_model_len=32768,
        num_scheduler_steps=1,
        multi_step_stream_outputs=False,
        enable_chunked_prefill=None,
        preemption_mode="swap",
        num_lookahead_slots=0,
        delay_factor=0.0,
        enable_prefix_caching=False,
        policy="fcfs",
        send_delta_data=False,
    )

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        worker_cls="auto",
        max_parallel_loading_workers=None,
        disable_custom_all_reduce=False,
        tokenizer_pool_config=None,
        ray_workers_use_nsight=False,
        placement_group=None,
        distributed_executor_backend=None,
    )

    compilation_config = CompilationConfig(
        level=0,
        backend="",
        custom_ops=[],
        splitting_ops=[],
        use_inductor=True,
        enable_fusion=True,
        use_cudagraph=False,
        cudagraph_num_of_warmups=0,
        cudagraph_capture_sizes=None,
        cudagraph_copy_inputs=False,
        use_cudagraph_for_prefill=False,
        enabled_custom_ops=None,
        disabled_custom_ops=None,
        inductor_compile_sizes=[],
        inductor_compile_config={},
        inductor_passes={},
        cudagraph_backend="flashinfer",
    )

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


# Backend name to class name prefix mapping
_BACKEND_NAME_MAP = {
    "flashattn_mla": "FlashAttnMLA",
    "flashmla": "FlashMLA",
    "flashinfer_mla": "FlashInferMLA",
    "cutlass_mla": "CutlassMLA",
}

# Special properties that differ from defaults
_BACKEND_PROPERTIES = {
    "flashmla": {
        "query_format": "concat",  # Single concatenated tensor (vs tuple)
        "block_size": 64,  # FlashMLA uses fixed block size
    },
}


def _get_backend_config(backend: str) -> dict:
    """
    Get backend configuration using naming conventions.

    All MLA backends follow the pattern:
    - Module: vllm.v1.attention.backends.mla.{backend}
    - Impl: {Name}Impl
    - Metadata: {Name}Metadata (or MLACommonMetadata)
    - DecodeMetadata: {Name}DecodeMetadata (or MLACommonDecodeMetadata)
    - MetadataBuilder: {Name}MetadataBuilder
    """
    if backend not in _BACKEND_NAME_MAP:
        raise ValueError(f"Unknown backend: {backend}")

    name = _BACKEND_NAME_MAP[backend]
    props = _BACKEND_PROPERTIES.get(backend, {})

    # Check if backend uses common metadata (FlashInfer, CUTLASS)
    uses_common = backend in ("flashinfer_mla", "cutlass_mla")

    return {
        "module": f"vllm.v1.attention.backends.mla.{backend}",
        "impl_class": f"{name}Impl",
        "metadata_class": "MLACommonMetadata" if uses_common else f"{name}Metadata",
        "decode_metadata_class": "MLACommonDecodeMetadata"
        if uses_common
        else f"{name}DecodeMetadata",
        "builder_class": f"{name}MetadataBuilder",
        "query_format": props.get("query_format", "tuple"),
        "block_size": props.get("block_size", None),
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
    q_start_cpu = np.array(
        [0] + [sum(q_lens[: i + 1]) for i in range(len(q_lens))],
        dtype=np.int32,
    )
    q_start_gpu = torch.from_numpy(q_start_cpu).to(device)

    # Build sequence lengths
    seq_lens_cpu = np.array(kv_lens, dtype=np.int32)
    seq_lens_gpu = torch.from_numpy(seq_lens_cpu).to(device)

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
        seq_lens_cpu=seq_lens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
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


def _create_query_tensors(
    total_q: int,
    mla_dims: dict,
    query_format: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Create query tensors in the appropriate format for the backend.

    Args:
        total_q: Total number of query tokens
        mla_dims: MLA dimension configuration
        query_format: Either "tuple" or "concat"
        device: Target device
        dtype: Tensor dtype

    Returns:
        Query tensor(s) - either (q_nope, q_pe) tuple or concatenated tensor
    """
    if query_format == "tuple":
        q_nope = torch.randn(
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
        return (q_nope, q_pe)
    else:  # concat
        return torch.randn(
            total_q,
            mla_dims["num_q_heads"],
            mla_dims["kv_lora_rank"] + mla_dims["qk_rope_head_dim"],
            device=device,
            dtype=dtype,
        )


# ============================================================================
# Backend Initialization
# ============================================================================


def _create_backend_impl(
    backend_cfg: dict,
    mla_dims: dict,
    vllm_config: VllmConfig,
    device: torch.device,
):
    """
    Create backend implementation instance.

    Args:
        backend_cfg: Backend configuration dict
        mla_dims: MLA dimension configuration
        vllm_config: VllmConfig instance
        device: Target device

    Returns:
        Tuple of (impl, layer, builder_instance)
    """
    # Import backend classes
    backend_module = importlib.import_module(backend_cfg["module"])
    impl_class = getattr(backend_module, backend_cfg["impl_class"])

    # Calculate scale
    scale = 1.0 / np.sqrt(mla_dims["qk_nope_head_dim"] + mla_dims["qk_rope_head_dim"])

    # Create impl
    impl = impl_class(
        num_heads=mla_dims["num_q_heads"],
        head_size=mla_dims["head_dim"],
        scale=scale,
        num_kv_heads=mla_dims["num_kv_heads"],
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=mla_dims["kv_lora_rank"],
        qk_nope_head_dim=mla_dims["qk_nope_head_dim"],
        qk_rope_head_dim=mla_dims["qk_rope_head_dim"],
        qk_head_dim=mla_dims["qk_nope_head_dim"] + mla_dims["qk_rope_head_dim"],
        v_head_dim=mla_dims["v_head_dim"],
        kv_b_proj=None,
    )

    # Initialize DCP attributes
    if not hasattr(impl, "dcp_world_size") or impl.dcp_world_size is None:
        impl.dcp_world_size = 1
        impl.dcp_rank = 0

    # Create mock layer
    layer = MockLayer(device, impl=impl)

    # Create builder instance if needed
    builder_instance = None
    if backend_cfg["builder_class"]:
        builder_class = getattr(backend_module, backend_cfg["builder_class"])

        from vllm.v1.kv_cache_interface import FullAttentionSpec

        kv_cache_spec = FullAttentionSpec(
            block_size=backend_cfg["block_size"] or vllm_config.cache_config.block_size,
            num_kv_heads=1,  # MLA uses 1 KV head
            head_size=576,  # MLA head dim
            dtype=torch.float16,
        )

        # Populate static_forward_context so builder can find the layer
        # MockLayer inherits from AttentionLayerBase, so isinstance checks pass
        vllm_config.compilation_config.static_forward_context = {"placeholder": layer}

        builder_instance = builder_class(
            kv_cache_spec=kv_cache_spec,
            layer_names=["placeholder"],
            vllm_config=vllm_config,
            device=device,
        )

    return impl, layer, builder_instance


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
) -> dict:
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

    Returns:
        Dict with timing statistics
    """
    # Parse batch spec
    requests = parse_batch_spec(config.batch_spec)
    q_lens = [r.q_len for r in requests]
    total_q = sum(q_lens)

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
        dtype=torch.float16,
    )

    # Create query tensors
    query = _create_query_tensors(
        total_q,
        mla_dims,
        backend_cfg["query_format"],
        device,
        torch.float16,
    )

    # Warmup
    for _ in range(config.warmup_iters):
        impl._forward_decode(query, kv_cache, metadata, layer)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(config.num_layers):
            impl._forward_decode(query, kv_cache, metadata, layer)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms / 1000.0 / config.num_layers)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "throughput": total_q / np.mean(times) if times else 0,
    }


def _run_mla_benchmark_batched(
    backend: str,
    configs_with_params: list[tuple],  # [(config, threshold, num_splits), ...]
) -> list[dict]:
    """
    Unified batched MLA benchmark runner for all backends.

    Works for: flashattn_mla, flashmla, flashinfer_mla, cutlass_mla

    This function reuses backend initialization across multiple benchmarks
    to avoid setup/teardown overhead.

    Args:
        backend: Backend name
        configs_with_params: List of (config, threshold, num_splits) tuples
            - threshold: reorder_batch_threshold (FlashAttn/FlashMLA only)
            - num_splits: num_kv_splits (CUTLASS only)

    Returns:
        List of dicts with timing statistics
    """
    if not configs_with_params:
        return []

    backend_cfg = _get_backend_config(backend)
    device = torch.device(configs_with_params[0][0].device)
    torch.cuda.set_device(device)

    # Determine block size
    config_block_size = configs_with_params[0][0].block_size
    block_size = backend_cfg["block_size"] or config_block_size

    # Create and set vLLM config for MLA (reused across all benchmarks)
    vllm_config = create_minimal_vllm_config(
        model_name="deepseek-v3",
        block_size=block_size,
    )

    # Setup MLA dimensions (reused)
    mla_dims = setup_mla_dims("deepseek-v3")

    results = []

    with set_current_vllm_config(vllm_config):
        # Create backend impl, layer, and builder (reused across benchmarks)
        impl, layer, builder_instance = _create_backend_impl(
            backend_cfg, mla_dims, vllm_config, device
        )

        # Get builder class for threshold setting (if applicable)
        builder_class = None
        if backend_cfg["builder_class"]:
            backend_module = importlib.import_module(backend_cfg["module"])
            builder_class = getattr(backend_module, backend_cfg["builder_class"])

        # Run each benchmark with the shared impl
        for config, threshold, num_splits in configs_with_params:
            # Set threshold for this benchmark (FlashAttn/FlashMLA only)
            original_threshold = None
            if threshold is not None and builder_class:
                original_threshold = builder_class.reorder_batch_threshold
                builder_class.reorder_batch_threshold = threshold

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
                )
                results.append(result)

            finally:
                # Restore original threshold
                if original_threshold is not None:
                    builder_class.reorder_batch_threshold = original_threshold

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
    reorder_batch_threshold: Optional[int] = None,
    num_kv_splits: Optional[int] = None,
) -> dict:
    """
    Unified MLA benchmark runner for all backends.

    Works for: flashattn_mla, flashmla, flashinfer_mla, cutlass_mla

    Always uses batched execution internally for optimal performance.

    Args:
        backend: Backend name (flashattn_mla, flashmla, flashinfer_mla, cutlass_mla)
        config: BenchmarkConfig or list of (BenchmarkConfig, param) tuples
        reorder_batch_threshold: Threshold override for FlashAttn/FlashMLA
                                 (single config mode only)
        num_kv_splits: Number of KV splits for CUTLASS (single config mode only)

    Returns:
        Dict with timing statistics (single mode) or list of dicts (batched mode)
    """
    # Normalize to batched mode: (config, threshold, num_splits)
    if isinstance(config, list):
        # Already in batched format
        if len(config) > 0 and isinstance(config[0], tuple):
            # Format: [(cfg, param), ...] where param is threshold or num_splits
            if backend in ("flashattn_mla", "flashmla"):
                configs_with_params = [(cfg, param, None) for cfg, param in config]
            else:  # cutlass_mla or flashinfer_mla
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
    results = _run_mla_benchmark_batched(backend, configs_with_params)

    # Return single result or list based on input
    return results[0] if return_single else results
