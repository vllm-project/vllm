import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    unified_kv_cache_update,
)

# Try to import FlashAttentionBackend
try:
    from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
except ImportError:
    HAS_FLASH_ATTN = False

logger = init_logger(__name__)


@pytest.mark.parametrize("use_optimization", [True])
def test_unified_kv_cache_optimization_index_logic(use_optimization):
    """
    Verifies that the index-based layer name resolution in unified_kv_cache_update
    works correctly using real Attention layers (no mocks).
    """
    logger.info("Starting test_unified_kv_cache_optimization_index_logic")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping test")
        pytest.skip("Test requires CUDA")

    if not HAS_FLASH_ATTN:
        logger.warning("FlashAttention not available, skipping test")
        pytest.skip("Test requires FlashAttention")

    # 1. Setup minimal VllmConfig
    model_config = ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=0,
    )
    # Use low memory utilization to avoid OOM
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.01,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        worker_use_ray=False,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        max_model_len=2048,
        is_encoder_decoder=False,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
    )

    # 2. Setup Attention layers
    logger.info("Setting up Attention layers...")
    with set_current_vllm_config(vllm_config):
        # Force FlashAttentionBackend to ensure we have do_kv_cache_update
        # and forward_includes_kv_cache_update = False

        layer0 = Attention(
            num_heads=4,
            head_size=16,
            scale=1.0,
            num_kv_heads=4,
            prefix="layer_0",
            attn_backend=FlashAttentionBackend,
        )
        layer1 = Attention(
            num_heads=4,
            head_size=16,
            scale=1.0,
            num_kv_heads=4,
            prefix="layer_1",
            attn_backend=FlashAttentionBackend,
        )

        # Move layers to CUDA to ensure buffers (like _k_scale) are on correct device
        layer0.to("cuda")
        layer1.to("cuda")

        # Verify backend properties
        assert not layer0.attn_backend.forward_includes_kv_cache_update

        # Manually allocate KV cache for the layers
        num_blocks = 16  # Small number of blocks
        block_size = 16
        num_kv_heads = 4
        head_size = 16
        dtype = torch.float16

        # Get KV cache shape
        kv_shape = layer0.attn_backend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

        # Allocate KV cache on GPU
        kv_cache_0 = torch.zeros(kv_shape, dtype=dtype, device="cuda")
        kv_cache_1 = torch.zeros(kv_shape, dtype=dtype, device="cuda")

        # Assign to layers (virtual_engine 0)
        layer0.kv_cache[0] = kv_cache_0
        layer1.kv_cache[0] = kv_cache_1

    # 3. Setup ForwardContext
    logger.info("Creating ForwardContext...")
    no_compile_layers = {"layer_0": layer0, "layer_1": layer1}
    all_kv_cache_update_layers = ["layer_0", "layer_1"]

    # Update compilation config
    vllm_config.compilation_config.static_forward_context = no_compile_layers

    # Create valid slot_mapping
    # k has shape (1, 4, 16) -> 1 token
    # Map to block 0, offset 0
    slot_mapping_tensor = torch.tensor([0], dtype=torch.long, device="cuda")

    forward_context = ForwardContext(
        no_compile_layers=no_compile_layers,
        remaining_moe_layers=[],
        all_kv_cache_update_layers=all_kv_cache_update_layers,
        virtual_engine=0,
        attn_metadata={},
        slot_mapping={"layer_0": slot_mapping_tensor, "layer_1": slot_mapping_tensor},
    )

    # 4. Verify Index Logic
    logger.info("Verifying index logic...")
    with set_current_vllm_config(vllm_config):
        # We need to set the global forward context
        # But ForwardContext is usually set via context manager.
        # However, for this test we want to manually control it or just set it globally
        # as the original test did.
        import vllm.forward_context

        original_context = vllm.forward_context._forward_context
        vllm.forward_context._forward_context = forward_context

        try:
            # Prepare input tensors
            # k: (num_tokens, num_kv_heads, head_size)
            k = torch.randn(1, 4, 16, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 4, 16, device="cuda", dtype=torch.float16)

            # --- First Call (Should resolve to layer_0) ---
            logger.info("Testing first call (expecting layer_0)...")
            assert forward_context.kv_cache_update_index == 0

            # This calls the real implementation!
            unified_kv_cache_update(k, v, "from_forward_context")

            assert forward_context.kv_cache_update_index == 1
            logger.info("First call passed.")

            # --- Second Call (Should resolve to layer_1) ---
            logger.info("Testing second call (expecting layer_1)...")
            unified_kv_cache_update(k, v, "from_forward_context")

            assert forward_context.kv_cache_update_index == 2
            logger.info("Second call passed.")

            # --- Out of Bounds Call ---
            logger.info("Testing out of bounds call...")
            with pytest.raises(
                AssertionError, match="expected the number of KV cache update layers"
            ):
                unified_kv_cache_update(k, v, "from_forward_context")
            logger.info("Out of bounds check passed.")

        except Exception as e:
            logger.error("Test failed with exception: %s", e)
            raise e
        finally:
            vllm.forward_context._forward_context = original_context

    logger.info("test_unified_kv_cache_optimization_index_logic passed successfully!")
