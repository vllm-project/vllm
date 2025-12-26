# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MLA test script for validating RoPE and quantization changes.

This script replicates the complete MLA (Multi-Head Latent Attention) code path:
1. Q/KV projections (fused_qkv_a_proj, q_b_proj, kv_a_layernorm)
2. RoPE application to Q_pe and K_pe
3. Sparse indexer (optional, for V3.2)
4. MLAAttention forward (prefill and decode paths)
5. Output projection (o_proj)

The test serves as a reference implementation before making changes to the actual
MLA implementation.

Backend Support:
- TRITON_MLA: Works on all GPUs
- FLASHINFER_MLA: Requires Blackwell GPU (SM 10.x)
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
    try_get_attention_backend,
)
from vllm import _custom_ops as ops
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

# Seed for reproducibility
torch.manual_seed(42)


@dataclass
class MLAConfig:
    """MLA dimension configuration matching DeepSeek V3."""

    hidden_size: int = 7168
    num_heads: int = 128
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def head_size(self) -> int:
        """KV cache head size = kv_lora_rank + qk_rope_head_dim."""
        return self.kv_lora_rank + self.qk_rope_head_dim


@dataclass
class MLALayers:
    """Container for MLA projection layers."""

    fused_qkv_a_proj: MergedColumnParallelLinear
    q_a_layernorm: RMSNorm
    q_b_proj: ColumnParallelLinear
    kv_a_layernorm: RMSNorm
    kv_b_proj: ColumnParallelLinear
    o_proj: RowParallelLinear
    rotary_emb: torch.nn.Module


@dataclass
class DebugTensors:
    """Container for capturing intermediate tensors for debugging."""

    # After Q projection
    q_c: torch.Tensor | None = None
    q_after_norm: torch.Tensor | None = None
    q_full: torch.Tensor | None = None

    # After KV projection
    kv_lora: torch.Tensor | None = None
    kv_c: torch.Tensor | None = None
    k_pe_before_rope: torch.Tensor | None = None
    kv_c_normed: torch.Tensor | None = None

    # After RoPE
    q_pe_after_rope: torch.Tensor | None = None
    k_pe_after_rope: torch.Tensor | None = None

    # After attention
    attn_output: torch.Tensor | None = None

    # After output projection
    final_output: torch.Tensor | None = None


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._prob_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


class MockMLAAttentionLayer(AttentionLayerBase):
    """A mock MLA attention layer for populating static_forward_context."""

    def __init__(self, impl):
        self.impl = impl

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        raise NotImplementedError


def _convert_dtype_to_torch(dtype) -> torch.dtype:
    """Convert ModelDType to torch.dtype."""
    if isinstance(dtype, str):
        if dtype == "auto":
            return torch.bfloat16
        elif dtype in STR_DTYPE_TO_TORCH_DTYPE:
            return STR_DTYPE_TO_TORCH_DTYPE[dtype]
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def create_mla_layers(
    config: MLAConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> MLALayers:
    """Create MLA projection layers with random weights.

    Uses disable_tp=True to avoid requiring distributed initialization.
    """

    # fused_qkv_a_proj: projects hidden_size ->
    # [q_lora_rank, kv_lora_rank + qk_rope_head_dim]
    fused_qkv_a_proj = MergedColumnParallelLinear(
        input_size=config.hidden_size,
        output_sizes=[
            config.q_lora_rank,
            config.kv_lora_rank + config.qk_rope_head_dim,
        ],
        bias=False,
        disable_tp=True,
    ).to(device=device, dtype=dtype)

    # q_a_layernorm: normalizes the Q latent
    q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps).to(
        device=device, dtype=dtype
    )

    # q_b_proj: projects q_lora_rank -> num_heads * qk_head_dim
    q_b_proj = ColumnParallelLinear(
        input_size=config.q_lora_rank,
        output_size=config.num_heads * config.qk_head_dim,
        bias=False,
        disable_tp=True,
    ).to(device=device, dtype=dtype)

    # kv_a_layernorm: normalizes the KV latent
    kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps).to(
        device=device, dtype=dtype
    )

    # kv_b_proj: projects kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
    kv_b_proj = ColumnParallelLinear(
        input_size=config.kv_lora_rank,
        output_size=config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
        bias=False,
        disable_tp=True,
    ).to(device=device, dtype=dtype)

    # o_proj: projects num_heads * v_head_dim -> hidden_size
    o_proj = RowParallelLinear(
        input_size=config.num_heads * config.v_head_dim,
        output_size=config.hidden_size,
        bias=False,
        disable_tp=True,
    ).to(device=device, dtype=dtype)

    # RoPE embedding (GPT-J style, is_neox_style=False for MLA)
    rotary_emb = get_rope(
        config.qk_rope_head_dim,
        max_position=config.max_position_embeddings,
        rope_parameters={"rope_type": "default"},
        is_neox_style=False,
    ).to(device=device)

    # Initialize weights with random values for testing
    # The vLLM linear layers create empty weights, so we need to initialize them
    for layer in [
        fused_qkv_a_proj,
        q_b_proj,
        kv_b_proj,
        o_proj,
    ]:
        if hasattr(layer, "weight") and layer.weight is not None:
            torch.nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)

    return MLALayers(
        fused_qkv_a_proj=fused_qkv_a_proj,
        q_a_layernorm=q_a_layernorm,
        q_b_proj=q_b_proj,
        kv_a_layernorm=kv_a_layernorm,
        kv_b_proj=kv_b_proj,
        o_proj=o_proj,
        rotary_emb=rotary_emb,
    )


def mla_preprocess(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    config: MLAConfig,
    layers: MLALayers,
    debug: DebugTensors | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLA preprocessing - replicates MultiHeadLatentAttentionWrapper.forward_native().

    Args:
        positions: Position indices [num_tokens]
        hidden_states: Input hidden states [num_tokens, hidden_size]
        config: MLA configuration
        layers: MLA projection layers
        debug: Optional container to capture intermediate tensors

    Returns:
        q: Query tensor [num_tokens, num_heads, qk_head_dim]
        kv_c_normed: Normalized KV latent [num_tokens, kv_lora_rank]
        k_pe: K position embeddings [num_tokens, 1, qk_rope_head_dim]
    """
    # 1. Project hidden states to Q and KV latents
    qkv_lora = layers.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_lora = qkv_lora.split(
        [config.q_lora_rank, config.kv_lora_rank + config.qk_rope_head_dim],
        dim=-1,
    )

    if debug is not None:
        debug.q_c = q_c.clone()
        debug.kv_lora = kv_lora.clone()

    # 2. Q path: normalize and project
    q_c_normed = layers.q_a_layernorm(q_c)
    if debug is not None:
        debug.q_after_norm = q_c_normed.clone()

    q = layers.q_b_proj(q_c_normed)[0]
    q = q.view(-1, config.num_heads, config.qk_head_dim)

    if debug is not None:
        debug.q_full = q.clone()

    # 3. KV path: split and normalize
    kv_c, k_pe = kv_lora.split([config.kv_lora_rank, config.qk_rope_head_dim], dim=-1)
    kv_c_normed = layers.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(1)

    if debug is not None:
        debug.kv_c = kv_c.clone()
        debug.k_pe_before_rope = k_pe.clone()
        debug.kv_c_normed = kv_c_normed.clone()

    # 4. Apply RoPE to Q_pe (last qk_rope_head_dim of q) and K_pe
    q_pe = q[..., config.qk_nope_head_dim :]
    q_pe_roped, k_pe_roped = layers.rotary_emb(positions, q_pe, k_pe)

    # Update q with roped Q_pe
    q[..., config.qk_nope_head_dim :] = q_pe_roped

    if debug is not None:
        debug.q_pe_after_rope = q_pe_roped.clone()
        debug.k_pe_after_rope = k_pe_roped.clone()

    return q, kv_c_normed, k_pe_roped


def create_and_prepopulate_kv_cache(
    kv_c_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_blocks: int,
    common_attn_metadata: CommonAttentionMetadata,
    randomize_blocks: bool = True,
    kv_cache_dtype: str = "auto",
    scale: float = 1.0,
) -> torch.Tensor:
    """Create and prepopulate an MLA KV cache with context data.

    Args:
        kv_cache_dtype: Cache dtype. Use "fp8_e4m3" for FP8 MLA cache format.
        scale: Scaling factor for FP8 quantization.
    """
    batch_size = len(kv_c_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    use_fp8 = kv_cache_dtype in ("fp8", "fp8_e4m3")

    if use_fp8:
        # FP8 MLA cache format: (num_blocks, block_size, head_size) as uint8
        # Each element is quantized to FP8 (1 byte per value)
        kv_cache = torch.zeros(
            num_blocks, block_size, head_size, dtype=torch.uint8, device=device
        )
        scale_tensor = torch.tensor(scale, dtype=torch.float32, device=device)
    else:
        # Create MLA KV cache: (num_blocks, block_size, head_size)
        kv_cache = torch.zeros(
            num_blocks, block_size, head_size, dtype=dtype, device=device
        )
        kv_cache_flat = kv_cache.view(-1, head_size)

    # Populate the cache with the context tokens
    start_block_idx = 1  # block_id=0 is the null block
    for i in range(batch_size):
        kv_c_context, k_pe_context = kv_c_contexts[i], k_pe_contexts[i]
        context_len = kv_c_context.shape[0]
        if context_len == 0:
            start_block_idx += cdiv(int(seq_lens[i]), block_size)
            continue

        start = start_block_idx * block_size

        if use_fp8:
            slots = torch.arange(context_len, device=device, dtype=torch.long) + start
            ops.concat_and_cache_mla(
                kv_c_context,
                k_pe_context.squeeze(1),
                kv_cache,
                slots,
                kv_cache_dtype=kv_cache_dtype,
                scale=scale_tensor,
            )
        else:
            kv_context = torch.cat([kv_c_context, k_pe_context.squeeze(1)], dim=-1)
            end = start + kv_context.shape[0]
            kv_cache_flat[start:end, ...] = kv_context

        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null)
    if randomize_blocks:
        perm = torch.randperm(blocks_end - 1) + 1
    else:
        perm = torch.arange(1, blocks_end)

    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    inv_perm[1:] = torch.argsort(perm) + 1
    kv_cache[1:blocks_end, ...] = kv_cache[perm, ...]

    # Construct the right block table
    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        block_table[i, num_blocks_for_seq:] = 0
        start_block_idx += num_blocks_for_seq

    # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[
            i, block_indices
        ] * block_size + token_inter_block_offsets.to(device)

    return kv_cache


def run_mla_attention_backend(
    backend: AttentionBackendEnum,
    kv_cache_spec: FullAttentionSpec,
    layer_names: list[str],
    vllm_config,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    query: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    config: MLAConfig,
    mock_kv_b_proj: ColumnParallelLinear,
    kv_cache_dtype: str = "auto",
) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    builder_cls, impl_cls = try_get_attention_backend(backend)

    with set_current_vllm_config(vllm_config):
        num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        head_size = vllm_config.model_config.get_head_size()
        scale = 1.0 / (head_size**0.5)

        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,  # MLA uses single KV head in latent space
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            qk_head_dim=config.qk_head_dim,
            v_head_dim=config.v_head_dim,
            kv_b_proj=mock_kv_b_proj,
        )

        # Process weights to create W_UK_T and W_UV attributes
        act_dtype = _convert_dtype_to_torch(vllm_config.model_config.dtype)
        impl.process_weights_after_loading(act_dtype)

        # Populate static_forward_context
        for layer_name in layer_names:
            vllm_config.compilation_config.static_forward_context[layer_name] = (
                MockMLAAttentionLayer(impl)
            )

        # Build metadata
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        # Create mock layer and output buffer
        mock_layer = MockAttentionLayer(device)
        num_tokens = query.shape[0]
        output = torch.empty(
            num_tokens,
            num_heads * config.v_head_dim,
            dtype=query.dtype,
            device=query.device,
        )

        # Run forward pass
        output = impl.forward(
            mock_layer, query, kv_c, k_pe, kv_cache, attn_metadata, output=output
        )

        return output


def compute_reference_sdpa(
    q: torch.Tensor,
    kv_c_full: torch.Tensor,
    k_pe_full: torch.Tensor,
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    config: MLAConfig,
    context_len: int,
    is_decode: bool,
    scale: float,
) -> torch.Tensor:
    """Compute reference attention output using PyTorch SDPA.

    Args:
        q: Query tensor [q_len, num_heads, qk_head_dim]
        kv_c_full: Full KV context [s_len, kv_lora_rank]
        k_pe_full: Full K position embeddings [s_len, 1, qk_rope_head_dim]
        W_UK: K projection weight [kv_lora_rank, num_heads, qk_nope_head_dim]
        W_UV: V projection weight [kv_lora_rank, num_heads, v_head_dim]
        kv_b_proj_weight: Combined kv_b_proj weight
            [kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim]
        config: MLA configuration
        context_len: Number of context tokens
        is_decode: Whether to use decode path (MQA) or prefill path (MHA)
        scale: Attention scale factor

    Returns:
        SDPA output [q_len, num_heads * v_head_dim]
    """
    q_len = q.shape[0]
    s_len = kv_c_full.shape[0]
    num_heads = config.num_heads
    device = q.device

    # Split q into nope and rope components
    q_nope = q[..., : config.qk_nope_head_dim]
    q_pe = q[..., config.qk_nope_head_dim :]

    # Create attention mask
    attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=device)
    causal_mask = torch.tril(torch.ones(q_len, q_len, device=device))
    attn_mask[:, context_len:] = causal_mask

    if is_decode:
        # Decode path: MQA-style attention in latent space
        # Transform q_nope to latent space: q_nope @ W_UK
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)

        # Build MQA attention inputs
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)
        k_mqa = torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
        k_mqa = k_mqa.unsqueeze(1).expand(-1, num_heads, -1)
        v_mqa = kv_c_full.unsqueeze(1).expand(-1, num_heads, -1)

        # SDPA expects (N, H, L, D)
        q_sdpa = q_mqa.unsqueeze(0).transpose(1, 2)
        k_sdpa = k_mqa.unsqueeze(0).transpose(1, 2)
        v_sdpa = v_mqa.unsqueeze(0).transpose(1, 2)

        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, scale=scale
        )
        sdpa_out = sdpa_out.transpose(1, 2).squeeze(
            0
        )  # [q_len, num_heads, kv_lora_rank]

        # Project back to output space
        sdpa_out = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)
        return sdpa_out.flatten(start_dim=-2)
    else:
        # Prefill path: MHA-style attention with full sequence
        kv_nope_full = torch.einsum("sl,lnh->snh", kv_c_full, kv_b_proj_weight)
        k_nope_full, v_full = kv_nope_full.split(
            [config.qk_nope_head_dim, config.v_head_dim], dim=-1
        )

        q_mha = torch.cat([q_nope, q_pe], dim=-1)
        k_pe_expanded = k_pe_full.expand(-1, num_heads, -1)
        k_full = torch.cat([k_nope_full, k_pe_expanded], dim=-1)

        # SDPA expects (N, H, L, D)
        q_sdpa = q_mha.unsqueeze(0).transpose(1, 2)
        k_sdpa = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa = v_full.unsqueeze(0).transpose(1, 2)

        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, scale=scale
        )
        sdpa_out = sdpa_out.transpose(1, 2).squeeze(0)
        return sdpa_out.flatten(start_dim=-2)


# ============================================================================
# Test Cases
# ============================================================================

# Determine available backends
BACKENDS_TO_TEST = [AttentionBackendEnum.TRITON_MLA]
# FLASHINFER_MLA requires Blackwell GPU (SM 10.x)
# Only add if we have a Blackwell GPU
if torch.cuda.is_available():
    cc_major = torch.cuda.get_device_properties(0).major
    if cc_major >= 10:
        BACKENDS_TO_TEST.append(AttentionBackendEnum.FLASHINFER_MLA)

BATCH_SPECS = {
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small": BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
}


@pytest.fixture
def mla_config() -> MLAConfig:
    """Create MLA configuration matching DeepSeek V3."""
    return MLAConfig()


@pytest.fixture
def device() -> torch.device:
    """Get CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def dtype() -> torch.dtype:
    """Default dtype for testing."""
    return torch.bfloat16


class TestMLARoPEApplication:
    """Test that RoPE is correctly applied to Q and K."""

    def test_rope_applies_to_q_pe_only(
        self, dist_init, mla_config: MLAConfig, device: torch.device, dtype: torch.dtype
    ):
        """Verify RoPE is applied only to the rope portion of Q."""
        layers = create_mla_layers(mla_config, dtype, device)
        debug = DebugTensors()

        num_tokens = 10
        hidden_states = torch.randn(
            num_tokens, mla_config.hidden_size, dtype=dtype, device=device
        )
        positions = torch.arange(num_tokens, device=device)

        q, kv_c_normed, k_pe = mla_preprocess(
            positions, hidden_states, mla_config, layers, debug
        )

        # Check that q_pe changed after RoPE
        assert debug.q_pe_after_rope is not None
        assert debug.q_full is not None
        q_pe_before = debug.q_full[..., mla_config.qk_nope_head_dim :]
        assert not torch.allclose(q_pe_before, debug.q_pe_after_rope, atol=1e-5)

        # Check that the nope portion of q wasn't modified by RoPE
        q_nope_before = debug.q_full[..., : mla_config.qk_nope_head_dim]
        q_nope_after = q[..., : mla_config.qk_nope_head_dim]
        assert torch.allclose(q_nope_before, q_nope_after, atol=1e-6)

    def test_rope_applies_to_k_pe(
        self, dist_init, mla_config: MLAConfig, device: torch.device, dtype: torch.dtype
    ):
        """Verify RoPE is applied to K position embeddings."""
        layers = create_mla_layers(mla_config, dtype, device)
        debug = DebugTensors()

        num_tokens = 10
        hidden_states = torch.randn(
            num_tokens, mla_config.hidden_size, dtype=dtype, device=device
        )
        positions = torch.arange(num_tokens, device=device)

        q, kv_c_normed, k_pe = mla_preprocess(
            positions, hidden_states, mla_config, layers, debug
        )

        # Check that k_pe changed after RoPE
        assert debug.k_pe_before_rope is not None
        assert debug.k_pe_after_rope is not None
        assert not torch.allclose(
            debug.k_pe_before_rope, debug.k_pe_after_rope, atol=1e-5
        )

    def test_rope_position_dependent(
        self, dist_init, mla_config: MLAConfig, device: torch.device, dtype: torch.dtype
    ):
        """Verify RoPE produces different results for different positions."""
        layers = create_mla_layers(mla_config, dtype, device)

        num_tokens = 10
        hidden_states = torch.randn(
            num_tokens, mla_config.hidden_size, dtype=dtype, device=device
        )

        # Run with different position offsets
        positions1 = torch.arange(num_tokens, device=device)
        positions2 = torch.arange(100, 100 + num_tokens, device=device)

        q1, _, k_pe1 = mla_preprocess(positions1, hidden_states, mla_config, layers)
        q2, _, k_pe2 = mla_preprocess(positions2, hidden_states, mla_config, layers)

        # The rope portions should be different
        assert not torch.allclose(
            q1[..., mla_config.qk_nope_head_dim :],
            q2[..., mla_config.qk_nope_head_dim :],
            atol=1e-5,
        )
        assert not torch.allclose(k_pe1, k_pe2, atol=1e-5)


class TestMLAKVCache:
    """Test KV cache population and retrieval."""

    def test_concat_and_cache_mla(
        self, mla_config: MLAConfig, device: torch.device, dtype: torch.dtype
    ):
        """Test the concat_and_cache_mla operation."""
        num_blocks = 10
        block_size = 16
        num_tokens = 5

        # Create KV cache
        kv_cache = torch.zeros(
            num_blocks, block_size, mla_config.head_size, dtype=dtype, device=device
        )

        # Create input tensors
        kv_c = torch.randn(
            num_tokens, mla_config.kv_lora_rank, dtype=dtype, device=device
        )
        k_pe = torch.randn(
            num_tokens, mla_config.qk_rope_head_dim, dtype=dtype, device=device
        )

        # Create slot mapping (put tokens in block 1)
        slot_mapping = torch.arange(
            block_size, block_size + num_tokens, dtype=torch.long, device=device
        )

        # Cache the KV
        ops.concat_and_cache_mla(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype="auto",
            scale=torch.tensor(1.0, device=device),
        )

        # Verify the cache was populated correctly
        expected = torch.cat([kv_c, k_pe], dim=-1)
        cached = kv_cache.view(-1, mla_config.head_size)[
            block_size : block_size + num_tokens
        ]
        assert torch.allclose(cached, expected, atol=1e-5)


class TestMLAFullForward:
    """Test complete MLA forward pass against SDPA reference."""

    @pytest.mark.parametrize("batch_spec_name", ["small_decode", "small_prefill"])
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_mla_full_forward(
        self,
        dist_init,
        batch_spec_name: str,
        backend: AttentionBackendEnum,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Test complete forward pass against SDPA reference."""
        batch_spec = BATCH_SPECS[batch_spec_name]

        # Use smaller config for faster testing
        config = MLAConfig(
            hidden_size=2048,
            num_heads=16,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=1536,
        )

        # Use block_size 64 for MLA backends (required by FLASHINFER_MLA decode)
        block_size = 64
        required_blocks = sum(
            (seq_len + block_size - 1) // block_size for seq_len in batch_spec.seq_lens
        )
        num_gpu_blocks = required_blocks + 1 + 100

        # Create vllm config - use local cached model
        vllm_config = create_vllm_config(
            model_name="/home/yming/.cache/huggingface/hub/models--nvidia--DeepSeek-R1-0528-FP4-v2/snapshots/25a138f28f49022958b9f2d205f9b7de0cdb6e18/",
            tensor_parallel_size=1,
            max_model_len=max(batch_spec.seq_lens),
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            hf_config_override={"num_attention_heads": config.num_heads},
        )

        # Create shared weight matrices
        W_UK = torch.randn(
            config.kv_lora_rank,
            config.num_heads,
            config.qk_nope_head_dim,
            dtype=dtype,
            device=device,
        )
        W_UV = torch.randn(
            config.kv_lora_rank,
            config.num_heads,
            config.v_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

        # Generate test data
        all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
        all_sdpa_outputs = []
        kv_c_contexts, k_pe_contexts = [], []

        scale = 1.0 / (config.head_size**0.5)

        for i in range(batch_spec.batch_size):
            s_len = batch_spec.seq_lens[i]
            q_len = batch_spec.query_lens[i]
            context_len = s_len - q_len

            # Generate tensors
            q = torch.randn(
                q_len, config.num_heads, config.qk_head_dim, dtype=dtype, device=device
            )
            kv_c_full = torch.randn(
                s_len, config.kv_lora_rank, dtype=dtype, device=device
            )
            k_pe_full = torch.randn(
                s_len, 1, config.qk_rope_head_dim, dtype=dtype, device=device
            )

            # Determine if decode or prefill
            is_decode = q_len <= 1

            # Compute reference
            sdpa_out = compute_reference_sdpa(
                q,
                kv_c_full,
                k_pe_full,
                W_UK,
                W_UV,
                kv_b_proj_weight,
                config,
                context_len,
                is_decode,
                scale,
            )
            all_sdpa_outputs.append(sdpa_out)

            # Inputs for vLLM MLA backends (only new tokens)
            all_q_vllm.append(q)
            all_kv_c_vllm.append(kv_c_full[context_len:])
            all_k_pe_vllm.append(k_pe_full[context_len:])

            # Context for KV cache
            kv_c_contexts.append(kv_c_full[:context_len])
            k_pe_contexts.append(k_pe_full[:context_len])

        # Concatenate all sequences
        query_vllm = torch.cat(all_q_vllm, dim=0)
        kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
        k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
        expected_output = torch.cat(all_sdpa_outputs, dim=0)

        # Create mock kv_b_proj
        mock_kv_b_proj = ColumnParallelLinear(
            input_size=config.kv_lora_rank,
            output_size=config.num_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            disable_tp=True,
        ).to(device=device, dtype=dtype)
        kv_b_proj_weight_flat = kv_b_proj_weight.view(
            config.kv_lora_rank,
            config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
        )
        mock_kv_b_proj.weight = torch.nn.Parameter(
            kv_b_proj_weight_flat.T, requires_grad=False
        )

        # Create metadata and KV cache
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, block_size, device
        )

        # Pad block table
        required_divisor = int(128 / block_size)
        current_block_num = common_attn_metadata.block_table_tensor.shape[1]
        if current_block_num % required_divisor != 0:
            padded_block_num = (
                (current_block_num + required_divisor - 1) // required_divisor
            ) * required_divisor
            padding_cols = padded_block_num - current_block_num
            padding = torch.zeros(
                (common_attn_metadata.block_table_tensor.shape[0], padding_cols),
                dtype=torch.int32,
                device=device,
            )
            common_attn_metadata.block_table_tensor = torch.cat(
                [common_attn_metadata.block_table_tensor, padding], dim=1
            )

        kv_cache = create_and_prepopulate_kv_cache(
            kv_c_contexts=kv_c_contexts,
            k_pe_contexts=k_pe_contexts,
            block_size=block_size,
            head_size=config.head_size,
            dtype=dtype,
            device=device,
            num_blocks=num_gpu_blocks,
            common_attn_metadata=common_attn_metadata,
            randomize_blocks=True,
        )

        kv_cache_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=config.head_size,
            dtype=dtype,
        )

        # Run backend
        backend_output = run_mla_attention_backend(
            backend,
            kv_cache_spec,
            ["test_layer"],
            vllm_config,
            device,
            common_attn_metadata,
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            config,
            mock_kv_b_proj,
        )

        # Compare
        assert backend_output.shape == expected_output.shape
        assert torch.isfinite(backend_output).all()

        # Use larger tolerance as there may be slight differences
        # in how the reference SDPA and backend handle the computation.
        # The existing test_mla_backends.py uses similar tolerances.
        rtol = 1e-2
        # Large tolerance to catch major bugs while allowing for algorithm diffs
        atol = 10.0
        max_diff = torch.max(torch.abs(backend_output - expected_output)).item()
        assert torch.allclose(backend_output, expected_output, rtol=rtol, atol=atol), (
            f"Backend {backend.name} output differs from SDPA. Max diff: {max_diff:.6f}"
        )


class TestMLADebugHooks:
    """Test debugging hooks for intermediate tensor inspection."""

    def test_debug_tensor_capture(
        self, dist_init, mla_config: MLAConfig, device: torch.device, dtype: torch.dtype
    ):
        """Verify all debug tensors are captured correctly."""
        layers = create_mla_layers(mla_config, dtype, device)
        debug = DebugTensors()

        num_tokens = 10
        hidden_states = torch.randn(
            num_tokens, mla_config.hidden_size, dtype=dtype, device=device
        )
        positions = torch.arange(num_tokens, device=device)

        q, kv_c_normed, k_pe = mla_preprocess(
            positions, hidden_states, mla_config, layers, debug
        )

        # Verify all debug tensors were captured
        assert debug.q_c is not None
        assert debug.q_after_norm is not None
        assert debug.q_full is not None
        assert debug.kv_lora is not None
        assert debug.kv_c is not None
        assert debug.k_pe_before_rope is not None
        assert debug.kv_c_normed is not None
        assert debug.q_pe_after_rope is not None
        assert debug.k_pe_after_rope is not None

        # Verify shapes
        assert debug.q_c.shape == (num_tokens, mla_config.q_lora_rank)
        assert debug.q_full.shape == (
            num_tokens,
            mla_config.num_heads,
            mla_config.qk_head_dim,
        )
        assert debug.kv_c.shape == (num_tokens, mla_config.kv_lora_rank)
        assert debug.k_pe_before_rope.shape == (
            num_tokens,
            1,
            mla_config.qk_rope_head_dim,
        )


class TestMLABackendComparison:
    """Test FLASHINFER_MLA against TRITON_MLA reference with tight tolerance."""

    @pytest.mark.parametrize("batch_spec_name", ["small_decode", "small_prefill"])
    def test_flashinfer_vs_triton(
        self,
        dist_init,
        batch_spec_name: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Compare FLASHINFER_MLA output against TRITON_MLA with small tolerance."""
        # Skip if FLASHINFER_MLA is not available (requires Blackwell GPU)
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        cc_major = torch.cuda.get_device_properties(0).major
        if cc_major < 10:
            pytest.skip("FLASHINFER_MLA requires Blackwell GPU (SM 10.x)")

        batch_spec = BATCH_SPECS[batch_spec_name]

        # Use smaller config for faster testing
        config = MLAConfig(
            hidden_size=2048,
            num_heads=16,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=1536,
        )

        # Use block_size 64 (required by FLASHINFER_MLA decode)
        block_size = 64
        required_blocks = sum(
            (seq_len + block_size - 1) // block_size for seq_len in batch_spec.seq_lens
        )
        num_gpu_blocks = required_blocks + 1 + 100

        # Create vllm config
        vllm_config = create_vllm_config(
            model_name="/home/yming/.cache/huggingface/hub/models--nvidia--DeepSeek-R1-0528-FP4-v2/snapshots/25a138f28f49022958b9f2d205f9b7de0cdb6e18/",
            tensor_parallel_size=1,
            max_model_len=max(batch_spec.seq_lens),
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            hf_config_override={"num_attention_heads": config.num_heads},
        )

        # Create shared weight matrices
        W_UK = torch.randn(
            config.kv_lora_rank,
            config.num_heads,
            config.qk_nope_head_dim,
            dtype=dtype,
            device=device,
        )
        W_UV = torch.randn(
            config.kv_lora_rank,
            config.num_heads,
            config.v_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

        # Generate test data
        all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
        kv_c_contexts, k_pe_contexts = [], []

        for i in range(batch_spec.batch_size):
            s_len = batch_spec.seq_lens[i]
            q_len = batch_spec.query_lens[i]
            context_len = s_len - q_len

            # Generate tensors
            q = torch.randn(
                q_len,
                config.num_heads,
                config.qk_head_dim,
                dtype=dtype,
                device=device,
            )
            kv_c_full = torch.randn(
                s_len, config.kv_lora_rank, dtype=dtype, device=device
            )
            k_pe_full = torch.randn(
                s_len, 1, config.qk_rope_head_dim, dtype=dtype, device=device
            )

            # Inputs for vLLM MLA backends (only new tokens)
            all_q_vllm.append(q)
            all_kv_c_vllm.append(kv_c_full[context_len:])
            all_k_pe_vllm.append(k_pe_full[context_len:])

            # Context for KV cache
            kv_c_contexts.append(kv_c_full[:context_len])
            k_pe_contexts.append(k_pe_full[:context_len])

        # Concatenate all sequences
        query_vllm = torch.cat(all_q_vllm, dim=0)
        kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
        k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)

        # Create mock kv_b_proj
        mock_kv_b_proj = ColumnParallelLinear(
            input_size=config.kv_lora_rank,
            output_size=config.num_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            disable_tp=True,
        ).to(device=device, dtype=dtype)
        kv_b_proj_weight_flat = kv_b_proj_weight.view(
            config.kv_lora_rank,
            config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
        )
        mock_kv_b_proj.weight = torch.nn.Parameter(
            kv_b_proj_weight_flat.T, requires_grad=False
        )

        # Create metadata and KV cache
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, block_size, device
        )

        # Pad block table if needed
        required_divisor = int(128 / block_size)
        current_block_num = common_attn_metadata.block_table_tensor.shape[1]
        if current_block_num % required_divisor != 0:
            padded_block_num = (
                (current_block_num + required_divisor - 1) // required_divisor
            ) * required_divisor
            padding_cols = padded_block_num - current_block_num
            padding = torch.zeros(
                (common_attn_metadata.block_table_tensor.shape[0], padding_cols),
                dtype=torch.int32,
                device=device,
            )
            common_attn_metadata.block_table_tensor = torch.cat(
                [common_attn_metadata.block_table_tensor, padding], dim=1
            )

        kv_cache = create_and_prepopulate_kv_cache(
            kv_c_contexts=kv_c_contexts,
            k_pe_contexts=k_pe_contexts,
            block_size=block_size,
            head_size=config.head_size,
            dtype=dtype,
            device=device,
            num_blocks=num_gpu_blocks,
            common_attn_metadata=common_attn_metadata,
            randomize_blocks=True,
        )

        kv_cache_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=config.head_size,
            dtype=dtype,
        )

        # Run TRITON_MLA as reference
        triton_output = run_mla_attention_backend(
            AttentionBackendEnum.TRITON_MLA,
            kv_cache_spec,
            ["test_layer"],
            vllm_config,
            device,
            common_attn_metadata,
            query_vllm.clone(),
            kv_c_vllm.clone(),
            k_pe_vllm.clone(),
            kv_cache.clone(),
            config,
            mock_kv_b_proj,
        )

        # Run FLASHINFER_MLA
        flashinfer_output = run_mla_attention_backend(
            AttentionBackendEnum.FLASHINFER_MLA,
            kv_cache_spec,
            ["test_layer"],
            vllm_config,
            device,
            common_attn_metadata,
            query_vllm.clone(),
            kv_c_vllm.clone(),
            k_pe_vllm.clone(),
            kv_cache.clone(),
            config,
            mock_kv_b_proj,
        )

        # Compare with tight tolerance
        assert flashinfer_output.shape == triton_output.shape
        assert torch.isfinite(flashinfer_output).all()
        assert torch.isfinite(triton_output).all()

        # Use smaller tolerance since both are MLA backends
        rtol = 1e-2
        atol = 5e-1
        max_diff = torch.max(torch.abs(flashinfer_output - triton_output)).item()
        assert torch.allclose(flashinfer_output, triton_output, rtol=rtol, atol=atol), (
            f"FLASHINFER_MLA differs from TRITON_MLA. Max diff: {max_diff:.6f}"
        )


class TestMLAFP8CacheInput:
    """Test FP8 input to concat_and_cache_mla (pre-quantized FP8 tensors)."""

    def test_fp8_input_concat_and_cache_mla(
        self, mla_config: MLAConfig, device: torch.device, dtype: torch.dtype
    ):
        """Test that concat_and_cache_mla accepts FP8 input tensors directly."""
        num_blocks = 10
        block_size = 16
        num_tokens = 5

        # Create FP8 KV cache
        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            mla_config.head_size,
            dtype=torch.uint8,
            device=device,
        )

        # Create BF16 input tensors first
        kv_c_bf16 = torch.randn(
            num_tokens, mla_config.kv_lora_rank, dtype=dtype, device=device
        )
        k_pe_bf16 = torch.randn(
            num_tokens, mla_config.qk_rope_head_dim, dtype=dtype, device=device
        )

        # Quantize to FP8 (pre-quantized input)
        kv_c_fp8 = kv_c_bf16.to(torch.float8_e4m3fn)
        k_pe_fp8 = k_pe_bf16.to(torch.float8_e4m3fn)

        # Create slot mapping (put tokens in block 1)
        slot_mapping = torch.arange(
            block_size, block_size + num_tokens, dtype=torch.long, device=device
        )

        # Cache with FP8 input - this should work now
        # With kAuto cache type, no quantization is performed (data is already FP8)
        ops.concat_and_cache_mla(
            kv_c_fp8,
            k_pe_fp8,
            kv_cache,
            slot_mapping,
            kv_cache_dtype="fp8_e4m3",
            scale=torch.tensor(1.0, device=device),
        )

        # Verify the cache was populated - read back as FP8 and compare
        cached = kv_cache.view(-1, mla_config.head_size)[
            block_size : block_size + num_tokens
        ]

        # The cached data should match the concatenated FP8 input
        expected_fp8 = torch.cat([kv_c_fp8, k_pe_fp8], dim=-1)
        expected_uint8 = expected_fp8.view(torch.uint8)

        assert torch.equal(cached, expected_uint8), (
            f"Cached data doesn't match FP8 input. "
            f"Max diff: {(cached.float() - expected_uint8.float()).abs().max()}"
        )

    def test_fp8_input_vs_bf16_input_quantized(
        self, mla_config: MLAConfig, device: torch.device, dtype: torch.dtype
    ):
        """Compare FP8 input path vs BF16 input with quantization."""
        num_blocks = 10
        block_size = 16
        num_tokens = 5

        # Create two KV caches
        kv_cache_from_fp8 = torch.zeros(
            num_blocks,
            block_size,
            mla_config.head_size,
            dtype=torch.uint8,
            device=device,
        )
        kv_cache_from_bf16 = torch.zeros(
            num_blocks,
            block_size,
            mla_config.head_size,
            dtype=torch.uint8,
            device=device,
        )

        # Create BF16 input tensors
        kv_c_bf16 = torch.randn(
            num_tokens, mla_config.kv_lora_rank, dtype=dtype, device=device
        )
        k_pe_bf16 = torch.randn(
            num_tokens, mla_config.qk_rope_head_dim, dtype=dtype, device=device
        )

        # Create slot mapping
        slot_mapping = torch.arange(
            block_size, block_size + num_tokens, dtype=torch.long, device=device
        )

        # Path 1: Quantize BF16 to FP8 first, then cache as FP8 input
        kv_c_fp8 = kv_c_bf16.to(torch.float8_e4m3fn)
        k_pe_fp8 = k_pe_bf16.to(torch.float8_e4m3fn)

        ops.concat_and_cache_mla(
            kv_c_fp8,
            k_pe_fp8,
            kv_cache_from_fp8,
            slot_mapping,
            kv_cache_dtype="fp8_e4m3",
            scale=torch.tensor(1.0, device=device),
        )

        # Path 2: Cache BF16 input with internal quantization
        ops.concat_and_cache_mla(
            kv_c_bf16,
            k_pe_bf16,
            kv_cache_from_bf16,
            slot_mapping,
            kv_cache_dtype="fp8_e4m3",
            scale=torch.tensor(1.0, device=device),
        )

        # Both paths should produce the same result (or very close)
        cached_from_fp8 = kv_cache_from_fp8.view(-1, mla_config.head_size)[
            block_size : block_size + num_tokens
        ]
        cached_from_bf16 = kv_cache_from_bf16.view(-1, mla_config.head_size)[
            block_size : block_size + num_tokens
        ]

        # The results should be identical since both use same quantization
        assert torch.equal(cached_from_fp8, cached_from_bf16), (
            f"FP8 input path differs from BF16 input path. "
            f"Diff count: {(cached_from_fp8 != cached_from_bf16).sum()}"
        )


class TestMLAFP8Attention:
    """Test FP8 KV cache attention comparing FLASHINFER_MLA vs CUTLASS_MLA."""

    @pytest.mark.parametrize("batch_spec_name", ["small_decode", "small_prefill"])
    def test_fp8_flashinfer_vs_cutlass(
        self,
        dist_init,
        batch_spec_name: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Compare FLASHINFER_MLA FP8 against CUTLASS_MLA FP8."""
        # Both backends require Blackwell GPU (SM 10.x)
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        cc_major = torch.cuda.get_device_properties(0).major
        if cc_major < 10:
            pytest.skip("FP8 MLA backends require Blackwell GPU (SM 10.x)")

        batch_spec = BATCH_SPECS[batch_spec_name]

        # Use smaller config for faster testing
        config = MLAConfig(
            hidden_size=2048,
            num_heads=16,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=1536,
        )

        # Use block_size 64 (required by FLASHINFER_MLA decode)
        block_size = 64
        required_blocks = sum(
            (seq_len + block_size - 1) // block_size for seq_len in batch_spec.seq_lens
        )
        num_gpu_blocks = required_blocks + 1 + 100

        # Create vllm config
        vllm_config = create_vllm_config(
            model_name="/home/yming/.cache/huggingface/hub/"
            "models--nvidia--DeepSeek-R1-0528-FP4-v2/snapshots/"
            "25a138f28f49022958b9f2d205f9b7de0cdb6e18/",
            tensor_parallel_size=1,
            max_model_len=max(batch_spec.seq_lens),
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
            hf_config_override={"num_attention_heads": config.num_heads},
        )

        # Create shared weight matrices
        W_UK = torch.randn(
            config.kv_lora_rank,
            config.num_heads,
            config.qk_nope_head_dim,
            dtype=dtype,
            device=device,
        )
        W_UV = torch.randn(
            config.kv_lora_rank,
            config.num_heads,
            config.v_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

        # Generate test data
        all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
        kv_c_contexts, k_pe_contexts = [], []

        for i in range(batch_spec.batch_size):
            s_len = batch_spec.seq_lens[i]
            q_len = batch_spec.query_lens[i]
            context_len = s_len - q_len

            # Generate tensors
            q = torch.randn(
                q_len,
                config.num_heads,
                config.qk_head_dim,
                dtype=dtype,
                device=device,
            )
            kv_c_full = torch.randn(
                s_len, config.kv_lora_rank, dtype=dtype, device=device
            )
            k_pe_full = torch.randn(
                s_len, 1, config.qk_rope_head_dim, dtype=dtype, device=device
            )

            # Inputs for vLLM MLA backends (only new tokens)
            all_q_vllm.append(q)
            all_kv_c_vllm.append(kv_c_full[context_len:])
            all_k_pe_vllm.append(k_pe_full[context_len:])

            # Context for KV cache
            kv_c_contexts.append(kv_c_full[:context_len])
            k_pe_contexts.append(k_pe_full[:context_len])

        # Concatenate all sequences
        query_vllm = torch.cat(all_q_vllm, dim=0)
        kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
        k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)

        # Create mock kv_b_proj
        mock_kv_b_proj = ColumnParallelLinear(
            input_size=config.kv_lora_rank,
            output_size=config.num_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            disable_tp=True,
        ).to(device=device, dtype=dtype)
        kv_b_proj_weight_flat = kv_b_proj_weight.view(
            config.kv_lora_rank,
            config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
        )
        mock_kv_b_proj.weight = torch.nn.Parameter(
            kv_b_proj_weight_flat.T, requires_grad=False
        )

        # Create metadata and FP8 KV cache (shared between both backends)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, block_size, device
        )

        # Pad block table if needed
        required_divisor = int(128 / block_size)
        current_block_num = common_attn_metadata.block_table_tensor.shape[1]
        if current_block_num % required_divisor != 0:
            padded_block_num = (
                (current_block_num + required_divisor - 1) // required_divisor
            ) * required_divisor
            padding_cols = padded_block_num - current_block_num
            padding = torch.zeros(
                (common_attn_metadata.block_table_tensor.shape[0], padding_cols),
                dtype=torch.int32,
                device=device,
            )
            common_attn_metadata.block_table_tensor = torch.cat(
                [common_attn_metadata.block_table_tensor, padding], dim=1
            )

        # Create FP8 KV cache
        kv_cache_fp8 = create_and_prepopulate_kv_cache(
            kv_c_contexts=kv_c_contexts,
            k_pe_contexts=k_pe_contexts,
            block_size=block_size,
            head_size=config.head_size,
            dtype=dtype,
            device=device,
            num_blocks=num_gpu_blocks,
            common_attn_metadata=common_attn_metadata,
            randomize_blocks=False,
            kv_cache_dtype="fp8_e4m3",
            scale=1.0,
        )

        kv_cache_spec_fp8 = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=config.head_size,
            dtype=torch.uint8,  # FP8 uses uint8 storage
        )

        # Run FLASHINFER_MLA with FP8 cache
        flashinfer_output = run_mla_attention_backend(
            AttentionBackendEnum.FLASHINFER_MLA,
            kv_cache_spec_fp8,
            ["test_layer_flashinfer"],
            vllm_config,
            device,
            common_attn_metadata,
            query_vllm.clone(),
            kv_c_vllm.clone(),
            k_pe_vllm.clone(),
            kv_cache_fp8.clone(),
            config,
            mock_kv_b_proj,
            kv_cache_dtype="fp8_e4m3",
        )

        # Run CUTLASS_MLA with FP8 cache
        cutlass_output = run_mla_attention_backend(
            AttentionBackendEnum.CUTLASS_MLA,
            kv_cache_spec_fp8,
            ["test_layer_cutlass"],
            vllm_config,
            device,
            common_attn_metadata,
            query_vllm.clone(),
            kv_c_vllm.clone(),
            k_pe_vllm.clone(),
            kv_cache_fp8.clone(),
            config,
            mock_kv_b_proj,
            kv_cache_dtype="fp8_e4m3",
        )

        # Compare outputs - both use same FP8 cache, should be very close
        assert flashinfer_output.shape == cutlass_output.shape
        assert torch.isfinite(flashinfer_output).all()
        assert torch.isfinite(cutlass_output).all()

        # Use reasonable tolerance - both backends read same FP8 data but may have
        # minor implementation differences in how they handle the computation
        rtol = 1e-2
        atol = 2.0  # Allow for some implementation differences
        max_diff = torch.max(torch.abs(flashinfer_output - cutlass_output)).item()
        assert torch.allclose(
            flashinfer_output, cutlass_output, rtol=rtol, atol=atol
        ), f"FLASHINFER_MLA FP8 differs from CUTLASS_MLA FP8. Max diff: {max_diff:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
