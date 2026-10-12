# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA counterpart of test_rocm_aiter_qk_norm_rope_kvcache_fusion: the
QK-norm + RoPE + unified_kv_cache_update sequence is fused into one
fused_qk_norm_rope_kvcache launch on the FlashAttention backend."""

import os

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm.compilation.passes.fusion.matcher_utils import ROTARY_OP
from vllm.compilation.passes.fusion.qk_norm_rope_kvcache_fusion import (
    QkNormRopeKvCacheFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.utility.scatter_split_replace import (
    ScatterSplitReplacementPass,
)
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

pytestmark = pytest.mark.skip_global_cleanup

if not current_platform.is_cuda():
    pytest.skip("fused_qk_norm_rope_kvcache is CUDA-only", allow_module_level=True)

INDEX_SELECT_OP = torch.ops.aten.index.Tensor


@pytest.fixture(scope="module", autouse=True)
def module_global_cleanup():
    from vllm.distributed import cleanup_dist_env_and_memory

    yield
    cleanup_dist_env_and_memory()


class QKNormRoPEKVCacheTestModel(torch.nn.Module):
    """q, k, v = split(qkv); q/k = rms_norm(...); q, k = rope(...);
    dummy = unified_kv_cache_update(k, v, layer_name)"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_backend: AttentionBackendEnum,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        rotary_dim: int,
        is_neox: bool,
        rms_norm_eps: float,
        dtype: torch.dtype,
        device: torch.device,
        prefix: str = "model.layers.0.self_attn.attn",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = vllm_config.cache_config.block_size
        self.q_size = num_heads * head_size
        self.kv_size = num_kv_heads * head_size
        self.dtype = dtype
        self.device = device
        self.layer_name = prefix
        self.q_norm = RMSNorm(head_size, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_size, eps=rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            dtype=dtype,
        )
        self.enable_rope_custom_op = self.rotary_emb.enabled()
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / head_size**0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            attn_backend=attn_backend.get_class(),
        )
        self.attn_backend: type[AttentionBackend] = self.attn.get_attn_backend()
        assert not self.attn_backend.forward_includes_kv_cache_update
        assert self.attn.impl.fused_qk_norm_rope_kvcache_supported()
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)
        self.kv_cache_spec = self.attn.attn_backend.customize_spec(
            AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
            )
        )
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=[self.attn.layer_name],
            vllm_config=vllm_config,
            device=device,
        )

    def build_attn_metadata(
        self, batch_size: int, layout: KVCacheLayout
    ) -> CommonAttentionMetadata:
        batch_spec = BatchSpec(seq_lens=[1] * batch_size, query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )
        num_blocks = batch_size * (
            (max(batch_spec.seq_lens) + self.block_size - 1) // self.block_size
        )
        raw_tensor = torch.zeros(
            num_blocks * self.kv_cache_spec.page_size_bytes,
            dtype=torch.int8,
            device=self.device,
        )
        self.attn.kv_cache = dense_kv_cache_views(
            raw_tensor, self.kv_cache_spec, num_blocks, num_layers=1, layout=layout
        )[0]
        return self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.clone()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_size)).view(
            -1, self.q_size
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_size)).view(
            -1, self.kv_size
        )
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(-1, self.num_heads, self.head_size)
        k = k.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size)
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            k, v, _encode_layer_name(self.layer_name)
        )
        return q, k, v, kv_cache_dummy_dep

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        rope_op = ROTARY_OP if self.enable_rope_custom_op else INDEX_SELECT_OP
        return [rope_op, torch.ops.vllm.unified_kv_cache_update.default]

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default]


_FUSION_CONFIGS = [
    pytest.param(32, 8, 128, 128, True, id="qwen3-neox"),
    pytest.param(32, 8, 128, 128, False, id="gptj"),
    pytest.param(16, 2, 64, 64, True, id="hd64"),
    pytest.param(8, 2, 256, 256, True, id="hd256"),
    pytest.param(32, 8, 128, 64, True, id="partial-rotary"),
]


@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, rotary_dim, is_neox", _FUSION_CONFIGS
)
@pytest.mark.parametrize("num_tokens", [5, 2048])
@pytest.mark.parametrize(
    "kv_layout",
    [
        pytest.param(KVCacheLayout.LBHNC, id="head_major"),
        pytest.param(KVCacheLayout.LBNHC, id="token_major"),
    ],
)
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("custom_op", ["+rotary_embedding", "+rms_norm"])
def test_qk_norm_rope_kvcache_fusion(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    is_neox: bool,
    num_tokens: int,
    kv_layout: KVCacheLayout,
    block_size: int,
    dtype: torch.dtype,
    custom_op: str,
):
    device = os.environ.get("VLLM_TEST_CUDA_DEVICE", "cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    rms_norm_eps = 1e-6

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(block_size=block_size, cache_dtype="auto"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=[custom_op],
            pass_config=PassConfig(
                fuse_qk_norm_rope_kvcache=True,
                eliminate_noops=True,
            ),
        ),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        model = QKNormRoPEKVCacheTestModel(
            vllm_config=vllm_config,
            attn_backend=AttentionBackendEnum.FLASH_ATTN,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            rotary_dim=rotary_dim,
            is_neox=is_neox,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
            device=torch.get_default_device(),
        )
        fusion_pass = QkNormRopeKvCacheFusionPass(vllm_config)
        backend = TestBackend(
            NoOpEliminationPass(vllm_config),
            SplitCoalescingPass(vllm_config),
            ScatterSplitReplacementPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        )
        qkv = torch.randn(num_tokens, (num_heads + 2 * num_kv_heads) * head_size)
        pos = torch.arange(num_tokens, dtype=torch.long)

        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(num_tokens, kv_layout)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_unfused, k_unfused, v_unfused, _ = model(qkv.clone(), pos.clone())
            kv_cache_unfused = forward_context.no_compile_layers[
                model.layer_name
            ].kv_cache.clone()

        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        with set_forward_context(None, vllm_config):
            model_fused = torch.compile(model, backend=backend)
            forward_context = get_forward_context()
            attn_metadata = model_fused.build_attn_metadata(num_tokens, kv_layout)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_fused, k_fused, v_fused, _ = model_fused(qkv, pos)
            kv_cache_fused = forward_context.no_compile_layers[
                model.layer_name
            ].kv_cache

        assert fusion_pass.matched_count == 1
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())

        # The fused kernel keeps the rotated values in fp32 between the norm
        # and the rotation, so it is not bit-identical to the two-step
        # reference; the tolerance matches test_fused_qk_norm_rope.
        atol = rtol = 1e-2
        torch.testing.assert_close(q_unfused, q_fused, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_unfused, k_fused, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_unfused, v_fused, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            kv_cache_unfused.float(), kv_cache_fused.float(), atol=atol, rtol=rtol
        )
