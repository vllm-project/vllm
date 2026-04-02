# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.compilation.passes.fusion.kv_cache_mla_rope_fusion import (
    KVCacheMLARoPEFusionPass,
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
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    MLAAttentionSpec,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

INDEX_SELECT_OP = torch.ops.aten.index.Tensor
VLLM_UNIFIED_KV_CACHE_UPDATE_OP = torch.ops.vllm.unified_kv_cache_update
FP8_DTYPE = current_platform.fp8_dtype()

ROTARY_OP = torch.ops._C.rotary_embedding
FLASHINFER_ROTARY_OP = torch.ops.vllm.flashinfer_rotary_embedding


class QKRoPEKVCacheMLATestModel(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_backend: AttentionBackendEnum,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        is_neox: bool,
        dtype: torch.dtype,
        device: torch.device,
        use_deepseek_scaling_rope: bool,
        use_flashinfer: bool,
        prefix: str = "model.layers.0.self_attn.attn",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj
        self.block_size = vllm_config.cache_config.block_size
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.is_neox = is_neox
        self.dtype = dtype
        self.device = device
        self.use_deepseek_scaling_rope = use_deepseek_scaling_rope
        self.layer_name = prefix
        self.num_kv_heads = 1

        if use_deepseek_scaling_rope:
            self.rotary_emb = DeepseekScalingRotaryEmbedding(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position_embeddings=4096,
                base=10000,
                is_neox_style=is_neox,
                scaling_factor=1.0,
                dtype=self.dtype,
            )
        else:
            if use_flashinfer:
                self.rotary_op = FLASHINFER_ROTARY_OP
            else:
                self.rotary_op = ROTARY_OP

            self.rotary_emb = RotaryEmbedding(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position_embeddings=4096,
                base=10000,
                is_neox_style=is_neox,
                dtype=self.dtype,
            )

        # Register layer metadata for the fusion pass via Attention.
        self.attn = MLAAttention(
            num_heads=self.num_heads,
            scale=self.scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )
        self.attn_backend: type[AttentionBackend] = attn_backend.get_class()
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)

        self.kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        self.kv_cache_dtype = self.dtype

        # Initialize attn MetadataBuilder
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=MLAAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_dtype,
            ),
            layer_names=[self.attn.layer_name],
            vllm_config=vllm_config,
            device=device,
        )

    def build_attn_metadata(self, batch_size: int) -> CommonAttentionMetadata:
        """Initialize attention metadata."""
        # Create common attn metadata
        batch_spec = BatchSpec(seq_lens=[1] * batch_size, query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )

        max_blocks = (max(batch_spec.seq_lens) + self.block_size - 1) // self.block_size
        num_blocks = batch_size * max_blocks

        # Fetch the attention backend and kv cache shape and stride order
        attn_backend = self.attn.attn_backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size
        )
        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]

        # Create dummy KV cache
        raw_tensor = torch.zeros(
            num_blocks * self.block_size * self.num_kv_heads * self.head_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
        )
        raw_tensor = raw_tensor.view(kv_cache_shape)
        kv_cache = raw_tensor.permute(*inv_order)

        self.attn.kv_cache = kv_cache

        # Build attn metadata
        attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return attn_metadata

    def forward(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_normed: torch.Tensor,
        mm: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q.clone()
        k_pe = k_pe.clone()
        kv_c_normed = kv_c_normed.clone()
        mm = mm.clone()
        positions = positions.clone()
        cos_sin_cache = cos_sin_cache.clone()
        k_scale = k_scale.clone()

        if self.use_deepseek_scaling_rope:
            h = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.rotary_emb.cos_sin_cache = cos_sin_cache
            q = mm.view(-1, self.num_heads, h)
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed,
                k_pe,
                self.layer_name,
                self.kv_cache_dtype_str,
                k_scale,
            )
            return dummy, q, k_pe, kv_c_normed
        else:
            self.rotary_op(
                positions, q, k_pe, self.qk_rope_head_dim, cos_sin_cache, self.is_neox
            )
            k = k_pe.squeeze(1)
            scatter = torch.ops.aten.slice_scatter.default(
                mm, k, 1, self.kv_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim
            )
            _, k2 = torch.ops.aten.split_with_sizes.default(
                scatter, [self.kv_lora_rank, self.qk_rope_head_dim], -1
            )
            k3 = k2.unsqueeze(1)
            dummy = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed, k3, self.layer_name, self.kv_cache_dtype_str, k_scale
            )
            return dummy, q, k3, kv_c_normed

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops = []
        ops.append(torch.ops.vllm.unified_mla_kv_cache_update.default)
        return ops

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.vllm.fused_concat_and_cache_mla_rope.default]


@pytest.mark.parametrize(
    "attn_backend",
    [AttentionBackendEnum.FLASH_ATTN_MLA, AttentionBackendEnum.FLASHINFER_MLA],
)
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("scale", [0.115])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("v_head_dim", [128])
@pytest.mark.parametrize("q_lora_rank", [None])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize("use_deepseek_scaling_rope", [True, False])
@pytest.mark.parametrize("use_flashinfer", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_rope_kvcache_mla_fusion(
    dist_init,
    attn_backend: AttentionBackendEnum,
    num_heads: int,
    scale: float,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    block_size: int,
    is_neox: bool,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    use_deepseek_scaling_rope: bool,
    use_flashinfer: bool,
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    if use_deepseek_scaling_rope == enabled:
        pytest.skip(
            "Enabled is only supported with use_deepseek_scaling_rope (and vice versa)"
        )

    if is_neox and use_deepseek_scaling_rope:
        pytest.skip("DeepseekScalingRotaryEmbedding is not matched with is_neox")

    if use_flashinfer and use_deepseek_scaling_rope:
        pytest.skip(
            "DeepseekScalingRotaryEmbedding is mutually exclusive with use_flashinfer"
        )

    if not enabled and use_flashinfer:
        pytest.skip("Flashinfer is only supported with enabled")

    custom_ops: list[str] = []
    if enabled:
        custom_ops.append("+rotary_embedding")

    vllm_config = VllmConfig(
        model_config=ModelConfig(model="deepseek-ai/DeepSeek-V2-Lite", dtype=dtype),
        cache_config=CacheConfig(
            block_size=block_size,
            cache_dtype=kv_cache_dtype,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            use_inductor_graph_partition=True,
            pass_config=PassConfig(
                enable_cache_mla_rope_fusion=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context():
        kv_b_proj = ColumnParallelLinear(
            input_size=512,
            output_size=4096,
            return_bias=False,
            disable_tp=True,
            gather_output=False,
            params_dtype=torch.bfloat16,
        )
        model = QKRoPEKVCacheMLATestModel(
            vllm_config=vllm_config,
            attn_backend=attn_backend,
            num_heads=num_heads,
            scale=scale,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            is_neox=is_neox,
            dtype=dtype,
            device=torch.get_default_device(),
            use_deepseek_scaling_rope=use_deepseek_scaling_rope,
            use_flashinfer=use_flashinfer,
        )

        fusion_pass = KVCacheMLARoPEFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            SplitCoalescingPass(vllm_config),
            ScatterSplitReplacementPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)

        T = 5
        L = 163840
        q = torch.randn(T, num_heads, qk_rope_head_dim, dtype=dtype)
        torch._dynamo.mark_dynamic(q, 0)

        k_pe = torch.randn(q.shape[0], 1, qk_rope_head_dim, dtype=dtype)
        kv_c_normed = torch.randn(q.shape[0], kv_lora_rank, dtype=dtype)
        mm_size = (
            num_heads * (qk_nope_head_dim + qk_rope_head_dim)
            if use_deepseek_scaling_rope
            else qk_rope_head_dim + kv_lora_rank
        )
        mm = torch.randn(q.shape[0], mm_size, dtype=dtype)
        pos = torch.arange(q.shape[0], dtype=torch.long)
        k_scale = torch.randn((1), device=k_pe.device, dtype=torch.float32)

        cos_sin_cache = torch.randn(
            L, qk_rope_head_dim, dtype=torch.float32 if use_flashinfer else dtype
        )
        torch._dynamo.mark_dynamic(cos_sin_cache, 0)

        torch._check(q.shape[0] == k_pe.shape[0])
        torch._check(q.shape[0] == kv_c_normed.shape[0])
        torch._check(q.shape[0] == mm.shape[0])
        torch._check(q.shape[0] == pos.shape[0])

        q_unfused = q.clone()
        k_pe_unfused = k_pe.clone()
        kv_c_normed_unfused = kv_c_normed.clone()
        mm_unfused = mm.clone()
        pos_unfused = pos.clone()
        cos_sin_cache_unfused = cos_sin_cache.clone()
        k_scale_unfused = k_scale.clone()

        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            ret_dummy, ret_query, ret_k_pe, ret_kv_c_normed = model(
                q_unfused,
                k_pe_unfused,
                kv_c_normed_unfused,
                mm_unfused,
                pos_unfused,
                cos_sin_cache_unfused,
                k_scale_unfused,
            )
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_unfused = attn_layer.kv_cache
        del ret_dummy

        with set_forward_context(None, vllm_config):
            model_fused = torch.compile(model, backend=backend)
            forward_context = get_forward_context()
            attn_metadata = model_fused.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            ret_dummy_fused, ret_query_fused, ret_k_pe_fused, ret_kv_c_normed_fused = (
                model_fused(q, k_pe, kv_c_normed, mm, pos, cos_sin_cache, k_scale)
            )
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_fused = attn_layer.kv_cache
        del ret_dummy_fused

        assert fusion_pass.matched_count == 1

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())

        ATOL, RTOL = (5e-2, 2e-3)

        torch.testing.assert_close(ret_query, ret_query_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(ret_k_pe, ret_k_pe_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            ret_kv_c_normed, ret_kv_c_normed_fused, atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            kv_cache_unfused, kv_cache_fused, atol=ATOL, rtol=RTOL
        )
