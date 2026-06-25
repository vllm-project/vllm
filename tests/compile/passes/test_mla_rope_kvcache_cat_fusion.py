# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.fusion.mla_rope_kvcache_cat_fusion import (
    MLARoPEKVCacheCatFusionPass,
)
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.backends.registry import AttentionBackendEnum

INDEX_SELECT_OP = torch.ops.aten.index.Tensor
VLLM_UNIFIED_MLA_KV_CACHE_UPDATE_OP = torch.ops.vllm.unified_mla_kv_cache_update
FP8_DTYPE = current_platform.fp8_dtype()


class MLARoPEKVCacheCatTestModel(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_backend: AttentionBackendEnum,
        use_deepseek_scaling_rope: bool,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        is_neox: bool,
        dtype: torch.dtype,
        device: torch.device,
        prefix: str = "model.layers.0.self_attn.attn",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.dtype = dtype
        self.device = device
        self.layer_name = prefix

        self.num_kv_heads = 1
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.block_size = vllm_config.cache_config.block_size
        self.scale = self.qk_head_dim**-0.5

        if use_deepseek_scaling_rope:
            self.rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size=qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position_embeddings=4096,
                base=10000,
                is_neox_style=is_neox,
                scaling_factor=1.0,
                dtype=dtype,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                head_size=qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position_embeddings=4096,
                base=10000,
                is_neox_style=is_neox,
                dtype=dtype,
            )

        # Initialize intermediate mm layers for unit test
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            prefix=f"{prefix}.q_b_proj",
        ).to(device)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            prefix=f"{prefix}.kv_b_proj",
        ).to(device)

        # ColumnParallelLinear default init in bf16 with seed 0 produces
        # near-zero weights (7/4.7M nonzero), making the GEMM output almost
        # entirely zero and masking correctness bugs. Reinitialize to get
        # dense outputs.
        with torch.no_grad():
            torch.nn.init.normal_(self.q_b_proj.weight, std=0.02)
            torch.nn.init.normal_(self.kv_b_proj.weight, std=0.02)

        # Register layer metadata for the fusion pass via MLAAttention
        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=self.scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            attn_backend=attn_backend.get_class(),
        )
        self.attn_backend: type[AttentionBackend] = self.mla_attn.get_attn_backend()
        self.mla_attn._k_scale = self.mla_attn._k_scale.to(device)
        self.mla_attn._v_scale = self.mla_attn._v_scale.to(device)

        # Keep both the string dtype (for ops) and torch dtype (for tensors)
        self.kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        self.kv_cache_dtype = (
            FP8_DTYPE if self.kv_cache_dtype_str.startswith("fp8") else self.dtype
        )

        # Initialize attn MetadataBuilder
        self.builder = self.attn_backend.get_builder_cls()(
            kv_cache_spec=self.mla_attn.get_kv_cache_spec(vllm_config),
            layer_names=[self.mla_attn.layer_name],
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
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size
        )
        try:
            kv_cache_stride_order = self.attn_backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]

        raw_tensor = torch.zeros(
            num_blocks * self.block_size * self.num_kv_heads * self.head_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
        )
        raw_tensor = raw_tensor.view(kv_cache_shape)
        kv_cache = raw_tensor.permute(*inv_order)

        self.mla_attn.kv_cache = kv_cache

        # Build attn metadata
        attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return attn_metadata

    def forward(
        self, qkv_lora: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_lora = qkv_lora.clone()
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q = self.q_b_proj(q_c)[0]
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)

        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        dummy = torch.ops.vllm.unified_mla_kv_cache_update(
            kv_c,
            k_pe,
            _encode_layer_name(self.layer_name),
            self.kv_cache_dtype_str,
            self.mla_attn._k_scale,
        )
        return q, kv_c, k_pe, dummy

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops = [
            INDEX_SELECT_OP,
            torch.ops.vllm.unified_mla_kv_cache_update.default,
        ]
        return ops

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.vllm.fused_rope_unified_mla_kv_cache_update.default]


MLA_BACKENDS = [AttentionBackendEnum.TRITON_MLA]
if flash_attn_supports_mla():
    MLA_BACKENDS += [AttentionBackendEnum.FLASH_ATTN_MLA]
if is_aiter_found_and_supported():
    MLA_BACKENDS += [AttentionBackendEnum.ROCM_AITER_MLA]


@pytest.mark.parametrize("attn_backend", MLA_BACKENDS)
@pytest.mark.parametrize("use_deepseek_scaling_rope", [True])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("v_head_dim", [128])
@pytest.mark.parametrize("q_lora_rank", [1536])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="MLA RoPE+KVCache+Cat fusion is only supported on CUDA and ROCm.",
)
def test_mla_rope_kvcache_cat_fusion(
    attn_backend: AttentionBackendEnum,
    use_deepseek_scaling_rope: bool,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    block_size: int,
    is_neox: bool,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="deepseek-ai/DeepSeek-V2-Lite",
            dtype=dtype,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            cache_dtype=kv_cache_dtype,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(
                fuse_rope_kvcache_cat_mla=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        if not torch.distributed.is_initialized():
            from vllm.distributed.parallel_state import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            from vllm.utils.system_utils import update_environment_variables

            update_environment_variables(
                {
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "WORLD_SIZE": "1",
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "54321",
                }
            )
            init_distributed_environment()
            initialize_model_parallel()

        if attn_backend == AttentionBackendEnum.ROCM_AITER_MLA:
            m.setenv("VLLM_ROCM_USE_AITER", "1")
            rocm_aiter_ops.refresh_env_variables()

        model = MLARoPEKVCacheCatTestModel(
            vllm_config=vllm_config,
            attn_backend=attn_backend,
            use_deepseek_scaling_rope=use_deepseek_scaling_rope,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            is_neox=is_neox,
            dtype=dtype,
            device=torch.get_default_device(),
        )

        fusion_pass = MLARoPEKVCacheCatFusionPass(vllm_config)
        # note: FixFunctionalizationPass is required to correctly lower
        # the fused op to its inplace version with auto-functionalization v1.
        # Without it, decompose_auto_functionalized calls clone_preserve_strides
        # on the non-contiguous q_pe slice directly, and inductor's lowering
        # of the resulting as_strided chain incorrectly drops the storage offset.
        # auto-functionalization v2 avoids this: it clones the contiguous base
        # tensor (_all_bases) and reconstructs the slice as a view, so the
        # offset is never passed through as_strided lowering.
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
            FixFunctionalizationPass(vllm_config),
        ]
        backend = TestBackend(*passes)

        T = 5

        qkv_lora = torch.randn(
            T,
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
        )
        pos = torch.arange(T, dtype=torch.long)

        qkv_unfused = qkv_lora.clone()
        pos_unfused = pos.clone()

        # Run unfused version
        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_unfused, kv_c_unfused, k_pe_unfused, dummy = model(
                qkv_unfused, pos_unfused
            )
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_unfused = attn_layer.kv_cache.clone()
        del dummy

        # Run fused version (compiled)
        torch._dynamo.mark_dynamic(qkv_lora, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        with set_forward_context(None, vllm_config):
            model_fused = torch.compile(model, backend=backend)
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_fused, kv_c_fused, k_pe_fused, dummy = model_fused(qkv_lora, pos)
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_fused = attn_layer.kv_cache
        del dummy

        assert fusion_pass.matched_count == 1

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())

        if dtype == torch.float16:
            ATOL, RTOL = (2e-3, 2e-3)
        else:
            ATOL, RTOL = (1e-2, 1e-2)

        torch.testing.assert_close(q_unfused, q_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(kv_c_unfused, kv_c_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(k_pe_unfused, k_pe_fused, atol=ATOL, rtol=RTOL)
        # Cannot compare fp8_* directly here, cast to model dtype instead
        torch.testing.assert_close(
            kv_cache_unfused.view(dtype),
            kv_cache_fused.view(dtype),
            atol=ATOL,
            rtol=RTOL,
        )
