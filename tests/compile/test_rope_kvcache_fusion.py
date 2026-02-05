# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.matcher_utils import ROTARY_OP
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec

INDEX_SELECT_OP = torch.ops.aten.index.Tensor
VLLM_UNIFIED_KV_CACHE_UPDATE_OP = torch.ops.vllm.unified_kv_cache_update


class QKRoPEKVCacheTestModel(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_backend: AttentionBackendEnum,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        is_neox: bool,
        dtype: torch.dtype,
        kv_cache_dtype: torch.dtype,
        device: torch.device,
        prefix: str = "model.layers.0.self_attn.attn",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.q_size = num_heads * head_size
        self.kv_size = num_kv_heads * head_size
        self.is_neox = is_neox
        self.dtype = dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.device = device
        self.layer_name = prefix

        self.rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim=head_size,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            dtype=self.dtype,
        )

        # Whether to check for the RoPE custom op or component index_select
        self.enable_rope_custom_op = self.rotary_emb.enabled()

        # Register layer metadata for the fusion pass via Attention.
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
        assert not self.attn_backend.forward_includes_kv_cache_update, (
            f"Attention backend {self.attn_backend} does not support fuse_rope_kvcache."
        )
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)

        # Initialize attn MetadataBuilder
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=head_size,
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
        backend = self.attn.backend

        # Create dummy KV cache for the selected backend
        if backend == AttentionBackendEnum.ROCM_ATTN:
            raise NotImplementedError
            # k/v as 1st dimention
            # HND: [num_blocks, num_kv_heads, block_size, head_size]
            kv_cache = torch.zeros(
                2,
                num_blocks,
                self.num_kv_heads,
                self.block_size,
                self.head_size,
                dtype=self.kv_cache_dtype,
                device=self.device,
            )
        elif backend == AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN:
            # k/v as 1st dimention
            # NHD: [num_blocks, block_size, num_kv_heads, head_size]
            kv_cache = torch.zeros(
                2,
                num_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_size,
                dtype=self.kv_cache_dtype,
                device=self.device,
            )
        elif backend == AttentionBackendEnum.TRITON_ATTN:
            # k/v as 2nd dimention
            # NHD: [num_blocks, block_size, num_kv_heads, head_size]
            kv_cache = torch.zeros(
                num_blocks,
                2,
                self.num_kv_heads,
                self.block_size,
                self.head_size,
                dtype=self.kv_cache_dtype,
                device=self.device,
            )
        elif backend == AttentionBackendEnum.ROCM_AITER_FA:
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.attn.kv_cache = [kv_cache]

        # Build attn metadata
        attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return attn_metadata

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Create copies so inplace ops do not modify the original tensors
        q = q.clone()
        k = k.clone()
        q, k = self.rotary_emb(positions, q, k)

        # Instead of a full forward pass, match only the KV cache update op here
        q = q.view(-1, self.num_heads, self.head_size)
        k = k.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size)
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            k, v, self.layer_name
        )
        # TODO (Rohan138) return and compare KV cache as well
        return q, k, v, kv_cache_dummy_dep

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops = []
        if self.enable_rope_custom_op:
            ops.append(ROTARY_OP)
        else:
            ops.append(INDEX_SELECT_OP)
        ops.append(torch.ops.vllm.unified_kv_cache_update)
        return ops

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [rocm_aiter_ops.get_qk_rope_reshape_and_cache_op()]


@pytest.mark.parametrize("attn_backend", [AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN])
@pytest.mark.parametrize("enable_rope_custom_op", [True])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("is_neox", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", [torch.bfloat16])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_rope_kvcache_fusion(
    attn_backend: AttentionBackendEnum,
    enable_rope_custom_op: bool,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    is_neox: bool,
    dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    custom_ops: list[str] = []
    if enable_rope_custom_op:
        custom_ops.append("+rotary_embedding")
    else:
        custom_ops.append("-rotary_embedding")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(
                fuse_rope_kvcache=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        from vllm.compilation.rocm_aiter_fusion import (
            ROCmAiterTritonRopeReshapeKVCacheFusionPass,
        )

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        fusion_pass = ROCmAiterTritonRopeReshapeKVCacheFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)

        model = QKRoPEKVCacheTestModel(
            vllm_config=vllm_config,
            attn_backend=attn_backend,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            is_neox=is_neox,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            device=torch.get_default_device(),
        )

        T = 5

        q = torch.randn(T, num_heads * head_size)
        k = torch.randn(T, num_kv_heads * head_size)
        v = torch.randn(T, num_kv_heads * head_size)
        pos = torch.arange(T, dtype=torch.long)

        q_unfused = q.clone()
        k_unfused = k.clone()
        v_unfused = v.clone()
        pos_unfused = pos.clone()

        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_unfused, k_unfused, v_unfused, dummy = model(
                q_unfused, k_unfused, v_unfused, pos_unfused
            )
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_unfused = attn_layer.kv_cache[forward_context.virtual_engine]
        del dummy

        torch._dynamo.mark_dynamic(q, 0)
        torch._dynamo.mark_dynamic(k, 0)
        torch._dynamo.mark_dynamic(v, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        with set_forward_context(None, vllm_config):
            model_fused = torch.compile(model, backend=backend)
            forward_context = get_forward_context()
            attn_metadata = model_fused.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_fused, k_fused, v_fused, dummy = model_fused(q, k, v, pos)
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_fused = attn_layer.kv_cache[forward_context.virtual_engine]
        del dummy

        if dtype == torch.float16:
            ATOL, RTOL = (2e-3, 2e-3)
        else:
            ATOL, RTOL = (1e-2, 1e-2)

        torch.testing.assert_close(q_unfused, q_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(k_unfused, k_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(v_unfused, v_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            kv_cache_unfused, kv_cache_fused, atol=ATOL, rtol=RTOL
        )

        assert fusion_pass.matched_count == 1

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
