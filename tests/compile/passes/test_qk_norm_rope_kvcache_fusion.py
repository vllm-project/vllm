# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.fusion.matcher_utils import RMS_OP, ROTARY_OP
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
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec

INDEX_SELECT_OP = torch.ops.aten.index.Tensor
FP8_DTYPE = current_platform.fp8_dtype()


class QKNormRoPEKVCacheTestModel(torch.nn.Module):
    """Minimal model that reproduces the QK-norm + RoPE + KV cache update
    pattern matched by QkNormRopeKvCacheFusionPass:

        q, k, v = split(qkv)
        q = rms_norm(q.view(heads, dim), q_weight).view(flat)
        k = rms_norm(k.view(heads, dim), k_weight).view(flat)
        q, k = rotary_emb(positions, q, k)
        q = q.view(num_heads, head_dim)
        k = k.view(num_kv_heads, head_dim)
        v = v.view(num_kv_heads, head_dim)
        dummy = unified_kv_cache_update(k, v, layer_name)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_backend: AttentionBackendEnum,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
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
        self.is_neox = is_neox
        self.dtype = dtype
        self.device = device
        self.layer_name = prefix

        self.q_norm = RMSNorm(head_size, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_size, eps=rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim=head_size,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            dtype=self.dtype,
        )

        self.enable_rope_custom_op = self.rotary_emb.enabled()
        self.enable_rms_custom_op = self.q_norm.enabled()

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
            f"Attention backend {self.attn_backend} does not support "
            "fuse_qk_norm_rope_kvcache."
        )
        kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        self.kv_cache_dtype = (
            FP8_DTYPE if kv_cache_dtype_str.startswith("fp8") else self.dtype
        )

        if self.kv_cache_dtype != self.dtype:
            self.attn._k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
            self.attn._v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
            self.attn._k_scale_float = 1.0
            self.attn._v_scale_float = 1.0
        else:
            self.attn._k_scale = self.attn._k_scale.to(device)
            self.attn._v_scale = self.attn._v_scale.to(device)

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
        batch_spec = BatchSpec(seq_lens=[1] * batch_size, query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )

        max_blocks = (max(batch_spec.seq_lens) + self.block_size - 1) // self.block_size
        num_blocks = batch_size * max_blocks

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

        raw_tensor = torch.zeros(
            2 * num_blocks * self.block_size * self.num_kv_heads * self.head_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
        )
        raw_tensor = raw_tensor.view(kv_cache_shape)
        kv_cache = raw_tensor.permute(*inv_order)

        self.attn.kv_cache = [kv_cache]

        attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return attn_metadata

    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.clone()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm: RMSNorm on per-head Q and K
        q = q.view(-1, self.num_heads, self.head_size)
        q = self.q_norm(q)
        q = q.view(-1, self.q_size)

        k = k.view(-1, self.num_kv_heads, self.head_size)
        k = self.k_norm(k)
        k = k.view(-1, self.kv_size)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Final views + KV cache update
        q = q.view(-1, self.num_heads, self.head_size)
        k = k.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size)
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            k, v, self.layer_name
        )
        return q, k, v, kv_cache_dummy_dep

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops: list[torch._ops.OpOverload] = []
        # RMSNorm custom op: when enabled on ROCm with AITER, the AITER
        # RMSNorm op appears in the graph.  When disabled (custom_ops has
        # "none"), it decomposes into primitive aten ops with no single
        # identifiable op node.
        if self.enable_rms_custom_op:
            if rocm_aiter_ops.is_rmsnorm_enabled():
                ops.append(rocm_aiter_ops.get_rmsnorm_op())
            else:
                ops.append(RMS_OP)
        # RoPE op
        if self.enable_rope_custom_op:
            if rocm_aiter_ops.is_triton_rotary_embed_enabled():
                ops.append(torch.ops.vllm.rocm_aiter_triton_rotary_embedding.default)
            else:
                ops.append(ROTARY_OP)
        else:
            ops.append(INDEX_SELECT_OP)
        ops.append(torch.ops.vllm.unified_kv_cache_update.default)
        return ops

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default]


@pytest.mark.parametrize(
    "attn_backend",
    [
        AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        AttentionBackendEnum.ROCM_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
    ],
)
@pytest.mark.parametrize("enable_rms_custom_op", [True, False])
@pytest.mark.parametrize("enable_aiter_triton_rope", [True, False])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("rms_norm_eps", [1e-5, 1e-6])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_qk_norm_rope_kvcache_fusion(
    attn_backend: AttentionBackendEnum,
    enable_rms_custom_op: bool,
    enable_aiter_triton_rope: bool,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    is_neox: bool,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    rms_norm_eps: float,
    monkeypatch: pytest.MonkeyPatch,
):
    device = os.environ.get("VLLM_TEST_CUDA_DEVICE", "cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    custom_ops: list[str] = ["+rotary_embedding"]
    if enable_rms_custom_op:
        custom_ops.append("+rms_norm")

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(
            block_size=block_size,
            cache_dtype=kv_cache_dtype,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(
                fuse_qk_norm_rope_kvcache=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv(
            "VLLM_ROCM_USE_AITER_TRITON_ROPE",
            "1" if enable_aiter_triton_rope else "0",
        )
        rocm_aiter_ops.refresh_env_variables()

        model = QKNormRoPEKVCacheTestModel(
            vllm_config=vllm_config,
            attn_backend=attn_backend,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            is_neox=is_neox,
            rms_norm_eps=rms_norm_eps,
            dtype=dtype,
            device=torch.get_default_device(),
        )

        fusion_pass = QkNormRopeKvCacheFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            SplitCoalescingPass(vllm_config),
            ScatterSplitReplacementPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)

        T = 5

        qkv = torch.randn(
            T,
            num_heads * head_size + 2 * num_kv_heads * head_size,
            dtype=dtype,
        )
        pos = torch.arange(T, dtype=torch.long)

        qkv_unfused = qkv.clone()
        pos_unfused = pos.clone()

        # Run unfused (eager) forward
        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_unfused, k_unfused, v_unfused, dummy = model(qkv_unfused, pos_unfused)
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_unfused = attn_layer.kv_cache[forward_context.virtual_engine]
        del dummy

        # Run fused (compiled) forward
        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        with set_forward_context(None, vllm_config):
            model_fused = torch.compile(model, backend=backend)
            forward_context = get_forward_context()
            attn_metadata = model_fused.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_fused, k_fused, v_fused, dummy = model_fused(qkv, pos)
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_fused = attn_layer.kv_cache[forward_context.virtual_engine]
        del dummy

        assert fusion_pass.matched_count == 1

        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())

        ATOL, RTOL = (1e-2, 1e-2)
        is_fp8_cache = model.kv_cache_dtype != dtype

        torch.testing.assert_close(q_unfused, q_fused, atol=ATOL, rtol=RTOL)

        if not is_fp8_cache:
            # The AITER PTS kernel populates k_out only for non-FP8 caches.
            # With FP8, the kernel writes quantized K directly to the cache
            # and may leave k_out uninitialised.  In production this is fine
            # because downstream attention reads K from the cache.
            torch.testing.assert_close(k_unfused, k_fused, atol=ATOL, rtol=RTOL)

        torch.testing.assert_close(v_unfused, v_fused, atol=ATOL, rtol=RTOL)

        uses_interleaved_v = getattr(model.attn.impl, "_use_interleaved_v_cache", False)
        cache_atol = 5e-2 if is_fp8_cache else ATOL
        cache_rtol = 1.0 if is_fp8_cache else RTOL

        # K-cache: same layout for both paths, always compare directly.
        torch.testing.assert_close(
            kv_cache_unfused[0].view(dtype),
            kv_cache_fused[0].view(dtype),
            atol=cache_atol,
            rtol=cache_rtol,
        )

        if uses_interleaved_v:
            # The fused AITER kernel writes V-cache in interleaved layout
            # [blocks, heads, block_size/x, head_dim, x] while the unfused
            # write_to_paged_cache uses standard [blocks, heads, head_dim,
            # block_size].  Transform interleaved → standard before comparing.
            #
            # split_kv_cache views the raw [n, BS, H, D] as [n, H, D, BS].
            # In that view the interleaved data is laid out as
            # [BS//x, D, x] per (block, head), so:
            #   reshape → [n, H, BS//x, D, x]
            #   permute → [n, H, D, BS//x, x]
            #   reshape → [n, H, D, BS]   (standard layout)
            x_il = 16 // kv_cache_fused.element_size()
            n_blk = kv_cache_fused.shape[1]

            v_unfused_view = kv_cache_unfused[1].view(
                n_blk, num_kv_heads, head_size, block_size
            )
            v_fused_view = kv_cache_fused[1].view(
                n_blk, num_kv_heads, head_size, block_size
            )
            v_fused_std = (
                v_fused_view.reshape(
                    n_blk, num_kv_heads, block_size // x_il, head_size, x_il
                )
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .reshape(n_blk, num_kv_heads, head_size, block_size)
            )
            torch.testing.assert_close(
                v_unfused_view.view(dtype),
                v_fused_std.view(dtype),
                atol=cache_atol,
                rtol=cache_rtol,
            )
        else:
            torch.testing.assert_close(
                kv_cache_unfused[1].view(dtype),
                kv_cache_fused[1].view(dtype),
                atol=cache_atol,
                rtol=cache_rtol,
            )
