# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.fusion.matcher_utils import QUANT_OPS, ROTARY_OP
from vllm.compilation.passes.fusion.rope_kvcache_fusion import RopeKVCacheFusionPass
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.utility.scatter_split_replace import (
    ScatterSplitReplacementPass,
)
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec

INDEX_SELECT_OP = torch.ops.aten.index.Tensor
VLLM_UNIFIED_KV_CACHE_UPDATE_OP = torch.ops.vllm.unified_kv_cache_update
FP8_DTYPE = current_platform.fp8_dtype()


class QKRoPEKVCacheTestModelBase(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        is_neox: bool,
        dtype: torch.dtype,
        device: torch.device,
        prefix: str = "model.layers.0.self_attn.attn",
        attn_backend: AttentionBackendEnum = None,
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
            attn_backend=attn_backend.get_class() if attn_backend is not None else None,
        )
        self.attn_backend: type[AttentionBackend] = self.attn.get_attn_backend()
        assert not self.attn_backend.forward_includes_kv_cache_update, (
            f"Attention backend {self.attn_backend} does not support fuse_rope_kvcache."
        )
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)

        kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        self.kv_cache_dtype = (
            FP8_DTYPE if kv_cache_dtype_str.startswith("fp8") else self.dtype
        )

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
            2 * num_blocks * self.block_size * self.num_kv_heads * self.head_size,
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
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        raise NotImplementedError

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        raise NotImplementedError


class QKRoPEKVCacheTestModel(QKRoPEKVCacheTestModelBase):
    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Create copy so inplace ops do not modify the original tensors
        qkv = qkv.clone()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # Instead of a full forward pass, match only the KV cache update op here
        q = q.view(-1, self.num_heads, self.head_size)
        k = k.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size)
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            k, v, self.layer_name
        )
        return q, k, v, kv_cache_dummy_dep

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops = []
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
        return [torch.ops.vllm.fused_rope_and_unified_kv_cache_update.default]


class QKRoPEQuantKVCacheTestModel(QKRoPEKVCacheTestModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.attn.query_quant is not None

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        qkv = qkv.clone()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return attn_output

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops = []
        if self.enable_rope_custom_op:
            if self.rotary_emb.use_flashinfer:
                ops.append(torch.ops.vllm.flashinfer_rotary_embedding.default)
            else:
                ops.append(ROTARY_OP)
        else:
            ops.append(INDEX_SELECT_OP)
        if self.attn.query_quant.enabled():
            ops.append(QUANT_OPS[kFp8StaticTensorSym])
        else:
            ops.append(torch.ops.aten.reciprocal)
        ops.append(torch.ops.vllm.unified_kv_cache_update.default)
        return ops

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.vllm.fused_rope_and_unified_kv_cache_update.default]


@pytest.mark.parametrize(
    "attn_backend",
    [
        AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.ROCM_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
    ],
)
@pytest.mark.parametrize("enable_rope_custom_op", [True])  # [True, False])
@pytest.mark.parametrize("enable_aiter_triton_rope", [True, False])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="Only test on ROCm with AITER installed and supported",
)
def test_rope_kvcache_fusion(
    attn_backend: AttentionBackendEnum,
    enable_rope_custom_op: bool,
    enable_aiter_triton_rope: bool,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    is_neox: bool,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    custom_ops: list[str] = []
    if enable_rope_custom_op:
        custom_ops.append("+rotary_embedding")

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
                fuse_rope_kvcache=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv(
            "VLLM_ROCM_USE_AITER_TRITON_ROPE", "1" if enable_aiter_triton_rope else "0"
        )
        rocm_aiter_ops.refresh_env_variables()

        model = QKRoPEKVCacheTestModel(
            vllm_config=vllm_config,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            is_neox=is_neox,
            dtype=dtype,
            device=torch.get_default_device(),
            attn_backend=attn_backend,
        )

        fusion_pass = RopeKVCacheFusionPass(vllm_config)
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
            T, num_heads * head_size + 2 * num_kv_heads * head_size, dtype=dtype
        )
        pos = torch.arange(T, dtype=torch.long)

        qkv_unfused = qkv.clone()
        pos_unfused = pos.clone()

        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_unfused, k_unfused, v_unfused, dummy = model(qkv_unfused, pos_unfused)
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_unfused = attn_layer.kv_cache
        del dummy

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
        torch.testing.assert_close(k_unfused, k_fused, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(v_unfused, v_fused, atol=ATOL, rtol=RTOL)
        # Cannot compare fp8_* directly here, cast to model dtype instead
        torch.testing.assert_close(
            kv_cache_unfused.view(dtype),
            kv_cache_fused.view(dtype),
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("attn_backend", [AttentionBackendEnum.FLASHINFER])
@pytest.mark.parametrize("model_name", ["openai/gpt-oss-20b"])
@pytest.mark.parametrize("enable_rope_custom_op", [True])
@pytest.mark.parametrize("enable_quant_custom_op", [True, False])
@pytest.mark.parametrize("enable_flashinfer_rope", [True, False])
@pytest.mark.parametrize("batch_size", [7, 64, 533])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["fp8"])
@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability((10, 0))
        and has_flashinfer()
    ),
    reason="Only test on CUDA Blackwell platform with FlashInfer installed",
)
def test_rope_quant_kvcache_fusion(
    attn_backend: AttentionBackendEnum,
    model_name: str,
    enable_rope_custom_op: bool,
    enable_quant_custom_op: bool,
    enable_flashinfer_rope: bool,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    is_neox: bool,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    if enable_flashinfer_rope:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_ROPE", "1")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    custom_ops: list[str] = []
    if enable_rope_custom_op:
        custom_ops.append("+rotary_embedding")
    if enable_quant_custom_op:
        custom_ops.append("+quant_fp8")

    model_config = ModelConfig(
        model=model_name,
        max_model_len=2048,
        dtype=dtype,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=1024,
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            cache_dtype=kv_cache_dtype,
        ),
        attention_config=AttentionConfig(
            backend=attn_backend,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(
                eliminate_noops=False,
                fuse_rope_kvcache=False,
            ),
        ),
    )

    hidden_size = head_size * (num_heads + num_kv_heads * 2)
    qkv = torch.randn(batch_size, hidden_size, dtype=dtype)
    pos = torch.arange(batch_size, dtype=torch.long)

    # Run model directly without fusion
    vllm_config_unfused = copy.deepcopy(vllm_config)
    with (
        set_current_vllm_config(vllm_config_unfused),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config_unfused),
    ):
        model_unfused = QKRoPEQuantKVCacheTestModel(
            vllm_config=vllm_config_unfused,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            is_neox=is_neox,
            dtype=dtype,
            device=torch.get_default_device(),
        )
        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_unfused.build_attn_metadata(batch_size)
        forward_ctx.slot_mapping = {
            model_unfused.layer_name: forward_ctx.attn_metadata.slot_mapping
        }
        compiled_unfused = torch.compile(model_unfused, fullgraph=True)
        result_unfused = compiled_unfused(qkv.clone(), pos.clone())

    # Run model with fusion enabled
    vllm_config.compilation_config.pass_config = PassConfig(
        eliminate_noops=True,
        fuse_rope_kvcache=True,
    )
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
    ):
        model_fused = QKRoPEQuantKVCacheTestModel(
            vllm_config=vllm_config,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            is_neox=is_neox,
            dtype=dtype,
            device=torch.get_default_device(),
        )
        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_fused.build_attn_metadata(batch_size)
        forward_ctx.slot_mapping = {
            model_fused.layer_name: forward_ctx.attn_metadata.slot_mapping
        }

        # Create test backend with fusion passes enabled
        fusion_pass = RopeKVCacheFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            SplitCoalescingPass(vllm_config),
            ScatterSplitReplacementPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)
        compiled_fused = torch.compile(model_fused, backend=backend, fullgraph=True)
        result_fused = compiled_fused(qkv.clone(), pos.clone())

        assert fusion_pass.matched_count == 1

        backend.check_before_ops(model_fused.ops_in_model_before())
        backend.check_after_ops(model_fused.ops_in_model_after())

    torch.testing.assert_close(result_unfused, result_fused, atol=1e-2, rtol=1e-2)
