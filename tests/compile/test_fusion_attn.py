# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch._dynamo

from tests.compile.backend import LazyInitPass, TestBackend
from tests.utils import flat_product
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import Attention
from vllm.attention.selector import global_force_attn_backend_context_manager
from vllm.compilation.fusion_attn import ATTN_OP, AttnFusionPass
from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.matcher_utils import QUANT_OPS
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.config import (
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Quant,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.kv_cache_interface import AttentionSpec

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


class AttentionQuantPatternModel(torch.nn.Module):
    """Base model for AttentionQuantPattern fusion."""

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_size: int,
        kv_cache_dtype: torch.dtype,
        device: torch.device,
        vllm_config: VllmConfig,
        **kwargs,
    ):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.device = device
        self.vllm_config = vllm_config

        self.attn = Attention(
            num_heads=self.num_qo_heads,
            head_size=self.head_size,
            scale=1.0 / (self.head_size**0.5),
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
        )
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)

        self.block_size = 16

        # Initialize attn MetadataBuilder
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_dtype,
            ),
            layer_names=[self.attn.layer_name],
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def build_attn_metadata(self, batch_size: int) -> AttentionMetadata:
        """Initialize attention metadata."""

        # Create common attn metadata
        batch_spec = BatchSpec(seq_lens=[1] * batch_size, query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )

        max_blocks = (max(batch_spec.seq_lens) + self.block_size - 1) // self.block_size
        num_blocks = batch_size * max_blocks
        backend = self.attn.backend

        # TODO(luka) use get_kv_cache_stride_order
        # Create dummy KV cache for the selected backend
        if backend == AttentionBackendEnum.ROCM_ATTN:
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
        elif backend == AttentionBackendEnum.FLASHINFER:
            kv_cache = torch.zeros(
                num_blocks,
                2,
                self.num_kv_heads,
                self.block_size,
                self.head_size,
                dtype=self.kv_cache_dtype,
                device=self.device,
            ).permute(0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.attn.kv_cache = [kv_cache]

        # Build attn metadata
        self.attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return self.attn_metadata


class TestAttentionFp8StaticQuantPatternModel(AttentionQuantPatternModel):
    """Test model for AttentionFp8StaticQuantPattern fusion."""

    quant_key = kFp8StaticTensorSym

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fp8_linear = Fp8LinearOp(
            act_quant_static=self.quant_key.scale.static,
            act_quant_group_shape=self.quant_key.scale.group_shape,
        )

        hidden_size = self.num_qo_heads * self.head_size
        self.w = kwargs.get(
            "w",
            {
                "weight": torch.randn(hidden_size, hidden_size)
                .to(dtype=FP8_DTYPE, device=self.device)
                .t(),
                "wscale": torch.tensor([1.0], dtype=torch.float32, device=self.device),
                "scale": torch.tensor([1.0], dtype=torch.float32, device=self.device),
            },
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass that creates the pattern to be fused."""
        attn_output = self.attn(q, k, v)
        return self.fp8_linear.apply(
            input=attn_output,
            weight=self.w["weight"],
            weight_scale=self.w["wscale"],
            input_scale=self.w["scale"],
        )


class TestAttentionNvfp4QuantPatternModel(AttentionQuantPatternModel):
    """Test model for AttentionNvfp4QuantPattern fusion."""

    quant_key = kNvfp4Quant

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden_size = self.num_qo_heads * self.head_size
        self.w = kwargs.get(
            "w",
            {
                "weight": torch.randint(
                    256,
                    (hidden_size, hidden_size // 2),
                    dtype=FP4_DTYPE,
                    device=self.device,
                ),
                "wscale_swizzled": torch.randn(hidden_size, hidden_size // 16).to(
                    dtype=FP8_DTYPE, device=self.device
                ),
                "wscale": torch.tensor([500], dtype=torch.float32, device=self.device),
                "scale": torch.tensor([0.002], dtype=torch.float32, device=self.device),
            },
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass that creates the pattern to be fused."""
        attn_output = self.attn(q, k, v)
        quant_output, output_block_scale = scaled_fp4_quant(
            attn_output, 1 / self.w["scale"]
        )
        return cutlass_scaled_fp4_mm(
            a=quant_output,
            b=self.w["weight"],
            block_scale_a=output_block_scale,
            block_scale_b=self.w["wscale_swizzled"],
            alpha=self.w["scale"] * self.w["wscale"],
            out_dtype=attn_output.dtype,
        )


MODELS_FP8: list[tuple[str, type]] = []
MODELS_FP4: list[tuple[str, type]] = []
HEADS: list[tuple[int, int]] = []
SPLIT_ATTENTION: list[bool] = []
BACKENDS_FP8: list[AttentionBackendEnum] = []
BACKENDS_FP4: list[AttentionBackendEnum] = []

if current_platform.is_cuda():
    HEADS = [(64, 8), (40, 8)]
    MODELS_FP8 = [
        (
            "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
            TestAttentionFp8StaticQuantPatternModel,
        )
    ]
    MODELS_FP4 = [
        (
            "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
            TestAttentionNvfp4QuantPatternModel,
        )
    ]
    BACKENDS_FP8 = [AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.FLASHINFER]
    BACKENDS_FP4 = [AttentionBackendEnum.FLASHINFER]

elif current_platform.is_rocm():
    HEADS = [(32, 8), (40, 8)]
    MODELS_FP8 = [
        ("amd/Llama-3.1-8B-Instruct-FP8-KV", TestAttentionFp8StaticQuantPatternModel)
    ]
    BACKENDS = [
        AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        AttentionBackendEnum.ROCM_ATTN,
        AttentionBackendEnum.TRITON_ATTN,
    ]


@pytest.mark.parametrize("num_qo_heads, num_kv_heads", HEADS)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "batch_size", [7, 256, 533] if current_platform.is_cuda() else [8]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "backend, model_name, model_class, custom_ops",
    # Test attention+quant_fp8 fusion with custom and torch impls of QuantFP8
    list(flat_product(BACKENDS_FP8, MODELS_FP8, ["+quant_fp8", "-quant_fp8"]))
    # quant_fp4 only has the custom impl
    + list(flat_product(BACKENDS_FP4, MODELS_FP4, [""])),
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test ROCm or CUDA"
)
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
def test_attention_quant_pattern(
    num_qo_heads: int,
    num_kv_heads: int,
    head_size: int,
    batch_size: int,
    dtype: torch.dtype,
    custom_ops: str,
    model_name: str,
    model_class: type[AttentionQuantPatternModel],
    backend: AttentionBackendEnum,
    dist_init,
):
    """Test AttentionStaticQuantPattern fusion pass"""
    if backend == AttentionBackendEnum.FLASHINFER and (
        not current_platform.is_device_capability((10, 0)) or not has_flashinfer()
    ):
        pytest.skip("FlashInfer attn fusion requires Blackwell and flashinfer")

    custom_ops_list = custom_ops.split(",") if custom_ops else []

    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

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
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops_list,
        ),
        cache_config=CacheConfig(cache_dtype="fp8"),
    )

    # Create test inputs
    q = torch.randn(batch_size, num_qo_heads * head_size, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_kv_heads * head_size, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_kv_heads * head_size, dtype=dtype, device=device)

    # Mark first dimension as dynamic for realistic testing
    torch._dynamo.mark_dynamic(q, 0)
    torch._dynamo.mark_dynamic(k, 0)
    torch._dynamo.mark_dynamic(v, 0)

    # Run model directly without compilation and fusion
    vllm_config_unfused = copy.deepcopy(vllm_config)
    with (
        set_current_vllm_config(vllm_config_unfused),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config_unfused),
        global_force_attn_backend_context_manager(backend),
    ):
        model_unfused = model_class(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            kv_cache_dtype=FP8_DTYPE,
            device=device,
            vllm_config=vllm_config_unfused,
        )
        model_unfused = model_unfused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_unfused.build_attn_metadata(batch_size)

        # Run model directly without fusion
        # Still compile so query QuantFP8 has closer numerics
        result_unfused = torch.compile(model_unfused, fullgraph=True)(q, k, v)

    # Run model with attn fusion enabled
    vllm_config.compilation_config.pass_config = PassConfig(
        enable_attn_fusion=True, enable_noop=True
    )
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
        global_force_attn_backend_context_manager(backend),
    ):
        model_fused = model_class(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            kv_cache_dtype=FP8_DTYPE,
            device=device,
            vllm_config=vllm_config,
            w=model_unfused.w,
        )
        model_fused = model_fused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_fused.build_attn_metadata(batch_size)

        # Create test backend with fusion passes enabled
        noop_pass = NoOpEliminationPass(vllm_config)
        attn_pass = LazyInitPass(AttnFusionPass, vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        test_backend = TestBackend(noop_pass, attn_pass, cleanup_pass)

        # Compile model with fusion enabled
        model_compiled = torch.compile(
            model_fused, backend=test_backend, fullgraph=True
        )
        assert model_compiled.attn._o_scale_float is None

        result_fused_1 = model_compiled(q, k, v)

        if backend == AttentionBackendEnum.FLASHINFER:
            # With the Flashinfer backend after the 1st round of the forward
            # pass, output quant scale should be loaded into the attn layer's
            # _o_scale_float, the 2nd round should reuse the loaded
            # _o_scale_float
            assert model_compiled.attn._o_scale_float is not None
            result_fused_2 = model_compiled(q, k, v)

            assert model_compiled.attn._o_scale_float is not None

            torch.testing.assert_close(
                result_unfused, result_fused_2, atol=1e-2, rtol=1e-2
            )

    # Check attn fusion support
    quant_key: QuantKey = model_class.quant_key
    attn_fusion_supported = [
        layer.impl.fused_output_quant_supported(quant_key)
        for key, layer in vllm_config.compilation_config.static_forward_context.items()
    ]
    assert sum(attn_fusion_supported) == len(attn_fusion_supported), (
        "All layers should support attention fusion"
    )

    # Check quantization ops in the graph before and after fusion
    quant_op = (
        torch.ops.aten.reciprocal
        if "-quant_fp8" in custom_ops_list
        else QUANT_OPS[quant_key]
    )

    # Note: for fp8, fully_replaced=False because query quant ops remain in graph.
    # Only output quant ops are fused into attention.
    test_backend.check_before_ops([quant_op], fully_replaced=quant_key is kNvfp4Quant)

    # access the underlying `AttnFusionPass` on the `LazyInitPass`
    assert attn_pass.pass_.matched_count == sum(attn_fusion_supported)

    # Check attention ops in the graph before and after fusion
    attn_nodes_pre = list(find_op_nodes(ATTN_OP, test_backend.graph_pre_pass))
    attn_nodes_post = list(find_op_nodes(ATTN_OP, test_backend.graph_post_pass))

    assert len(attn_nodes_pre) > 0, "Should have attention nodes before fusion"
    assert len(attn_nodes_pre) == len(attn_nodes_post), (
        "Should have same number of attention nodes before and after fusion"
    )
    assert attn_nodes_pre[0].kwargs.get("output_scale") is None, (
        "Attention should not have output_scale before fusion"
    )
    assert attn_nodes_post[0].kwargs.get("output_scale") is not None, (
        "Attention should have output_scale after fusion"
    )

    assert attn_nodes_pre[0].kwargs.get("output_block_scale") is None, (
        "Attention should not have output_block_scale before fusion"
    )
    if quant_key.dtype == FP8_DTYPE:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is None, (
            "Attention should not have output_block_scale after FP8 fusion"
        )
    elif quant_key.dtype == FP4_DTYPE:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is not None, (
            "Attention should have output_block_scale after FP4 fusion"
        )

    # Check that results are close
    torch.testing.assert_close(result_unfused, result_fused_1, atol=1e-2, rtol=1e-2)
