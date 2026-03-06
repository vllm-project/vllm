# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch._dynamo

from tests.compile.backend import LazyInitPass, TestBackend
from tests.utils import TestFP8Layer, flat_product
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.passes.fusion.matcher_utils import QUANT_OPS
from vllm.compilation.passes.fusion.mla_attn_quant_fusion import (
    MLA_ATTN_OP,
    MLAAttnFusionPass,
)
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
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
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


class MLAAttentionQuantPatternModel(torch.nn.Module):
    """Base model for MLA AttentionQuantPattern fusion."""

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        kv_cache_dtype: torch.dtype,
        device: torch.device,
        vllm_config: VllmConfig,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.output_dim = num_heads * v_head_dim
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.kv_cache_dtype = kv_cache_dtype
        self.device = device
        self.vllm_config = vllm_config

        # Create kv_b_proj (ColumnParallelLinear) on device
        kv_b_proj = ColumnParallelLinear(
            input_size=kv_lora_rank,
            output_size=num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            prefix="model.layers.0.self_attn.kv_b_proj",
        ).to(device)

        # Create MLAAttention
        self.mla_attn = MLAAttention(
            num_heads=num_heads,
            scale=1.0 / (self.qk_head_dim**0.5),
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
        )
        self.mla_attn._k_scale = self.mla_attn._k_scale.to(device)
        self.mla_attn._v_scale = self.mla_attn._v_scale.to(device)

        # Initialize W_UK_T and W_UV from kv_b_proj weights
        self.mla_attn.process_weights_after_loading(torch.get_default_dtype())

        self.block_size = 16

        # Initialize MLA MetadataBuilder
        self.builder = self.mla_attn.attn_backend.get_builder_cls()(
            kv_cache_spec=MLAAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=self.kv_cache_dtype,
            ),
            layer_names=[self.mla_attn.layer_name],
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def build_attn_metadata(self, batch_size: int) -> AttentionMetadata:
        """Initialize MLA attention metadata."""

        batch_spec = BatchSpec(seq_lens=[1] * batch_size, query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )

        max_blocks = (max(batch_spec.seq_lens) + self.block_size - 1) // self.block_size
        num_blocks = batch_size * max_blocks

        # MLA KV cache is 3D: (num_blocks, block_size, head_size)
        attn_backend = self.mla_attn.attn_backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, 1, self.head_size
        )
        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        ordered_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]

        raw_tensor = torch.zeros(
            ordered_shape, dtype=self.kv_cache_dtype, device=self.device
        )
        kv_cache = raw_tensor.permute(*inv_order)

        self.mla_attn.kv_cache = [kv_cache]

        self.attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return self.attn_metadata


class TestMLAAttentionFp8StaticQuantPatternModel(MLAAttentionQuantPatternModel):
    """Test model for MLA Attention + FP8 static quant fusion."""

    quant_key = kFp8StaticTensorSym

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fp8_linear = TestFP8Layer(
            weight_shape=(self.output_dim, self.output_dim),
            activation_quant_key=self.quant_key,
            weight_quant_key=self.quant_key,
            device=self.device,
        )

        w = kwargs.get("w")
        if w is not None:
            self.fp8_linear.weight = w["weight"]
            self.fp8_linear.weight_scale = w["wscale"]
            self.fp8_linear.input_scale = w["scale"]

        self.w = {
            "weight": self.fp8_linear.weight,
            "wscale": self.fp8_linear.weight_scale,
            "scale": self.fp8_linear.input_scale,
        }

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
    ):
        """Forward pass that creates the MLA attention + FP8 quant pattern."""
        attn_output = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(q.shape[0], self.output_dim),
        )
        return self.fp8_linear(attn_output)


class TestMLAAttentionNvfp4QuantPatternModel(MLAAttentionQuantPatternModel):
    """Test model for MLA Attention + NVFP4 quant fusion."""

    quant_key = kNvfp4Dynamic

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w = kwargs.get(
            "w",
            {
                "weight": torch.randint(
                    256,
                    (self.output_dim, self.output_dim // 2),
                    dtype=FP4_DTYPE,
                    device=self.device,
                ),
                "wscale_swizzled": torch.randn(
                    self.output_dim, self.output_dim // 16
                ).to(dtype=FP8_DTYPE, device=self.device),
                "wscale": torch.tensor([500], dtype=torch.float32, device=self.device),
                "scale": torch.tensor([0.002], dtype=torch.float32, device=self.device),
            },
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
    ):
        """Forward pass that creates the MLA attention + NVFP4 quant pattern."""
        attn_output = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(q.shape[0], self.output_dim),
        )
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


# MLA test configuration
MLA_DIMS: list[tuple[int, int, int, int, int]] = []
PATTERN_TEST_MODELS_MLA_FP8: list[tuple[str, type]] = []
PATTERN_TEST_MODELS_MLA_FP4: list[tuple[str, type]] = []
BACKENDS_MLA_FP8: list[AttentionBackendEnum] = []
BACKENDS_MLA_FP4: list[AttentionBackendEnum] = []

if current_platform.is_cuda():
    # (num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank)
    MLA_DIMS = [(16, 128, 64, 128, 512)]
    PATTERN_TEST_MODELS_MLA_FP8 = [
        (
            "deepseek-ai/DeepSeek-V2-Lite",
            TestMLAAttentionFp8StaticQuantPatternModel,
        )
    ]
    PATTERN_TEST_MODELS_MLA_FP4 = [
        (
            "deepseek-ai/DeepSeek-V2-Lite",
            TestMLAAttentionNvfp4QuantPatternModel,
        )
    ]
    BACKENDS_MLA_FP8 = [AttentionBackendEnum.TRITON_MLA]
    BACKENDS_MLA_FP4 = []  # TODO: add when FP4 MLA backends are available


@pytest.mark.parametrize(
    "num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank",
    MLA_DIMS,
)
@pytest.mark.parametrize("batch_size", [7, 256] if current_platform.is_cuda() else [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "backend, model_name, model_class, custom_ops",
    list(
        flat_product(
            BACKENDS_MLA_FP8,
            PATTERN_TEST_MODELS_MLA_FP8,
            ["+quant_fp8", "-quant_fp8"],
        )
    )
    + list(flat_product(BACKENDS_MLA_FP4, PATTERN_TEST_MODELS_MLA_FP4, [""])),
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test ROCm or CUDA"
)
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
def test_mla_attention_quant_pattern(
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    batch_size: int,
    dtype: torch.dtype,
    custom_ops: str,
    model_name: str,
    model_class: type[MLAAttentionQuantPatternModel],
    backend: AttentionBackendEnum,
    dist_init,
    monkeypatch,
    use_fresh_inductor_cache,
):
    """Test MLA AttentionQuantPattern fusion pass"""
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

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
        cache_config=CacheConfig(cache_dtype="auto"),
        attention_config=AttentionConfig(backend=backend),
    )

    # MLA inputs: q(B, N, qk_head_dim), kv_c_normed(B, L), k_pe(B, 1, R)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q = torch.randn(batch_size, num_heads, qk_head_dim, dtype=dtype, device=device)
    kv_c_normed = torch.randn(batch_size, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(batch_size, 1, qk_rope_head_dim, dtype=dtype, device=device)

    # Mark first dimension as dynamic
    torch._dynamo.mark_dynamic(q, 0)
    torch._dynamo.mark_dynamic(kv_c_normed, 0)
    torch._dynamo.mark_dynamic(k_pe, 0)

    # Run model without fusion
    vllm_config_unfused = copy.deepcopy(vllm_config)
    with (
        set_current_vllm_config(vllm_config_unfused),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config_unfused),
    ):
        model_unfused = model_class(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            kv_cache_dtype=dtype,
            device=device,
            vllm_config=vllm_config_unfused,
        )
        model_unfused = model_unfused.to(device)
        # HACK: See #131044
        result_unfused_0 = model_unfused(q, kv_c_normed, k_pe)  # noqa: F841

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_unfused.build_attn_metadata(batch_size)

        compiled_unfused = torch.compile(model_unfused, fullgraph=True)
        result_unfused = compiled_unfused(q, kv_c_normed, k_pe)

    # Run model with attn fusion enabled
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_attn_quant=True, eliminate_noops=True
    )
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
    ):
        model_fused = model_class(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            kv_cache_dtype=dtype,
            device=device,
            vllm_config=vllm_config,
            w=model_unfused.w,
        )
        model_fused = model_fused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_fused.build_attn_metadata(batch_size)

        # Create test backend with fusion passes
        noop_pass = NoOpEliminationPass(vllm_config)
        attn_pass = LazyInitPass(MLAAttnFusionPass, vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        test_backend = TestBackend(noop_pass, attn_pass, cleanup_pass)
        # HACK: See https://github.com/vllm-project/vllm/issues/31044
        result_fused_0 = model_fused(q, kv_c_normed, k_pe)  # noqa: F841

        compiled_fused = torch.compile(
            model_fused, backend=test_backend, fullgraph=True
        )

        result_fused = compiled_fused(q, kv_c_normed, k_pe)

    # Check attn fusion support
    quant_key: QuantKey = model_class.quant_key
    attn_fusion_supported = [
        layer.impl.fused_output_quant_supported(quant_key)
        for key, layer in vllm_config.compilation_config.static_forward_context.items()
        if isinstance(layer, MLAAttention)
    ]
    assert sum(attn_fusion_supported) == len(attn_fusion_supported), (
        "All MLA layers should support attention fusion"
    )

    # Check quantization ops in the graph
    quant_op = (
        torch.ops.aten.reciprocal
        if "-quant_fp8" in custom_ops_list
        else QUANT_OPS[quant_key]
    )
    test_backend.check_before_ops([quant_op], fully_replaced=quant_key is kNvfp4Dynamic)

    assert attn_pass.pass_.matched_count == sum(attn_fusion_supported)

    # Check MLA attention ops in the graph
    attn_nodes_pre = list(find_op_nodes(MLA_ATTN_OP, test_backend.graph_pre_pass))
    attn_nodes_post = list(find_op_nodes(MLA_ATTN_OP, test_backend.graph_post_pass))

    assert len(attn_nodes_pre) > 0, "Should have MLA attention nodes before fusion"
    assert len(attn_nodes_pre) == len(attn_nodes_post), (
        "Should have same number of MLA attention nodes before and after fusion"
    )
    assert attn_nodes_pre[0].kwargs.get("output_scale") is None, (
        "MLA attention should not have output_scale before fusion"
    )
    assert attn_nodes_post[0].kwargs.get("output_scale") is not None, (
        "MLA attention should have output_scale after fusion"
    )

    assert attn_nodes_pre[0].kwargs.get("output_block_scale") is None, (
        "MLA attention should not have output_block_scale before fusion"
    )

    if quant_key.dtype == FP8_DTYPE:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is None, (
            "MLA attention should not have output_block_scale after FP8 fusion"
        )
    elif quant_key.dtype == FP4_DTYPE:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is not None, (
            "MLA attention should have output_block_scale after FP4 fusion"
        )

    # Check numerical correctness
    torch.testing.assert_close(result_unfused, result_fused, atol=1e-2, rtol=1e-2)
