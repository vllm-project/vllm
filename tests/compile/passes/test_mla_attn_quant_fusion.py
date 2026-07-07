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
    MLAAttnQuantFusionPass,
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
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    create_fp8_quant_key,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8
DEVICE_TYPE = current_platform.device_type


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
        self.dtype = vllm_config.model_config.dtype

        kv_b_proj = ColumnParallelLinear(
            input_size=kv_lora_rank,
            output_size=num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            prefix="model.layers.0.self_attn.kv_b_proj",
        ).to(device)
        kv_b_proj_weight = kwargs.get("kv_b_proj_weight")
        if kv_b_proj_weight is not None:
            kv_b_proj.weight.data.copy_(kv_b_proj_weight)
        else:
            kv_b_proj.weight.data.normal_()

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
            quant_config=self.quant_config,
            prefix="model.layers.0.self_attn.attn",
        )
        self.mla_attn._k_scale = self.mla_attn._k_scale.to(device)
        self.mla_attn._v_scale = self.mla_attn._v_scale.to(device)

        # Initialize W_UK_T and W_UV from kv_b_proj weights
        self.mla_attn.process_weights_after_loading(torch.get_default_dtype())
        self.kv_b_proj_weight = kv_b_proj.weight.data.clone()

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

    def build_attn_metadata(
        self, batch_size: int, query_len: int = 1
    ) -> AttentionMetadata:
        """Initialize MLA attention metadata.

        ``query_len == 1`` is a decode-only batch (forward_mqa). ``query_len > 1``
        is a pure-prefill batch (seq_len == query_len, no context) which routes
        through forward_mha — needed to exercise the fused per-group output path.
        """

        batch_spec = BatchSpec(
            seq_lens=[query_len] * batch_size, query_lens=[query_len] * batch_size
        )
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

        self.mla_attn.kv_cache = kv_cache

        self.attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return self.attn_metadata


class TestMLAAttentionFp8StaticQuantPatternModel(MLAAttentionQuantPatternModel):
    """Test model for MLA Attention + FP8 static quant fusion."""

    quant_key = kFp8StaticTensorSym
    quant_config = Fp8Config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fp8_linear = TestFP8Layer(
            weight_shape=(self.output_dim, self.output_dim),
            activation_quant_key=self.quant_key,
            weight_quant_key=self.quant_key,
            device=self.device,
            input_dtype=self.dtype,
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
    quant_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=False,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

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


class TestMLAAttentionFp8GroupQuantPatternModel(MLAAttentionQuantPatternModel):
    """Test model for MLA Attention + per-group FP8 (block quant) fusion."""

    quant_key = kFp8Dynamic128Sym
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    # o_proj block-scaled MM kernel; subclasses override to change the scale layout.
    block_kernel = CutlassFp8BlockScaledMMKernel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        weight_quant_key = create_fp8_quant_key(
            static=True, group_shape=GroupShape(128, 128)
        )
        device = kwargs.get("device", torch.device("cuda:0"))

        # Subclass to set weight_block_size before process_weights_after_loading
        class _BlockFP8Layer(TestFP8Layer):
            def __init__(self, *a, **kw):
                self.weight_block_size = [128, 128]
                super().__init__(*a, **kw)

        # Force a block-scaled kernel that emits per_token_group_fp8_quant (not the
        # deepgemm packed variant) so the fusion pattern matches.
        self.block_fp8_linear = _BlockFP8Layer(
            weight_shape=(self.output_dim, self.output_dim),
            activation_quant_key=self.quant_key,
            weight_quant_key=weight_quant_key,
            input_dtype=self.dtype,
            device=device,
            force_kernel=self.block_kernel,
        )

        w = kwargs.get("w")
        if w is not None:
            self.block_fp8_linear.weight = w["weight"]
            # Block-wise uses weight_scale_inv, not weight_scale
            self.block_fp8_linear.weight_scale_inv = w["wscale"]

        self.w = {
            "weight": self.block_fp8_linear.weight,
            "wscale": self.block_fp8_linear.weight_scale_inv,
        }

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
    ):
        """Forward pass: MLA attention -> block FP8 linear (group quant)."""
        attn_output = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(q.shape[0], self.output_dim),
        )
        return self.block_fp8_linear(attn_output)


class TestMLAAttentionFp8GroupQuantPatternModelTriton(
    TestMLAAttentionFp8GroupQuantPatternModel
):
    """Per-group FP8 with the Triton block-scaled o_proj, which produces plain row-major
    (non-ue8m0, non-col-major) scales. That layout satisfies the FA4 fused-output gate
    (col == ue8m0 == tma), so the prefill forward_mha fused path engages."""

    block_kernel = TritonFp8BlockScaledMMKernel


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


# MLA test configuration
MLA_DIMS: list[tuple[int, int, int, int, int]] = []
PATTERN_TEST_MODELS_MLA_FP8: list[tuple[str, type]] = []
PATTERN_TEST_MODELS_MLA_GROUP_FP8: list[tuple[str, type]] = []
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
    PATTERN_TEST_MODELS_MLA_GROUP_FP8 = [
        (
            "deepseek-ai/DeepSeek-V3",
            TestMLAAttentionFp8GroupQuantPatternModel,
        )
    ]
    PATTERN_TEST_MODELS_MLA_FP4 = [
        (
            "deepseek-ai/DeepSeek-V2-Lite",
            TestMLAAttentionNvfp4QuantPatternModel,
        )
    ]
    BACKENDS_MLA_FP8 = [AttentionBackendEnum.TRITON_MLA]
    BACKENDS_MLA_FP4 = [AttentionBackendEnum.TRITON_MLA]


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
    + list(
        flat_product(
            BACKENDS_MLA_FP8,
            PATTERN_TEST_MODELS_MLA_GROUP_FP8,
            ["+quant_fp8"],
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
    if (
        model_class is TestMLAAttentionNvfp4QuantPatternModel
        and not is_nvfp4_supported()
    ):
        pytest.skip("NVFP4 is not supported on this GPU (requires SM 100+).")

    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    custom_ops_list = custom_ops.split(",") if custom_ops else []

    device = torch.device(f"{DEVICE_TYPE}:0")
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
            kv_b_proj_weight=model_unfused.kv_b_proj_weight,
        )
        model_fused = model_fused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_fused.build_attn_metadata(batch_size)

        # Create test backend with fusion passes
        noop_pass = NoOpEliminationPass(vllm_config)
        attn_pass = LazyInitPass(MLAAttnQuantFusionPass, vllm_config)
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
    is_per_group = quant_key.scale.group_shape.is_per_group()
    quant_op = (
        torch.ops.aten.reciprocal
        if "-quant_fp8" in custom_ops_list
        else QUANT_OPS[quant_key]
    )
    test_backend.check_before_ops([quant_op], fully_replaced=is_per_group)

    assert attn_pass.pass_.matched_count == sum(attn_fusion_supported)

    # Check MLA attention ops in the graph
    attn_nodes_pre = list(find_op_nodes(MLA_ATTN_OP, test_backend.graph_pre_pass))
    attn_nodes_post = list(find_op_nodes(MLA_ATTN_OP, test_backend.graph_post_pass))

    assert len(attn_nodes_pre) > 0, "Should have MLA attention nodes before fusion"
    assert len(attn_nodes_pre) == len(attn_nodes_post), (
        "Should have same number of MLA attention nodes before and after fusion"
    )

    # Before fusion: neither scale should be set
    assert attn_nodes_pre[0].kwargs.get("output_scale") is None
    assert attn_nodes_pre[0].kwargs.get("output_block_scale") is None

    # After fusion: derive expected scale presence from quant_key properties.
    # - output_scale: present for static quant or non-FP8 (NVFP4 carries input_scale)
    # - output_block_scale: present when quant uses per-group/block scaling
    has_output_scale = attn_nodes_post[0].kwargs.get("output_scale") is not None
    has_block_scale = attn_nodes_post[0].kwargs.get("output_block_scale") is not None

    expects_output_scale = quant_key.scale.static or quant_key.dtype != FP8_DTYPE
    assert has_output_scale == expects_output_scale, (
        f"output_scale: expected present={expects_output_scale}, got {has_output_scale}"
    )
    assert has_block_scale == is_per_group, (
        f"output_block_scale: expected present={is_per_group}, got {has_block_scale}"
    )

    # Check numerical correctness
    torch.testing.assert_close(result_unfused, result_fused, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="FA4 fused per-group output requires Blackwell (SM100/SM110).",
)
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("query_len", [32])
def test_mla_prefill_pergroup_fused_output(
    batch_size: int,
    query_len: int,
    dist_init,
    monkeypatch,
    use_fresh_inductor_cache,
):
    """Exercise the FA4 *prefill* per-group FP8 fused-output path (forward_mha).

    ``test_mla_attention_quant_pattern`` is decode-only (forward_mqa -> post-quant). A
    prefill batch routes through ``forward_mha``, and the Triton block-scaled o_proj
    (plain row-major scales) makes ``forward_impl``'s layout gate engage the fused path.

    Asserts (a) the fused path ran — the FA prefill call received ``output_scales`` —
    and (b) the fused fp8 output is correct: dequantizing it matches a bf16 reference
    attention on the same inputs within fp8 per-group tolerance. The downstream o_proj
    GEMM's *consumption* of the scales is a separate concern that this synthetic harness
    can't set up faithfully (the block-scaled GEMM needs real weight processing), so the
    GEMM result itself is not asserted.
    """
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    num_heads, qk_nope, qk_rope, v_head_dim, kv_lora = 16, 128, 64, 128, 512
    qk_head_dim = qk_nope + qk_rope
    num_tokens = batch_size * query_len
    dtype = torch.bfloat16
    device = torch.device(f"{DEVICE_TYPE}:0")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    model_config = ModelConfig(
        model="deepseek-ai/DeepSeek-V3", max_model_len=2048, dtype=dtype
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=1024,
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["+quant_fp8"]
        ),
        cache_config=CacheConfig(cache_dtype="auto"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_MLA),
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_attn_quant=True, eliminate_noops=True
    )

    q = torch.randn(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    kv_c_normed = torch.randn(num_tokens, kv_lora, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, 1, qk_rope, dtype=dtype, device=device)
    torch._dynamo.mark_dynamic(q, 0)
    torch._dynamo.mark_dynamic(kv_c_normed, 0)
    torch._dynamo.mark_dynamic(k_pe, 0)

    # Spy on the FA prefill call: confirm the fused path ran (output_scales passed) and
    # that the fused fp8 output dequantizes to a bf16 reference attention (same inputs).
    orig_run = FlashAttnPrefillBackend.run_prefill_new_tokens
    fused_ran = False
    fa_err: dict = {}

    def spy_run(self, *args, out=None, output_scale=None, output_scales=None, **kwargs):
        nonlocal fused_ran
        if output_scales is None:
            return orig_run(self, *args, out=out, output_scale=output_scale, **kwargs)
        fused_ran = True
        bf16_ref = orig_run(self, *args, out=None, output_scale=None, **kwargs)
        ret = orig_run(
            self,
            *args,
            out=out,
            output_scale=output_scale,
            output_scales=output_scales,
            **kwargs,
        )
        deq = out.float() * output_scales.float()
        fa_err["abs"] = (deq - bf16_ref.float()).abs().max().item()
        fa_err["amax"] = bf16_ref.float().abs().max().item()
        return ret

    monkeypatch.setattr(FlashAttnPrefillBackend, "run_prefill_new_tokens", spy_run)

    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
    ):
        model = TestMLAAttentionFp8GroupQuantPatternModelTriton(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora,
            kv_cache_dtype=dtype,
            device=device,
            vllm_config=vllm_config,
        ).to(device)
        model(q, kv_c_normed, k_pe)  # HACK: warmup, see #131044
        get_forward_context().attn_metadata = model.build_attn_metadata(
            batch_size, query_len
        )
        test_backend = TestBackend(
            NoOpEliminationPass(vllm_config),
            LazyInitPass(MLAAttnQuantFusionPass, vllm_config),
            PostCleanupPass(vllm_config),
        )
        # forward_impl -> forward_mha (prefill) -> FA fused output -> o_proj GEMM.
        torch.compile(model, backend=test_backend, fullgraph=True)(q, kv_c_normed, k_pe)

    assert fused_ran, (
        "fused per-group prefill path was not exercised: the FA prefill call never "
        "received output_scales (forward_mha fell back to the post-quant path)"
    )
    # fp8-e4m3 per-group quant: max abs error is a few % of the block amax.
    assert fa_err["abs"] <= 0.1 * fa_err["amax"], (
        f"fused per-group fp8 output diverges from bf16 attention: "
        f"max_abs={fa_err['abs']:.3f} vs amax={fa_err['amax']:.3f}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="FA4 fused NVFP4 output requires Blackwell (SM100/SM110).",
)
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("query_len", [32])
def test_mla_prefill_nvfp4_fused_output(
    batch_size: int,
    query_len: int,
    dist_init,
    monkeypatch,
    use_fresh_inductor_cache,
):
    """Exercise the FA4 *prefill* NVFP4 fused-output path (forward_mha).

    Mirrors ``test_mla_prefill_pergroup_fused_output`` for NVFP4: a pure-prefill
    batch routes through ``forward_mha`` and FA4 writes packed e2m1 codes plus the
    128x4-swizzled e4m3 block scales directly. Asserts (a) the fused path ran — the
    FA prefill call received ``output_scales`` — and (b) dequantizing the fused
    NVFP4 output matches a bf16 reference attention within fp4 tolerance.
    """
    from tests.kernels.quantization.nvfp4_utils import dequantize_nvfp4_to_dtype

    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

    num_heads, qk_nope, qk_rope, v_head_dim, kv_lora = 16, 128, 64, 128, 512
    qk_head_dim = qk_nope + qk_rope
    num_tokens = batch_size * query_len
    dtype = torch.bfloat16
    device = torch.device(f"{DEVICE_TYPE}:0")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    model_config = ModelConfig(
        model="deepseek-ai/DeepSeek-V3", max_model_len=2048, dtype=dtype
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=1024,
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["+quant_fp8"]
        ),
        cache_config=CacheConfig(cache_dtype="auto"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_MLA),
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_attn_quant=True, eliminate_noops=True
    )

    q = torch.randn(num_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    kv_c_normed = torch.randn(num_tokens, kv_lora, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, 1, qk_rope, dtype=dtype, device=device)
    torch._dynamo.mark_dynamic(q, 0)
    torch._dynamo.mark_dynamic(kv_c_normed, 0)
    torch._dynamo.mark_dynamic(k_pe, 0)

    orig_run = FlashAttnPrefillBackend.run_prefill_new_tokens
    fused_ran = False
    fa_err: dict = {}

    def spy_run(self, *args, out=None, output_scale=None, output_scales=None, **kwargs):
        nonlocal fused_ran
        if output_scales is None:
            return orig_run(self, *args, out=out, output_scale=output_scale, **kwargs)
        fused_ran = True
        bf16_ref = orig_run(self, *args, out=None, output_scale=None, **kwargs)
        ret = orig_run(
            self,
            *args,
            out=out,
            output_scale=output_scale,
            output_scales=output_scales,
            **kwargs,
        )
        deq = dequantize_nvfp4_to_dtype(
            out.view(torch.uint8).flatten(start_dim=-2),
            output_scales,
            output_scale,
            torch.float32,
            out.device,
        )
        ref = bf16_ref.float().flatten(start_dim=-2)
        fa_err["abs"] = (deq - ref).abs().max().item()
        fa_err["amax"] = ref.abs().max().item()
        return ret

    monkeypatch.setattr(FlashAttnPrefillBackend, "run_prefill_new_tokens", spy_run)

    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
    ):
        model = TestMLAAttentionNvfp4QuantPatternModel(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora,
            kv_cache_dtype=dtype,
            device=device,
            vllm_config=vllm_config,
        ).to(device)
        model(q, kv_c_normed, k_pe)  # HACK: warmup, see #131044
        get_forward_context().attn_metadata = model.build_attn_metadata(
            batch_size, query_len
        )
        test_backend = TestBackend(
            NoOpEliminationPass(vllm_config),
            LazyInitPass(MLAAttnQuantFusionPass, vllm_config),
            PostCleanupPass(vllm_config),
        )
        # forward_impl -> forward_mha (prefill) -> FA fused NVFP4 output -> o_proj GEMM.
        torch.compile(model, backend=test_backend, fullgraph=True)(q, kv_c_normed, k_pe)

    assert fused_ran, (
        "fused NVFP4 prefill path was not exercised: the FA prefill call never "
        "received output_scales (forward_mha fell back to the post-quant path)"
    )
    # e2m1 has a 1-bit mantissa: per-element quant error is bounded by half the gap
    # to the next grid point (<= block_amax / 6), plus e4m3 scale rounding.
    assert fa_err["abs"] <= 0.25 * fa_err["amax"], (
        f"fused NVFP4 output diverges from bf16 attention: "
        f"max_abs={fa_err['abs']:.3f} vs amax={fa_err['amax']:.3f}"
    )
