# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch._dynamo

from tests.compile.backend import LazyInitPass, TestBackend
from tests.utils import TestFP8Layer, flat_product
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.passes.fusion.attn_output_gate_fusion import (
    AttnOutputGateFusionPass,
)
from vllm.compilation.passes.fusion.attn_quant_fusion import (
    ATTN_OP,
    AttnQuantFusionPass,
)
from vllm.compilation.passes.fusion.matcher_utils import QUANT_OPS
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
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    get_kv_quant_mode,
)

DEVICE_TYPE = current_platform.device_type
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


class AttentionQuantPatternModel(torch.nn.Module):
    """Base model for AttentionQuantPattern fusion."""

    def __init__(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
        vllm_config: VllmConfig,
        block_size: int,
        **kwargs,
    ):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.device = device
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype

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

        self.block_size = block_size

        # Initialize attn MetadataBuilder (match Attention.get_kv_cache_spec)
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.attn.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.attn.kv_cache_dtype),
            ),
            layer_names=[self.attn.layer_name],
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def build_attn_metadata(self, batch_size: int) -> AttentionMetadata:
        """Initialize attention metadata."""
        # TODO (Rohan138) reuse utils from vllm/v1/worker/gpu/attn_utils.py

        # Create common attn metadata
        batch_spec = BatchSpec(seq_lens=[1] * batch_size, query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )

        max_blocks = (max(batch_spec.seq_lens) + self.block_size - 1) // self.block_size
        num_blocks = batch_size * max_blocks

        spec = self.attn.attn_backend.customize_spec(
            AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.attn.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.attn.kv_cache_dtype),
            )
        )
        supported = get_supported_kv_cache_layouts([self.attn.attn_backend])
        layout = resolve_kv_cache_layout(
            self.vllm_config, [[m.name for m in supported]]
        )
        raw_tensor = torch.zeros(
            num_blocks * spec.page_size_bytes,
            dtype=torch.int8,
            device=self.device,
        )
        kv_cache = dense_kv_cache_views(
            raw_tensor,
            spec,
            num_blocks,
            num_layers=1,
            layout=layout,
        )[0]

        self.attn.kv_cache = kv_cache

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

        hidden_size = self.num_qo_heads * self.head_size
        self.fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass that creates the pattern to be fused."""
        attn_output = self.attn(q, k, v)
        return self.fp8_linear(attn_output)


class TestAttentionAiterFp8StaticQuantPatternModel(
    TestAttentionFp8StaticQuantPatternModel
):
    aiter_quant = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp8_linear.kernel.quant_fp8.use_aiter = True
        assert self.fp8_linear.input_scale is not None
        self.fp8_linear.input_scale.fill_(0.125)


class TestAttentionNvfp4QuantPatternModel(AttentionQuantPatternModel):
    """Test model for AttentionNvfp4QuantPattern fusion."""

    quant_key = kNvfp4Dynamic

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


PATTERN_TEST_MODELS_FP8: list[tuple[str, type]] = []
PATTERN_TEST_MODELS_FP4: list[tuple[str, type]] = []
HEADS: list[tuple[int, int]] = []
BACKENDS_FP8: list[AttentionBackendEnum] = []
BACKENDS_FP4: list[AttentionBackendEnum] = []

if current_platform.is_cuda():
    HEADS = [(64, 8), (40, 8)]
    PATTERN_TEST_MODELS_FP8 = [
        (
            "RedHatAI/Meta-Llama-3.1-8B-FP8",
            TestAttentionFp8StaticQuantPatternModel,
        )
    ]
    PATTERN_TEST_MODELS_FP4 = [
        (
            "nvidia/Llama-3.1-8B-Instruct-NVFP4",
            TestAttentionNvfp4QuantPatternModel,
        )
    ]
    BACKENDS_FP8 = [AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.FLASHINFER]
    BACKENDS_FP4 = [AttentionBackendEnum.FLASHINFER]

elif current_platform.is_rocm():
    HEADS = [(32, 8), (40, 8)]
    PATTERN_TEST_MODELS_FP8 = [
        ("amd/Llama-3.1-8B-Instruct-FP8-KV", TestAttentionFp8StaticQuantPatternModel)
    ]
    BACKENDS_FP8 = [
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
@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "nvfp4"])
@pytest.mark.parametrize(
    "backend, model_name, model_class, custom_ops",
    # Test attention+quant_fp8 fusion with custom and torch impls of QuantFP8
    list(
        flat_product(
            BACKENDS_FP8, PATTERN_TEST_MODELS_FP8, ["+quant_fp8", "-quant_fp8"]
        )
    )
    # quant_fp4 only has the custom impl
    + list(flat_product(BACKENDS_FP4, PATTERN_TEST_MODELS_FP4, [""])),
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
    kv_cache_dtype: CacheDType,
    custom_ops: str,
    model_name: str,
    model_class: type[AttentionQuantPatternModel],
    backend: AttentionBackendEnum,
    dist_init,
    disable_vllm_compile_cache,
):
    """Test AttentionStaticQuantPattern fusion pass."""
    if backend == AttentionBackendEnum.FLASHINFER and (
        not (current_platform.is_device_capability_family(100)) or not has_flashinfer()
    ):
        # This also captures the FP4 case
        pytest.skip("FlashInfer attn fusion requires SM100/SM107 and flashinfer")
    if kv_cache_dtype == "nvfp4" and backend != AttentionBackendEnum.FLASHINFER:
        pytest.skip("NVFP4 KV cache is only supported by FlashInfer")

    # trtllm-gen has no NVFP4-KV kernel with NVFP4 output, so with an NVFP4 KV
    # cache the FP4 output fusion must be rejected: the standalone
    # scaled_fp4_quant stays in the graph and the result is unchanged. FP8 output
    # fusion with NVFP4 KV is supported.
    expect_fusion = not (
        kv_cache_dtype == "nvfp4" and model_class.quant_key is kNvfp4Dynamic
    )

    custom_ops_list = custom_ops.split(",") if custom_ops else []

    device = torch.device(f"{DEVICE_TYPE}:0")
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    backend_cls = backend.get_class()
    block_size = backend_cls.get_preferred_block_size(16)

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
        cache_config=CacheConfig(cache_dtype=kv_cache_dtype),
        attention_config=AttentionConfig(backend=backend),
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
    ):
        model_unfused = model_class(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            device=device,
            vllm_config=vllm_config_unfused,
            block_size=block_size,
        )
        model_unfused = model_unfused.to(device)
        result_unfused_0 = model_unfused(q, k, v)  # noqa: F841  HACK: See #131044

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_unfused.build_attn_metadata(batch_size)

        # Run model directly without fusion
        # Still compile so query QuantFP8 has closer numerics
        compiled_unfused = torch.compile(model_unfused, fullgraph=True)
        result_unfused = compiled_unfused(q, k, v)

    # Run model with attn fusion enabled
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_attn_quant=True, eliminate_noops=True
    )
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
    ):
        model_fused = model_class(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            device=device,
            vllm_config=vllm_config,
            w=model_unfused.w,
            block_size=block_size,
        )
        model_fused = model_fused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_fused.build_attn_metadata(batch_size)

        # Create test backend with fusion passes enabled
        noop_pass = NoOpEliminationPass(vllm_config)
        attn_pass = LazyInitPass(AttnQuantFusionPass, vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        test_backend = TestBackend(noop_pass, attn_pass, cleanup_pass)
        # HACK: See https://github.com/vllm-project/vllm/issues/31044
        result_fused_0 = model_fused(q, k, v)  # noqa: F841

        # Compile model with fusion enabled
        compiled_fused = torch.compile(
            model_fused, backend=test_backend, fullgraph=True
        )
        assert compiled_fused.attn._o_scale_float is None

        result_fused = compiled_fused(q, k, v)

        if backend == AttentionBackendEnum.FLASHINFER and expect_fusion:
            # With the Flashinfer backend after the 1st round of the forward
            # pass, output quant scale should be loaded into the attn layer's
            # _o_scale_float, the 2nd round should reuse the loaded
            # _o_scale_float
            assert compiled_fused.attn._o_scale_float is not None
            result_fused_2 = compiled_fused(q, k, v)

            assert compiled_fused.attn._o_scale_float is not None

            torch.testing.assert_close(
                result_unfused, result_fused_2, atol=1e-2, rtol=1e-2
            )
        elif not expect_fusion:
            assert compiled_fused.attn._o_scale_float is None

    # Check attn fusion support
    quant_key: QuantKey = model_class.quant_key
    attn_fusion_supported = [
        layer.impl.fused_output_quant_supported(quant_key)
        for key, layer in vllm_config.compilation_config.static_forward_context.items()
    ]
    if expect_fusion:
        assert sum(attn_fusion_supported) == len(attn_fusion_supported), (
            "All layers should support attention fusion"
        )
    else:
        assert sum(attn_fusion_supported) == 0, (
            "No layer should support attention fusion"
        )

    # Check quantization ops in the graph before and after fusion
    if "-quant_fp8" in custom_ops_list:
        quant_op = torch.ops.aten.reciprocal
    elif getattr(model_class, "aiter_quant", False):
        quant_op = torch.ops.vllm.rocm_aiter_per_tensor_quant.default
    else:
        quant_op = QUANT_OPS[quant_key]

    if expect_fusion:
        # Note: for fp8, fully_replaced=False because query quant ops remain in
        # graph. Only output quant ops are fused into attention.
        test_backend.check_before_ops(
            [quant_op], fully_replaced=quant_key is kNvfp4Dynamic
        )
    else:
        # Fusion rejected: the standalone output quant must survive untouched.
        num_pre = test_backend.op_count(quant_op, before=True)
        assert num_pre > 0, f"Op {quant_op} not found in pre-pass graph"
        assert test_backend.op_count(quant_op) == num_pre, (
            "Output quant op should remain when fusion is rejected"
        )

    # access the underlying `AttnQuantFusionPass` on the `LazyInitPass`
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
    if expect_fusion:
        assert attn_nodes_post[0].kwargs.get("output_scale") is not None, (
            "Attention should have output_scale after fusion"
        )
    else:
        assert attn_nodes_post[0].kwargs.get("output_scale") is None, (
            "Attention should not have output_scale when fusion is rejected"
        )

    assert attn_nodes_pre[0].kwargs.get("output_block_scale") is None, (
        "Attention should not have output_block_scale before fusion"
    )

    kv_cache_dummy_dep_pre_is_none = (
        attn_nodes_pre[0].kwargs.get("kv_cache_dummy_dep") is None
    )
    kv_cache_dummy_dep_post_is_none = (
        attn_nodes_post[0].kwargs.get("kv_cache_dummy_dep") is None
    )
    assert not (kv_cache_dummy_dep_pre_is_none ^ kv_cache_dummy_dep_post_is_none), (
        "The kv_cache_dummy_dep should be consistent before and after fusion"
    )

    if quant_key.dtype == FP8_DTYPE or not expect_fusion:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is None, (
            "Attention should not have output_block_scale unless FP4 fusion happened"
        )
    elif quant_key.dtype == FP4_DTYPE:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is not None, (
            "Attention should have output_block_scale after FP4 fusion"
        )

    # Check that results are close
    torch.testing.assert_close(result_unfused, result_fused, atol=1e-2, rtol=1e-2)
    if getattr(model_class, "aiter_quant", False):
        # Query quantization remains; exactly the output quant op is removed.
        assert test_backend.op_count(quant_op, before=True) == (
            test_backend.op_count(quant_op) + 1
        )
        for model in (model_unfused, model_fused):
            torch.testing.assert_close(
                model.fp8_linear.input_scale,
                torch.full_like(model.fp8_linear.input_scale, 0.125),
                atol=0,
                rtol=0,
            )


@pytest.mark.skipif(
    not current_platform.is_rocm() or not is_aiter_found_and_supported(),
    reason="Requires ROCm AITER attention and quantization",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rocm_aiter_static_quant_attention(
    dtype, dist_init, disable_vllm_compile_cache, monkeypatch
):
    """Fuse the actual AITER op without changing its static scale or outputs."""
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_ROCM_USE_AITER", "1")
            rocm_aiter_ops.refresh_env_variables()
            test_attention_quant_pattern(
                num_qo_heads=32,
                num_kv_heads=8,
                head_size=128,
                batch_size=8,
                dtype=dtype,
                kv_cache_dtype="fp8",
                custom_ops="+quant_fp8",
                model_name="amd/Llama-3.1-8B-Instruct-FP8-KV",
                model_class=TestAttentionAiterFp8StaticQuantPatternModel,
                backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
                dist_init=dist_init,
                disable_vllm_compile_cache=disable_vllm_compile_cache,
            )
    finally:
        rocm_aiter_ops.refresh_env_variables()


class TestAttentionOutputGatePatternModel(torch.nn.Module):
    """Gated attention (Qwen3-Next): q and the gate packed per head as [q | gate]
    (gate_by_head, the model's chunk), or as [q heads | gate heads] (flat), then
    attn * sigmoid(gate). Qwen3.8 full-attention geometry."""

    num_heads, num_kv_heads, head_size = 24, 4, 256

    def __init__(self, vllm_config, device, gate_by_head: bool):
        super().__init__()
        self.block_size = vllm_config.cache_config.block_size
        self.device = device
        self.gate_by_head = gate_by_head
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.head_size**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
        )
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)
        self.kv_cache_spec = self.attn.attn_backend.customize_spec(
            AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=torch.bfloat16,
            )
        )
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=[self.attn.layer_name],
            vllm_config=vllm_config,
            device=device,
        )

    def build_attn_metadata(
        self, num_tokens: int, seq_len: int
    ) -> tuple[AttentionMetadata, torch.Tensor]:
        """Decode batch of num_tokens sequences of seq_len, on a random KV cache;
        returns the metadata and the slot mapping."""
        batch_spec = BatchSpec(
            seq_lens=[seq_len] * num_tokens, query_lens=[1] * num_tokens
        )
        common = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )
        num_blocks = num_tokens * -(-seq_len // self.block_size)
        raw = torch.zeros(
            num_blocks * self.kv_cache_spec.page_size_bytes,
            dtype=torch.int8,
            device=self.device,
        )
        kv_cache = dense_kv_cache_views(
            raw,
            self.kv_cache_spec,
            num_blocks,
            num_layers=1,
            layout=KVCacheLayout.LBHNC,
        )[0]
        generator = torch.Generator(device=self.device).manual_seed(1)
        kv_cache.copy_(
            torch.randn(
                kv_cache.shape,
                generator=generator,
                dtype=kv_cache.dtype,
                device=self.device,
            )
        )
        self.attn.kv_cache = kv_cache
        metadata = self.builder.build(common_prefix_len=0, common_attn_metadata=common)
        return metadata, common.slot_mapping

    def forward(self, q_gate: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        hidden = self.num_heads * self.head_size
        if self.gate_by_head:
            q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_size)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q, gate = q.reshape(-1, hidden), gate.reshape(-1, hidden)
        else:
            q, gate = q_gate.split([hidden, hidden], dim=-1)
        return self.attn(q, k, v) * torch.sigmoid(gate)


def _run_attention_output_gate_fusion(
    backend: AttentionBackendEnum, num_tokens: int, seq_len: int, gate_by_head: bool
):
    """The eager reference and the compiled run with AttnOutputGateFusionPass, of
    one decode batch."""
    device = torch.device(f"{DEVICE_TYPE}:0")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    model_cls = TestAttentionOutputGatePatternModel
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            dtype=torch.bfloat16,
            hf_overrides={
                "num_attention_heads": model_cls.num_heads,
                "num_key_value_heads": model_cls.num_kv_heads,
                "head_dim": model_cls.head_size,
            },
        ),
        cache_config=CacheConfig(block_size=16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(fuse_attn_output_gate=True, eliminate_noops=True),
        ),
        attention_config=AttentionConfig(backend=backend),
    )
    with set_current_vllm_config(vllm_config):
        model = model_cls(vllm_config, device, gate_by_head)
        gate_pass = LazyInitPass(AttnOutputGateFusionPass, vllm_config)
        test_backend = TestBackend(
            NoOpEliminationPass(vllm_config), gate_pass, PostCleanupPass(vllm_config)
        )
        hidden = model_cls.num_heads * model_cls.head_size
        kv_hidden = model_cls.num_kv_heads * model_cls.head_size
        q_gate = torch.randn(num_tokens, 2 * hidden)
        k = torch.randn(num_tokens, kv_hidden)
        v = torch.randn(num_tokens, kv_hidden)

        def run(fn):
            with set_forward_context(None, vllm_config):
                attn_metadata, slot_mapping = model.build_attn_metadata(
                    num_tokens, seq_len
                )
                forward_ctx = get_forward_context()
                forward_ctx.attn_metadata = attn_metadata
                forward_ctx.slot_mapping = {model.attn.layer_name: slot_mapping}
                return fn(q_gate.clone(), k.clone(), v.clone())

        reference = run(model)
        for x in (q_gate, k, v):
            torch._dynamo.mark_dynamic(x, 0)
        fused = run(torch.compile(model, backend=test_backend, fullgraph=True))
    return gate_pass.pass_, test_backend, reference, fused


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test ROCm or CUDA"
)
@pytest.mark.parametrize("gate_by_head", [True, False], ids=["by_head", "flat"])
def test_attention_output_gate_pattern_unsupported_backend(
    gate_by_head: bool, dist_init, disable_vllm_compile_cache
):
    """A backend without fused_output_gate_supported() keeps attn * sigmoid(gate)."""
    gate_pass, backend, reference, fused = _run_attention_output_gate_fusion(
        AttentionBackendEnum.TRITON_ATTN, 7, 37, gate_by_head
    )
    assert gate_pass.matched_count == 0
    assert backend.op_count(torch.ops.aten.sigmoid.default) == 1
    (attn,) = find_op_nodes(ATTN_OP, backend.graph_post_pass)
    assert attn.kwargs.get("output_gate") is None
    torch.testing.assert_close(fused, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not (
        current_platform.is_rocm()
        and is_aiter_found_and_supported()
        and rocm_aiter_ops.pa_decode_gluon_supports_output_gate()
    ),
    reason="Requires ROCm AITER whose pa_decode_gluon takes output_gate",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 128])
@pytest.mark.parametrize("seq_len", [37, 3000])
@pytest.mark.parametrize("gate_by_head", [True, False], ids=["by_head", "flat"])
def test_attention_output_gate_pattern(
    num_tokens: int,
    seq_len: int,
    gate_by_head: bool,
    dist_init,
    disable_vllm_compile_cache,
    monkeypatch,
):
    """ROCM_AITER_FA_GLUON takes the gate in the attention op: no sigmoid left and,
    for the per-head gate, no copy flattening it. 128 tokens runs the one-split
    path, where the impl applies the gate itself."""
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_ROCM_USE_AITER", "1")
            patch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1")
            rocm_aiter_ops.refresh_env_variables()
            gate_pass, backend, reference, fused = _run_attention_output_gate_fusion(
                AttentionBackendEnum.ROCM_AITER_FA_GLUON,
                num_tokens,
                seq_len,
                gate_by_head,
            )
    finally:
        rocm_aiter_ops.refresh_env_variables()
    assert gate_pass.matched_count == 1
    assert backend.op_count(torch.ops.aten.sigmoid.default) == 0
    if gate_by_head:
        clone = torch.ops.aten.clone.default
        assert backend.op_count(clone) < backend.op_count(clone, before=True)
    (attn,) = find_op_nodes(ATTN_OP, backend.graph_post_pass)
    assert attn.kwargs.get("output_gate") is not None
    torch.testing.assert_close(fused, reference, atol=2e-2, rtol=2e-2)
