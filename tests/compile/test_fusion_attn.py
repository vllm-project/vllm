# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Optional

import pytest
import torch._dynamo

from tests.compile.backend import TestBackend
from tests.models.utils import check_outputs_equal
from tests.v1.attention.utils import (BatchSpec, _Backend,
                                      create_common_attn_metadata)
from vllm import LLM, SamplingParams
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.attention import Attention
from vllm.attention.selector import global_force_attn_backend_context_manager
from vllm.compilation.fusion import QUANT_OPS
from vllm.compilation.fusion_attn import ATTN_OP, AttnFusionPass
from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import (CacheConfig, CompilationConfig, CompilationLevel,
                         ModelConfig, PassConfig, SchedulerConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey, kFp8StaticTensorSym, kNvfp4Quant)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp)
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer
from vllm.v1.kv_cache_interface import AttentionSpec

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

# globals needed for string-import custom Dynamo backend field
backend: Optional[TestBackend] = None
backend_unfused: Optional[TestBackend] = None


@pytest.mark.parametrize(
    "model, quant_key",
    [("amd/Llama-3.1-8B-Instruct-FP8-KV", kFp8StaticTensorSym)])
@pytest.mark.parametrize("use_triton_fa", [True, False])
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="V0 attn quant fusion only on ROCm")
def test_attention_fusion_v0(example_prompts, monkeypatch, model: str,
                             quant_key: QuantKey, use_triton_fa: bool):
    # Clean Dynamo cache to avoid reusing other test cases
    # (for some reason the reset at the end is not enough)
    torch._dynamo.reset()

    # Use global backends
    global backend, backend_unfused

    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_USE_TRITON_FLASH_ATTN", str(int(use_triton_fa)))

    # Prompt 4 seems too open-ended, differs between fused and unfused
    # (both outputs look reasonable though)
    prompts = example_prompts[:4] + example_prompts[5:]

    compile_config = CompilationConfig(
        # DYNAMO_AS_IS triggers custom backend & does full Dynamo compilation
        # DYNAMO_ONCE does not properly propagate shapes.
        level=CompilationLevel.DYNAMO_AS_IS,
        backend="tests.compile.test_fusion_attn.backend_unfused",
        custom_ops=["+quant_fp8"],
    )
    vllm_config = VllmConfig(compilation_config=compile_config,
                             model_config=ModelConfig(
                                 model=model,
                                 dtype=torch.bfloat16,
                             ))
    backend_unfused = TestBackend(NoOpEliminationPass(vllm_config))

    llm = LLM(model,
              enforce_eager=True,
              compilation_config=compile_config,
              gpu_memory_utilization=0.5,
              max_model_len=2048)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=10,
                                     top_p=0.95)

    unfused_output = llm.generate(prompts, sampling_params)
    backend_unfused = None  # Reset backend to make sure llm gets released
    del llm

    compile_config = CompilationConfig(
        # DYNAMO_AS_IS triggers custom backend & does full Dynamo compilation
        # DYNAMO_ONCE does not properly propagate shapes.
        level=CompilationLevel.DYNAMO_AS_IS,
        backend="tests.compile.test_fusion_attn.backend",
        custom_ops=["+quant_fp8"],
    )
    vllm_config = VllmConfig(compilation_config=compile_config,
                             model_config=ModelConfig(
                                 model=model,
                                 dtype=torch.bfloat16,
                             ))

    # AttnFusionPass needs attention layers to be registered in config upon init
    # so we initialize it during compilation.
    attn_pass = lambda *args, **kw: AttnFusionPass(vllm_config)(*args, **kw)
    backend = TestBackend(NoOpEliminationPass(vllm_config), attn_pass)
    llm2 = LLM(model,
               enforce_eager=True,
               compilation_config=compile_config,
               gpu_memory_utilization=0.5,
               max_model_len=2048)

    # check support
    attn_fusion_supported = [
        layer.impl.fused_output_quant_supported(quant_key)
        for key, layer in compile_config.static_forward_context.items()
    ]

    print(f"{attn_fusion_supported=}")
    if any(attn_fusion_supported):
        # Check quant ops
        backend.check_before_ops([QUANT_OPS[quant_key]], fully_replaced=False)

    # attention ops present in both, just output_scale param changes
    attn_nodes_pre = list(find_op_nodes(ATTN_OP, backend.graph_pre_pass))
    attn_nodes_post = list(find_op_nodes(ATTN_OP, backend.graph_post_pass))
    assert len(attn_nodes_pre) == len(attn_nodes_post)

    for i in range(len(attn_nodes_pre)):
        assert attn_nodes_pre[i].kwargs["output_scale"] is None
        fused = attn_nodes_post[i].kwargs["output_scale"] is not None
        assert fused == attn_fusion_supported[i], \
            f"Node {i} {'' if fused else 'not '} expected " \
            f"to have fused output quant"

    # check outputs
    fused_output = llm2.generate(prompts, sampling_params)

    # transform outputs to format expected by check_outputs_equal
    sample_outs = lambda s: (list(s.token_ids), s.text)
    outs_lst = lambda ros: [sample_outs(ro.outputs[0]) for ro in ros]

    check_outputs_equal(
        outputs_0_lst=outs_lst(unfused_output),
        outputs_1_lst=outs_lst(fused_output),
        name_0="unfused",
        name_1="fused",
    )

    # Clean Dynamo cache to avoid polluting other case(s)
    torch._dynamo.reset()

    # Reset backend to make sure llm2 gets released
    backend = None


class AttentionQuantPatternModel(torch.nn.Module):
    """Base model for AttentionQuantPattern fusion."""

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_size: int,
                 kv_cache_dtype: torch.dtype, device: torch.device,
                 vllm_config: VllmConfig, **kwargs):
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
                use_mla=False,
            ),
            layer_names=[self.attn.layer_name],
            vllm_config=self.vllm_config,
            device=self.device,
        )

    def build_attn_metadata(self, batch_size: int, use_hnd: bool):
        """Initialize attention metadata."""

        # Create common attn metadata
        batch_spec = BatchSpec(seq_lens=[1] * batch_size,
                               query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec,
            self.block_size,
            self.device,
            arange_block_indices=True)

        max_blocks = (max(batch_spec.seq_lens) + self.block_size -
                      1) // self.block_size
        num_blocks = batch_size * max_blocks

        # Create dummy KV cache for FlashInfer TRTLLM
        #   - NHD: [num_blocks, block_size, num_kv_heads, head_size]
        #   - HND: [num_blocks, num_kv_heads, block_size, head_size]
        kv_cache = torch.zeros(num_blocks,
                               2,
                               self.num_kv_heads,
                               self.block_size,
                               self.head_size,
                               dtype=self.kv_cache_dtype,
                               device=self.device)
        if current_platform.is_rocm():
            # k/v as 1st dimention
            if use_hnd:
                kv_cache = kv_cache.permute(1, 0, 2, 3, 4)
            else:
                kv_cache = kv_cache.permute(1, 0, 3, 2, 4)
        else:
            # k/v as 2nd dimention
            # Create kv_cache in HND layout and permute to NHD layout
            # (later will be permuted back to HND layout in forward pass)
            kv_cache = kv_cache.permute(0, 1, 3, 2, 4)
        self.attn.kv_cache = [kv_cache]

        # Build attn metadata
        self.attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata)

        return self.attn_metadata


class TestAttentionFp8StaticQuantPatternModel(AttentionQuantPatternModel):
    """Test model for AttentionFp8StaticQuantPattern fusion."""

    quant_key = kFp8StaticTensorSym

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fp8_linear = Fp8LinearOp(
            act_quant_static=self.quant_key.scale.static,
            act_quant_group_shape=self.quant_key.scale.group_shape)

        hidden_size = self.num_qo_heads * self.head_size
        self.w = kwargs.get(
            "w", {
                "weight":
                torch.randn(hidden_size, hidden_size).to(
                    dtype=FP8_DTYPE, device=self.device).t(),
                "wscale":
                torch.tensor([1.0], dtype=torch.float32, device=self.device),
                "scale":
                torch.tensor([1.0], dtype=torch.float32, device=self.device),
            })

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass that creates the pattern to be fused."""
        attn_output = self.attn(q, k, v)
        return self.fp8_linear.apply(input=attn_output,
                                     weight=self.w["weight"],
                                     weight_scale=self.w["wscale"],
                                     input_scale=self.w["scale"])


class TestAttentionNvfp4QuantPatternModel(AttentionQuantPatternModel):
    """Test model for AttentionNvfp4QuantPattern fusion."""

    quant_key = kNvfp4Quant

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden_size = self.num_qo_heads * self.head_size
        self.w = kwargs.get(
            "w", {
                "weight":
                torch.randint(256, (hidden_size, hidden_size // 2),
                              dtype=FP4_DTYPE,
                              device=self.device),
                "wscale_swizzled":
                torch.randn(hidden_size, hidden_size // 16).to(
                    dtype=FP8_DTYPE, device=self.device),
                "wscale":
                torch.tensor([500], dtype=torch.float32, device=self.device),
                "scale":
                torch.tensor([0.002], dtype=torch.float32, device=self.device),
            })

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass that creates the pattern to be fused."""
        attn_output = self.attn(q, k, v)
        quant_output, output_block_scale = scaled_fp4_quant(
            attn_output, 1 / self.w["scale"])
        return cutlass_scaled_fp4_mm(a=quant_output,
                                     b=self.w["weight"],
                                     block_scale_a=output_block_scale,
                                     block_scale_b=self.w["wscale_swizzled"],
                                     alpha=self.w["scale"] * self.w["wscale"],
                                     out_dtype=attn_output.dtype)


if current_platform.is_cuda():
    MODELS = [("nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
               TestAttentionFp8StaticQuantPatternModel),
              ("nvidia/Llama-4-Scout-17B-16E-Instruct-FP4",
               TestAttentionNvfp4QuantPatternModel)]
    HEADS = [(64, 8), (40, 8)]
elif current_platform.is_rocm():
    MODELS = [("amd/Llama-3.1-8B-Instruct-FP8-KV",
               TestAttentionFp8StaticQuantPatternModel)]
    HEADS = [(32, 8), (40, 8)]
else:
    MODELS = []
    HEADS = []


@pytest.mark.parametrize("num_qo_heads, num_kv_heads", HEADS)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("batch_size",
                         [7, 256, 533] if current_platform.is_cuda() else [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("model_name, model_class", MODELS)
@pytest.mark.parametrize("backend",
                         [_Backend.FLASHINFER] if current_platform.is_cuda()
                         else [_Backend.TRITON_ATTN_VLLM_V1])
@pytest.mark.parametrize(
    "split_attention",
    [False, True] if current_platform.is_rocm() else [False])
# TODO(boyuan): test inductor graph partition on rocm
@pytest.mark.parametrize(
    "use_inductor_graph_partition",
    [False] if current_platform.is_rocm() else [False, True])
@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="Only test ROCm or CUDA")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.skipif(current_platform.is_cuda()
                    and not current_platform.is_device_capability((10, 0)),
                    reason="On CUDA only test on SM100(Blackwell)")
@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="Only test ROCm or CUDA")
def test_attention_quant_pattern(num_qo_heads: int, num_kv_heads: int,
                                 head_size: int, batch_size: int,
                                 dtype: torch.dtype, model_name: str,
                                 model_class: type[AttentionQuantPatternModel],
                                 backend: _Backend, split_attention: bool,
                                 use_inductor_graph_partition: bool,
                                 monkeypatch, dist_init, caplog_vllm):
    """Test AttentionStaticQuantPattern fusion pass"""

    if use_inductor_graph_partition and not is_torch_equal_or_newer(
            "2.9.0.dev"):
        pytest.skip("inductor graph partition is only available "
                    "in PyTorch 2.9+")

    monkeypatch.setenv("VLLM_USE_V1", "1")
    if split_attention:
        monkeypatch.setenv("VLLM_V1_USE_PREFILL_DECODE_ATTENTION", "1")

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=model_name,
            max_model_len=2048,
            dtype=dtype,
        ),
        scheduler_config=SchedulerConfig(max_num_seqs=1024),
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            custom_ops=["+quant_fp8"],
            use_inductor_graph_partition=use_inductor_graph_partition,
        ),
        cache_config=CacheConfig(cache_dtype="fp8"))

    # Create test inputs
    q = torch.randn(batch_size,
                    num_qo_heads * head_size,
                    dtype=dtype,
                    device=device)
    k = torch.randn(batch_size,
                    num_kv_heads * head_size,
                    dtype=dtype,
                    device=device)
    v = torch.randn(batch_size,
                    num_kv_heads * head_size,
                    dtype=dtype,
                    device=device)

    # Mark first dimension as dynamic for realistic testing
    torch._dynamo.mark_dynamic(q, 0)
    torch._dynamo.mark_dynamic(k, 0)
    torch._dynamo.mark_dynamic(v, 0)

    # Run model directly without compilation and fusion
    vllm_config_unfused = copy.deepcopy(vllm_config)
    with set_current_vllm_config(vllm_config_unfused), set_forward_context(
            attn_metadata=None, vllm_config=vllm_config_unfused
    ), global_force_attn_backend_context_manager(backend):
        model_unfused = model_class(num_qo_heads=num_qo_heads,
                                    num_kv_heads=num_kv_heads,
                                    head_size=head_size,
                                    kv_cache_dtype=FP8_DTYPE,
                                    device=device,
                                    vllm_config=vllm_config_unfused)
        model_unfused = model_unfused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_unfused.build_attn_metadata(
            batch_size, use_hnd=split_attention)

        # Run model directly without compilation and fusion
        result_unfused = model_unfused(q, k, v)

    # Run model with attn fusion enabled
    vllm_config.compilation_config.pass_config = PassConfig(
        enable_attn_fusion=True, enable_noop=True)
    with set_current_vllm_config(vllm_config), set_forward_context(
            attn_metadata=None, vllm_config=vllm_config
    ), global_force_attn_backend_context_manager(backend):
        model_fused = model_class(num_qo_heads=num_qo_heads,
                                  num_kv_heads=num_kv_heads,
                                  head_size=head_size,
                                  kv_cache_dtype=FP8_DTYPE,
                                  device=device,
                                  vllm_config=vllm_config,
                                  w=model_unfused.w)
        model_fused = model_fused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_fused.build_attn_metadata(
            batch_size, use_hnd=split_attention)

        # Create test backend with fusion passes enabled
        noop_pass = NoOpEliminationPass(vllm_config)
        attn_pass = lambda *args, **kw: AttnFusionPass(vllm_config)(*args, **kw
                                                                    )
        test_backend = TestBackend(noop_pass, attn_pass)

        # Compile model with fusion enabled
        model_compiled = torch.compile(model_fused,
                                       backend=test_backend,
                                       fullgraph=True)
        assert model_compiled.attn._o_scale_float is None

        result_fused_1 = model_compiled(q, k, v)

        if backend == _Backend.FLASHINFER:
            # With the Flashinfer backend after the 1st round of the forward
            # pass, output quant scale should be loaded into the attn layer's
            # _o_scale_float, the 2nd round should reuse the loaded
            # _o_scale_float
            assert model_compiled.attn._o_scale_float is not None
            result_fused_2 = model_compiled(q, k, v)

            assert model_compiled.attn._o_scale_float is not None

            torch.testing.assert_close(result_unfused,
                                       result_fused_2,
                                       atol=1e-2,
                                       rtol=1e-2)

    # Check attn fusion support
    quant_key = model_class.quant_key
    attn_fusion_supported = [
        layer.impl.fused_output_quant_supported(quant_key) for key, layer in
        vllm_config.compilation_config.static_forward_context.items()
    ]
    if any(attn_fusion_supported):
        # Check quantization ops in the graph before and after fusion
        test_backend.check_before_ops([QUANT_OPS[quant_key]],
                                      fully_replaced=True)

    # Check attention ops in the graph before and after fusion
    attn_nodes_pre = list(find_op_nodes(ATTN_OP, test_backend.graph_pre_pass))
    attn_nodes_post = list(find_op_nodes(ATTN_OP,
                                         test_backend.graph_post_pass))

    assert len(attn_nodes_pre) > 0, "Should have attention nodes before fusion"
    assert len(attn_nodes_pre) == len(attn_nodes_post), \
        "Should have same number of attention nodes before and after fusion"
    assert attn_nodes_pre[0].kwargs.get("output_scale") is None, \
        "Attention should not have output_scale before fusion"
    assert attn_nodes_post[0].kwargs.get("output_scale") is not None, \
        "Attention should have output_scale after fusion"

    assert attn_nodes_pre[0].kwargs.get("output_block_scale") is None, \
        "Attention should not have output_block_scale before fusion"
    if quant_key.dtype == FP8_DTYPE:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is None, \
            "Attention should not have output_block_scale after FP8 fusion"
    elif quant_key.dtype == FP4_DTYPE:
        assert attn_nodes_post[0].kwargs.get("output_block_scale") is not None, \
            "Attention should have output_block_scale after FP4 fusion"  # noqa: E501

    # Check that results are close
    torch.testing.assert_close(result_unfused,
                               result_fused_1,
                               atol=1e-2,
                               rtol=1e-2)
