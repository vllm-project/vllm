# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Optional

import pytest
import torch._dynamo

from tests.compile.backend import TestBackend
from tests.models.utils import check_outputs_equal
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm import LLM, SamplingParams
from vllm.attention import Attention
from vllm.compilation.fusion import QUANT_OPS, QuantKey, kFp8StaticTensorSym
from vllm.compilation.fusion_attn import ATTN_OP, AttnFusionPass
from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import (CacheConfig, CompilationConfig, CompilationLevel,
                         ModelConfig, PassConfig, SchedulerConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

FP8_DTYPE = current_platform.fp8_dtype()

# globals needed for string-import custom Dynamo backend field
backend: Optional[TestBackend] = None
backend_unfused: Optional[TestBackend] = None


@pytest.mark.parametrize(
    "model, quant_key",
    [("amd/Llama-3.1-8B-Instruct-FP8-KV", kFp8StaticTensorSym)])
@pytest.mark.parametrize(
    "use_triton_fa", [True, False] if current_platform.is_rocm() else [False])
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="Only test CUDA and ROCm")
def test_attention_fusion(example_prompts, monkeypatch, model: str,
                          quant_key: QuantKey, use_triton_fa: bool):
    # Clean Dynamo cache to avoid reusing other test cases
    # (for some reason the reset at the end is not enough)
    torch._dynamo.reset()

    # Use global backends
    global backend, backend_unfused

    use_v1 = False  # can be made a param once V1 support added
    monkeypatch.setenv("VLLM_USE_V1", str(int(use_v1)))
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
    vllm_config = VllmConfig(compilation_config=compile_config)
    backend_unfused = TestBackend(NoOpEliminationPass(vllm_config))

    llm = LLM(model,
              enforce_eager=True,
              compilation_config=compile_config,
              gpu_memory_utilization=0.9,
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
    vllm_config = VllmConfig(compilation_config=compile_config)

    # AttnFusionPass needs attention layers to be registered in config upon init
    # so we initialize it during compilation.
    attn_pass = lambda *args, **kw: AttnFusionPass(vllm_config)(*args, **kw)
    backend = TestBackend(NoOpEliminationPass(vllm_config), attn_pass)
    llm2 = LLM(model,
               enforce_eager=True,
               compilation_config=compile_config,
               gpu_memory_utilization=0.9,
               max_model_len=2048)

    # check support
    attn_fusion_supported = [
        layer.impl.fused_output_quant_supported(quant_key.dtype,
                                                quant_key.static,
                                                quant_key.group_shape)
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


class TestAttentionStaticQuantPatternModel(torch.nn.Module):
    """Test model for AttentionStaticQuantPattern fusion."""

    def __init__(self,
                 num_qo_heads: int,
                 num_kv_heads: int,
                 head_size: int,
                 kv_cache_dtype: torch.dtype,
                 device: torch.device,
                 vllm_config: VllmConfig,
                 w: Optional[torch.Tensor] = None,
                 kv_cache: Optional[torch.Tensor] = None):
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

        self.fp8_linear = Fp8LinearOp(
            act_quant_static=True, act_quant_group_shape=GroupShape.PER_TENSOR)

        hidden_size = num_qo_heads * head_size
        if w is not None:
            self.w = w
        else:
            self.w = torch.randn(hidden_size, hidden_size).to(FP8_DTYPE).t()
        self.wscale = torch.tensor([1.0], dtype=torch.float32)
        self.scale = torch.tensor([1.0], dtype=torch.float32)

        self.block_size = 16
        self.kv_cache = None
        if kv_cache is not None:
            self.kv_cache = kv_cache

    def build_attn_metadata(self, batch_size: int):
        """Initialize Flashinfer attention metadata."""

        # Create common attn metadata
        batch_spec = BatchSpec(seq_lens=[1] * batch_size,
                               query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec,
            self.block_size,
            self.device,
            arange_block_indices=True)

        # Create kv cache
        if self.kv_cache is None:
            max_blocks = (max(batch_spec.seq_lens) + self.block_size -
                          1) // self.block_size
            num_blocks = batch_size * max_blocks

            # Create dummy KV cache for FlashInfer TRTLLM
            #   - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
            #   - HND: [num_blocks, 2,  num_kv_heads, block_size, head_size]
            # Create kv_cache in HND layout and permute to NHD layout
            # (later will be permuted back to HND layout in forward pass)
            self.kv_cache = torch.randn(num_blocks, 2, self.num_kv_heads,
                                        self.block_size, self.head_size).to(
                                            dtype=self.kv_cache_dtype,
                                            device=self.device)
            self.kv_cache = self.kv_cache.permute(0, 1, 3, 2, 4)

        self.attn.kv_cache = [self.kv_cache]

        # Initialize FlashInferMetadataBuilder
        builder = FlashInferMetadataBuilder(
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

        # Build FlashInferMetadata
        self.attn_metadata = builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata)

        return self.attn_metadata

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass that creates the pattern to be fused."""
        attn_output = self.attn(q, k, v)
        return self.fp8_linear.apply(input=attn_output,
                                     weight=self.w,
                                     weight_scale=self.wscale,
                                     input_scale=self.scale)


@pytest.mark.parametrize("num_heads", [(64, 8), (40, 8)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("batch_size", [7, 256, 533])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "model_quant_dtype",
    [("nvidia/Llama-4-Scout-17B-16E-Instruct-FP8", FP8_DTYPE)])
@pytest.mark.parametrize("test_backend", ["FLASHINFER"])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test CUDA")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.skipif(not current_platform.is_device_capability((10, 0)),
                    reason="Only test on SM100(Blackwell)")
def test_attention_quant_pattern(num_heads: tuple[int, int], head_size: int,
                                 batch_size: int, dtype: torch.dtype,
                                 model_quant_dtype: tuple[str, torch.dtype],
                                 test_backend, monkeypatch, dist_init):
    """Test AttentionStaticQuantPattern fusion pass"""

    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", test_backend)

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    num_qo_heads, num_kv_heads = num_heads
    model_name, quant_dtype = model_quant_dtype

    # The quant op to check the fusion happenes or not
    quant_op = None
    if quant_dtype == FP8_DTYPE:
        quant_op = QUANT_OPS[kFp8StaticTensorSym]
    else:
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}")

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=model_name,
            max_model_len=2048,
        ),
        scheduler_config=SchedulerConfig(max_num_seqs=1024),
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            custom_ops=["+quant_fp8"],
            full_cuda_graph=True,
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
            attn_metadata=None, vllm_config=vllm_config_unfused):
        model_unfused = TestAttentionStaticQuantPatternModel(
            num_qo_heads, num_kv_heads, head_size, FP8_DTYPE, device,
            vllm_config_unfused)
        model_unfused = model_unfused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_unfused.build_attn_metadata(
            batch_size)

        # Run model directly without compilation and fusion
        result_unfused = model_unfused(q, k, v)

    # Run model with attn fusion enabled
    vllm_config.compilation_config.pass_config = PassConfig(
        enable_attn_fusion=True, enable_noop=True)
    with set_current_vllm_config(vllm_config), set_forward_context(
            attn_metadata=None, vllm_config=vllm_config):
        model_fused = TestAttentionStaticQuantPatternModel(
            num_qo_heads, num_kv_heads, head_size, FP8_DTYPE, device,
            vllm_config, model_unfused.w, model_unfused.kv_cache)
        model_fused = model_fused.to(device)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model_fused.build_attn_metadata(batch_size)

        # Create test backend with fusion passes enabled
        noop_pass = NoOpEliminationPass(vllm_config)
        attn_pass = lambda *args, **kw: AttnFusionPass(vllm_config)(*args, **kw
                                                                    )
        backend = TestBackend(noop_pass, attn_pass)

        # Compile model with fusion enabled
        model_compiled = torch.compile(model_fused,
                                       backend=backend,
                                       fullgraph=True)
        result_fused = model_compiled(q, k, v)

    # Check quantization ops in the graph before and after fusion
    backend.check_before_ops([quant_op], fully_replaced=True)

    # Check attention ops in the graph before and after fusion
    attn_nodes_pre = list(find_op_nodes(ATTN_OP, backend.graph_pre_pass))
    attn_nodes_post = list(find_op_nodes(ATTN_OP, backend.graph_post_pass))

    assert len(attn_nodes_pre) > 0, "Should have attention nodes before fusion"
    assert len(attn_nodes_pre) == len(attn_nodes_post), \
        "Should have same number of attention nodes before and after fusion"
    assert attn_nodes_pre[0].kwargs.get("output_scale") is None, \
        "Attention should not have output_scale before fusion"
    assert attn_nodes_post[0].kwargs.get("output_scale") is not None, \
        "Attention should have output_scale after fusion"

    # Check that results are closed
    torch.testing.assert_close(result_unfused.to(dtype),
                               result_fused.to(dtype),
                               atol=1e-2,
                               rtol=1e-2)
