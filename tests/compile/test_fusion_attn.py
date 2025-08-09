# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import pytest
import torch._dynamo

from tests.compile.backend import TestBackend
from tests.models.utils import check_outputs_equal
from vllm import LLM, SamplingParams
from vllm.attention import Attention
from vllm.compilation.fusion import QUANT_OPS, QuantKey, kFp8StaticTensorSym
from vllm.compilation.fusion_attn import ATTN_OP, AttnFusionPass
from vllm.compilation.fx_utils import find_op_nodes, is_auto_func
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.config import (CacheConfig, CompilationConfig, CompilationLevel,
                         ModelConfig, ParallelConfig, PassConfig,
                         SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
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


class TestQuantAttentionQuantPatternModel(torch.nn.Module):
    """Test model for QuantAttentionQuantPattern fusion."""

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_size: int,
                 kv_cache_dtype: torch.dtype, quant_dtype: torch.dtype,
                 device: torch.device, vllm_config: VllmConfig):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size = num_qo_heads * head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.quant_dtype = quant_dtype
        self.device = device
        self.vllm_config = vllm_config

        # Create the attention layer with proper prefix pattern for fusion
        # registration.
        # The fusion pass looks for layers.{layer_id}.self_attn.attn pattern
        self.attn = Attention(
            num_heads=self.num_qo_heads,
            head_size=self.head_size,
            scale=1.0 / (self.head_size**0.5),
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
        )

        # Create the o_proj layer that fusion pass requires to find input_scale
        # The fusion pass looks for layers.{layer_id}.self_attn.o_proj pattern
        self.o_proj = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix="model.layers.0.self_attn.o_proj",
        )

        # Initialize weights for the o_proj layer
        with torch.no_grad():
            self.o_proj.input_scale.fill_(1.0)
            self.o_proj.weight_scale.fill_(1.0)

            # Create a temporary float tensor then copy to FP8 weight
            temp_weight = torch.randn_like(self.o_proj.weight,
                                           dtype=torch.float16)
            torch.nn.init.normal_(temp_weight, mean=0.0, std=0.01)
            self.o_proj.weight.copy_(temp_weight.to(self.o_proj.weight.dtype))

            # Process weights to handle transposition and other setup required
            # by quantization method.
            self.o_proj.quant_method.process_weights_after_loading(self.o_proj)

    def build_attn_metadata(self, batch_size: int):
        """Initialize attention metadata for the given batch configuration."""
        # Create basic attention metadata
        query_start_loc = torch.arange(0,
                                       batch_size + 1,
                                       dtype=torch.int32,
                                       device=self.device)
        seq_lens = torch.ones(batch_size,
                              dtype=torch.int32,
                              device=self.device)

        # Create simple block table and slot mapping for testing
        block_size = 16
        seq_len = batch_size  # num_tokens = batch_size for simplicity
        num_blocks = max(1, (seq_len + block_size - 1) // block_size)
        block_table = torch.arange(num_blocks,
                                   dtype=torch.int32,
                                   device=self.device).unsqueeze(0).repeat(
                                       batch_size, 1)
        slot_mapping = torch.arange(batch_size,
                                    dtype=torch.long,
                                    device=self.device)

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            num_computed_tokens_cpu=torch.zeros(batch_size, dtype=torch.int32),
            num_reqs=batch_size,
            num_actual_tokens=batch_size,
            max_query_len=1,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
        )

        # Mock the KV cache for testing with correct layout for FlashInfer
        #   - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
        #   - HND: [num_blocks, 2,  num_kv_heads, block_size, head_size]
        # Create kv_cache in HND layout and permute to NHD layout for now
        # (later will be permute back to HND layout in forward pass)
        kv_cache = torch.zeros(num_blocks,
                               2,
                               self.num_kv_heads,
                               block_size,
                               self.head_size,
                               dtype=self.kv_cache_dtype,
                               device=self.device)
        kv_cache = kv_cache.permute(0, 1, 3, 2, 4)
        self.attn.kv_cache = [kv_cache]

        # Initialize FlashInferMetadataBuilder
        builder = FlashInferMetadataBuilder(
            kv_cache_spec=AttentionSpec(
                block_size=block_size,
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
        o_proj_output, _ = self.o_proj(attn_output)
        return o_proj_output

    def op_in_model_before(self):
        """Operations expected before fusion."""
        # Before fusion: o_proj has internal FP8 quantization ops
        if self.quant_dtype == FP8_DTYPE:
            quant_key = kFp8StaticTensorSym
        else:
            raise ValueError(f"Unsupported quant_dtype: {self.quant_dtype}")
        return QUANT_OPS[quant_key]

    def op_in_model_after(self):
        """Operations expected after fusion."""
        # After fusion:
        # - Output quantization should be fused into attention
        # - Query quantization should be inserted before attention
        pre_quant_key = kFp8StaticTensorSym
        return QUANT_OPS[pre_quant_key]


@pytest.mark.parametrize("num_heads", [(64, 8), (40, 8)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("num_tokens", [7, 256, 533])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("quant_dtype", [FP8_DTYPE])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test CUDA")
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Need FP8")
@pytest.mark.skipif(not current_platform.is_device_capability((10, 0)),
                    reason="Only test on SM100(Blackwell)")
def test_quant_attention_quant_pattern(num_heads: tuple[int,
                                                        int], head_size: int,
                                       num_tokens: int, dtype: torch.dtype,
                                       quant_dtype: torch.dtype, monkeypatch,
                                       dist_init):
    """Test QuantAttentionQuantPattern fusion pass with FlashInfer V1 backend"""

    # Enable FlashInfer v1 backend for this test
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    monkeypatch.setenv("VLLM_USE_TRTLLM_ATTN", "1")

    device = torch.device("cuda")
    torch.manual_seed(42)

    num_qo_heads, num_kv_heads = num_heads

    kv_cache_dtype = FP8_DTYPE
    kv_cache_dtype_str = "fp8"

    if quant_dtype == FP8_DTYPE:
        quant_config = ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
    else:
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}")

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
            max_model_len=2048,
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        scheduler_config=SchedulerConfig(max_num_seqs=2048),
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            custom_ops=["+quant_fp8"],
            full_cuda_graph=True,
            pass_config=PassConfig(enable_attn_fusion=True, enable_noop=True),
        ),
        cache_config=CacheConfig(cache_dtype=kv_cache_dtype_str),
        quant_config=quant_config,
    )

    with set_current_vllm_config(vllm_config), set_forward_context(
            attn_metadata=None, vllm_config=vllm_config):
        model = TestQuantAttentionQuantPatternModel(num_qo_heads, num_kv_heads,
                                                    head_size, kv_cache_dtype,
                                                    quant_dtype, device,
                                                    vllm_config)

        # Create test inputs
        q = torch.rand(num_tokens,
                       num_qo_heads * head_size,
                       dtype=dtype,
                       device=device)
        k = torch.rand(num_tokens,
                       num_kv_heads * head_size,
                       dtype=dtype,
                       device=device)
        v = torch.rand(num_tokens,
                       num_kv_heads * head_size,
                       dtype=dtype,
                       device=device)

        # Mark first dimension as dynamic for realistic testing
        torch._dynamo.mark_dynamic(q, 0)
        torch._dynamo.mark_dynamic(k, 0)
        torch._dynamo.mark_dynamic(v, 0)

        forward_ctx = get_forward_context()

        # Run model directly without compilation and fusion
        forward_ctx.attn_metadata = model.build_attn_metadata(
            batch_size=num_tokens)
        result_unfused = model(q, k, v)

        # Create test backend with fusion passes enabled
        noop_pass = NoOpEliminationPass(vllm_config)
        attn_pass = lambda *args, **kw: AttnFusionPass(vllm_config)(*args, **kw
                                                                    )
        backend = TestBackend(noop_pass, attn_pass)

        # Compile model without attn metadata for fusion detection
        forward_ctx.attn_metadata = None
        model_compiled = torch.compile(model, backend=backend, fullgraph=True)
        result_fused = model_compiled(q, k, v)

        assert model_compiled.attn.fused_quant, \
            "Attention layer should have fused_quant=True"\
            "after the first compilation"

        # The attn metadata is build according to the result of the fusion pass
        # for getting the correct kernel implementation.
        forward_ctx.attn_metadata = model.build_attn_metadata(
            batch_size=num_tokens)
        result_fused = model_compiled(q, k, v)

    # Check quantization ops in the graph before and after fusion
    quant_ops_pre = list(
        find_op_nodes(model.op_in_model_before(), backend.graph_pre_pass))
    query_quant_ops_post = list(
        find_op_nodes(model.op_in_model_after(), backend.graph_post_pass))

    assert len(quant_ops_pre) > 0, "Should have quantization ops before fusion"
    assert len(quant_ops_pre) == len(query_quant_ops_post), \
        "Should have same number of quantization ops before and after fusion"

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

    # Check the relative position of attention and quantization ops in the graph
    # before and after fusion.
    attn_quant_ops_pre = []
    for node in backend.graph_pre_pass.nodes:
        if is_auto_func(node, ATTN_OP):
            attn_quant_ops_pre.append(ATTN_OP)
        elif is_auto_func(node, model.op_in_model_before()):
            attn_quant_ops_pre.append(model.op_in_model_before())

    assert (attn_quant_ops_pre[0] == ATTN_OP and
            attn_quant_ops_pre[1] == model.op_in_model_before()), \
        "Attention should be before the output quantization op before fusion"

    quant_attn_ops_post = []
    for node in backend.graph_post_pass.nodes:
        if is_auto_func(node, ATTN_OP):
            quant_attn_ops_post.append(ATTN_OP)
        elif is_auto_func(node, model.op_in_model_after()):
            quant_attn_ops_post.append(model.op_in_model_after())

    assert (quant_attn_ops_post[0] == model.op_in_model_after() and
            quant_attn_ops_post[1] == ATTN_OP), \
        "Attention should be after the query quantization op after fusion"

    # Check that results are closed
    torch.testing.assert_close(result_unfused,
                               result_fused,
                               atol=1e-2,
                               rtol=1e-2)
