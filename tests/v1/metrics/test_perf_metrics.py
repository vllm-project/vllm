# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the analytic estimators in metrics/flops.py.
"""

import types
from types import SimpleNamespace

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.llama4.configuration_llama4 import (
    Llama4Config,
    Llama4TextConfig,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from vllm.config.model import ModelConfig, get_hf_text_config
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)
from vllm.v1.metrics.perf import (
    AttentionMetrics,
    BaseConfigParser,
    ExecutionContext,
    FfnMetrics,
    ModelMetrics,
    ParsedArgs,
    UnembedMetrics,
)


class MockModelConfig:
    """Mock ModelConfig that implements the getter methods used by parsers."""

    def __init__(self, hf_config, dtype):
        self.hf_config = hf_config
        self.hf_text_config = get_hf_text_config(hf_config)
        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
            self.hf_config.model_type, ModelArchConfigConvertorBase
        )
        self.model_arch_config = convertor_cls(
            self.hf_config, self.hf_text_config
        ).convert()
        self.dtype = dtype
        self.is_attention_free = False

    def __getattr__(self, name):
        # 1. Check if ModelConfig actually has this attribute
        if not hasattr(ModelConfig, name):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"and neither does 'ModelConfig'."
            )

        # 2. Fetch the attribute from the ModelConfig CLASS
        attr = getattr(ModelConfig, name)

        # 3. Case A: It is a @property
        if isinstance(attr, property):
            # Manually invoke the property's getter, passing 'self' (this mock instance)
            return attr.__get__(self, self.__class__)

        # 4. Case B: It is a standard method (function)
        if isinstance(attr, types.FunctionType):
            # Bind the function to 'self' so it acts like a method of
            # this instance. This creates a bound method where 'self' is
            # automatically passed as the first arg.
            return types.MethodType(attr, self)

        # 5. Case C: It is a class attribute / static variable
        return attr


def create_mock_vllm_config(
    hf_config,
    model_dtype="bfloat16",
    cache_dtype="auto",
    quant_config=None,
    data_parallel_size=1,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    enable_expert_parallel=False,
) -> SimpleNamespace:
    vllm_config = SimpleNamespace()
    vllm_config.model_config = MockModelConfig(hf_config, model_dtype)

    vllm_config.cache_config = SimpleNamespace()
    vllm_config.cache_config.cache_dtype = cache_dtype

    vllm_config.quant_config = quant_config

    vllm_config.parallel_config = SimpleNamespace()
    vllm_config.parallel_config.data_parallel_size = data_parallel_size
    vllm_config.parallel_config.tensor_parallel_size = tensor_parallel_size
    vllm_config.parallel_config.pipeline_parallel_size = pipeline_parallel_size
    vllm_config.parallel_config.enable_expert_parallel = enable_expert_parallel

    return vllm_config


#### Parser Tests ####


def test_base_config_parser():
    """Test BaseConfigParser extracts base model attributes correctly."""
    hf_config = Qwen3Config(
        vocab_size=50000,
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=24,
    )
    vllm_config = create_mock_vllm_config(hf_config, model_dtype="float16")

    parser = BaseConfigParser()
    args = ParsedArgs()
    result = parser.parse(args, vllm_config)

    assert result.vocab_size == 50000
    assert result.hidden_size == 2048
    assert result.num_attention_heads == 16
    assert result.num_hidden_layers == 24
    assert result.weight_byte_size == 2  # float16 is 2 bytes
    assert result.activation_byte_size == 2  # default activation size


def test_base_attention_config_parser_with_gqa():
    """Test BaseAttentionConfigParser with grouped query attention."""
    hf_config = Qwen3Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA with 4:1 ratio
        head_dim=128,
    )
    vllm_config = create_mock_vllm_config(hf_config)

    parser_chain = AttentionMetrics.get_parser()
    result = parser_chain.parse(vllm_config)

    assert result.num_key_value_heads == 8
    assert result.head_dim == 128


def test_base_attention_config_parser_without_gqa():
    """
    Test BaseAttentionConfigParser defaults to MHA when num_key_value_heads not
    specified.
    """
    hf_config = Qwen3Config(
        hidden_size=4096,
        num_attention_heads=32,
        # No num_key_value_heads specified
    )
    vllm_config = create_mock_vllm_config(hf_config)

    parser_chain = AttentionMetrics.get_parser()
    result = parser_chain.parse(vllm_config)

    # Should default to MHA (num_key_value_heads = num_attention_heads)
    assert result.num_key_value_heads == 32


def test_base_ffn_config_parser_dense():
    """Test BaseFfnConfigParser for dense FFN."""
    hf_config = Qwen3Config(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
    )
    vllm_config = create_mock_vllm_config(hf_config)

    parser_chain = FfnMetrics.get_parser()
    result = parser_chain.parse(vllm_config)

    assert result.intermediate_size == 11008
    assert result.num_experts == 0
    assert result.num_experts_per_tok == 0
    assert result.num_moe_layers == 0  # No MoE


def test_base_ffn_config_parser_moe():
    """Test BaseFfnConfigParser for MoE FFN."""
    hf_config = Qwen3MoeConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=14336,
        n_shared_experts=2,
    )
    vllm_config = create_mock_vllm_config(hf_config)

    parser_chain = FfnMetrics.get_parser()
    result = parser_chain.parse(vllm_config)

    assert result.num_experts == 64
    assert result.num_experts_per_tok == 8
    assert result.moe_intermediate_size == 14336
    assert result.num_shared_experts == 2
    assert result.num_moe_layers == 32  # All layers are MoE by default


def test_interleave_moe_layer_step_parser():
    """Test InterleaveMoeLayerStepParser correctly computes MoE layer count."""
    hf_config = Llama4Config(
        text_config=Llama4TextConfig(
            num_hidden_layers=32,
            num_local_experts=64,
            interleave_moe_layer_step=4,  # Every 4th layer is MoE
        ),
    )

    vllm_config = create_mock_vllm_config(hf_config)

    parser_chain = FfnMetrics.get_parser()
    result = parser_chain.parse(vllm_config)

    assert result.num_moe_layers == 8


def test_moe_layer_freq_parser():
    """Test MoeLayerFreqParser correctly computes MoE layer count."""
    hf_config = DeepseekV3Config(
        num_hidden_layers=30,
        n_routed_experts=64,
        moe_layer_freq=3,  # Every 3rd layer after first_k_dense_replace
        first_k_dense_replace=6,  # First 6 layers are dense
    )
    vllm_config = create_mock_vllm_config(hf_config)

    parser_chain = FfnMetrics.get_parser()
    result = parser_chain.parse(vllm_config)

    # Layers >= 6 and divisible by 3: 6, 9, 12, 15, 18, 21, 24, 27
    expected_moe_layers = len(
        [layer for layer in range(30) if layer >= 6 and layer % 3 == 0]
    )
    assert expected_moe_layers == 8
    assert result.num_moe_layers == expected_moe_layers


#### ComponentMetrics Tests ####


def test_attention_metrics_scaling():
    """Test that attention metrics scale proportionally with model dimensions."""
    base_hf_config = Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=12,
        head_dim=128,
    )

    base_vllm_config = create_mock_vllm_config(base_hf_config)
    base_metrics = AttentionMetrics.from_vllm_config(base_vllm_config)

    # Test scaling with number of layers
    double_layers_hf_config = Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=24,  # Double the layers
        head_dim=128,
    )
    double_layers_vllm_config = create_mock_vllm_config(double_layers_hf_config)
    double_layers_metrics = AttentionMetrics.from_vllm_config(double_layers_vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # FLOPS should double when layers double
    base_flops = base_metrics.get_num_flops(ctx)
    double_flops = double_layers_metrics.get_num_flops(ctx)
    assert double_flops == 2 * base_flops

    # Read/write bytes should also scale proportionally
    base_read = base_metrics.get_read_bytes(ctx)
    double_read = double_layers_metrics.get_read_bytes(ctx)
    assert double_read == 2 * base_read

    base_write = base_metrics.get_write_bytes(ctx)
    double_write = double_layers_metrics.get_write_bytes(ctx)
    assert double_write == 2 * base_write


def test_attention_metrics_grouped_query():
    """Test attention metrics handle grouped query attention correctly."""
    mha_hf_config = Qwen3Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,  # MHA
        num_hidden_layers=1,
    )
    mha_config = create_mock_vllm_config(mha_hf_config)

    gqa_hf_config = Qwen3Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA with 4:1 ratio
        num_hidden_layers=1,
    )
    gqa_config = create_mock_vllm_config(gqa_hf_config)

    mha_metrics = AttentionMetrics.from_vllm_config(mha_config)
    gqa_metrics = AttentionMetrics.from_vllm_config(gqa_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=1, context_len=1024, is_prefill=False
    )

    # GQA should have less KV cache reads since fewer KV heads
    mha_read = mha_metrics.get_read_bytes(ctx)
    gqa_read = gqa_metrics.get_read_bytes(ctx)
    assert gqa_read < mha_read


def test_ffn_metrics_scaling():
    """Test FFN metrics scale proportionally with model dimensions."""
    base_hf_config = Qwen3Config(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
    )
    base_vllm_config = create_mock_vllm_config(base_hf_config)
    base_metrics = FfnMetrics.from_vllm_config(base_vllm_config)

    # Test scaling with intermediate size
    larger_ffn_hf_config = Qwen3Config(
        hidden_size=2048,
        intermediate_size=16384,  # Double intermediate size
        num_hidden_layers=12,
    )
    larger_ffn_vllm_config = create_mock_vllm_config(larger_ffn_hf_config)
    larger_ffn_metrics = FfnMetrics.from_vllm_config(larger_ffn_vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # FLOPS should double when intermediate size doubles
    base_flops = base_metrics.get_num_flops(ctx)
    larger_flops = larger_ffn_metrics.get_num_flops(ctx)
    assert larger_flops == base_flops * 2


def test_moe_metrics_vs_dense():
    """Test MoE metrics versus dense metrics."""
    dense_hf_config = Qwen3Config(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
    )
    dense_config = create_mock_vllm_config(dense_hf_config)

    moe_hf_config = Qwen3MoeConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
        num_experts=64,
        num_experts_per_tok=2,  # 2 routed expert
        moe_intermediate_size=8192,
        n_shared_experts=0,
    )
    moe_config = create_mock_vllm_config(moe_hf_config)

    dense_metrics = FfnMetrics.from_vllm_config(dense_config)
    moe_metrics = FfnMetrics.from_vllm_config(moe_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # MoE should have different compute/memory characteristics
    dense_flops = dense_metrics.get_num_flops(ctx)
    moe_flops = moe_metrics.get_num_flops(ctx)

    # 2 routed experts vs 1 dense.
    assert moe_flops == dense_flops * 2


def test_unembed_metrics_scaling():
    """Test unembedding metrics scale with vocab size."""
    small_vocab_hf_config = Qwen3Config(
        hidden_size=2048,
        vocab_size=32000,
    )
    small_vocab_config = create_mock_vllm_config(small_vocab_hf_config)

    large_vocab_hf_config = Qwen3Config(
        hidden_size=2048,
        vocab_size=64000,  # Double vocab size
    )
    large_vocab_config = create_mock_vllm_config(large_vocab_hf_config)

    small_vocab_metrics = UnembedMetrics.from_vllm_config(small_vocab_config)
    large_vocab_metrics = UnembedMetrics.from_vllm_config(large_vocab_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # FLOPS should double when vocab size doubles
    small_flops = small_vocab_metrics.get_num_flops(ctx)
    large_flops = large_vocab_metrics.get_num_flops(ctx)
    assert large_flops == 2 * small_flops


def test_prefill_vs_decode_differences():
    """Test that prefill and decode have different memory access patterns."""
    hf_config = Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=1,
    )
    config = create_mock_vllm_config(hf_config)

    metrics = AttentionMetrics.from_vllm_config(config)

    prefill_ctx = ExecutionContext.from_single_request(
        num_tokens=512, context_len=512, is_prefill=True
    )
    decode_ctx = ExecutionContext.from_single_request(
        num_tokens=1, context_len=512, is_prefill=False
    )

    prefill_read = metrics.get_read_bytes(prefill_ctx)
    decode_read = metrics.get_read_bytes(decode_ctx)

    assert prefill_read != decode_read


def test_model_metrics_aggregation():
    """Test ModelMetrics correctly aggregates across components."""
    hf_config = Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=12,
        vocab_size=32000,
        intermediate_size=8192,
    )
    config = create_mock_vllm_config(hf_config)

    model_metrics = ModelMetrics(config)
    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # Should have metrics for attention, ffn, and unembed
    total_flops = model_metrics.get_num_flops(ctx)
    breakdown = model_metrics.get_num_flops_breakdown(ctx)

    # Breakdown should sum to total
    assert total_flops == sum(breakdown.values())


def test_moe_expert_activation_proportional_scaling():
    """Test that routed expert metrics scale proportionally with num_experts_per_tok."""
    base_moe_config = Qwen3MoeConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
        num_experts=64,
        num_experts_per_tok=1,  # 1 expert per token
        moe_intermediate_size=8192,
        n_shared_experts=2,
    )

    double_experts_config = Qwen3MoeConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
        num_experts=64,
        num_experts_per_tok=2,  # 2 experts per token (double)
        moe_intermediate_size=8192,
        n_shared_experts=2,  # Same shared experts
    )

    triple_experts_config = Qwen3MoeConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
        num_experts=64,
        num_experts_per_tok=3,  # 3 experts per token (triple)
        moe_intermediate_size=8192,
        n_shared_experts=2,  # Same shared experts
    )

    base_vllm_config = create_mock_vllm_config(base_moe_config)
    double_vllm_config = create_mock_vllm_config(double_experts_config)
    triple_vllm_config = create_mock_vllm_config(triple_experts_config)

    base_metrics = FfnMetrics.from_vllm_config(base_vllm_config)
    double_metrics = FfnMetrics.from_vllm_config(double_vllm_config)
    triple_metrics = FfnMetrics.from_vllm_config(triple_vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # Get total metrics - the key insight is that differences should be proportional
    base_flops = base_metrics.get_num_flops(ctx)
    double_flops = double_metrics.get_num_flops(ctx)
    triple_flops = triple_metrics.get_num_flops(ctx)

    # The difference between double and base should equal one additional expert
    one_expert_diff = double_flops - base_flops

    # The difference between triple and base should equal two additional experts
    two_expert_diff = triple_flops - base_flops

    # Proportional scaling: 2 * (1 expert diff) should equal (2 expert diff)
    assert two_expert_diff == 2 * one_expert_diff

    # Same logic applies to memory operations
    base_read = base_metrics.get_read_bytes(ctx)
    double_read = double_metrics.get_read_bytes(ctx)
    triple_read = triple_metrics.get_read_bytes(ctx)

    one_expert_read_diff = double_read - base_read
    two_expert_read_diff = triple_read - base_read

    assert two_expert_read_diff == 2 * one_expert_read_diff

    # Same for write bytes
    base_write = base_metrics.get_write_bytes(ctx)
    double_write = double_metrics.get_write_bytes(ctx)
    triple_write = triple_metrics.get_write_bytes(ctx)

    one_expert_write_diff = double_write - base_write
    two_expert_write_diff = triple_write - base_write

    assert two_expert_write_diff == 2 * one_expert_write_diff


def test_quantization_config_parser_fp8():
    """Test quantization parsers with fp8."""

    class MockQuantConfig:
        def get_name(self):
            return "fp8"

    hf_config = Qwen3Config(
        hidden_size=2048, num_attention_heads=16, num_hidden_layers=1
    )
    vllm_config = create_mock_vllm_config(hf_config, quant_config=MockQuantConfig())

    attn_result = AttentionMetrics.get_parser().parse(vllm_config)
    assert attn_result.weight_byte_size == 1  # fp8

    ffn_result = FfnMetrics.get_parser().parse(vllm_config)
    assert ffn_result.weight_byte_size == 1  # fp8


def test_quantization_config_parser_mxfp4():
    """Test quantization parsers with mxfp4."""

    class MockQuantConfig:
        def get_name(self):
            return "mxfp4"

    hf_config = Qwen3Config(
        hidden_size=2048, intermediate_size=8192, num_hidden_layers=1
    )
    vllm_config = create_mock_vllm_config(hf_config, quant_config=MockQuantConfig())

    ffn_result = FfnMetrics.get_parser().parse(vllm_config)
    assert ffn_result.weight_byte_size == 0.5  # mxfp4


#### Per-GPU Tests ####


def test_attention_per_gpu_with_tensor_parallelism():
    """Test attention metrics with tensor parallelism - per_gpu vs global."""
    hf_config = Qwen3Config(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=24,
    )

    # Test with TP=4
    vllm_config = create_mock_vllm_config(hf_config, tensor_parallel_size=4)
    metrics = AttentionMetrics.from_vllm_config(vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=128, context_len=1024, is_prefill=True
    )

    # Get global and per-gpu metrics
    global_flops = metrics.get_num_flops(ctx, per_gpu=False)
    per_gpu_flops = metrics.get_num_flops(ctx, per_gpu=True)

    # With TP=4, global flops should be 4x per-gpu flops (heads divided by 4)
    assert global_flops == 4 * per_gpu_flops

    # Same for read/write bytes
    global_read = metrics.get_read_bytes(ctx, per_gpu=False)
    per_gpu_read = metrics.get_read_bytes(ctx, per_gpu=True)
    # Reads should scale similarly (weight reads are divided by TP)
    assert global_read > per_gpu_read

    global_write = metrics.get_write_bytes(ctx, per_gpu=False)
    per_gpu_write = metrics.get_write_bytes(ctx, per_gpu=True)
    assert global_write > per_gpu_write


def test_attention_per_gpu_with_pipeline_parallelism():
    """Test attention metrics with pipeline parallelism - per_gpu vs global."""
    hf_config = Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=32,
    )

    # Test with PP=4
    vllm_config = create_mock_vllm_config(hf_config, pipeline_parallel_size=4)
    metrics = AttentionMetrics.from_vllm_config(vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=False
    )

    # Get global and per-gpu metrics
    global_flops = metrics.get_num_flops(ctx, per_gpu=False)
    per_gpu_flops = metrics.get_num_flops(ctx, per_gpu=True)

    # With PP=4, global flops should be 4x per-gpu flops (layers divided by 4)
    assert global_flops == 4 * per_gpu_flops

    global_read = metrics.get_read_bytes(ctx, per_gpu=False)
    per_gpu_read = metrics.get_read_bytes(ctx, per_gpu=True)
    assert global_read == 4 * per_gpu_read


def test_ffn_per_gpu_with_tensor_parallelism():
    """Test FFN metrics with tensor parallelism - per_gpu vs global."""
    hf_config = Qwen3Config(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
    )

    # Test with DP=2, TP=4 (ffn_tp_size will be 8)
    vllm_config = create_mock_vllm_config(
        hf_config,
        data_parallel_size=2,
        tensor_parallel_size=4,
    )
    metrics = FfnMetrics.from_vllm_config(vllm_config)

    # ffn_tp_size should be dp_size * tp_size = 8 (when EP not enabled)
    assert metrics.ffn_tp_size == 8

    ctx = ExecutionContext.from_single_request(
        num_tokens=128, context_len=2048, is_prefill=True
    )

    # Get global and per-gpu metrics
    global_flops = metrics.get_num_flops(ctx, per_gpu=False)
    per_gpu_flops = metrics.get_num_flops(ctx, per_gpu=True)

    # With ffn_tp_size=8, global should be 8x per-gpu
    assert global_flops == 8 * per_gpu_flops


def test_ffn_per_gpu_with_pipeline_parallelism():
    """Test FFN metrics with pipeline parallelism - per_gpu vs global."""
    hf_config = Qwen3Config(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
    )

    # Test with PP=6
    vllm_config = create_mock_vllm_config(hf_config, pipeline_parallel_size=6)
    metrics = FfnMetrics.from_vllm_config(vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # Get global and per-gpu metrics
    global_flops = metrics.get_num_flops(ctx, per_gpu=False)
    per_gpu_flops = metrics.get_num_flops(ctx, per_gpu=True)

    # With PP=6, global should be 6x per-gpu (layers divided by 6)
    assert global_flops == 6 * per_gpu_flops


def test_moe_per_gpu_with_expert_parallelism():
    """
    Test MoE metrics with expert parallelism - verifies num_activated_experts bug fix.
    """
    hf_config = Qwen3MoeConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=14336,
        n_shared_experts=2,
    )

    # Test with DP=2, TP=4, EP enabled (ffn_ep_size will be 8)
    vllm_config = create_mock_vllm_config(
        hf_config,
        data_parallel_size=2,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
    )
    metrics = FfnMetrics.from_vllm_config(vllm_config)

    # When EP enabled, ffn_ep_size = dp_size * tp_size = 8
    assert metrics.ffn_ep_size == 8
    assert metrics.ffn_tp_size == 1

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # Get per-gpu metrics
    per_gpu_read_breakdown = metrics.get_read_bytes_breakdown(ctx, per_gpu=True)
    global_read_breakdown = metrics.get_read_bytes_breakdown(ctx, per_gpu=False)

    # Verify that routed expert weight reads are reasonable
    # With per_gpu=True, each GPU has 64/8 = 8 experts
    # T=100, E_per_gpu=8/8=1, so T*E=100 expert activations
    # num_activated_experts should be min(100, 8) = 8

    # Check that weight reads scale appropriately
    # Global has all 64 experts, per-gpu has 8 experts
    # So weight reads should reflect this difference
    if "routed_up_gate_weights" in per_gpu_read_breakdown:
        per_gpu_weight_reads = per_gpu_read_breakdown["routed_up_gate_weights"]
        global_weight_reads = global_read_breakdown["routed_up_gate_weights"]

        # The ratio should reflect the expert count difference
        # This verifies the bug fix works correctly
        assert per_gpu_weight_reads < global_weight_reads

        # Global should read more experts than per-gpu
        # Exact ratio depends on num_activated_experts calculation
        ratio = global_weight_reads / per_gpu_weight_reads
        # Should be > 1 since global has more experts to read
        assert ratio > 1


def test_moe_per_gpu_expert_activation_accounting():
    """
    Test that MoE correctly accounts for expert activations with small batch sizes.
    """
    hf_config = Qwen3MoeConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=12,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=14336,
        n_shared_experts=0,  # No shared experts for this test
    )

    # Test with EP=8
    vllm_config = create_mock_vllm_config(
        hf_config,
        data_parallel_size=8,
        enable_expert_parallel=True,
    )
    metrics = FfnMetrics.from_vllm_config(vllm_config)

    # Small batch: T=10, E_per_gpu=8/8=1
    # Each GPU: T*E = 10*1 = 10 activations
    # Experts per GPU: 64/8 = 8
    # So num_activated_experts should be min(10, 8) = 8
    small_ctx = ExecutionContext.from_single_request(
        num_tokens=10, context_len=512, is_prefill=True
    )
    small_read = metrics.get_read_bytes_breakdown(small_ctx, per_gpu=True)

    # Large batch: T=1000, E_per_gpu=1
    # Each GPU: T*E = 1000*1 = 1000 activations
    # Experts per GPU: 8
    # So num_activated_experts should be min(1000, 8) = 8 (all experts activated)
    large_ctx = ExecutionContext.from_single_request(
        num_tokens=1000, context_len=512, is_prefill=True
    )
    large_read = metrics.get_read_bytes_breakdown(large_ctx, per_gpu=True)

    # Weight reads should be similar (both activate all 8 experts per GPU)
    # But activation reads should differ (proportional to T*E)
    if "routed_up_gate_weights" in small_read:
        small_weight = small_read["routed_up_gate_weights"]
        large_weight = large_read["routed_up_gate_weights"]

        # Weight reads should be the same (both read all 8 experts)
        assert small_weight == large_weight

        # But input activation reads should scale with T*E
        small_input = small_read["routed_up_gate_input"]
        large_input = large_read["routed_up_gate_input"]
        assert large_input == 100 * small_input  # 1000/10 = 100x


def test_unembed_per_gpu_with_tensor_parallelism():
    """Test unembed metrics with tensor parallelism - per_gpu vs global."""
    hf_config = Qwen3Config(
        hidden_size=4096,
        vocab_size=128000,
    )

    # Test with TP=8
    vllm_config = create_mock_vllm_config(hf_config, tensor_parallel_size=8)
    metrics = UnembedMetrics.from_vllm_config(vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # Get global and per-gpu metrics
    global_flops = metrics.get_num_flops(ctx, per_gpu=False)
    per_gpu_flops = metrics.get_num_flops(ctx, per_gpu=True)

    # With TP=8, vocab is divided by 8, so global should be 8x per-gpu
    assert global_flops == 8 * per_gpu_flops

    # For read bytes, weight reads scale with TP but input reads don't (replicated)
    global_read_breakdown = metrics.get_read_bytes_breakdown(ctx, per_gpu=False)
    per_gpu_read_breakdown = metrics.get_read_bytes_breakdown(ctx, per_gpu=True)

    # Input reads should be the same (replicated across TP ranks)
    assert global_read_breakdown["input"] == per_gpu_read_breakdown["input"]

    # Weight reads should scale 8x (divided by TP)
    assert global_read_breakdown["weight"] == 8 * per_gpu_read_breakdown["weight"]


def test_model_metrics_per_gpu_aggregation():
    """Test ModelMetrics correctly aggregates per_gpu metrics across components."""
    hf_config = Qwen3Config(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=12,
        vocab_size=32000,
        intermediate_size=8192,
    )

    # Test with mixed parallelism: TP=2, PP=2
    vllm_config = create_mock_vllm_config(
        hf_config,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )

    model_metrics = ModelMetrics(vllm_config)
    ctx = ExecutionContext.from_single_request(
        num_tokens=100, context_len=512, is_prefill=True
    )

    # Get breakdowns for both modes
    per_gpu_breakdown = model_metrics.get_num_flops_breakdown(ctx, per_gpu=True)
    global_breakdown = model_metrics.get_num_flops_breakdown(ctx, per_gpu=False)

    # Verify breakdown sums match totals
    per_gpu_total = model_metrics.get_num_flops(ctx, per_gpu=True)
    global_total = model_metrics.get_num_flops(ctx, per_gpu=False)

    assert per_gpu_total == sum(per_gpu_breakdown.values())
    assert global_total == sum(global_breakdown.values())

    # Global should be larger than per-gpu due to parallelism
    assert global_total > per_gpu_total

    # With TP=2 and PP=2, the ratio depends on which parallelism applies to
    # which component but we can verify that global is reasonably larger
    ratio = global_total / per_gpu_total
    assert ratio > 1  # Should be between PP and TP*PP depending on component mix


def test_attention_per_gpu_heads_not_evenly_divisible():
    """Test attention with heads not evenly divisible by TP."""
    hf_config = Qwen3Config(
        hidden_size=2048,
        num_attention_heads=17,  # Not divisible by 4
        num_key_value_heads=5,  # Not divisible by 4
        num_hidden_layers=8,
    )

    vllm_config = create_mock_vllm_config(hf_config, tensor_parallel_size=4)
    metrics = AttentionMetrics.from_vllm_config(vllm_config)

    ctx = ExecutionContext.from_single_request(
        num_tokens=64, context_len=256, is_prefill=True
    )

    # Should not crash and should handle max(1, ...) correctly
    per_gpu_flops = metrics.get_num_flops(ctx, per_gpu=True)
    global_flops = metrics.get_num_flops(ctx, per_gpu=False)

    # Both should be positive
    assert per_gpu_flops > 0
    assert global_flops > 0
    assert global_flops > per_gpu_flops
