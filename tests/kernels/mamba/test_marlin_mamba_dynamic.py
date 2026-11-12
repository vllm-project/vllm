# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The real Mamba owner must keep token-count routing inside its opaque op."""

import pytest
import torch
from transformers import NemotronHConfig

from vllm.compilation.wrapper import compile_model_with_stock_torch
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.kernels.linear.nvfp4.b12x import (
    B12xNvFp4W4A16LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
from vllm.model_executor.layers.mamba.ops.marlin_input_conv import (
    _batched_shape_is_profitable,
    _runtime_metadata,
    can_use_marlin_input_conv,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionBackend,
    Mamba2AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.replayssm_utils import (
    get_replayssm_block_copy_tensors,
    initialize_replayssm_ring_trackers,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    bind_kv_cache_to_layers,
    clear_layer_kv_caches,
    get_uniform_decode_token_count,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="requires NVIDIA CUDA",
)


def _decode_metadata(query, seq, block_table, *, batches, tokens, max_query, physical):
    """Keep every real-builder test on the same ordinary decode metadata ABI."""
    return CommonAttentionMetadata(
        query_start_loc=query.cuda(),
        query_start_loc_cpu=query,
        seq_lens=seq.cuda(),
        num_reqs=batches,
        num_actual_tokens=tokens,
        max_query_len=max_query,
        max_seq_len=256,
        block_table_tensor=block_table,
        slot_mapping=torch.full((physical,), -1, device="cuda", dtype=torch.int64),
        is_prefilling=torch.zeros(batches, dtype=torch.bool),
        seq_lens_cpu_upper_bound=seq,
    )


@pytest.fixture
def owned_mamba(tmp_path, dist_init, request, monkeypatch):
    options = getattr(request, "param", 8)
    options = options if isinstance(options, dict) else {"configured_tokens": options}
    if options.get("disable_marlin", False):
        monkeypatch.setenv("VLLM_DISABLED_KERNELS", "MarlinNvFp4LinearKernel")
    configured_tokens = options.get("configured_tokens", 8)
    capture_mode = options.get("capture_mode", CUDAGraphMode.NONE)
    expected_admission = options.get("expected_admission", True)
    max_num_seqs = options.get("max_num_seqs", 2)
    capture_tokens = options.get("capture_tokens", 8)
    state_dtype = options.get("state_dtype", "auto")
    sr_rng = options.get("sr_rng")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor, props.multi_processor_count) != (12, 1, 48):
        pytest.skip("requires the supported SM121/48SM device")
    if get_conv_state_layout() != "SD":
        pytest.skip("requires SD convolution cache layout")
    if not hasattr(torch.ops._C, "mamba_marlin_input_conv"):
        pytest.skip("requires vLLM build with Marlin input-conv support")
    NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        layers_block_type=["linear_attention"],
        vocab_size=128,
        max_position_embeddings=1024,
    ).save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        enforce_eager=False,
        max_model_len=1024,
    )
    parallel = ParallelConfig()
    cache = CacheConfig(
        block_size=16,
        mamba_block_size=1024,
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype=(
            "auto"
            if state_dtype != "auto"
            else options.get("checkpoint_dtype", "bfloat16")
        ),
        replayssm_state_dtype=state_dtype,
        use_replayssm=True,
        replayssm_buffer_len=16,
        enable_prefix_caching=False,
    )
    config = VllmConfig(
        model_config=model,
        parallel_config=parallel,
        cache_config=cache,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_model_len=1024,
            is_encoder_decoder=False,
            async_scheduling=False,
            enable_chunked_prefill=True,
        ),
        speculative_config=(
            SpeculativeConfig(
                method="ngram",
                num_speculative_tokens=configured_tokens - 1,
                enable_adaptive_verification=options.get("adaptive", False),
                target_model_config=model,
                target_parallel_config=parallel,
            )
            if configured_tokens > 1
            else None
        ),
        mamba_config=MambaConfig(
            backend=MambaBackendEnum.FLASHINFER,
            enable_stochastic_rounding=sr_rng is not None,
            stochastic_rounding_philox_rounds=(
                options.get("sr_rounds", 5) if sr_rng is not None else 0
            ),
            stochastic_rounding_rng=sr_rng or "philox",
        ),
        compilation_config=CompilationConfig(
            mode=(
                CompilationMode.NONE
                if options.get("admission_only", False)
                else CompilationMode.STOCK_TORCH_COMPILE
            ),
            backend="eager",
            splitting_ops=["vllm::mamba_mixer2_with_input_projection"],
            cudagraph_mode=capture_mode,
            cudagraph_capture_sizes=[capture_tokens],
            max_cudagraph_capture_size=capture_tokens,
            decode_cudagraph_capture_request_sizes=options.get("decode_request_sizes"),
        ),
        kernel_config=KernelConfig(
            linear_backend=options.get("linear_backend", "auto"),
            moe_backend=options.get("moe_backend", "auto"),
            linear_backend_per_quant=(
                None if "linear_backend" in options else {"nvfp4_w4a16": "marlin"}
            ),
        ),
    )
    quant = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
        with torch.device(device):
            layer = MambaMixer2(
                hidden_size=4096,
                ssm_state_size=128,
                conv_kernel_size=4,
                intermediate_size=8192,
                use_conv_bias=True,
                use_bias=False,
                n_groups=8,
                model_config=model,
                cache_config=cache,
                quant_config=quant,
                prefix="model.layers.0.mixer",
            )
            # Real checkpoint-format parameters and normal post-load conversion.
            # Preserve W4A16's unused NaN input_scale compatibility placeholders.
            with torch.no_grad():
                for name, parameter in layer.named_parameters():
                    if name.endswith("input_scale"):
                        continue
                    parameter.fill_(
                        1 if "scale" in name or name == "norm.weight" else 0
                    )
                for projection in (layer.in_proj, layer.out_proj):
                    if options.get("nonzero_weights", False):
                        projection.weight.fill_(0x11)
                    expected_kernel = (
                        B12xNvFp4W4A16LinearKernel
                        if options.get("disable_marlin", False)
                        else MarlinNvFp4LinearKernel
                    )
                    assert type(projection.quant_method.kernel) is expected_kernel
                    projection.quant_method.process_weights_after_loading(projection)
                    assert projection.input_scale.retired
                    assert torch.isnan(projection.input_scale).all()
                if options.get("nonzero_weights", False):
                    layer.conv1d.weight.fill_(0.125)
                    layer.D.fill_(1)
            layer.eval()
            spec = layer.get_kv_cache_spec(config)
            num_blocks = max_num_seqs + 1
            pages = torch.zeros(
                (num_blocks, 1, 1, spec.page_size_bytes), dtype=torch.int8
            )
            context = config.compilation_config.static_forward_context
            bind_kv_cache_to_layers({layer.prefix: pages}, context)
            kv = KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=[KVCacheGroupSpec([layer.prefix], spec)],
            )
            initialize_replayssm_ring_trackers(kv, context)
            starts, history = get_replayssm_block_copy_tensors(context)
            initialize_mamba_ssu_backend(config.mamba_config, kv, use_replayssm=True)
            builder = Mamba2AttentionMetadataBuilder(
                spec, [layer.prefix], config, device
            )
            hidden = torch.ones(
                (max(16, max_num_seqs * configured_tokens), 4096),
                dtype=torch.bfloat16,
            )
            if options.get("nonzero_weights", False):
                hidden.div_(4096)
        assert can_use_marlin_input_conv(layer, hidden[:8], None) == expected_admission
        try:
            yield config, layer, pages, starts, history, builder, hidden
        finally:
            clear_layer_kv_caches([layer])


@pytest.mark.parametrize(
    "owned_mamba",
    [dict(linear_backend="b12x", moe_backend="b12x", nonzero_weights=True)],
    indirect=True,
)
def test_gb10_mamba_ownership_does_not_change_attention_or_global_backend(owned_mamba):
    """Only Mamba's two methods select Marlin before their single post-load pack."""
    config, layer, *_ = owned_mamba
    assert type(layer.in_proj.quant_method.kernel) is MarlinNvFp4LinearKernel
    assert type(layer.out_proj.quant_method.kernel) is MarlinNvFp4LinearKernel
    quant = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    with set_current_vllm_config(config), torch.device("cuda"):
        attention = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[4096],
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix="model.layers.1.attention.q_proj",
        )
    assert type(attention.quant_method.kernel) is B12xNvFp4W4A16LinearKernel
    assert config.kernel_config.linear_backend == "b12x"
    assert config.kernel_config.moe_backend == "b12x"
    assert config.kernel_config.linear_backend_per_quant is None


@pytest.mark.parametrize(
    "owned_mamba",
    [dict(linear_backend="b12x", disable_marlin=True, expected_admission=False)],
    indirect=True,
)
def test_gb10_mamba_respects_disabled_marlin_kernel(owned_mamba):
    """A disabled kernel keeps each real Mamba projection on its selected backend."""
    config, layer, *_ = owned_mamba
    assert type(layer.in_proj.quant_method.kernel) is B12xNvFp4W4A16LinearKernel
    assert type(layer.out_proj.quant_method.kernel) is B12xNvFp4W4A16LinearKernel
    assert config.kernel_config.linear_backend == "b12x"
    assert config.kernel_config.linear_backend_per_quant is None


@pytest.mark.parametrize(
    "owned_mamba",
    [
        dict(checkpoint_dtype="float16", sr_rng="shared_seed", nonzero_weights=True),
        dict(
            checkpoint_dtype="float16",
            sr_rng="shared_seed",
            sr_rounds=0,
            nonzero_weights=True,
        ),
        dict(state_dtype="int8", sr_rng="philox", nonzero_weights=True),
        dict(state_dtype="int8", sr_rng="shared_seed", nonzero_weights=True),
    ],
    indirect=True,
    ids=[
        "fp16-shared-seed",
        "fp16-shared-default-rounds",
        "int8-philox",
        "int8-shared-seed",
    ],
)
@pytest.mark.parametrize(("batches", "query_tokens"), [(1, 8), (2, 8), (1, 4)])
@torch.inference_mode()
def test_fused_layer_uses_bound_checkpoint_and_scale_cache(
    owned_mamba, batches, query_tokens
):
    """Real bound caches and both SR policies survive fused decode and flush."""
    config, layer, pages, starts, history, builder, hidden = owned_mamba
    tokens = batches * query_tokens
    quantized = layer.replayssm_state_dtype != "auto"
    assert len(layer.kv_cache) == (6 if quantized else 5)
    slots = torch.arange(1, batches + 1, device="cuda", dtype=torch.int32)
    query = torch.arange(0, tokens + 1, query_tokens, dtype=torch.int32)
    seq = torch.full((batches,), 256, dtype=torch.int32)
    common = _decode_metadata(
        query,
        seq,
        slots[:, None],
        batches=batches,
        tokens=tokens,
        max_query=query_tokens,
        physical=tokens,
    )
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=torch.full((batches,), 4, dtype=torch.int32, device="cuda"),
    )
    assert _runtime_metadata(layer, metadata) is not None
    pages.zero_()
    if quantized:
        layer.initialize_replayssm_state(
            torch.full(
                (batches, 128, 64, 128), 0.25, dtype=torch.float32, device="cuda"
            ),
            slots,
        )
        assert (layer.kv_cache[5][1 : batches + 1] > 0).all()
    else:
        layer.kv_cache[1][1 : batches + 1].fill_(0.25)
    for state in (layer.kv_cache[0], layer.kv_cache[2], layer.kv_cache[4]):
        state[1 : batches + 1].fill_(0.125)
    layer.kv_cache[3][1 : batches + 1].fill_(0.0625)
    initial_pages = pages.clone()
    rng = torch.cuda.get_rng_state()

    def invoke(live, *, fused):
        pages.copy_(initial_pages)
        starts.fill_(5)
        history.zero_()
        history[1 : batches + 1] = live
        torch.cuda.set_rng_state(rng)
        with set_forward_context({layer.prefix: metadata}, config, num_tokens=tokens):
            if fused:
                output = layer(hidden[:tokens])
            else:
                projected, _ = layer.in_proj(hidden[:tokens])
                ssm_output = torch.empty(
                    (tokens, 8192), device="cuda", dtype=torch.bfloat16
                )
                layer.conv_ssm_forward(projected, ssm_output)
                normalized = layer.norm(ssm_output, projected[:, :8192])
                # Hold the independently selected output policy common while
                # comparing fused versus separate input/convolution paths.
                from vllm.model_executor.layers.mamba.ops.marlin_input_conv import (
                    can_use_marlin_output,
                    run_marlin_output,
                )

                if can_use_marlin_output(layer, normalized):
                    output = run_marlin_output(layer, normalized)
                else:
                    output, _ = layer.out_proj(normalized)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all() and output.any()
        return output.clone(), pages.clone(), starts.clone(), history.clone()

    for live in (4, 15):
        reference = invoke(live, fused=False)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            actual = invoke(live, fused=True)
        for expected, result in zip(reference, actual):
            torch.testing.assert_close(expected, result, rtol=0, atol=0)
        counts = {event.key: event.count for event in prof.key_averages()}
        assert counts.get("_C::mamba_marlin_input_conv", 0) == 1


@pytest.mark.parametrize("trace_tokens", [8, 16])
@torch.inference_mode()
def test_dynamic_tokens_keep_real_marlin_mamba_dispatch(owned_mamba, trace_tokens):
    config, layer, pages, starts, history, builder, hidden = owned_mamba
    inputs = {8: hidden[:8], 16: hidden}

    def invoke(tokens):
        batch = tokens // 8
        pages.zero_()
        starts.fill_(5)
        history.zero_()
        history[1] = 9
        query = torch.arange(0, tokens + 1, 8, dtype=torch.int32)
        seq = torch.full((batch,), 256, dtype=torch.int32)
        common = _decode_metadata(
            query,
            seq,
            torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).reshape(
                batch, 1
            ),
            batches=batch,
            tokens=tokens,
            max_query=8,
            physical=tokens,
        )
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.full(
                (batch,), 4, dtype=torch.int32, device="cuda"
            ),
        )
        assert metadata.num_decodes == batch and metadata.num_decode_tokens == tokens
        rng = torch.cuda.get_rng_state()
        with set_forward_context({layer.prefix: metadata}, config, num_tokens=tokens):
            output = layer(inputs[tokens])
        torch.cuda.synchronize()
        assert torch.equal(rng, torch.cuda.get_rng_state())
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(state).all() for state in layer.kv_cache)
        assert not pages[0].any() and (batch == 2 or not pages[2].any())
        return output.clone(), pages.clone(), starts.clone(), history.clone()

    # Ordinary eager references and warmup, using the same real owner/cache.
    expected = {tokens: invoke(tokens) for tokens in (8, 16)}
    for value in inputs.values():
        torch._dynamo.mark_dynamic(value, 0)
    assert not torch._dynamo.config.suppress_errors
    compile_model_with_stock_torch(layer, config)
    invoke(trace_tokens)  # Compile with a dynamic row count.
    # Warm both compiled input paths before profiling runtime dispatch.
    invoke(8)
    invoke(16)
    for tokens in (8, 16):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            actual = invoke(tokens)
        for reference, candidate in zip(expected[tokens], actual):
            torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
        counts = {event.key: event.count for event in prof.key_averages()}
        assert counts.get("vllm::mamba_mixer2_with_input_projection", 0) == 1
        assert counts.get("_C::mamba_marlin_input_conv", 0) == 1
        assert counts.get("vllm::mamba_mixer2_output_projection", 0) == 1
        assert counts.get("_C::mamba_marlin_output", 0) == (tokens == 8)


@pytest.mark.parametrize("owned_mamba", range(1, 9), indirect=True)
@torch.inference_mode()
def test_configured_short_drafts_and_runtime_shorter_queries(owned_mamba):
    """Real config T-1 plus shorter current queries retain opaque runtime fusion."""
    config, layer, pages, starts, history, builder, hidden = owned_mamba
    maximum = 1 + layer.num_spec
    inputs = {tokens: hidden[:tokens] for tokens in range(1, maximum + 1)}

    def invoke(tokens, *, legacy=False):
        pages.zero_()
        # Nontrivial carry makes full-cache comparison observable despite the
        # synthetic projection's deterministic checkpoint-format weights.
        layer.kv_cache[0][1].fill_(0.125)
        starts.fill_(5)
        history.zero_()
        query = torch.tensor([0, tokens], dtype=torch.int32)
        seq = torch.tensor([256], dtype=torch.int32)
        common = _decode_metadata(
            query,
            seq,
            torch.tensor([[1]], device="cuda", dtype=torch.int32),
            batches=1,
            tokens=tokens,
            max_query=tokens,
            physical=tokens,
        )
        accepted = (
            torch.tensor([maximum], device="cuda", dtype=torch.int32)
            if layer.num_spec > 0
            else None
        )
        metadata = builder.build(0, common, num_accepted_tokens=accepted)
        assert metadata.num_decodes == 1 and metadata.num_decode_tokens == tokens
        allowed = layer._marlin_input_conv_allowed
        if legacy:
            layer._marlin_input_conv_allowed = False
        rng = torch.cuda.get_rng_state()
        try:
            with set_forward_context(
                {layer.prefix: metadata}, config, num_tokens=tokens
            ):
                output = layer(inputs[tokens])
        finally:
            layer._marlin_input_conv_allowed = allowed
        torch.cuda.synchronize()
        assert torch.equal(rng, torch.cuda.get_rng_state())
        return output.clone(), pages.clone(), starts.clone(), history.clone()

    references = {t: invoke(t, legacy=True) for t in inputs}
    for value in inputs.values():
        if value.shape[0] > 1:
            torch._dynamo.mark_dynamic(value, 0)
    compile_model_with_stock_torch(layer, config)
    invoke(maximum)
    # Normal Torch0/1 specialization is permitted, not suppressed or forced.
    for tokens in inputs:
        invoke(tokens)
    for tokens in (*inputs, maximum):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            actual = invoke(tokens)
        for reference, candidate in zip(references[tokens], actual):
            torch.testing.assert_close(reference, candidate, rtol=0, atol=0)
        counts = {event.key: event.count for event in prof.key_averages()}
        assert counts.get("vllm::mamba_mixer2_with_input_projection", 0) == 1
        assert counts.get("_C::mamba_marlin_input_conv", 0) == 1


@pytest.mark.parametrize(
    "owned_mamba",
    [
        dict(
            admission_only=True,
            adaptive=True,
            capture_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            expected_admission=False,
        ),
        dict(
            admission_only=True,
            capture_mode=CUDAGraphMode.FULL,
            expected_admission=False,
        ),
        dict(
            admission_only=True,
            capture_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            expected_admission=True,
        ),
    ],
    indirect=True,
    ids=["adaptive-excluded", "mixed-full-excluded", "uniform-decode-eligible"],
)
@torch.inference_mode()
def test_normal_constructor_capture_admission_and_padded_fallback(owned_mamba):
    """Uniform decode may fuse; padded/adaptive contracts retain original code."""
    config, layer, pages, starts, history, builder, hidden = owned_mamba
    expected_fusion = not (
        config.speculative_config.enable_adaptive_verification
        or config.compilation_config.cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
    )
    assert layer._marlin_input_conv_allowed == expected_fusion

    def invoke(logical_tokens, legacy=False):
        pages.zero_()
        layer.kv_cache[0][1].fill_(0.125)
        starts.fill_(5)
        history.zero_()
        query = torch.tensor([0, logical_tokens], dtype=torch.int32)
        seq = torch.tensor([256], dtype=torch.int32)
        common = _decode_metadata(
            query,
            seq,
            torch.tensor([[1]], device="cuda", dtype=torch.int32),
            batches=1,
            tokens=logical_tokens,
            max_query=logical_tokens,
            physical=8,
        )
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.tensor([8], device="cuda", dtype=torch.int32),
        )
        assert metadata.num_decode_tokens == logical_tokens
        allowed = layer._marlin_input_conv_allowed
        if legacy:
            layer._marlin_input_conv_allowed = False
        rng = torch.cuda.get_rng_state()
        try:
            with set_forward_context({layer.prefix: metadata}, config, num_tokens=8):
                output = layer(hidden[:8])
        finally:
            layer._marlin_input_conv_allowed = allowed
        torch.cuda.synchronize()
        assert torch.equal(rng, torch.cuda.get_rng_state())
        # Original padding rows have no semantic output. Compare all live rows
        # plus the full cache/trackers, rather than inventing padding semantics.
        return (
            output[:logical_tokens].clone(),
            pages.clone(),
            starts.clone(),
            history.clone(),
        )

    for logical_tokens in (8,) if expected_fusion else (1, 7, 8):
        reference = invoke(logical_tokens, legacy=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            actual = invoke(logical_tokens)
        for old, new in zip(reference, actual):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        counts = {event.key: event.count for event in prof.key_averages()}
        assert counts.get("_C::mamba_marlin_input_conv", 0) == expected_fusion
        assert (
            counts.get("vllm::mamba_mixer2_with_input_projection", 0) == expected_fusion
        )


@pytest.mark.parametrize(
    "owned_mamba,batches,lengths,physical",
    [
        (
            dict(configured_tokens=1, max_num_seqs=4, admission_only=True),
            4,
            [1, 1, 1, 1],
            4,
        ),
        (
            dict(configured_tokens=2, max_num_seqs=4, admission_only=True),
            4,
            [2, 2, 2, 2],
            8,
        ),
        (dict(configured_tokens=3, max_num_seqs=3, admission_only=True), 2, [3, 3], 6),
        (dict(configured_tokens=3, max_num_seqs=3, admission_only=True), 2, [3, 2], 6),
        (dict(configured_tokens=3, max_num_seqs=3, admission_only=True), 2, [2, 2], 4),
        (
            dict(configured_tokens=2, max_num_seqs=4, admission_only=True),
            3,
            [2, 2, 2],
            8,
        ),
        (
            dict(configured_tokens=5, max_num_seqs=3, admission_only=True),
            3,
            [5, 5, 5],
            15,
        ),
        (
            dict(configured_tokens=8, max_num_seqs=3, admission_only=True),
            3,
            [8, 8, 8],
            24,
        ),
        (
            dict(configured_tokens=8, max_num_seqs=4, admission_only=True),
            4,
            [8, 8, 8, 8],
            32,
        ),
    ],
    indirect=["owned_mamba"],
    ids=[
        "regular-b4",
        "full-b4",
        "odd-t3",
        "ragged",
        "uniform-short",
        "padded",
        "m16",
        "m24",
        "m32",
    ],
)
@torch.inference_mode()
def test_batched_real_builder_admission_and_fallback(
    owned_mamba, batches, lengths, physical
):
    """Real owners/builders fuse only full uniform logical and physical rows."""
    config, layer, pages, starts, history, builder, hidden = owned_mamba
    assert layer._marlin_input_conv_allowed
    total = sum(lengths)
    query = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    seq = torch.full((batches,), 256, dtype=torch.int32)
    common = _decode_metadata(
        query,
        seq,
        torch.arange(1, batches + 1, device="cuda", dtype=torch.int32).reshape(
            batches, 1
        ),
        batches=batches,
        tokens=total,
        max_query=max(lengths),
        physical=physical,
    )
    accepted = (
        torch.arange(1, batches + 1, device="cuda", dtype=torch.int32).clamp_max(
            1 + layer.num_spec
        )
        if layer.num_spec
        else None
    )
    metadata = builder.build(0, common, num_accepted_tokens=accepted)
    assert metadata.num_decodes == batches and metadata.num_decode_tokens == total
    logical_uniform = total == batches * (1 + layer.num_spec)
    assert (_runtime_metadata(layer, metadata) is not None) == logical_uniform
    expected_fusion = (
        logical_uniform
        and physical == total
        and _batched_shape_is_profitable(batches, 1 + layer.num_spec)
    )

    def invoke(legacy=False):
        pages.zero_()
        for page in range(1, batches + 1):
            layer.kv_cache[0][page].fill_(page * 0.125)
        starts.fill_(5)
        history.zero_()
        allowed = layer._marlin_input_conv_allowed
        if legacy:
            layer._marlin_input_conv_allowed = False
        rng = torch.cuda.get_rng_state()
        try:
            with set_forward_context(
                {layer.prefix: metadata}, config, num_tokens=physical
            ):
                output = layer(hidden[:physical])
        finally:
            layer._marlin_input_conv_allowed = allowed
        torch.cuda.synchronize()
        assert torch.equal(rng, torch.cuda.get_rng_state())
        return output[:total].clone(), pages.clone(), starts.clone(), history.clone()

    reference = invoke(legacy=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        actual = invoke()
    for old, new in zip(reference, actual):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    fused = sum(
        event.count
        for event in prof.key_averages()
        if event.key.startswith("_C::mamba_marlin_input_conv")
    )
    assert fused == int(expected_fusion)


@pytest.mark.parametrize(
    "owned_mamba",
    [
        dict(
            configured_tokens=tokens,
            max_num_seqs=4,
            admission_only=True,
            capture_tokens=4 * tokens,
            capture_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        )
        for tokens in (2, 3, 8)
    ],
    indirect=True,
)
@torch.inference_mode()
def test_batched_normal_graph_replay_reads_padded_null_requests(owned_mamba):
    """Full-capacity graph padding uses real builder buffers and null slots."""
    config, layer, pages, starts, history, builder, hidden = owned_mamba
    assert layer._marlin_input_conv_allowed
    tokens = 1 + layer.num_spec
    rows = 4 * tokens
    query = torch.arange(0, rows + 1, tokens, dtype=torch.int32)
    seq = torch.full((4,), 256, dtype=torch.int32)
    common = _decode_metadata(
        query,
        seq,
        torch.arange(1, 5, device="cuda", dtype=torch.int32).reshape(4, 1),
        batches=4,
        tokens=rows,
        max_query=tokens,
        physical=rows,
    )
    metadata = builder.build_for_cudagraph_capture(common)
    assert _runtime_metadata(layer, metadata) is not None
    index_pointer = metadata.state_indices_tensor_d.data_ptr()
    acceptance_pointer = metadata.num_accepted_tokens.data_ptr()

    def reset():
        pages.zero_()
        for page in range(1, 5):
            layer.kv_cache[0][page].fill_(page * 0.125)
        starts.fill_(5)
        history.zero_()

    def invoke(legacy=False):
        allowed = layer._marlin_input_conv_allowed
        if legacy:
            layer._marlin_input_conv_allowed = False
        try:
            with set_forward_context(
                {layer.prefix: metadata},
                config,
                num_tokens=rows,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
            ):
                return layer(hidden[:rows])
        finally:
            layer._marlin_input_conv_allowed = allowed

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            reset()
            invoke()
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = invoke()
    torch.cuda.current_stream().wait_stream(stream)
    try:
        for slots, live in (([1, 2, 3, 0], 3), ([4, 2, 0, 0], 2)):
            common.block_table_tensor.copy_(
                torch.tensor(slots, device="cuda", dtype=torch.int32).reshape(4, 1)
            )
            accepted = torch.tensor(
                [1 if slot else -999 for slot in slots],
                device="cuda",
                dtype=torch.int32,
            )
            current = builder.build(0, common, num_accepted_tokens=accepted)
            assert current.state_indices_tensor_d.data_ptr() == index_pointer
            assert current.num_accepted_tokens.data_ptr() == acceptance_pointer
            reset()
            expected_output = invoke(legacy=True)[: live * tokens].clone()
            expected = pages.clone(), starts.clone(), history.clone()
            reset()
            pristine = pages.clone()
            rng = torch.cuda.get_rng_state()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(rng, torch.cuda.get_rng_state())
            torch.testing.assert_close(
                captured_output[: live * tokens], expected_output, rtol=0, atol=0
            )
            for actual, reference in zip((pages, starts, history), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for page in set(range(5)) - set(slots):
                assert torch.equal(pages[page], pristine[page])
    finally:
        graph.reset()


@pytest.mark.parametrize(
    "owned_mamba",
    [
        dict(
            configured_tokens=t,
            max_num_seqs=4,
            admission_only=True,
            capture_tokens=4 * t,
            capture_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            decode_request_sizes=[4],
        )
        for t in (2, 3, 8)
    ],
    indirect=True,
)
@torch.inference_mode()
def test_batched_normal_worker_preparation_replays_live_requests(owned_mamba):
    """Actual worker preparation pads liveB4→3→2 for one normal B4 graph."""
    config, layer, pages, starts, history, builder, hidden = owned_mamba
    assert layer._marlin_input_conv_allowed
    assert config.cache_config.mamba_cache_mode != "align"
    tokens, rows = 1 + layer.num_spec, 4 * (1 + layer.num_spec)
    device = hidden.device
    buffers = InputBuffers(4, rows, device)
    tables = BlockTables([1024], 4, rows, [1], device)
    for request in range(4):
        tables.append_block_ids(request, ([request + 1],), overwrite=True)
    tables.apply_staged_writes()

    class LayerModel(torch.nn.Module):
        """Give the real single layer the normal model-forward signature."""

        def __init__(self):
            super().__init__()
            self.mixer = layer
            self.register_buffer("hidden", hidden)
            self.captured_metadata = None

        def forward(self, input_ids, positions, intermediate_tensors=None):
            self.captured_metadata = get_forward_context().attn_metadata[layer.prefix]
            return self.mixer(self.hidden[: input_ids.shape[0]])

    model = LayerModel()
    # Fixture already used the normal binder/ring initialization. Actual manager
    # capture and runtime preparation share this real model-state owner.
    state = MambaHybridModelState(config, model, None, device)
    state.num_accepted_tokens_gpu.copy_(
        torch.arange(1, 5, device=device, dtype=torch.int32).clamp_max(tokens)
    )
    spec = layer.get_kv_cache_spec(config)
    group = AttentionGroup(
        Mamba2AttentionBackend,
        [layer.prefix],
        spec,
        kv_cache_group_id=0,
        metadata_builders=[builder],
    )
    kv = KVCacheConfig(
        num_blocks=5,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([layer.prefix], spec)],
    )
    manager = ModelCudaGraphManager(
        config,
        device,
        CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_lens={tokens: 4},
        varlen_decode=False,
    )

    def prepare(live, for_capture=False):
        batch = InputBatch.make_dummy(
            live, live * tokens, buffers, max_query_len=tokens
        )
        batch.num_reqs_after_padding, batch.num_tokens_after_padding = 4, rows
        # The normal helper produced these persistent GPU tails; don't fill them
        # by hand. Copies expose their actual contents to the ordinary builder.
        batch.query_start_loc = buffers.query_start_loc
        batch.query_start_loc_np = buffers.query_start_loc.cpu().numpy()
        batch.seq_lens = buffers.seq_lens
        batch.seq_lens_cpu_upper_bound = buffers.seq_lens.cpu()
        batch.positions = buffers.positions[:rows]
        batch.num_draft_tokens_per_req = batch.num_scheduled_tokens - 1
        blocks = tables.gather_block_tables(batch.idx_mapping, num_reqs_padded=4)
        slots = tables.compute_slot_mappings(
            batch.idx_mapping,
            batch.query_start_loc,
            batch.positions,
            num_tokens_padded=rows,
        )
        result = state.prepare_attn(
            batch,
            CUDAGraphMode.FULL,
            blocks,
            slots,
            [[group]],
            kv,
            for_capture=for_capture,
        )[layer.prefix]
        assert result.num_decodes == 4 and result.num_decode_tokens == rows
        assert torch.equal(blocks[0][live:], torch.zeros_like(blocks[0][live:]))
        assert torch.equal(
            result.num_accepted_tokens[live:],
            torch.ones_like(result.num_accepted_tokens[live:]),
        )
        assert not result.state_indices_tensor_d[live:].any()
        assert not result.replayssm_state_indices_d[live:].any()
        assert torch.all(batch.query_start_loc[live:] == live * tokens)
        assert (
            get_uniform_decode_token_count(live, live * tokens, tokens, True) == tokens
        )
        return result

    def reset():
        pages.zero_()
        for page in range(1, 5):
            layer.kv_cache[0][page].fill_(page * 0.125)
        starts.fill_(5)
        history.zero_()

    def invoke(metadata, mode=CUDAGraphMode.NONE, legacy=False):
        allowed = layer._marlin_input_conv_allowed
        if legacy:
            layer._marlin_input_conv_allowed = False
        try:
            with set_forward_context(
                {layer.prefix: metadata},
                config,
                num_tokens=rows,
                cudagraph_runtime_mode=mode,
            ):
                return layer(hidden[:rows])
        finally:
            layer._marlin_input_conv_allowed = allowed

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            manager.capture(model, state, buffers, None, tables, [[group]], kv)
        fused_calls = sum(
            event.count
            for event in prof.key_averages()
            if event.key.startswith("_C::mamba_marlin_input_conv")
        )
        assert bool(fused_calls) == _batched_shape_is_profitable(4, tokens)
        assert len(manager.graphs) == 1
        original = model.captured_metadata
        assert original is not None
        assert original.num_decodes == 4 and original.num_decode_tokens == rows
        for live in (4, 3, 2):
            metadata = prepare(live)
            desc = manager.dispatch(
                live, live * tokens, tokens, 0, max_query_len=tokens
            )
            assert desc.cg_mode == CUDAGraphMode.FULL and desc.num_reqs == 4
            for field in (
                "state_indices_tensor_d",
                "replayssm_state_indices_d",
                "query_start_loc_d",
                "num_accepted_tokens",
            ):
                assert (
                    getattr(metadata, field).data_ptr()
                    == getattr(original, field).data_ptr()
                )
            for actual, reference in zip(
                metadata.replayssm_scratch, original.replayssm_scratch
            ):
                assert actual.data_ptr() == reference.data_ptr()
            reset()
            expected_output = invoke(metadata, legacy=True)[: live * tokens].clone()
            expected = pages.clone(), starts.clone(), history.clone()
            reset()
            pristine = pages.clone()
            rng = torch.cuda.get_rng_state()
            actual_output = manager.run_fullgraph(desc)
            torch.cuda.synchronize()
            assert torch.equal(rng, torch.cuda.get_rng_state())
            torch.testing.assert_close(
                actual_output[: live * tokens], expected_output, rtol=0, atol=0
            )
            for actual, reference in zip((pages, starts, history), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert torch.equal(pages[live + 1 :], pristine[live + 1 :])
    finally:
        for graph in manager.graphs.values():
            graph.reset()
        manager.release_graphs()
