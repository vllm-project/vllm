# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm.models.minicpm5_moe.xpu.specialized_decode.config import (
    MAX_CONTEXT_LEN,
    SHAPES,
    native_context_limit,
    native_rows,
    requested,
    serves,
    unsupported,
)
from vllm.models.minicpm5_moe.xpu.specialized_decode.sampling import (
    Candidates,
    CandidateSampler,
    candidate_head_supported,
    request_ok,
)
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    ("head_dtype", "width", "start", "supported"),
    [
        (torch.float16, 65280, 65280, True),
        (torch.bfloat16, 65280, 65280, True),
        (torch.float32, 65280, 65280, False),
        (torch.float16, 65024, 65024, False),
        (torch.float16, 65280, 0, False),
    ],
)
def test_candidate_head_matches_native_dtype_and_vocab(
    head_dtype, width, start, supported
):
    head = NS(
        tp_size=2,
        num_embeddings_per_partition=width,
        num_embeddings_padded=2 * width,
        org_vocab_size=2 * width,
        shard_indices=NS(org_vocab_start_index=start),
    )
    processor = NS(
        org_vocab_size=head.org_vocab_size,
        head_dtype=head_dtype,
        logits_as_input=False,
    )
    assert candidate_head_supported(head, processor, 1) is supported


def test_async_step_fault_rejects_output_after_copy_event(monkeypatch):
    from contextlib import nullcontext

    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu import async_utils

    events = []
    copy_event = NS(
        record=lambda stream: events.append("copy recorded"),
        synchronize=lambda: events.append("copy completed"),
    )
    monkeypatch.setattr(torch, "Event", lambda **kwargs: copy_event)
    monkeypatch.setattr(async_utils, "stream", lambda *_: nullcontext())
    copy_stream = NS(wait_stream=lambda _stream: None)
    sampler_output = NS(
        sampled_token_ids=torch.tensor([[123]]),
        logprobs_tensors=None,
        num_nans=None,
        sampling_mask_tensors=None,
    )
    failure_flag = torch.zeros(1, dtype=torch.int32)
    for failed in (True, False):
        failure_flag.fill_(failed)
        output = ModelRunnerOutput(req_ids=["request"], sampled_token_ids=None)
        async_output = async_utils.AsyncOutput(
            model_runner_output=output,
            sampler_output=sampler_output,
            num_sampled_tokens=torch.tensor([1]),
            main_stream=object(),
            copy_stream=copy_stream,
            check_ep_fault=False,
            pending_aux_output=None,
            step_failure=(failure_flag, "native failure"),
        )
        failure_flag.zero_()
        if failed:
            with pytest.raises(RuntimeError, match="native failure"):
                async_output.get_output()
            assert output.sampled_token_ids is None
        else:
            assert async_output.get_output().sampled_token_ids == [[123]]
    assert events == ["copy recorded", "copy completed"] * 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("logprobs", 1),
        ("presence_penalty", 0.1),
        ("frequency_penalty", 0.1),
        ("repetition_penalty", 1.1),
        ("logit_bias", {1: 1.0}),
        ("bad_words", ["city"]),
        ("allowed_token_ids", [1, 2]),
        ("min_tokens", 1),
        ("min_p", 0.1),
        ("thinking_token_budget", 8),
        ("structured_outputs", object()),
        ("logprob_token_ids", [1]),
    ],
)
def test_full_vocabulary_sampling_features_refuse_candidates(field, value):
    params = SamplingParams(temperature=0)
    setattr(params, field, value)
    assert not request_ok(params)


def test_seeded_random_sampling_retains_ordinary_draws_and_greedy_is_eligible():
    assert request_ok(SamplingParams(temperature=0, seed=7))
    assert not request_ok(SamplingParams(temperature=0.8, top_k=64))
    assert not request_ok(SamplingParams(temperature=0.8, top_k=65))
    assert not request_ok(SamplingParams(temperature=0.8, top_k=8, seed=7))


def test_candidate_plan_is_consumed_and_request_slot_reuse_does_not_leak():
    candidates = Candidates.__new__(Candidates)
    candidates.enabled = True
    candidates.ok = np.zeros(4, dtype=bool)
    candidates.indices = torch.ones(2, 128)
    candidates.add_request(1, SamplingParams(temperature=0))
    candidates.add_request(3, SamplingParams(temperature=0))
    batch = NS(
        has_prefill=False,
        has_structured_output_reqs=False,
        num_draft_tokens=0,
        num_reqs=2,
        num_scheduled_tokens=np.array([1, 1]),
        idx_mapping_np=np.array([3, 1]),
    )
    candidates.batch = batch
    assert candidates.plan(2, None) == 1
    assert candidates.indices is None
    assert candidates.plan(2, None) is None
    candidates.add_request(3, SamplingParams(temperature=0, logprobs=1))
    candidates.batch = batch
    assert candidates.plan(2, None) is None
    candidates.add_request(3, SamplingParams(temperature=0))
    batch.has_prefill = True
    candidates.batch = batch
    assert candidates.plan(2, None) is None
    batch.has_prefill = False
    candidates.batch = batch
    assert candidates.plan(2, object()) is None


def test_candidate_sampler_maps_global_ids_once_and_preserves_fallback():
    class Greedy:
        def __call__(self, logits, batch):
            return NS(sampled_token_ids=logits.argmax(-1, keepdim=True).int())

    candidates = Candidates.__new__(Candidates)
    candidates.indices = torch.tensor([[4, 10], [5, 11]])
    sampler = CandidateSampler(Greedy(), candidates)
    logits = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    assert sampler(logits, None).sampled_token_ids.tolist() == [[10], [5]]
    assert sampler(logits, None).sampled_token_ids.tolist() == [[1], [0]]
    candidates.indices = torch.ones(2, 128)
    with pytest.raises(RuntimeError, match="token IDs disagree"):
        sampler(logits, None)


def deployment(tp):
    return NS(
        model_config=NS(
            dtype="torch.bfloat16",
            quantization="fp8_per_tensor",
            enable_sleep_mode=False,
            max_model_len=32768,
            enforce_eager=True,
            architectures=["MiniCPM5MoEForCausalLM"],
            hf_text_config=NS(**SHAPES),
        ),
        parallel_config=NS(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
            enable_batch_sharded_sampling=False,
        ),
        cache_config=NS(cache_dtype="auto", block_size=64),
        compilation_config=NS(mode=NS(name="NONE"), cudagraph_mode=None),
        kernel_config=NS(monokernel=False),
        additional_config={},
        lora_config=None,
        kv_transfer_config=None,
        weight_transfer_config=None,
        offload_config=NS(uva=NS(cpu_offload_gb=0), prefetch=NS(offload_group_size=0)),
        aux_output_config=NS(enable_return_routed_experts=False),
        speculative_config=None,
    )


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_exact_model_shape_is_eligible_at_both_tp_sizes(tp, dtype):
    config = deployment(tp)
    config.model_config.dtype = "torch." + dtype
    assert unsupported(config) == []
    config.cache_config.cache_dtype = dtype
    assert unsupported(config) == []
    config.cache_config.cache_dtype = "float16" if dtype == "bfloat16" else "bfloat16"
    assert "KV cache must match activation dtype" in unsupported(config)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("tensor_parallel_size", 4, "TP1 or TP2 required"),
        ("enable_expert_parallel", True, "expert parallelism"),
        ("enable_eplb", True, "EPLB"),
    ],
)
def test_unsupported_parallel_feature_is_refused(field, value, reason):
    config = deployment(1)
    setattr(config.parallel_config, field, value)
    assert reason in unsupported(config)


def test_reports_shape_and_precision_errors_together():
    config = deployment(1)
    config.model_config.hf_text_config.n_routed_experts = 256
    config.model_config.dtype = "torch.float32"
    reasons = unsupported(config)
    assert "n_routed_experts must be 160" in reasons
    assert "BF16 or FP16 activations required" in reasons


@pytest.mark.parametrize("draft_tokens", [1, 3, 7])
@pytest.mark.parametrize("requests", [1, 2, 8])
def test_dspark_verify_is_eligible_without_enabling_other_speculative_methods(
    draft_tokens,
    requests,
):
    config = deployment(2)
    config.speculative_config = NS(method="dspark", num_speculative_tokens=draft_tokens)
    assert unsupported(config) == []
    metadata = NS(
        max_query_len=1 + draft_tokens,
        num_actual_tokens=requests * (1 + draft_tokens),
        max_seq_len=65536,
        minicpm5_decode_eligible=True,
        minicpm5_verify_eligible=True,
    )
    rows = requests * (1 + draft_tokens)
    assert serves(metadata, rows, 65536)
    metadata.causal = False
    assert not serves(metadata, rows, 65536)
    metadata.causal = True
    metadata.minicpm5_verify_eligible = False
    assert not serves(metadata, rows, 65536)


@pytest.mark.parametrize("method,tokens", [("eagle3", 7), ("dflash", 7), ("dspark", 8)])
def test_unqualified_speculative_method_or_native_width_is_refused(method, tokens):
    config = deployment(2)
    config.speculative_config = NS(method=method, num_speculative_tokens=tokens)
    assert "speculative decoding requires DSpark with 1..7 draft tokens" in unsupported(
        config
    )


def test_mono_returns_the_same_target_features_as_the_ordinary_decoder(monkeypatch):
    from vllm.model_executor.models import minicpm5_moe
    from vllm.models.minicpm5_moe.xpu.specialized_decode.decode import SpecializedDecode

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.prefix = str(index)

        def forward(self, positions, hidden, residual):
            if residual is None:
                residual = hidden
            return hidden * (1 + self.index / 128), residual + self.index / 64

        def launch(self, metadata, positions, hidden, residual):
            return self(positions, hidden, residual)

    class Norm(torch.nn.Module):
        def forward(self, hidden, residual):
            return (hidden + residual) * 0.5, residual

    model = minicpm5_moe.MiniCPM5Model.__new__(minicpm5_moe.MiniCPM5Model)
    torch.nn.Module.__init__(model)
    model.do_not_compile = True
    model.start_layer, model.end_layer = 0, 28
    model.layers = torch.nn.ModuleList([Layer(i) for i in range(28)])
    model.norm = Norm()
    model.aux_hidden_state_layers = (0, 2, 8, 14, 20, 26)
    monkeypatch.setattr(
        minicpm5_moe, "get_pp_group", lambda: NS(is_first_rank=True, is_last_rank=True)
    )
    positions = torch.arange(8)
    hidden = torch.randn(8, 16, dtype=torch.float16)
    expected, expected_aux = model(None, positions, inputs_embeds=hidden)
    decode = SpecializedDecode.__new__(SpecializedDecode)
    decode.layers = dict(enumerate(model.layers))
    decode.plan = lambda *args: None
    decode.attach(model)
    fallback, _ = model(None, positions, inputs_embeds=hidden)
    torch.testing.assert_close(fallback, expected, atol=0, rtol=0)
    decode.plan = lambda *args: {str(i): object() for i in range(28)}
    actual, actual_aux = model(None, positions, inputs_embeds=hidden)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(actual_aux) == 6
    for observed, reference in zip(actual_aux, expected_aux, strict=True):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [8, 16])
def test_verify_graph_expands_live_request_ownership_and_rollback(rows):
    from vllm.models.minicpm5_moe.xpu.specialized_decode.metadata import VerifyMetadata
    from vllm.platforms import current_platform

    if not current_platform.is_xpu():
        pytest.skip("XPU graph replay requires an Intel GPU")
    device = "xpu:0"
    torch.xpu.set_device(device)
    buffers = VerifyMetadata(256, 64, device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)
    metadata = NS(
        query_start_loc=torch.tensor([0, 3, rows], dtype=torch.int32, device=device),
        block_table=torch.tensor(
            [[2, 1, 0], [5, 4, 3]], dtype=torch.int32, device=device
        ),
        slot_mapping=torch.full((rows,), -1, dtype=torch.int64, device=device),
    )
    assert buffers.supports(metadata)
    buffers.expand(metadata, positions)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=torch.xpu.current_stream()):
        observed = buffers.expand(metadata, positions)
    for starts, offsets in (
        ([0, 3, rows], [129, 64]),
        ([0, 2, 6], [127, 63]),
        ([0, 0, 4], [0, 126]),
    ):
        pos = torch.zeros(rows, dtype=torch.int64)
        slots = torch.full((rows,), -1, dtype=torch.int64)
        expected_blocks = torch.zeros(rows, 4, dtype=torch.int32)
        expected_lengths = torch.zeros(rows, dtype=torch.int32)
        pages = metadata.block_table.cpu()
        for req, (start, end) in enumerate(zip(starts, starts[1:])):
            for row in range(start, end):
                position = offsets[req] + row - start
                pos[row] = position
                slots[row] = int(pages[req, position // 64]) * 64 + position % 64
                expected_lengths[row] = position + 1
                count = (position + 64) // 64
                expected_blocks[row, :count] = pages[req, :count]
        positions.copy_(pos)
        metadata.slot_mapping.copy_(slots)
        metadata.query_start_loc.copy_(torch.tensor(starts, dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(
            observed.seq_lens.cpu(), expected_lengths, atol=0, rtol=0
        )
        torch.testing.assert_close(
            observed.block_table.cpu(), expected_blocks, atol=0, rtol=0
        )


def test_automatic_block_size_waits_for_attention_backend_resolution():
    config = deployment(1)
    config.cache_config.block_size = 16
    config.cache_config.user_specified_block_size = False
    assert unsupported(config) == []
    config.cache_config.user_specified_block_size = True
    assert "cache block size must be a multiple of 64" in unsupported(config)


def test_normalized_default_rope_is_eligible_but_scaled_rope_is_refused():
    config = deployment(1)
    rope = {"rope_type": "default", "rope_theta": 5000000}
    config.model_config.hf_text_config.rope_parameters = rope
    assert unsupported(config) == []
    rope["rope_type"] = "linear"
    rope["factor"] = 2.0
    assert "attention bias or scaled RoPE" in unsupported(config)


@pytest.mark.parametrize("name", ["rms_norm", "fused_add_rms_norm"])
def test_different_normalization_rounding_provider_is_refused(name):
    config = deployment(1)
    config.kernel_config.ir_op_priority = NS(**{name: ["native"]})
    assert f"{name} must use the vllm_c rounding contract" in unsupported(config)


def test_mono_requires_an_explicit_request():
    config = deployment(1)
    assert not requested(config)
    config.additional_config = {"decode_backend": "latency_specialized"}
    assert requested(config)
    config.additional_config = {}
    config.kernel_config.monokernel = True
    assert requested(config)


@pytest.mark.parametrize("rows", range(1, 9))
def test_padded_whole_request_decode_is_served(rows):
    metadata = NS(
        max_query_len=1,
        num_actual_tokens=max(1, rows - 1),
        num_prefill_tokens=0,
        max_seq_len=32768,
        minicpm5_decode_eligible=True,
    )
    assert serves(metadata, rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_query_len", 2),
        ("num_actual_tokens", 0),
        ("num_actual_tokens", 9),
        ("num_prefill_tokens", 1),
        ("mm_prefix_query_range_tensor", object()),
        ("rswa_prefix_lens", object()),
        ("max_seq_len", 32769),
        ("minicpm5_decode_eligible", False),
    ],
)
def test_unsupported_step_falls_back_before_submission(field, value):
    metadata = NS(
        max_query_len=1,
        num_actual_tokens=4,
        num_prefill_tokens=0,
        max_seq_len=1024,
        minicpm5_decode_eligible=True,
    )
    setattr(metadata, field, value)
    assert not serves(metadata, 4)


def test_missing_metadata_and_uncompiled_row_count_fall_back():
    assert not serves(None, 1)
    assert not serves(NS(max_query_len=1, num_actual_tokens=3, max_seq_len=1024), 3)


def test_native_rows_preserves_old_abi3_capabilities():
    assert native_rows(NS()) == frozenset((1, 2, 4, 8))
    assert native_rows(NS(supported_rows=lambda: list(range(1, 9)))) == frozenset(
        range(1, 9)
    )


@pytest.mark.parametrize("length", [32769, 65536, 131072, MAX_CONTEXT_LEN])
def test_long_context_uses_the_configured_native_capability(length):
    config = deployment(2)
    config.model_config.max_model_len = length
    assert unsupported(config) == []
    metadata = NS(
        max_query_len=1,
        num_actual_tokens=1,
        num_prefill_tokens=0,
        max_seq_len=length,
        minicpm5_decode_eligible=True,
    )
    assert not serves(metadata, 1)
    assert serves(metadata, 1, length)
    assert not serves(metadata, 1, length - 1)
    metadata.max_seq_len = length + 1
    assert not serves(metadata, 1, length)


def test_native_context_limit_preserves_old_libraries_and_rejects_excess():
    assert native_context_limit(NS()) == 32768
    assert (
        native_context_limit(NS(max_context_len=lambda: MAX_CONTEXT_LEN))
        == MAX_CONTEXT_LEN
    )
    config = deployment(1)
    config.model_config.max_model_len = MAX_CONTEXT_LEN + 1
    assert f"mono context limit is {MAX_CONTEXT_LEN}" in unsupported(config)


@pytest.fixture
def bmg_device(monkeypatch):
    device = 4
    monkeypatch.setattr(torch.xpu, "current_device", lambda: device)
    monkeypatch.setattr(
        torch.ops._xpu_C, "is_bmg", lambda index: index == device, raising=False
    )
    return device


@pytest.mark.parametrize("api", ["legacy", "no_rope", "v1", "indexed"])
def test_deployment_checks_installed_context_capability_before_loading_state(
    api, monkeypatch, bmg_device
):
    from vllm_xpu_kernels import minicpm5_mono as native

    from vllm import distributed, envs
    from vllm.models.minicpm5_moe.xpu.specialized_decode import config as cfg
    from vllm.models.minicpm5_moe.xpu.specialized_decode import peer
    from vllm.platforms import current_platform

    ops = NS(supported=lambda: True, router_logits_dtype=lambda: "float32")
    if api != "legacy":
        ops.max_context_len = lambda: MAX_CONTEXT_LEN
    if api in ("v1", "indexed"):
        ops.rotary_cache_version = lambda: 1
    if api == "indexed":
        ops.supported = lambda: False
        ops.supports_device = lambda index: index == bmg_device
    monkeypatch.setattr(native, "load", lambda: ops)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(distributed, "get_tp_group", lambda: NS(cpu_group=object()))

    def agree(reasons, group, phase):
        if reasons:
            raise RuntimeError("; ".join(reasons))

    monkeypatch.setattr(peer, "agree", agree)
    config = deployment(2)
    config.model_config.max_model_len = 65536
    config.compilation_config.cudagraph_capture_sizes = []
    config.kernel_config.monokernel = True
    if api == "legacy":
        with pytest.raises(RuntimeError, match="installed limit is 32768"):
            cfg.configure(config)
    elif api == "no_rope":
        with pytest.raises(RuntimeError, match="canonical RoPE cache support"):
            cfg.configure(config)
    else:
        assert cfg.configure(config).max_context_len == 65536
        del ops.router_logits_dtype
        with pytest.raises(RuntimeError, match="FP32 router logits"):
            cfg.configure(config)


def test_tail_only_is_restricted_to_opted_in_tp1_fp16_graph(monkeypatch, bmg_device):
    from vllm_xpu_kernels import minicpm5_mono as native

    from vllm import distributed, envs
    from vllm.config import CUDAGraphMode
    from vllm.models.minicpm5_moe.xpu.specialized_decode import config as cfg
    from vllm.models.minicpm5_moe.xpu.specialized_decode import peer
    from vllm.platforms import current_platform

    ops = NS(
        supported=lambda: True,
        activation_dtypes=lambda: ["float16", "bfloat16"],
        router_logits_dtype=lambda: "float32",
    )
    monkeypatch.setattr(native, "load", lambda: ops)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(distributed, "get_tp_group", lambda: NS(cpu_group=object()))

    def agree(reasons, group, phase):
        assert not reasons, reasons

    monkeypatch.setattr(peer, "agree", agree)
    monkeypatch.delenv("MINICPM5_MONO_TAIL_ONLY", raising=False)
    monkeypatch.delenv("MINICPM5_MONO_TAIL_ROWS4", raising=False)
    monkeypatch.delenv("MINICPM5_MONO_PRE_O_SPLIT", raising=False)
    monkeypatch.delenv("MINICPM5_MONO_POST_O_SPLIT", raising=False)
    monkeypatch.delenv("MINICPM5_MONO_PRE_O_FA2_ADAPTIVE_SPLIT1", raising=False)
    config = deployment(1)
    config.model_config.dtype = torch.float16
    config.model_config.enforce_eager = False
    config.cache_config.enable_prefix_caching = False
    config.compilation_config.cudagraph_capture_sizes = [8]
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    assert cfg.configure(config) is None
    config.kernel_config.monokernel = True
    assert cfg.configure(config).tail_only
    assert cfg.configure(config).tail_rows4
    config.speculative_config = NS(method="dspark", num_speculative_tokens=7)
    assert not cfg.configure(config).tail_only
    assert not cfg.configure(config).tail_rows4
    config.speculative_config = None
    monkeypatch.setenv("MINICPM5_MONO_TAIL_ONLY", "0")
    assert not cfg.configure(config).tail_only
    assert cfg.configure(config).tail_rows4
    monkeypatch.setenv("MINICPM5_MONO_TAIL_ROWS4", "0")
    assert not cfg.configure(config).tail_rows4
    monkeypatch.delenv("MINICPM5_MONO_TAIL_ROWS4")
    monkeypatch.delenv("MINICPM5_MONO_TAIL_ONLY")
    config.cache_config.enable_prefix_caching = True
    assert not cfg.configure(config).tail_only
    assert not cfg.configure(config).tail_rows4
    config.cache_config.enable_prefix_caching = False
    config.parallel_config.tensor_parallel_size = 2
    assert not cfg.configure(config).tail_only
    assert not cfg.configure(config).tail_rows4
    assert cfg.configure(config).pre_o_split
    assert cfg.configure(config).post_o_split
    assert not cfg.configure(config).pre_o_fa2_adaptive_split1
    ops.post_o_abi_version = lambda: 1
    assert not cfg.configure(config).pre_o_fa2_adaptive_split1
    ops.decode_post_o = lambda *args: None
    assert cfg.configure(config).pre_o_fa2_adaptive_split1
    monkeypatch.setenv("MINICPM5_MONO_PRE_O_FA2_ADAPTIVE_SPLIT1", "0")
    assert cfg.configure(config).post_o_split
    assert not cfg.configure(config).pre_o_fa2_adaptive_split1
    monkeypatch.delenv("MINICPM5_MONO_PRE_O_FA2_ADAPTIVE_SPLIT1")
    monkeypatch.setenv("MINICPM5_MONO_POST_O_SPLIT", "0")
    assert cfg.configure(config).pre_o_split
    assert not cfg.configure(config).post_o_split
    assert not cfg.configure(config).pre_o_fa2_adaptive_split1
    monkeypatch.delenv("MINICPM5_MONO_POST_O_SPLIT")
    monkeypatch.setenv("MINICPM5_MONO_PRE_O_SPLIT", "0")
    assert not cfg.configure(config).pre_o_split
    assert not cfg.configure(config).post_o_split
    assert not cfg.configure(config).pre_o_fa2_adaptive_split1
    monkeypatch.setenv("MINICPM5_MONO_POST_O_SPLIT", "1")
    assert cfg.configure(config).post_o_split
    assert cfg.configure(config).pre_o_fa2_adaptive_split1
    monkeypatch.delenv("MINICPM5_MONO_PRE_O_SPLIT")
    config.cache_config.enable_prefix_caching = True
    assert not cfg.configure(config).pre_o_split
    assert not cfg.configure(config).post_o_split
    assert not cfg.configure(config).pre_o_fa2_adaptive_split1
    config.cache_config.enable_prefix_caching = False
    monkeypatch.delenv("MINICPM5_MONO_POST_O_SPLIT")
    config.kernel_config.monokernel = False
    config.additional_config = {"decode_backend": "latency_specialized"}
    assert cfg.configure(config).pre_o_split
    assert not cfg.configure(config).post_o_split
    assert not cfg.configure(config).pre_o_fa2_adaptive_split1
    config.parallel_config.tensor_parallel_size = 1
    assert cfg.configure(config).tail_only
    assert not cfg.configure(config).tail_rows4


@pytest.mark.parametrize(
    "pre_o,post_o,post_abi,rows,has_pre_o_rows4,expected",
    [
        (True, False, 1, 8, False, "pre"),
        (False, True, 1, 8, False, "post"),
        (True, True, 1, 8, True, "post"),
        (True, True, 0, 8, True, "pre"),
        (True, True, 1, 4, True, "pre4"),
        (True, True, 1, 4, False, "full"),
        (False, True, 1, 4, True, "full"),
    ],
)
def test_o_split_uses_native_capability_and_keeps_fa2_split_scoped(
    monkeypatch, pre_o, post_o, post_abi, rows, has_pre_o_rows4, expected
):
    from vllm.models.minicpm5_moe.xpu.specialized_decode import decode as runtime
    from vllm.models.minicpm5_moe.xpu.specialized_decode.config import DecodeConfig

    cache = torch.zeros(1, 1, 64, 256, dtype=torch.float16)
    kv = torch.zeros(1, 64, 1, 128, dtype=torch.float16)
    table = torch.zeros(rows, 1, dtype=torch.int32)
    positions = torch.arange(rows)
    slots = positions.clone()
    if rows == 4:
        slots[-1] = -1
    hidden = torch.ones(rows, 2048, dtype=torch.float16)
    fa2_splits, native_calls = [], []
    projection_calls: list[int] = []

    def fa2(**kwargs):
        fa2_splits.append(kwargs["num_splits"])
        return kwargs["q"].reshape_as(hidden)

    class BoundAttention:
        layer_name = "layer"
        kv_cache = cache
        impl = NS(vllm_flash_attn_version=2, dense_attention_kernel=fa2)

        def __call__(self, q, k, v):
            return self.impl.dense_attention_kernel(
                q=q.reshape(rows, 16, 128),
                k=kv,
                v=kv,
                block_table=table,
                fa_version=2,
                max_seqlen_q=1,
                num_splits=0,
            )

    def local_o(layer, tensor, bias):
        assert layer is projection and bias is None
        projection_calls.append(1)
        return tensor + 3

    projection = NS(
        input_is_parallel=True, bias=None, tp_size=2, quant_method=NS(apply=local_o)
    )
    layer = NS(
        mlp=NS(experts=object()),
        input_layernorm=lambda value: value,
        self_attn=NS(
            attn=BoundAttention(),
            o_proj=projection,
            gated=False,
            q_size=2048,
            kv_size=128,
            qkv_proj=lambda x: (torch.cat((x, x[:, :128], x[:, :128]), dim=-1), None),
            rotary_emb=lambda pos, q, k: (q, k),
        ),
    )

    def native(kind, value, residual, slot_mapping, out, next_residual, *args):
        assert slot_mapping is slots
        native_calls.append(kind)
        out.copy_(value)
        next_residual.copy_(residual)

    def full(value, residual, pos, blocks, lengths, slots, out, next_res, *args, **kw):
        assert kw["ignore_padding"] and pos is positions and blocks is table
        if residual is None:
            residual = value
        native("full", value, residual, slots, out, next_res)

    ops = NS(
        decode=full,
        pre_o_abi_version=lambda: 1,
        post_o_abi_version=lambda: post_abi,
        decode_pre_o=lambda *args: native("pre", *args),
        decode_post_o=lambda *args: native("post", *args),
        control_words=lambda: 304,
        validate=lambda *args: None,
        padding_mask_supported_tps=lambda: [2],
        padding_mask_version=lambda: 1,
        padding_mask_rows=lambda: [4, 8],
    )
    if has_pre_o_rows4:
        ops.decode_pre_o_rows4 = lambda *args: native("pre4", *args)
    monkeypatch.setattr(runtime, "layer_weights", lambda layer: [hidden])
    monkeypatch.setattr(runtime, "tiled_qkv_supported", lambda *args: False)
    config = DecodeConfig(
        tp=2,
        graph=True,
        pre_o_split=pre_o,
        post_o_split=post_o,
        pre_o_fa2_adaptive_split1=True,
    )
    mono = runtime.MonoLayer(layer, 1, config, [], None, ops)
    mono.bind()
    assert layer.self_attn.attn.impl.dense_attention_kernel is mono.pre_o_fa2
    metadata = NS(
        block_table=table,
        max_query_len=1,
        seq_lens=torch.ones(rows, dtype=torch.int32),
        slot_mapping=slots,
    )
    for force_one in (False, True, False):
        # The alternate Graph toggles the same wrapper only while capturing.
        mono.pre_o_fa2.force_one = force_one
        output, residual = mono.launch(metadata, positions, hidden, None)
        assert torch.equal(output, hidden + (3 if expected == "post" else 0))
        assert torch.equal(residual, hidden)
    assert fa2_splits == {"pre4": [0, 0, 0], "full": []}.get(expected, [0, 1, 0])
    assert native_calls == [expected] * 3
    assert len(projection_calls) == (3 if expected == "post" else 0)
    assert mono.pre_o_hits[rows] == (3 if expected in ("pre", "pre4") else 0)
    assert mono.post_o_hits[rows] == (3 if expected == "post" else 0)


@pytest.mark.parametrize(
    "case", ["eligible", "prefill", "rows4", "not_fa2", "window", "dtype", "page"]
)
def test_pre_o_fa2_split_one_is_scoped_to_exact_shape(case):
    from vllm.models.minicpm5_moe.xpu.specialized_decode.pre_o import PreOFa2SplitOne

    seen = []
    hook = PreOFa2SplitOne(lambda **kwargs: seen.append(kwargs.get("num_splits")))
    other = PreOFa2SplitOne(lambda **kwargs: seen.append(kwargs.get("num_splits")))
    hook.force_one = other.force_one = True
    rows = 4 if case == "rows4" else 8
    q = torch.empty(rows, 16, 128, dtype=torch.float16)
    k = torch.empty(1, 32 if case == "page" else 64, 1, 128, dtype=torch.float16)
    v = torch.empty_like(k, dtype=torch.bfloat16 if case == "dtype" else k.dtype)
    metadata = dict(
        fa_version=3 if case == "not_fa2" else 2,
        max_seqlen_q=4 if case == "prefill" else 1,
        max_seqlen_k=65536,
        block_table=torch.empty(rows, 1024, dtype=torch.int32),
        window_size=(63, 0) if case == "window" else (-1, -1),
    )

    def attention(q, k, v):
        other(q=q, k=k, v=v, **metadata)
        hook(q=q, k=k, v=v, **metadata)
        raise ValueError("test scope reset")

    hook(q=q, k=k, v=v, **metadata)
    with pytest.raises(ValueError, match="test scope reset"):
        hook.call_attention(attention, q, k, v)
    hook(q=q, k=k, v=v, **metadata)
    assert seen == [None, None, 1 if case == "eligible" else None, None]
    assert hook.calls == (case == "eligible")
    assert other.calls == 0


def test_pre_o_fa2_split_one_reaches_native_adapter(monkeypatch):
    from vllm import _xpu_ops as runtime
    from vllm.models.minicpm5_moe.xpu.specialized_decode.pre_o import PreOFa2SplitOne

    seen = []

    def kernel(**kwargs):
        seen.append(kwargs["num_splits_kv"])
        return kwargs["out"]

    monkeypatch.setattr(runtime, "flash_attn_varlen_func", kernel, raising=False)
    hook = PreOFa2SplitOne(runtime.xpu_ops.flash_attn_varlen_func)
    hook.force_one = True
    q = torch.empty(8, 16, 128, dtype=torch.float16)
    k = torch.empty(8, 64, 1, 128, dtype=torch.float16)

    def attention(q, k, v):
        return hook(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.arange(9, dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_k=65536,
            num_splits=0,
            fa_version=2,
            seqused_k=torch.full((8,), 128, dtype=torch.int32),
            block_table=torch.zeros(8, 1024, dtype=torch.int32),
        )

    attention(q, k, k)
    hook.call_attention(attention, q, k, k)
    assert seen == [None, 1]


@pytest.mark.parametrize(
    "rows,padded,enabled,version,verify,dense,expected",
    [
        (1, False, True, 1, False, False, "full"),
        (2, False, True, 1, False, False, "full"),
        (4, False, True, 1, False, False, "tail4"),
        (4, True, True, 1, False, False, "tail4"),
        (4, False, False, 1, False, False, "full"),
        (4, False, True, None, False, False, "full"),
        (4, False, True, 2, False, False, "full"),
        (4, False, True, 1, True, False, "full"),
        (4, False, True, 1, False, True, "full"),
        (8, False, True, None, False, False, "tail8"),
    ],
)
def test_rows4_tail_uses_independent_capability_and_preserves_mask(
    rows, padded, enabled, version, verify, dense, expected, monkeypatch
):
    from vllm.models.minicpm5_moe.xpu.specialized_decode import decode

    events = []
    hidden = torch.ones(rows, 16, dtype=torch.float16)
    positions = torch.arange(rows)
    cache = torch.empty(2, 2, 64, 256, dtype=torch.float16)
    slots = positions.clone()
    if padded:
        slots[1] = -1

    def full(*args, **kwargs):
        events.append("full")

    def tail(kind, normed, seen_slots, output, *args):
        torch.testing.assert_close(normed, torch.full_like(hidden, 6))
        assert seen_slots is slots
        output.copy_(normed)
        events.append(kind)

    def attention(pos, normed):
        assert pos is positions
        return normed + 2

    attention.attn = NS(layer_name="layer.0")  # type: ignore[attr-defined]
    layer = NS(
        mlp=NS() if dense else NS(experts=object()),
        self_attn=attention,
        input_layernorm=lambda x: x + 1,
        post_attention_layernorm=lambda x, residual: (x + 2, residual),
    )
    ops = NS(
        decode=full,
        tail_only_abi_version=lambda: 1,
        decode_tail=lambda *args: tail("tail8", *args),
        padding_mask_version=lambda: 1,
        padding_mask_supported_tps=lambda: [1, 2],
        padding_mask_rows=lambda: [4, 8],
        control_words=lambda: 304,
    )
    if version is not None:
        ops.tail_rows4_abi_version = lambda: version
        ops.decode_tail_rows4 = lambda *args: tail("tail4", *args)
    monkeypatch.setattr(decode, "layer_weights", lambda layer: [torch.empty(1)])
    monkeypatch.setattr(decode, "tiled_qkv_supported", lambda *args: False)
    monkeypatch.setattr(decode, "cache_tensor", lambda layer: cache)
    config = NS(tp=1, tail_only=True, tail_rows4=enabled)
    native = decode.MonoLayer(layer, 0, config, [], object(), ops)
    native.key = native.cache_key(cache)
    visible = NS(
        block_table=torch.zeros(rows, 4, dtype=torch.int32),
        slot_mapping=slots,
        seq_lens=torch.ones(rows, dtype=torch.int32),
        max_query_len=1,
        minicpm5_verify_eligible=verify,
    )
    _, passed_residual = native.launch(visible, positions, hidden, None)
    assert events == [expected]
    if expected.startswith("tail"):
        assert native.tail_hits[rows] == 1
        assert passed_residual is hidden
    else:
        assert not native.tail_hits


@pytest.mark.parametrize("gated", [False, True])
def test_pre_o_cut_matches_stock_attention_projection_input(gated):
    from vllm.model_executor.models.minicpm5_moe import MiniCPM5Attention
    from vllm.models.minicpm5_moe.xpu.specialized_decode.pre_o import (
        attention_before_output_projection,
    )

    positions = torch.arange(8)
    hidden = torch.arange(64).reshape(8, 8).half() / 32
    width = 8 * (2 if gated else 1) + 8
    projection = torch.arange(8 * width).reshape(8, width).half() / 512
    received = []

    def output_projection(value):
        received.append(value)
        return value * 2, None

    def rotary(position_ids, query, key):
        assert position_ids is positions
        return query + 1, key + 2

    layer = NS(
        q_size=8,
        kv_size=4,
        head_dim=4,
        num_heads=2,
        gated=gated,
        qkv_proj=lambda value: (value @ projection, None),
        rotary_emb=rotary,
        attn=lambda q, k, v: q + (k + v).repeat(1, 2),
        o_proj=output_projection,
    )
    stock = MiniCPM5Attention.forward(layer, positions, hidden)
    cut = attention_before_output_projection(layer, positions, hidden)
    assert len(received) == 1
    assert torch.equal(cut, received[0]) and torch.equal(stock, cut * 2)


@pytest.mark.parametrize("layout", ["valid", "short", "width", "dtype", "strided"])
def test_long_context_binds_model_rope_before_capture_and_passes_it_to_native(
    layout, monkeypatch
):
    from vllm.models.minicpm5_moe.xpu.specialized_decode import decode as runtime

    calls = []
    weights = [torch.ones(2048, dtype=torch.float16)]
    cache = torch.zeros(1, 1, 64, 256, dtype=torch.float16)
    shape = (
        65535 if layout == "short" else 65536,
        129 if layout == "width" else 128,
    )
    rope = torch.zeros(
        shape, dtype=torch.float32 if layout == "dtype" else torch.float16
    )
    if layout == "strided":
        rope = torch.zeros(65536, 256, dtype=torch.float16)[:, ::2]
    ops = NS(
        control_words=lambda: 304,
        validate=lambda *args: None,
        rotary_cache_version=lambda: 1,
        decode=lambda *args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(runtime, "cache_tensor", lambda layer: cache)
    monkeypatch.setattr(runtime, "layer_weights", lambda layer: weights)
    layer = runtime.MonoLayer(
        NS(
            mlp=NS(experts=object()),
            self_attn=NS(
                attn=NS(layer_name="layer0"),
                rotary_emb=NS(_match_cos_sin_cache_dtype=lambda query: rope),
            ),
        ),
        0,
        NS(tp=2, max_context_len=65536),
        [],
        object(),
        ops,
    )
    if layout != "valid":
        with pytest.raises(RuntimeError, match="canonical RoPE cache"):
            layer.bind()
        assert layer.key is None and not calls
        return
    layer.bind()
    assert layer.rope_cache is rope
    metadata = NS(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.long),
    )
    layer.launch(metadata, torch.zeros(1, dtype=torch.long), torch.zeros(1, 2048), None)
    assert calls == [{"rope_cache": rope}]


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("api", ["legacy", "v1", "unknown"])
def test_padding_contract_requires_native_capability_and_verified_tp_rows(
    tp, rows, api, monkeypatch
):
    """Legacy ABI3 and unverified shapes must never receive the new keyword."""
    from vllm.models.minicpm5_moe.xpu.specialized_decode import (
        decode as runtime,
    )

    calls = []
    ops = NS(
        control_words=lambda: 304,
        decode=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    if api != "legacy":
        ops.padding_mask_version = lambda: 1 if api == "v1" else 2
        ops.padding_mask_rows = lambda: [4, 8]
    cache = torch.zeros(1, 1, 64, 256)
    monkeypatch.setattr(runtime, "cache_tensor", lambda layer: cache)
    monkeypatch.setattr(runtime, "layer_weights", lambda layer: [torch.ones(2048)])
    layer = runtime.MonoLayer(
        NS(mlp=NS(experts=object()), self_attn=NS(attn=NS(layer_name="layer0"))),
        0,
        NS(tp=tp),
        [],
        object(),
        ops,
    )
    layer.key = layer.cache_key(cache)
    metadata = NS(
        block_table=torch.zeros(rows, 1, dtype=torch.int32),
        seq_lens=torch.ones(rows, dtype=torch.int32),
        slot_mapping=torch.full((rows,), -1, dtype=torch.long),
    )
    out, res = layer.launch(metadata, torch.zeros(rows), torch.zeros(rows, 2048), None)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[6] is out and args[7] is res
    assert kwargs == (
        {"ignore_padding": True} if tp == 2 and rows in (4, 8) and api == "v1" else {}
    )


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "disabled",
        "old_abi",
        "missing_op",
        "verify",
        "shared_table",
        "rows4",
        "bad_slots",
    ],
)
def test_tail_only_uses_original_attention_modules_and_residual(case, monkeypatch):
    from vllm.models.minicpm5_moe.xpu.specialized_decode import decode as runtime
    from vllm.models.minicpm5_moe.xpu.specialized_decode.config import DecodeConfig

    rows = 4 if case == "rows4" else 8
    hidden = torch.zeros(rows, 2048, dtype=torch.float16)
    residual = torch.ones_like(hidden)
    cache = torch.zeros(1, 2, 64, 256, dtype=torch.float16)
    positions = torch.arange(rows, dtype=torch.int64)
    slots = positions if case != "bad_slots" else positions.int()
    table = torch.zeros(rows, 1, dtype=torch.int32)
    if case == "shared_table":
        table = table[:1].expand(rows, -1)
    metadata = NS(
        block_table=table,
        max_query_len=2 if case == "verify" else 1,
        slot_mapping=slots,
        seq_lens=positions.int(),
    )
    calls = []

    def input_norm(value, residual_in):
        assert value is hidden and residual_in is residual
        calls.append("input_norm")
        return value + 3, residual_in + 4

    def attention(positions_in, value):
        assert positions_in is positions and (value == 3).all()
        calls.append("attention")
        return value + 10

    attention.attn = NS(layer_name="layer0", kv_cache=cache)  # type: ignore[attr-defined]

    def post_norm(value, residual_in):
        assert (value == 13).all() and (residual_in == 5).all()
        calls.append("post_norm")
        return value + 20, residual_in + 30

    def tail(
        normed, slots_in, output, workspace, control, weights, region, mailbox_slot
    ):
        assert (normed == 33).all() and slots_in is slots
        assert not workspace and control.numel() == 304 and mailbox_slot == 1
        assert weights[0].dtype == torch.float16 and region is not None
        calls.append("tail")
        output.copy_(normed + 40)

    def full(*args, **kwargs):
        calls.append("full")
        args[6].fill_(2)
        args[7].fill_(5)

    model = NS(
        mlp=NS(experts=object()),
        self_attn=attention,
        input_layernorm=input_norm,
        post_attention_layernorm=post_norm,
    )
    monkeypatch.setattr(
        runtime, "layer_weights", lambda _: [torch.empty(1, dtype=torch.float16)]
    )
    ops = NS(
        control_words=lambda: 304,
        validate=lambda *args: None,
        decode=full,
        padding_mask_supported_tps=lambda: (1,),
        padding_mask_version=lambda: 1,
        padding_mask_rows=lambda: [4, 8],
    )
    if case != "old_abi":
        ops.tail_only_abi_version = lambda: 1
    if case != "missing_op":
        ops.decode_tail = tail
    layer = runtime.MonoLayer(
        model,
        1,
        DecodeConfig(tp=1, graph=True, tail_only=case != "disabled"),
        [],
        object(),
        ops,
    )
    layer.bind()
    output, res = layer.launch(metadata, positions, hidden, residual)
    assert calls == (
        ["input_norm", "attention", "post_norm", "tail"]
        if case == "enabled"
        else ["full"]
    )
    assert torch.all(output == (73 if case == "enabled" else 2))
    assert torch.all(res == (35 if case == "enabled" else 5))
    assert layer.hits == {rows: 1} and layer.tail_hits == (
        {8: 1} if case == "enabled" else {}
    )


def test_uncompiled_rows_fall_back_before_stream_or_metadata_access():
    from vllm.models.minicpm5_moe.xpu.specialized_decode import (
        decode as runtime,
    )

    decode = runtime.SpecializedDecode.__new__(runtime.SpecializedDecode)
    decode.bound = True
    decode.rows = frozenset((1, 2, 4, 8))
    assert decode.plan(torch.zeros(3), None) is None
    assert decode.plan(torch.zeros(9), None) is None


def test_attention_prefill_counter_does_not_prove_pure_decode():
    metadata = NS(
        max_query_len=1,
        num_actual_tokens=1,
        num_prefill_tokens=0,
        max_seq_len=10,
    )
    assert not serves(metadata, 1)


@pytest.mark.parametrize(
    "graph,capturing", [(True, False), (False, False), (True, True)]
)
def test_capture_warmup_orders_stream_transition(graph, capturing, monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm.models.minicpm5_moe.xpu.specialized_decode import (
        decode as runtime,
    )

    class Stream:
        def wait_stream(self, previous):
            self.previous = previous

    original, current = Stream(), Stream()
    decode = runtime.SpecializedDecode.__new__(runtime.SpecializedDecode)
    decode.bound = True
    decode.rows = frozenset(range(1, 9))
    decode.stream = original
    decode.config = NS(graph=graph)
    rebinds = []
    decode.region = NS(rebind=lambda: rebinds.append(True))
    decode.layers = {}
    context = NS(cudagraph_runtime_mode=CUDAGraphMode.NONE, attn_metadata={})
    monkeypatch.setattr(runtime, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda: current)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: capturing)
    if graph and not capturing:
        assert decode.plan(torch.zeros(1), None) == {}
        assert current.previous is original
        assert decode.stream is current
        assert rebinds == [True]
    else:
        with pytest.raises(RuntimeError, match="different stream"):
            decode.plan(torch.zeros(1), None)


@pytest.mark.parametrize("distinct", [False, True])
@pytest.mark.parametrize("limit", [32768, MAX_CONTEXT_LEN])
def test_decode_checks_each_distinct_metadata_before_any_submission(
    distinct, limit, monkeypatch
):
    from vllm.config import CUDAGraphMode
    from vllm.models.minicpm5_moe.xpu.specialized_decode import (
        decode as runtime,
    )

    stream = object()
    decode = runtime.SpecializedDecode.__new__(runtime.SpecializedDecode)
    decode.bound = True
    decode.rows = frozenset(range(1, 9))
    decode.stream = stream
    decode.config = NS(graph=True, max_context_len=limit)
    decode.layers = {i: NS(prefix=str(i)) for i in range(28)}
    first, last = NS(eligible=True), NS(eligible=False)
    metadata = {str(i): first for i in range(28)}
    if distinct:
        metadata["27"] = last
    context = NS(cudagraph_runtime_mode=CUDAGraphMode.NONE, attn_metadata=metadata)
    calls = []

    def serves(item, rows, max_context_len):
        assert max_context_len == limit
        calls.append(item)
        return item.eligible

    monkeypatch.setattr(runtime, "serves", serves)
    monkeypatch.setattr(runtime, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda: stream)
    assert decode.plan(torch.zeros(8), None) == (None if distinct else metadata)
    assert calls == ([first, last] if distinct else [first])


def failstop_fixture(monkeypatch, rank=0, world=2):
    from vllm.models.minicpm5_moe.xpu.specialized_decode import guards

    zeros = torch.zeros
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kw: zeros(
            *args, **{k: v for k, v in kw.items() if k != "pin_memory"}
        ),
    )
    events, votes = [], []

    class Event:
        ready = False

        def __init__(self):
            events.append(self)

        def record(self):
            pass

        def query(self):
            return self.ready

    monkeypatch.setattr(torch.xpu, "Event", Event)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        guards.dist,
        "all_reduce",
        lambda value, **kw: votes.append(value.clone()),
    )
    health = torch.zeros(8, dtype=torch.int32)
    return guards.FailStop(health, rank, world, object()), health, events, votes


def test_shared_failstop_preserves_inflight_copy_and_sticky_completed_failure(
    monkeypatch,
):
    watch, health, events, votes = failstop_fixture(monkeypatch)
    watch.after_step()
    health[0] = 1
    for _ in range(3):
        watch.after_step()
    assert len(events) == 1 and not watch.seen
    events[-1].ready = True
    watch.after_step()  # queue failure readback, the prior landed copy was healthy
    events[-1].ready = True
    watch.after_step()
    assert watch.seen
    health[0] = 0  # a later healthy readback cannot undo a detected failure
    events[-1].ready = True
    for _ in range(9):
        watch.after_step()
    with pytest.raises(RuntimeError, match=r"TP ranks \[0\]"):
        watch.after_step()
    assert votes[0].tolist() == [1, 0]


def test_shared_failstop_reports_peer_failure_on_matching_cpu_collective(
    monkeypatch,
):
    from vllm.models.minicpm5_moe.xpu.specialized_decode import guards

    watch, _, _, _ = failstop_fixture(monkeypatch)

    def peer_failed(votes, **kw):
        votes[1] = 1

    monkeypatch.setattr(guards.dist, "all_reduce", peer_failed)
    for _ in range(15):
        watch.after_step()
    with pytest.raises(RuntimeError, match=r"TP ranks \[1\]"):
        watch.after_step()


def test_shared_failstop_does_not_schedule_host_work_inside_capture(
    monkeypatch,
):
    watch, _, events, votes = failstop_fixture(monkeypatch)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True)
    watch.after_step()
    assert watch.steps == 0 and not events and not votes


@pytest.mark.parametrize(
    "blocked", ["prefilling", "expanded", "processing", "logprobs"]
)
def test_direct_greedy_requires_one_completed_decode_logit(blocked):
    candidates = Candidates.__new__(Candidates)
    candidates.enabled = True
    candidates.ok = np.ones(1, dtype=bool)
    candidates.indices = None
    candidates.direct_sampler = NS(
        needs_logits_processing=np.array([blocked == "processing"]),
        get_logprobs_dims=lambda _: (1, 0) if blocked == "logprobs" else None,
    )
    batch = NS(
        has_prefill=False,
        has_structured_output_reqs=False,
        num_draft_tokens=0,
        num_reqs=1,
        num_scheduled_tokens=np.array([1]),
        idx_mapping_np=np.array([0]),
        is_prefilling_np=np.array([blocked == "prefilling"]),
        cu_num_logits_np=np.array([0, 2 if blocked == "expanded" else 1]),
    )
    candidates.batch = batch
    assert candidates.plan(1, None) == 1
    assert not candidates.direct_step
    batch.is_prefilling_np[:] = False
    batch.cu_num_logits_np[-1] = 1
    candidates.direct_sampler.needs_logits_processing[:] = False
    candidates.direct_sampler.get_logprobs_dims = lambda _: None
    candidates.batch = batch
    assert candidates.plan(1, None) == 1 and candidates.direct_step
    assert candidates.plan(1, None) is None and not candidates.direct_step


def test_direct_greedy_returns_counts_once_and_preserves_stock_fallback():
    class Greedy:
        def __call__(self, logits, batch):
            return NS(sampled_token_ids=logits.argmax(-1, keepdim=True))

    candidates = Candidates.__new__(Candidates)
    candidates.indices = torch.tensor([[4, 10]])
    sampler = CandidateSampler(Greedy(), candidates)
    logits, batch = torch.tensor([[1.0, 2.0]]), object()
    tokens = torch.tensor([[10]], dtype=torch.int64)
    counts, rejected = (
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    candidates.direct_outputs = tokens, counts, rejected
    candidates.direct_values, candidates.direct_batch = logits, batch
    out = sampler(logits, batch)
    assert out.sampled_token_ids is tokens and out.num_sampled is counts
    assert out.num_rejected is rejected
    assert out.logprobs_tensors is None and out.num_nans is None
    assert out.sampling_mask_tensors is None
    assert candidates.indices is None and candidates.direct_outputs is None
    assert sampler(logits, batch).sampled_token_ids.tolist() == [[1]]


def test_tiled_qkv_preserves_all_raw_fp8_bytes_and_kernel_view():
    from vllm.models.minicpm5_moe.xpu.specialized_decode.weights import tiled_qkv_weight

    raw = torch.arange(1024, dtype=torch.int32).remainder(256).byte().reshape(32, 32)
    weight = raw.view(torch.float8_e4m3fn).t()
    packed = tiled_qkv_weight(weight)
    expected = torch.cat(
        [
            raw[:16, :16].flatten(),
            raw[:16, 16:].flatten(),
            raw[16:, :16].flatten(),
            raw[16:, 16:].flatten(),
        ]
    )
    assert packed.dtype == weight.dtype and packed.shape == weight.shape
    assert packed.stride() == weight.stride()
    assert torch.equal(packed.t().view(torch.uint8).flatten(), expected)
    assert torch.equal(weight.t().view(torch.uint8), raw)


@pytest.mark.parametrize(
    "tp,dtype,bf_version,fp_version,capability_tps,expected",
    [
        (2, torch.bfloat16, 1, None, None, True),
        (2, torch.bfloat16, None, 1, None, False),
        (2, torch.bfloat16, 2, 1, None, False),
        (2, torch.float16, 1, 1, None, True),
        (2, torch.float16, 1, None, None, False),
        (2, torch.float16, 1, 0, None, False),
        (1, torch.bfloat16, 1, 1, (1, 2), False),
        (2, torch.float32, 1, 1, (1, 2), False),
        (1, torch.float16, 1, 1, None, False),
        (1, torch.float16, 1, 1, (1, 2), True),
        (1, torch.float16, 1, 1, (2,), False),
        (1, torch.float16, 1, 0, (1, 2), False),
        (3, torch.float16, 1, 1, (1, 2), False),
    ],
)
def test_tiled_layout_requires_per_dtype_native_capability(
    tp, dtype, bf_version, fp_version, capability_tps, expected
):
    from vllm.models.minicpm5_moe.xpu.specialized_decode.weights import (
        tiled_qkv_supported,
    )

    ops = NS()
    if bf_version is not None:
        ops.qkv_layout_version = lambda: bf_version
    if fp_version is not None:
        ops.fp16_qkv_layout_version = lambda: fp_version
    if capability_tps is not None:
        ops.fp16_qkv_supported_tps = lambda: capability_tps
    assert tiled_qkv_supported(ops, tp, dtype) is expected
