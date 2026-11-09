# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny TP numerical tests and a manual real-checkpoint smoke test."""

import json
import math
import os
from copy import deepcopy
from functools import partial
from operator import attrgetter

import pytest
import torch
from transformers import NemotronHConfig, NemotronHForCausalLM

from vllm import SamplingParams
from vllm.model_executor.models.nemotron_h_elastic import ElasticLinear
from vllm.sampling_params import BeamSearchParams, RequestOutputKind
from vllm.v1.core.kv_cache_utils import get_group_id
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec


@pytest.fixture
def tiny_elastic_checkpoint(tmp_path, request):
    torch.manual_seed(0)
    full_intermediate, sliced_intermediate, use_bias = getattr(
        request, "param", (128, 64, False)
    )
    config = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=full_intermediate,
        vocab_size=256,
        num_hidden_layers=3,
        hybrid_override_pattern="M*E",
        mamba_num_heads=4,
        mamba_head_dim=64,
        ssm_state_size=16,
        n_groups=2 if use_bias else 1,
        conv_kernel=4,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_shared_expert_intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        eos_token_id=2,
        bos_token_id=1,
        elastic_use_lazy_slicing=True,
        elastic_thinking={
            "elastic_hidden_size": 64,
            "elastic_moe_intermediate_size": sliced_intermediate,
        },
        elastic_think_end_token_id=13,
        use_bias=use_bias,
    )
    full = NemotronHForCausalLM(config).to(torch.bfloat16).eval()
    # Avoid near-tied expert choices changing between BF16 kernel implementations.
    full.model.layers[2].mixer.gate.e_score_correction_bias.copy_(
        torch.arange(config.n_routed_experts)
    )
    if config.use_bias:
        with torch.no_grad():
            full.model.layers[0].mixer.out_proj.bias.copy_(
                torch.linspace(-0.1, 0.1, config.hidden_size)
            )
    full.save_pretrained(tmp_path)
    sliced_config = deepcopy(config)
    sliced_config.hidden_size = 64
    sliced_config.moe_intermediate_size = sliced_intermediate
    sliced = NemotronHForCausalLM(sliced_config).to(torch.bfloat16).eval()
    # Materialize the sliced reference; inference does not use the Elastic wrappers.
    full_weights = full.state_dict()
    sliced.load_state_dict(
        {
            name: full_weights[name][
                tuple(slice(0, dim) for dim in tensor.shape)
            ].clone()
            for name, tensor in sliced.state_dict().items()
        }
    )
    return str(tmp_path), {"full": full, "sliced": sliced}


def dense_logprobs(output):
    return torch.tensor(
        [[row[i].logprob for i in range(256)] for row in output.logprobs]
    )


def joint_cache_block_size(llm):
    # TP workers update their local cache config; use the scheduler's final specs.
    groups = (
        llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config.kv_cache_groups
    )
    return math.lcm(*(group.kv_cache_spec.block_size for group in groups))


@torch.inference_mode()
def reference_logprobs(
    models, prompt, trace, phase, reset_after=None, producer=None, cached_tokens=0
):
    current = "sliced" if phase == "auto" else phase
    cache = None
    inputs = torch.tensor([prompt])
    if cached_tokens:
        cache = models[producer](
            inputs[:, :cached_tokens], use_cache=True
        ).past_key_values
        inputs = inputs[:, cached_tokens:]
    scores = []
    for index, token in enumerate(trace):
        if index == reset_after:
            cache = None
            inputs = torch.tensor([prompt + trace[:index]])
        result = models[current](inputs, past_key_values=cache, use_cache=True)
        scores.append(result.logits[0, -1].float().log_softmax(-1))
        cache = result.past_key_values
        inputs = torch.tensor([[token]])
        if phase == "auto" and token == 13:
            current = "full"
    return torch.stack(scores)


@pytest.fixture(
    params=[1, pytest.param(2, marks=pytest.mark.distributed(num_gpus=2))],
    ids=["tp1", "tp2"],
)
def tensor_parallel_size(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["eager", "graphs"])
def tiny_elastic_engine(
    vllm_runner, monkeypatch, tiny_elastic_checkpoint, tensor_parallel_size, request
):
    path, models = tiny_elastic_checkpoint
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with vllm_runner(
        path,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        distributed_timeout_seconds=120,
        cpu_distributed_timeout_seconds=120,
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=80,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        gpu_memory_utilization=0.08,
        async_scheduling=False,
        enforce_eager=request.param,
        enable_prefix_caching=True,
        enable_prompt_embeds=True,
        enable_chunked_prefill=True,
        enable_trace_replay=True,
        max_logprobs=256,
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": [1, 2],
        },
    ) as runner:
        yield runner.llm, models


@pytest.mark.hybrid_model
def test_elastic_logits_and_recovery(tiny_elastic_engine, record_property):
    """Check logits, including after a real EngineCore cache reset/preemption."""
    llm, models = tiny_elastic_engine
    errors = []
    trace = [4, 13, 5, 6]
    block_size = joint_cache_block_size(llm)
    for prompt_len in (
        17,
        block_size - 1,
        block_size,
        block_size + 1,
        2 * block_size + 1,
    ):
        prompt = [1] + [42] * (prompt_len - 1)
        for phase in ("sliced", "full", "auto"):
            params = SamplingParams(
                temperature=0,
                max_tokens=len(trace),
                ignore_eos=True,
                logprobs=256,
                trace_decode_token_ids=trace,
                extra_args={"elastic_phase": phase},
            )
            assert llm.reset_prefix_cache()
            result = llm.generate({"prompt_token_ids": prompt}, params)[0].outputs[0]
            expected = reference_logprobs(models, prompt, trace, phase)
            repeated_reference = reference_logprobs(models, prompt, trace, phase)
            torch.testing.assert_close(expected, repeated_reference, rtol=0, atol=0)
            errors.append(
                {
                    "prompt_tokens": prompt_len,
                    "phase": phase,
                    "max_logprob_error": (dense_logprobs(result) - expected)
                    .abs()
                    .max()
                    .item(),
                }
            )
            torch.testing.assert_close(
                dense_logprobs(result), expected, rtol=0, atol=0.01
            )
            if prompt_len == 17:
                assert llm.reset_prefix_cache()
                repeated = llm.generate({"prompt_token_ids": prompt}, params)[0]
                torch.testing.assert_close(
                    dense_logprobs(result),
                    dense_logprobs(repeated.outputs[0]),
                    rtol=0,
                    atol=0,
                )
            if phase != "auto":
                continue
            for reset_after in (1, 2, 3):
                assert llm.reset_prefix_cache()
                engine = llm.llm_engine
                recovery_params = deepcopy(params)
                recovery_params.output_kind = RequestOutputKind.CUMULATIVE
                engine.add_request(
                    "recover", {"prompt_token_ids": prompt}, recovery_params
                )
                reset = False
                for _ in range(32):
                    outputs = engine.step()
                    if not outputs:
                        continue
                    output = outputs[0]
                    if not reset and len(output.outputs[0].token_ids) >= reset_after:
                        assert not output.finished
                        assert len(output.outputs[0].token_ids) == reset_after
                        assert llm.reset_prefix_cache(reset_running_requests=True)
                        reset = True
                    if output.finished:
                        break
                assert output.finished and reset
                assert output.outputs[0].token_ids == trace
                recovered = reference_logprobs(
                    models, prompt, trace, phase, reset_after=reset_after
                )
                torch.testing.assert_close(
                    dense_logprobs(output.outputs[0]), recovered, rtol=0, atol=0.01
                )
    record_property("numerical_errors", json.dumps(errors))
    # Input markers do not transition; missing/repeated generated markers and
    # stopping on the marker use the same one-way boundary.
    for selected, tokens, stop, final_phase in (
        (None, [13, 4], [], "full"),
        ("full", [4, 5], [], "full"),
        ("sliced", [13, 4], [], "sliced"),
        ("auto", [4, 5], [], "sliced"),
        ("auto", [13, 13, 4], [], "full"),
        ("auto", [13, 4], [13], "sliced"),
    ):
        assert llm.reset_prefix_cache()
        edge_prompt = [1, 13, 42]
        edge_params = SamplingParams(
            temperature=0,
            max_tokens=len(tokens),
            logprobs=None if stop else 256,
            trace_decode_token_ids=None if stop else tokens,
            allowed_token_ids=[13] if stop else None,
            stop_token_ids=stop,
            extra_args={"elastic_phase": selected} if selected else None,
        )
        edge = llm.generate({"prompt_token_ids": edge_prompt}, edge_params)[0]
        generated = tokens[:1] if stop else tokens
        assert edge.outputs[0].token_ids == generated
        assert llm.apply_model(attrgetter("model.phase")) == [final_phase] * (
            llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
        )
        if stop:
            assert edge.outputs[0].finish_reason == "stop"
        else:
            torch.testing.assert_close(
                dense_logprobs(edge.outputs[0]),
                reference_logprobs(models, edge_prompt, generated, selected or "full"),
                rtol=0,
                atol=0.01,
            )
    assert llm.reset_prefix_cache()
    embeddings = models["full"].get_input_embeddings()(torch.tensor(prompt)).detach()
    embedded = llm.generate({"prompt_embeds": embeddings}, params)[0].outputs[0]
    torch.testing.assert_close(
        dense_logprobs(embedded), dense_logprobs(result), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="Beam search.*Elastic"):
        llm.beam_search(
            [{"prompt_token_ids": [1, 4]}],
            BeamSearchParams(beam_width=2, max_tokens=1),
        )


@pytest.mark.hybrid_model
@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("tiny_elastic_checkpoint", [(192, 128, True)], indirect=True)
def test_elastic_partial_expert_shard_matches_global_prefix(tiny_elastic_engine):
    """Check a 32-wide rank-one tail and nonzero row bias in eager/graph modes."""
    llm, models = tiny_elastic_engine
    prompt = [1, 13, 42]
    trace = [4, 13, 5]
    for phase in ("full", "sliced", "auto"):
        assert llm.reset_prefix_cache()
        output = llm.generate(
            {"prompt_token_ids": prompt},
            SamplingParams(
                temperature=0,
                max_tokens=len(trace),
                logprobs=256,
                trace_decode_token_ids=trace,
                extra_args={"elastic_phase": phase},
            ),
        )[0]
        torch.testing.assert_close(
            dense_logprobs(output.outputs[0]),
            reference_logprobs(models, prompt, trace, phase),
            rtol=0,
            atol=0.01,
        )


@pytest.mark.hybrid_model
def test_elastic_reuse_matches_producer_state(tiny_elastic_engine, record_property):
    """Compare all slice pairs at the actual joint attention/Mamba hit boundary."""
    llm, models = tiny_elastic_engine
    rows = []
    block_size = joint_cache_block_size(llm)
    for producer in ("sliced", "full"):
        for consumer in ("sliced", "full"):
            for size in (
                block_size - 1,
                block_size,
                block_size + 1,
                2 * block_size + 1,
            ):
                prompt = [1] + [42] * (size - 1)
                assert llm.reset_prefix_cache()
                seed = llm.generate(
                    {"prompt_token_ids": prompt},
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        extra_args={"elastic_phase": producer},
                    ),
                )[0]
                assert seed.num_cached_tokens == 0
                suffix_prompt = prompt + [45, 46, 47]
                trace = [4, 5, 6]
                params = SamplingParams(
                    temperature=0,
                    max_tokens=len(trace),
                    logprobs=256,
                    trace_decode_token_ids=trace,
                    extra_args={"elastic_phase": consumer},
                )
                result = llm.generate({"prompt_token_ids": suffix_prompt}, params)[0]
                cached = result.num_cached_tokens
                assert cached is not None
                # The producer keeps its last block writable for continuation.
                assert cached == (size - 1) // block_size * block_size
                expected = reference_logprobs(
                    models,
                    suffix_prompt,
                    trace,
                    consumer,
                    producer=producer,
                    cached_tokens=cached,
                )
                actual = dense_logprobs(result.outputs[0])
                rows.append(
                    {
                        "producer": producer,
                        "consumer": consumer,
                        "prompt_tokens": len(suffix_prompt),
                        "cached_tokens": cached,
                        "max_logprob_error": (actual - expected).abs().max().item(),
                    }
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0.01)

            # Existing request isolation and prompt-scoring restrictions still apply.
            for request_fields, sampling_fields in (
                ({"cache_salt": "another-tenant"}, {}),
                ({}, {"prompt_logprobs": 1}),
            ):
                miss = llm.generate(
                    {"prompt_token_ids": suffix_prompt, **request_fields},
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        extra_args={"elastic_phase": consumer},
                        **sampling_fields,
                    ),
                )[0]
                assert miss.num_cached_tokens == 0
            assert llm.reset_prefix_cache()
            cold = llm.generate({"prompt_token_ids": suffix_prompt}, params)[0]
            assert cold.num_cached_tokens == 0
            torch.testing.assert_close(
                dense_logprobs(cold.outputs[0]),
                reference_logprobs(models, suffix_prompt, trace, consumer),
                rtol=0,
                atol=0.01,
            )
    record_property("cross_slice_reuse", json.dumps(rows))


def snapshot_blocks(model, groups, block_ids):
    snapshots: list[torch.Tensor] = []
    for group, ids in zip(groups, block_ids):
        cache = model.get_submodule(group.layer_names[0]).kv_cache
        if isinstance(group.kv_cache_spec, MambaSpec):
            snapshots.extend(tensor[ids].detach().cpu().clone() for tensor in cache)
        else:
            # FlashAttention's current layout puts blocks first and packs K/V
            # together in the last dimension.
            assert cache.shape[-1] == 2 * group.kv_cache_spec.head_size
            snapshots.append(cache[ids].detach().cpu().clone())
    return snapshots


def weight_storage(model):
    for layer in model.modules():
        if isinstance(layer, ElasticLinear):
            assert layer.weight is layer.linear.weight
            view = layer.weight[:64, :64]
            assert (
                view.untyped_storage().data_ptr()
                == layer.weight.untyped_storage().data_ptr()
            )
    return {
        p.untyped_storage().data_ptr(): p.untyped_storage().nbytes()
        for p in model.parameters()
    }


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.hybrid_model
def test_elastic_isolation_and_cleanup(tiny_elastic_engine, record_property):
    """Source snapshots stay unchanged; private branches and aborts free state."""
    llm, models = tiny_elastic_engine
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager
    groups = scheduler.kv_cache_config.kv_cache_groups
    pool = manager.block_pool
    initial_free = pool.get_num_free_blocks()
    initial_weights = llm.apply_model(weight_storage)
    # Transformers registers the expert correction bias as a buffer; vLLM uses
    # a Parameter. Count the complete checkpoint state in the reference.
    checkpoint_numel = sum(p.numel() for p in models["full"].state_dict().values())
    rank_numel = llm.apply_model(parameter_count)
    if len(rank_numel) == 1:
        assert rank_numel == [checkpoint_numel]
    else:
        assert len(set(rank_numel)) == 1
        assert all(count < checkpoint_numel for count in rank_numel)
    cached_tokens = 2 * joint_cache_block_size(llm)
    prefix = [1] + [42] * cached_tokens
    checks = 0
    for producer in ("sliced", "full"):
        assert llm.reset_prefix_cache()
        llm.generate(
            {"prompt_token_ids": prefix},
            SamplingParams(
                temperature=0, max_tokens=1, extra_args={"elastic_phase": producer}
            ),
        )
        source_ids = [
            [
                b.block_id
                for b in pool.blocks
                if b.block_hash is not None and get_group_id(b.block_hash) == group_id
            ]
            for group_id in range(len(groups))
        ]
        assert all(source_ids)
        source = llm.apply_model(
            partial(snapshot_blocks, groups=groups, block_ids=source_ids)
        )
        branches = {}
        for index, consumer in enumerate(("sliced", "full")):
            prompt = prefix + [50 + index, 60 + index]
            trace = [4 + index, 6 + index, 8 + index, 10 + index]
            name = f"branch-{index}"
            internal = engine.add_request(
                name,
                {"prompt_token_ids": prompt},
                SamplingParams(
                    temperature=0,
                    max_tokens=4,
                    logprobs=256,
                    output_kind=RequestOutputKind.CUMULATIVE,
                    trace_decode_token_ids=trace,
                    extra_args={"elastic_phase": consumer},
                ),
            )
            branches[name] = (internal, prompt, trace, consumer)
        private: dict[str, tuple[int, list[list[torch.Tensor]]]] = {}
        finished = {}
        for _ in range(32):
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = output
            now = llm.apply_model(
                partial(snapshot_blocks, groups=groups, block_ids=source_ids)
            )
            for rank_source, rank_now in zip(source, now, strict=True):
                for before, after in zip(rank_source, rank_now, strict=True):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
            for name, (internal, _, _, _) in branches.items():
                req = scheduler.requests.get(internal)
                if req is None or not req.num_computed_tokens:
                    continue
                ids = manager.get_block_ids(internal)
                state = llm.apply_model(
                    partial(snapshot_blocks, groups=groups, block_ids=ids)
                )
                if name in private and private[name][0] == req.num_computed_tokens:
                    for rank_before, rank_after in zip(
                        private[name][1], state, strict=True
                    ):
                        for before, after in zip(rank_before, rank_after, strict=True):
                            torch.testing.assert_close(before, after, rtol=0, atol=0)
                    checks += 1
                private[name] = (req.num_computed_tokens, state)
            if not engine.has_unfinished_requests():
                break
        assert len(finished) == 2 and checks > 0
        for name, (_, prompt, trace, consumer) in branches.items():
            output = finished[name]
            assert output.num_cached_tokens == cached_tokens
            torch.testing.assert_close(
                dense_logprobs(output.outputs[0]),
                reference_logprobs(
                    models,
                    prompt,
                    trace,
                    consumer,
                    producer=producer,
                    cached_tokens=cached_tokens,
                ),
                rtol=0,
                atol=0.01,
            )
        assert pool.get_num_free_blocks() == initial_free
        assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)

    # Abort in a prefill chunk, immediately after the marker, and during full decode.
    for _ in range(3):
        for generated in (0, 2, 3):
            assert llm.reset_prefix_cache()
            internal = engine.add_request(
                "abort",
                {"prompt_token_ids": prefix},
                SamplingParams(
                    temperature=0,
                    max_tokens=6,
                    output_kind=RequestOutputKind.CUMULATIVE,
                    trace_decode_token_ids=[4, 13, 5, 6, 7, 8],
                    extra_args={"elastic_phase": "auto"},
                ),
            )
            for _ in range(16):
                engine.step()
                req = scheduler.requests[internal]
                if req.num_output_tokens == generated:
                    break
            else:
                pytest.fail("Did not reach the cancellation boundary")
            assert not req.is_finished()
            if generated == 0:
                assert 0 < req.num_computed_tokens < req.num_prompt_tokens
            else:
                assert scheduler.elastic.request_phase(req) == "full"
            engine.abort_request(["abort"])
            assert not engine.has_unfinished_requests()
            assert not scheduler.requests
            assert all(
                not m.req_to_blocks for m in manager.coordinator.single_type_managers
            )
            for m in manager.coordinator.single_type_managers:
                if isinstance(m, MambaManager):
                    assert not m.last_state_block_idx
                    assert not m._allocated_block_reqs and not m._checkpoints
            assert pool.get_num_free_blocks() == initial_free
            assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)
            assert llm.reset_prefix_cache()
            assert len(pool.cached_block_hash_to_block) == 0
    assert llm.apply_model(weight_storage) == initial_weights
    record_property(
        "state_checks",
        json.dumps(
            {
                "unchanged_other_branch_steps": checks,
                "abort_cycles": 9,
                "free_blocks": initial_free,
                "weight_storage_bytes_per_rank": [
                    sum(weights.values()) for weights in initial_weights
                ],
            }
        ),
    )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_elastic_switching_and_shared_cache(vllm_runner, monkeypatch, enforce_eager):
    # Keep scheduler inspection in-process; TP workers use serializable callables.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = os.environ.get(
        "ELASTIC_TEST_MODEL", "nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16"
    )
    public_model = model == "nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16"
    # The public checkpoint needs an explicit overlay for runtime slicing.
    hf_overrides = (
        {
            "elastic_use_lazy_slicing": True,
            "elastic_thinking": {
                "elastic_hidden_size": 1920,
                "elastic_moe_intermediate_size": 960,
            },
            "elastic_think_end_token_id": 13,
        }
        if public_model
        else {}
    )
    tp_size = int(os.environ.get("ELASTIC_TEST_TP_SIZE", "1"))
    with vllm_runner(
        model,
        revision="4bcc4326801e1ef6b8ec1c281edd69cb2e041964" if public_model else None,
        hf_overrides=hf_overrides,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        distributed_timeout_seconds=120,
        cpu_distributed_timeout_seconds=120,
        trust_remote_code=False,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=float(os.environ.get("ELASTIC_TEST_GPU_MEMORY", "0.88")),
        async_scheduling=False,
        enforce_eager=enforce_eager,
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": [1, 2, 4, 8],
        },
        enable_trace_replay=True,
    ) as runner:
        llm = runner.llm
        markers = llm.apply_model(attrgetter("config.elastic_think_end_token_id"))
        assert len(markers) == tp_size and len(set(markers)) == 1
        marker = markers[0]
        for mode, expected in (
            ("sliced", "sliced"),
            ("full", "full"),
            ("auto", "full"),
        ):
            params = SamplingParams(
                temperature=0,
                max_tokens=3,
                trace_decode_token_ids=[marker, 42, 43],
                extra_args={"elastic_phase": mode},
            )
            output = llm.generate("Complete this sentence:", params)[0]
            assert output.outputs[0].token_ids == [marker, 42, 43]
            assert llm.apply_model(attrgetter("model.phase")) == [expected] * tp_size
        # Both widths must progress when submitted together.
        params = [
            SamplingParams(
                temperature=0,
                max_tokens=4,
                ignore_eos=True,
                extra_args={"elastic_phase": mode},
            )
            for mode in ("sliced", "full") * 4
        ]
        outputs = llm.generate([f"Count from {i}:" for i in range(8)], params)
        assert len(outputs) == 8
        assert all(len(o.outputs[0].token_ids) == 4 for o in outputs)
        # The second width reuses the first width's real cache blocks.
        llm.reset_prefix_cache()
        prefix = {"prompt_token_ids": [42] * 5000}
        cached = []
        for mode in ("sliced", "full", "sliced"):
            output = llm.generate(
                prefix,
                SamplingParams(
                    temperature=0, max_tokens=1, extra_args={"elastic_phase": mode}
                ),
            )[0]
            cached.append(output.num_cached_tokens)
        assert cached[0] == 0
        assert cached[1] > 0 and cached[2] > 0
