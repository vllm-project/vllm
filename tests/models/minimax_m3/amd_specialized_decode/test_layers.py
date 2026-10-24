# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer outputs and MonoKernel execution on a seeded five-layer TP4 MiniMax-M3.

Compare native/mono layer states; exercise FULL replay, EAGLE3 and fail-stop.
MXFP8 arithmetic is checked against an independent reference in test_numerics.py.
"""

import zlib

import pytest
import torch

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

LAYERS = 5
SPARSE = (3, 4)
VOCAB = 4096
STEPS = 8
BATCHES = (1, 3, 16)
SEED = 0
ENV = {
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
    # deterministic BF16 GEMMs (hipBLASLt) for the stock layers: the split-K
    # skinny and AITER tuned GEMMs reduce in a run-dependent order
    "VLLM_ROCM_USE_SKINNY_GEMM": "0",
    "VLLM_ROCM_USE_AITER_LINEAR": "0",
    "HIP_ONLINE_TUNING": "0",
}


def _supported() -> bool:
    if not current_platform.is_rocm() or torch.accelerator.device_count() < 4:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = [
    pytest.mark.distributed(num_gpus=4),
    pytest.mark.skipif(not _supported(), reason="needs 4 gfx950 GPUs"),
]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config

    layers = [0] * 3 + [1] * (LAYERS - 3)
    mx = dict(
        dtype="fp4",
        qscheme="per_group",
        ch_axis=-1,
        group_size=32,
        scale_type="float",
        scale_format="e8m0",
        scale_calculation_mode="even",
        round_method="half_even",
    )
    path = tmp_path_factory.mktemp("minimax_m3_mini")
    MiniMaxM3Config(
        text_config=dict(
            num_hidden_layers=LAYERS,
            vocab_size=VOCAB,
            moe_layer_freq=layers,
            sparse_attention_config=dict(
                use_sparse_attention=True,
                sparse_index_dim=128,
                sparse_num_index_heads=4,
                sparse_topk_blocks=16,
                sparse_block_size=128,
                sparse_disable_index_value=layers,
                sparse_score_type="max",
                sparse_init_block=0,
                sparse_local_block=1,
                sparse_attention_freq=layers,
            ),
        ),
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
        # the checkpoint's: MXFP4 experts (A16W4 on this path), BF16 elsewhere
        quantization_config=dict(
            quant_method="quark",
            global_quant_config=dict(
                weight={**mx, "is_dynamic": False},
                input_tensors={**mx, "is_dynamic": True},
                output_tensors=None,
                bias=None,
            ),
            layer_quant_config={},
            layer_type_quant_config={},
            export=dict(kv_cache_group=[], pack_method="reorder"),
            exclude=[
                "lm_head",
                "re:.*self_attn.*",
                r"re:.*\.mlp\..*",
                r"re:.*block_sparse_moe\.gate$",
            ],
        ),
    ).save_pretrained(path)
    return str(path)


@torch.no_grad()
def _randomize_model(model, *, stable_prefill=False, fixed_routes=False):
    """Weights at unit-activation scales; TP shards seeded per rank, replicated
    tensors (norms, router, bias) alike on every rank."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )
    from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_rocm

    rank = get_tensor_model_parallel_rank()

    def generator(name, sharded=True):
        key = f"{SEED}/{name}/{rank if sharded else 0}"
        return torch.Generator(device="cuda").manual_seed(zlib.crc32(key.encode()))

    for name, p in model.named_parameters():
        if ".experts." in name:
            continue
        if name.endswith("weight_scale"):
            p.fill_(1.0)
            continue
        replicated = p.ndim == 1 or ".gate." in name
        if p.ndim == 1:
            std = 0.1  # Gemma norms scale by 1 + w; the router bias
        elif "embed_tokens" in name:
            std = 1.0
        else:
            std = p.shape[1] ** -0.5
        g = generator(name, not replicated)
        p.copy_(torch.randn(p.shape, device=p.device, generator=g).mul(std))
    for name, b in model.named_buffers():
        # dummy loading leaves the sparse layers' FP8 KV scales random
        if name.endswith(("_k_scale", "_v_scale", "_q_scale", "_prob_scale")):
            b.fill_(1.0)
    for layer in model.model.layers:
        # q = 0, uniform attention: the FP8 KV cache rounds native's run-to-run
        # last-bit differences into whole FP8 steps, which softmax over q.k
        # would amplify; the model comparison needs no attention content
        layer.self_attn.q_norm.weight.fill_(-1.0)
        if layer.layer_id not in SPARSE:
            if stable_prefill:
                layer.mlp.down_proj.weight.zero_()
            continue
        moe = layer.block_sparse_moe
        if fixed_routes:
            # Keep a routing margin while retaining nonuniform sigmoid weights.
            moe.e_score_correction_bias.fill_(-4)
            first = 4 * (layer.layer_id - SPARSE[0])
            moe.e_score_correction_bias[first : first + 4] = 4
        if moe.gate._bf16x3_weight is not None:
            moe.gate._bf16x3_weight.copy_(
                bf16x3_router_gemm_rocm.split_bf16x3(moe.gate.weight)
            )
        experts = moe.experts.routed_experts
        g = generator(f"experts.{layer.layer_id}")
        new = {}
        for key in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            dst = getattr(experts, key)
            # e8m0 2^-9..2^-7: unit gate/up pre-activations, MoE outputs on the
            # residual's scale (native's BF16 atomic sums round to its size)
            lo, hi = (118, 121) if "scale" in key else (0, 256)
            new[key] = torch.nn.Parameter(
                torch.randint(
                    lo, hi, dst.shape, dtype=torch.uint8, device="cuda", generator=g
                ),
                requires_grad=False,
            )
        if stable_prefill:
            # Native prefill sums BF16 down-projection tiles atomically. One
            # nonzero shared-expert tile gives graph replay a stable cache
            # reference; the independent op tests cover all routed experts.
            new["w2_weight"][:128].zero_()
            new["w2_weight"][128, :, 16:].zero_()
        converted = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            layer=torch.nn.Module(),
            **new,
        )
        for key, src in zip(new, converted):
            getattr(experts, key).view(torch.uint8).copy_(src.view(torch.uint8))
    torch.accelerator.synchronize()


@register_model_loader("minimax_m3_seeded_dummy")
class SeededDummyModelLoader(DummyModelLoader):
    stable_prefill = True
    fixed_routes = False

    def load_model(self, vllm_config, model_config, prefix=""):
        model = super().load_model(vllm_config, model_config, prefix)
        if "MiniMaxM3SparseForCausalLM" in model_config.hf_config.architectures:
            _randomize_model(
                model,
                stable_prefill=self.stable_prefill,
                fixed_routes=self.fixed_routes,
            )
        return model


@register_model_loader("minimax_m3_layer_comparison")
class LayerComparisonLoader(SeededDummyModelLoader):
    stable_prefill = False
    fixed_routes = True


class SeededWeightsWorkerExtension:
    """Import the seeded loader in each worker before device initialization."""


def _requests():
    set_random_seed(SEED)
    requests = []
    for n in BATCHES:
        prompts = [
            {"prompt_token_ids": torch.randint(VOCAB, (int(size),)).tolist()}
            for size in torch.randint(16, 48, (n,))
        ]
        traces = [torch.randint(VOCAB, (STEPS,)).tolist() for _ in range(n)]
        requests.append((n, prompts, traces))
    return requests


def _decode(llm):
    """Full-vocabulary decode logprobs along fixed teacher-forced tokens."""
    from vllm import SamplingParams

    rows = []
    for n, prompts, traces in _requests():
        params = [
            SamplingParams(
                max_tokens=STEPS,
                temperature=0,
                logprobs=-1,
                trace_decode_token_ids=trace,
                detokenize=False,
                ignore_eos=True,
            )
            for trace in traces
        ]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        for out, trace in zip(outputs, traces):
            assert list(out.outputs[0].token_ids) == trace
            for step in out.outputs[0].logprobs[1:]:
                rows.append([step[v].logprob for v in range(VOCAB)])
    return torch.tensor(rows)


def _logit_errors(got, want):
    """Cosine over all rows and each row's relative L2 error of the logits
    (logprobs less their row mean)."""
    got, want = (x.float() - x.float().mean(-1, keepdim=True) for x in (got, want))
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    return cos.item(), (got - want).norm(dim=-1) / want.norm(dim=-1)


def _engine(
    vllm_runner,
    model_dir,
    mono=True,
    graphs=False,
    max_num_seqs=None,
    load_format="minimax_m3_seeded_dummy",
    enforce_eager=None,
    ir_op_priority=None,
    cudagraph_capture_sizes=None,
    **extra,
):
    compilation_config = {"cudagraph_mode": "FULL_DECODE_ONLY" if graphs else "NONE"}
    if cudagraph_capture_sizes is not None:
        compilation_config["cudagraph_capture_sizes"] = cudagraph_capture_sizes
    kernel_config = {
        "ir_op_priority": ir_op_priority
        or {"rms_norm": ["aiter"], "fused_add_rms_norm": ["aiter"]},
        "enable_jit_warmup": False,
        "moe_backend": "aiter",
        "monokernel": mono,
    }
    return vllm_runner(
        model_dir,
        load_format=load_format,
        skip_tokenizer_init=True,
        tensor_parallel_size=4,
        max_num_seqs=max(BATCHES) if max_num_seqs is None else max_num_seqs,
        kv_cache_dtype="fp8",
        block_size=128,
        attention_backend="TRITON_ATTN",
        attention_config={"indexer_kv_dtype": "fp8"},
        kernel_config=kernel_config,
        enforce_eager=False if enforce_eager is None else enforce_eager,
        compilation_config=compilation_config,
        enable_prefix_caching=False,
        enable_trace_replay=True,
        max_logprobs=-1,
        gpu_memory_utilization=0.1,
        worker_extension_cls=(
            "tests.models.minimax_m3.amd_specialized_decode.test_layers."
            "SeededWeightsWorkerExtension"
        ),
        quantization_config={
            "targets": {
                "*self_attn*qkv_proj": "fp8_per_channel",
                "*self_attn*o_proj": "fp8_per_channel",
            }
        },
        **extra,
    )


@pytest.fixture(scope="module", autouse=True)
def _env():
    from vllm._aiter_ops import rocm_aiter_ops

    try:
        with pytest.MonkeyPatch.context() as mp:
            for name, value in ENV.items():
                mp.setenv(name, value)
            rocm_aiter_ops.refresh_env_variables()
            yield
    finally:
        rocm_aiter_ops.refresh_env_variables()


def _executor(worker):
    return worker.model_runner.model_state.backbone.specialized_decode.executor


def _full_layer_state(worker):
    from vllm.models.minimax_m3.amd.specialized_decode.runtime.layer_executor import (
        LayerExecutor,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    executor = _executor(worker)
    splits = {
        builder.num_par_softmax_segments
        for groups in worker.model_runner.attn_groups
        for group in groups
        for builder in group.metadata_builders
        if isinstance(builder, TritonAttentionMetadataBuilder)
    }
    return (
        isinstance(executor, LayerExecutor),
        executor.outputs,
        executor.health().failed,
        sorted(splits),
    )


def _capture_layer_states(worker):
    """Observe each layer's residual stream through the model's aux-state API."""
    from vllm.forward_context import get_forward_context

    model = worker.model_runner.get_model()
    model.set_aux_hidden_state_layers(tuple(range(1, LAYERS + 1)))
    backbone = worker.model_runner.model_state.backbone
    worker.layer_states = []

    def keep(module, args, kwargs, output):
        metadata = get_forward_context().attn_metadata
        sparse = metadata[backbone.layers[SPARSE[0]].self_attn.layer_name]
        if sparse.num_prefills == 0:
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            worker.layer_states.append(
                (positions.clone(), [state.clone() for state in output[1]])
            )
        # Sampling consumes the normal output; only this probe needs aux states.
        return output[0]

    backbone.register_forward_hook(keep, with_kwargs=True)


def _save_layer_states(worker, path):
    records = [
        (positions.cpu(), [state.cpu() for state in states])
        for positions, states in worker.layer_states
    ]
    torch.save(records, f"{path}.rank{worker.rank}")
    decode = worker.model_runner.model_state.backbone.specialized_decode
    return None if decode is None else decode.executor.launches


def test_mono_layer_states_match_native(vllm_runner, model_dir, monkeypatch, tmp_path):
    """Every layer/rank/decode row matches native along teacher-forced tokens.

    All dense and expert outputs are active. Separated routing biases avoid
    expert-set changes, isolating MXFP8 rounding across the two sparse layers.
    The residual-stream bound allows 4% relative L2 and cosine above 0.999.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    for mono in (False, True):
        with _engine(
            vllm_runner,
            model_dir,
            mono=mono,
            load_format="minimax_m3_layer_comparison",
            enforce_eager=True,
            async_scheduling=False,
            disable_custom_all_reduce=True,
        ) as runner:
            runner.llm.collective_rpc(_capture_layer_states)
            _decode(runner.llm)
            launches = runner.llm.collective_rpc(
                _save_layer_states, args=(str(tmp_path / str(mono)),)
            )
            if mono:
                assert all(
                    counts == dict.fromkeys(SPARSE, len(BATCHES) * (STEPS - 1))
                    for counts in launches
                ), launches
            else:
                assert launches == [None] * 4

    for rank in range(4):
        native = torch.load(f"{tmp_path / 'False'}.rank{rank}")
        mono = torch.load(f"{tmp_path / 'True'}.rank{rank}")
        assert len(native) == len(mono) == len(BATCHES) * (STEPS - 1)
        for step, ((positions, states), (ref_positions, refs)) in enumerate(
            zip(mono, native)
        ):
            assert torch.equal(positions, ref_positions), (rank, step)
            assert len(states) == len(refs) == LAYERS
            for layer, (got, want) in enumerate(zip(states, refs)):
                got, want = got.float(), want.float()
                relative = (got - want).norm(dim=-1) / want.norm(dim=-1)
                cosine = torch.nn.functional.cosine_similarity(got, want, dim=-1)
                assert relative.max() < 0.04 and cosine.min() > 0.999, (
                    rank,
                    step,
                    layer,
                    relative.max().item(),
                    cosine.min().item(),
                )


def test_default_monokernel_graph_replay_matches_uncaptured_forward(
    vllm_runner, model_dir, monkeypatch
):
    """Default --monokernel selects full-layer MXFP8 math under FULL replay.

    Expert arithmetic has an independent MXFP8 reference in test_numerics.py.
    Keep native providers and physical batch shapes identical; seed before
    weight packing and graph capture. Padding is checked by the layer op tests.
    """
    # Admit every request before stepping so both engines see the same batches.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    outputs = []
    for graphs in (False, True):
        with _engine(
            vllm_runner,
            model_dir,
            graphs=graphs,
            max_num_seqs=8,
            enforce_eager=False,
            async_scheduling=False,
            disable_custom_all_reduce=True,
            cudagraph_capture_sizes=list(range(1, 9)) if graphs else None,
            ir_op_priority={
                "rms_norm": ["aiter"],
                "fused_add_rms_norm": ["aiter"],
            },
            load_format="minimax_m3_seeded_dummy",
        ) as runner:
            outputs.append(_decode(runner.llm))
            for full_layer, count, failed, splits in runner.llm.collective_rpc(
                _full_layer_state
            ):
                assert full_layer and count > 0 and not failed
                assert splits == [256]
    cos, errors = _logit_errors(outputs[1], outputs[0])
    assert cos > 0.99 and (errors < 3e-2).float().mean().item() > 0.8


def _arm_fault(worker, rank):
    executor = _executor(worker)
    if executor.rank == rank:
        executor.abort()


def test_host_abort_returns_no_tokens_from_full_graphs(
    vllm_runner, model_dir, capfd, monkeypatch
):
    """A host abort on one rank stops generation before any token returns."""
    from vllm import SamplingParams
    from vllm.v1.engine.exceptions import EngineDeadError

    monkeypatch.setenv("HSA_DISABLE_COREDUMP_ON_EXCEPTION", "1")
    prompt = {"prompt_token_ids": list(range(1, 33))}
    params = SamplingParams(max_tokens=8, temperature=0, detokenize=False)
    with _engine(vllm_runner, model_dir, graphs=True) as runner:
        llm = runner.llm
        assert len(llm.generate(prompt, params)[0].outputs[0].token_ids) == 8
        capfd.readouterr()
        llm.collective_rpc(_arm_fault, args=(2,))
        outputs = None
        with pytest.raises(EngineDeadError):
            outputs = llm.generate(prompt, params)
        assert outputs is None
    logs = "".join(capfd.readouterr())
    # the error-word assert trapped the queue ahead of the step's sampling
    assert "hipErrorLaunchFailure" in logs or "HSA_STATUS_ERROR_EXCEPTION" in logs


# EAGLE3 verify steps: 1 + SPEC_TOKENS rows a request, up to 4 requests (16
# rows) fused
SPEC_TOKENS = 3
SPEC_BATCHES = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def draft_dir(tmp_path_factory):
    """A random one-layer EAGLE3 draft at the checkpoint draft's attention
    shapes and the mini model's vocabulary, its auxiliary states taken after
    layer 1 (stock) and layers 3 and 4 (fused)."""
    import json

    path = tmp_path_factory.mktemp("minimax_m3_eagle3_mini")
    config = dict(
        architectures=["LlamaForCausalLMEagle3"],
        model_type="llama",
        hidden_size=6144,
        intermediate_size=4096,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        max_position_embeddings=4096,
        vocab_size=VOCAB,
        draft_vocab_size=VOCAB,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        fc_norm=True,
        norm_output=True,
        eagle_aux_hidden_state_layer_ids=[2, 4, 5],
    )
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _epoch(worker):
    return int(_executor(worker).epoch[0])


def _spec_requests():
    set_random_seed(SEED + 1)
    return [
        [
            {"prompt_token_ids": torch.randint(VOCAB, (int(k),)).tolist()}
            for k in torch.randint(16, 48, (n,))
        ]
        for n in SPEC_BATCHES
    ]


def test_eagle3_verify_steps_match_target(
    vllm_runner, model_dir, draft_dir, monkeypatch
):
    """Accepted verify rows match the target's teacher-forced logprobs."""
    from tests.v1.e2e.spec_decode.utils import compute_acceptance_rate
    from vllm import SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    greedy = SamplingParams(
        max_tokens=STEPS, temperature=0, logprobs=-1, detokenize=False, ignore_eos=True
    )
    tokens, mono_rows = [], []
    with _engine(
        vllm_runner,
        model_dir,
        graphs=True,
        max_num_seqs=max(SPEC_BATCHES),
        cudagraph_capture_sizes=list(range(1, 17)),
        async_scheduling=False,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
        speculative_config={
            "method": "eagle3",
            "model": draft_dir,
            "num_speculative_tokens": SPEC_TOKENS,
            # A random draft needs forced acceptance to exercise accepted rows.
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [1.0, 1.0, 0.0],
        },
    ) as runner:
        llm = runner.llm
        for prompts in _spec_requests():
            before = llm.collective_rpc(_epoch)
            outputs = llm.generate(prompts, greedy, use_tqdm=False)
            moved = {a - b for a, b in zip(llm.collective_rpc(_epoch), before)}
            assert len(moved) == 1
            assert moved.pop() > 0
            tokens.append([list(out.outputs[0].token_ids) for out in outputs])
            for out in outputs:
                for step in out.outputs[0].logprobs[1:]:
                    mono_rows.append([step[v].logprob for v in range(VOCAB)])
        for full_layer, count, failed, _ in llm.collective_rpc(_full_layer_state):
            assert full_layer and count > 0 and not failed
        assert compute_acceptance_rate(llm.get_metrics()) > 0

    target_rows = []
    with _engine(
        vllm_runner,
        model_dir,
        max_num_seqs=max(SPEC_BATCHES),
        async_scheduling=False,
        disable_custom_all_reduce=True,
    ) as runner:
        llm = runner.llm
        for prompts, traces in zip(_spec_requests(), tokens):
            params = [
                SamplingParams(
                    max_tokens=STEPS,
                    temperature=0,
                    logprobs=-1,
                    trace_decode_token_ids=trace,
                    detokenize=False,
                    ignore_eos=True,
                )
                for trace in traces
            ]
            for out in llm.generate(prompts, params, use_tqdm=False):
                for step in out.outputs[0].logprobs[1:]:
                    target_rows.append([step[v].logprob for v in range(VOCAB)])
    cos, rows = _logit_errors(torch.tensor(mono_rows), torch.tensor(target_rows))
    close = (rows < 3e-2).float().mean().item()
    print(
        f"EAGLE3 cos {cos:.5f}, median row {rows.median():.2e}, within 3e-2 {close:.3f}"
    )
    assert cos > 0.99 and close > 0.8
