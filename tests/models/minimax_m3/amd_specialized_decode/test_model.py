# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer- and model-level precision of --specialized-decode on a mini MiniMax M3 at
the real per-layer shapes, TP4, with random weights (no checkpoint): layers
0-2 dense and stock, 3-4 sparse and fused, the last one ending in the step-end
handshake. Engines load dummy weights, so vLLM's real MXFP4 / AITER post-load
path runs; every rank then overwrites them in place at sensible scales,
distinct per layer and per TP shard, the experts generated in the checkpoint's
MXFP4 format and converted by that same path.

- layer: each sparse ``MiniMaxM3DecoderLayer``'s FFN half (seam all-reduce and
  residual, post-attention norm, MoE) on captured decode-step inputs, native
  vs the fused launch;
- model: teacher-forced decode (trace replay) with full-vocabulary logprobs,
  --specialized-decode eager and under FULL graphs (with frequent epoch resets) vs
  native eager;
- fail-stop: a late rank fault under FULL graphs returns no token."""

import zlib
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

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
def _randomize(worker):
    """Weights at unit-activation scales; TP shards seeded per rank, replicated
    tensors (norms, router, bias) alike on every rank."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )
    from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_rocm

    rank = get_tensor_model_parallel_rank()
    model = worker.model_runner.get_model()

    def generator(name, sharded=True):
        key = f"{SEED}/{name}/{rank if sharded else 0}"
        return torch.Generator(device="cuda").manual_seed(zlib.crc32(key.encode()))

    for name, p in model.named_parameters():
        if ".experts." in name:
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
        # would amplify; the FFN under test needs no attention content
        layer.self_attn.q_norm.weight.fill_(-1.0)
        if layer.layer_id not in SPARSE:
            continue
        moe = layer.block_sparse_moe
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
        converted = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            layer=torch.nn.Module(),
            **new,
        )
        for key, src in zip(new, converted):
            getattr(experts, key).view(torch.uint8).copy_(src.view(torch.uint8))
    torch.accelerator.synchronize()


def _requests():
    g = torch.Generator().manual_seed(SEED)
    for n in BATCHES:
        prompts = [
            {
                "prompt_token_ids": torch.randint(
                    VOCAB, (int(size),), generator=g
                ).tolist()
            }
            for size in torch.randint(16, 48, (n,), generator=g)
        ]
        traces = [
            torch.randint(VOCAB, (STEPS,), generator=g).tolist() for _ in range(n)
        ]
        yield n, prompts, traces


def _decode(llm, capture=False):
    """Full-vocabulary logprobs of every decode step (the first output token
    comes from prefill, which runs stock), every token forced; ``capture``:
    also each sparse layer's inputs at the first decode step of each batch."""
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
        if capture:
            llm.collective_rpc(_install, args=(n,))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if capture:
            assert llm.collective_rpc(_stash, args=(n,)) == [list(SPARSE)] * 4
        for out, trace in zip(outputs, traces):
            assert list(out.outputs[0].token_ids) == trace
            for step in out.outputs[0].logprobs[1:]:
                rows.append([step[v].logprob for v in range(VOCAB)])
    return torch.tensor(rows)


def _metrics(got, want):
    got, want = got.float(), want.float()
    err = got - want
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    rel = err.norm() / want.norm()
    row = (err.norm(dim=-1) / want.norm(dim=-1)).max()
    return cos.item(), rel.item(), row.item()


def _logit_errors(got, want):
    """Cosine over all rows and each row's relative L2 error of the logits
    (logprobs less their row mean)."""
    got, want = (x.float() - x.float().mean(-1, keepdim=True) for x in (got, want))
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    return cos.item(), (got - want).norm(dim=-1) / want.norm(dim=-1)


def _backbone(worker):
    return worker.model_runner.get_model().model


def _install(worker, s):
    worker._ffn_cap, worker._ffn_hooks = {}, []
    for layer in _backbone(worker).layers:
        if layer.layer_id not in SPARSE:
            continue

        def pre(mod, args, lid=layer.layer_id):
            if args[1].shape[0] == s and lid not in worker._ffn_cap:
                worker._ffn_cap[lid] = [args[1].clone(), args[2].clone(), None]

        def post(mod, args, kwargs, out, lid=layer.layer_id):
            cap = worker._ffn_cap.get(lid)
            if cap is not None and cap[2] is None:
                cap[2] = out.clone()

        worker._ffn_hooks += [
            layer.register_forward_pre_hook(pre),
            layer.self_attn.register_forward_hook(post, with_kwargs=True),
        ]


def _stash(worker, s):
    for hook in worker._ffn_hooks:
        hook.remove()
    worker._ffn_caps = getattr(worker, "_ffn_caps", {})
    worker._ffn_caps[s] = worker._ffn_cap
    return sorted(worker._ffn_cap)


def _replay(worker):
    """Each captured sparse layer's FFN half, native and fused, on every rank."""
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        fused_allreduce_gemma_rms_norm,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.runtime.executor import (
        FfnExecutor,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.runtime.weights import (
        MonoLayerWeights,
    )

    layers = {
        layer.layer_id: layer
        for layer in _backbone(worker).layers
        if layer.layer_id in SPARSE
    }
    weights = {i: MonoLayerWeights.from_layer(layer) for i, layer in layers.items()}
    executor = FfnExecutor(list(layers.values()), weights, BATCHES, 4)
    results = {}
    with torch.no_grad(), set_forward_context(None, worker.vllm_config):
        for s, caps in worker._ffn_caps.items():
            for i, (hidden, residual, part) in caps.items():
                layer = layers[i]
                _, residual = layer.input_layernorm(hidden, residual)
                x, want_res = fused_allreduce_gemma_rms_norm(
                    part, residual, layer.post_attention_layernorm
                )
                want = layer.block_sparse_moe(x)
                slots = torch.arange(s, dtype=torch.int64, device=part.device)
                got, got_res = executor.runner(part, residual, weights[i], slots)
                results[(s, i)] = (*_metrics(got, want), torch.equal(got_res, want_res))
    failed = executor.health().failed
    executor.close()
    return results, failed


def _engine(vllm_runner, model_dir, mono, graphs=False, **extra):
    return vllm_runner(
        model_dir,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=4,
        max_num_seqs=max(BATCHES),
        kv_cache_dtype="fp8",
        block_size=128,
        attention_backend="TRITON_ATTN",
        attention_config={"indexer_kv_dtype": "fp8"},
        kernel_config={
            "enable_jit_warmup": False,
            "moe_backend": "aiter",
            "specialized_decode": mono,
        },
        enforce_eager=not graphs,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY" if graphs else "NONE"},
        enable_prefix_caching=False,
        enable_trace_replay=True,
        max_logprobs=-1,
        gpu_memory_utilization=0.1,
        **extra,
    )


@pytest.fixture(scope="module", autouse=True)
def _env():
    with pytest.MonkeyPatch.context() as mp:
        for name, value in ENV.items():
            mp.setenv(name, value)
        yield


@pytest.fixture(scope="module")
def native(vllm_runner, model_dir):
    """Native eager: the reference logprobs and the layer replays."""
    with _engine(vllm_runner, model_dir, mono=False) as runner:
        runner.llm.collective_rpc(_randomize)
        logprobs = _decode(runner.llm, capture=True)
        layers = runner.llm.collective_rpc(_replay)
    return SimpleNamespace(logprobs=logprobs, layers=layers)


def test_sparse_layers_match_native(native):
    """Every rank, both sparse layers, decode widths 1, 3 and 16: the fused
    launch against the native FFN half; the residual exactly."""
    for results, failed in native.layers:
        assert not failed
        for cos, rel, row, residual_equal in results.values():
            assert residual_equal
            assert cos > 0.9999 and rel < 1e-2 and row < 1e-2


def _executor(worker):
    return worker.model_runner.model_state.backbone.specialized_decode.executor


def _reset_every(worker, steps):
    executor = _executor(worker)
    executor._reset_launches = steps * len(executor._layers)
    executor._since_reset = 0
    return executor.resets


def _reset_state(worker):
    executor = _executor(worker)
    return executor.resets, executor.health().failed


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "full_graphs"])
def test_model_matches_native(vllm_runner, model_dir, native, graphs):
    """Teacher-forced decode with --specialized-decode against native eager. Under
    FULL graphs (async scheduling) the epoch resets every 3 sampled steps,
    on every rank at the same steps, and no wait times out."""
    with _engine(
        vllm_runner,
        model_dir,
        mono=True,
        graphs=graphs,
        async_scheduling=True if graphs else None,
    ) as runner:
        llm = runner.llm
        llm.collective_rpc(_randomize)
        if graphs:
            before = llm.collective_rpc(_reset_every, args=(3,))
        logprobs = _decode(llm)
        after = llm.collective_rpc(_reset_state)
    cos, rows = _logit_errors(logprobs, native.logprobs)
    close = (rows < 3e-2).float().mean().item()
    print(f"cos {cos:.5f}, median row {rows.median():.2e}, within 3e-2 {close:.3f}")
    # native's MoE sums in BF16 atomics: native reruns differ by ~1e-2 a row,
    # and by ~0.2 on the 4-13% of rows where that flips a near-tie expert
    assert cos > 0.99 and close > 0.8
    assert not any(failed for _, failed in after)
    if graphs:
        resets = {n - b for (n, _), b in zip(after, before)}
        assert len(resets) == 1 and resets.pop() >= len(BATCHES) * STEPS // 3 - 1


def _arm_late_fault(worker, rank, ticks):
    _executor(worker).runner.inject.copy_(
        torch.tensor([rank + 1, ticks], dtype=torch.int32)
    )


def test_late_rank_fault_returns_no_tokens_from_full_graphs(
    vllm_runner, model_dir, capfd
):
    """During a replayed step, rank 2 fails 5 ms after the rest of the step's
    last launch, so the output rank is ahead of it; that step returns no
    token."""
    from vllm import SamplingParams
    from vllm.v1.engine.exceptions import EngineDeadError

    prompt = {"prompt_token_ids": list(range(1, 33))}
    params = SamplingParams(max_tokens=8, temperature=0, detokenize=False)
    with _engine(vllm_runner, model_dir, mono=True, graphs=True) as runner:
        llm = runner.llm
        assert len(llm.generate(prompt, params)[0].outputs[0].token_ids) == 8
        capfd.readouterr()
        llm.collective_rpc(_arm_late_fault, args=(2, 500_000))
        outputs = None
        with pytest.raises(EngineDeadError):
            outputs = llm.generate(prompt, params)
        assert outputs is None
    logs = "".join(capfd.readouterr())
    # the error-word assert trapped the queue ahead of the step's sampling
    assert "hipErrorLaunchFailure" in logs or "HSA_STATUS_ERROR_EXCEPTION" in logs
