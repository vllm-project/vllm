# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 parity of the FFN launch against the native ``MiniMaxM3DecoderLayer``
FFN half on the real checkpoint: each sparse layer's decode-step inputs are
captured from a native engine, then replayed on every rank through vLLM's
``fused_allreduce_gemma_rms_norm`` + ``block_sparse_moe`` and through the
launch, alone and as a two-layer chain."""

import os
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform

CHECKPOINT = os.environ.get("MINIMAX_M3_MXFP4_PATH", "/models/MiniMax-M3-MXFP4")
WIDTHS = (1, 4, 16)
SPARSE = range(3, 60)


def _supported() -> bool:
    if not current_platform.is_rocm() or torch.accelerator.device_count() < 4:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950() and Path(CHECKPOINT, "config.json").exists()


pytestmark = pytest.mark.skipif(
    not _supported(), reason="needs 4 gfx950 GPUs and the MiniMax-M3-MXFP4 checkpoint"
)


def _backbone(worker):
    model = worker.model_runner.get_model()
    return getattr(model, "language_model", model).model


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
    executor = FfnExecutor(list(layers.values()), weights, WIDTHS, 4)

    def native(i, part, residual):
        layer = layers[i]
        x, residual = fused_allreduce_gemma_rms_norm(
            part, residual, layer.post_attention_layernorm
        )
        return layer.block_sparse_moe(x), residual

    def fused(i, part, residual):
        slots = torch.arange(part.shape[0], dtype=torch.int64, device=part.device)
        return executor.runner(part, residual, weights[i], slots)

    def metrics(got, want):
        got, want = got.float(), want.float()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), want.flatten(), dim=0
        ).item()
        rel = ((got - want).norm() / want.norm()).item()
        row = ((got - want).norm(dim=-1) / want.norm(dim=-1)).max().item()
        return cos, rel, row

    results: dict[str, dict] = {"layers": {}, "chain": {}}
    with torch.no_grad(), set_forward_context(None, worker.vllm_config):
        for s, caps in worker._ffn_caps.items():
            residuals = {}
            for i, (hidden, residual, part) in caps.items():
                _, residual = layers[i].input_layernorm(hidden, residual)
                residuals[i] = residual
                want, want_res = native(i, part, residual)
                got, got_res = fused(i, part, residual)
                results["layers"][(s, i)] = (
                    *metrics(got, want),
                    torch.equal(got_res, want_res),
                )
            outs = {}
            for path, run in (("native", native), ("fused", fused)):
                out, res = run(3, caps[3][2], residuals[3])
                _, res = layers[4].input_layernorm(out, res)
                outs[path] = run(4, caps[4][2], res)
            results["chain"][s] = (
                metrics(outs["fused"][0], outs["native"][0]),
                metrics(outs["fused"][1], outs["native"][1]),
            )
    health = executor.health()
    executor.close()
    return results, health.failed


def test_ffn_launch_matches_native_layers(monkeypatch):
    for name, value in {
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_MOE": "1",
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=CHECKPOINT,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=max(WIDTHS),
        kv_cache_dtype="fp8",
        block_size=128,
        moe_backend="aiter",
        attention_backend="TRITON_ATTN",
        attention_config={"indexer_kv_dtype": "fp8"},
        enforce_eager=True,
        language_model_only=True,
        enable_prefix_caching=False,
    )
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did she sell altogether?"
    )
    for s in WIDTHS:
        llm.collective_rpc(_install, args=(s,))
        llm.generate(
            [f"Problem {i}. {question}" for i in range(s)],
            SamplingParams(max_tokens=8, temperature=0),
        )
        assert llm.collective_rpc(_stash, args=(s,)) == [list(SPARSE)] * 4

    for results, failed in llm.collective_rpc(_replay):
        assert not failed
        for cos, rel, row, residual_equal in results["layers"].values():
            assert residual_equal
            assert cos > 0.9999 and rel < 1e-2 and row < 2e-2
        for out, res in results["chain"].values():
            assert out[0] > 0.9999 and out[1] < 1e-2
            assert res[0] > 0.99999
