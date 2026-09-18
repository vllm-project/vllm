# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.model_executor.layers import logits_processor
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_lilicorr import LiLiCorrConfig, LiLiCorrHead
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator


@pytest.fixture(autouse=True)
def replicated_head_tp(monkeypatch):
    # The head has no collectives; unit tests only need TP metadata.
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )


def _config(**overrides):
    config = dict(
        candidate_topk=4,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        factor_dim=4,
        vector_eps=1e-6,
        logit_scale=3.0,
    )
    config.update(overrides)
    return LiLiCorrConfig(**config)


def _reference_scores(head, embeddings, log_probs, hidden, anchor, valid):
    """Unfused exported head: MHA attention, concatenated factors, separate edges."""
    batch, slots, top_k = log_probs.shape
    rank = torch.arange(top_k, device=log_probs.device) / max(top_k - 1, 1)
    top1 = (torch.arange(top_k, device=log_probs.device) == 0).float()
    features = torch.stack(
        (
            log_probs,
            log_probs.exp(),
            log_probs - log_probs.amax(-1, keepdim=True),
            rank.expand_as(log_probs),
            top1.expand_as(log_probs),
        ),
        -1,
    )
    x = head.token_proj(embeddings) + head.pass_hidden_proj(hidden).unsqueeze(-2)
    x = x + head.feature_mlp(features.to(x.dtype))
    x = x + head.slot_embedding[:, 0] + head.rank_embedding[:, 0]
    x = x.flatten(1, 2)
    positions = torch.arange(slots, device=x.device).repeat_interleave(top_k)
    relative = positions[:, None] - positions[None, :]
    bias = head.relative_slot_bias[:, relative + head.block_size - 1]
    bias = bias + (relative == 0) * head.same_slot_bias[:, None, None]
    bias = bias.repeat(batch, 1, 1)
    for layer in head.layers:
        attention = nn.MultiheadAttention(
            head.hidden_size,
            head.num_heads,
            batch_first=True,
            dtype=x.dtype,
            device=x.device,
        )
        attention.load_state_dict(
            {
                name.replace("in_proj.weight", "in_proj_weight").replace(
                    "in_proj.bias", "in_proj_bias"
                ): value
                for name, value in layer.attn.state_dict().items()
            }
        )
        normalized = layer.attn_norm(x)
        x = (
            x
            + attention(
                normalized, normalized, normalized, attn_mask=bias, need_weights=False
            )[0]
        )
        x = x + layer.mlp(layer.mlp_norm(x))
    x = head.output_norm(x).view(batch, slots, top_k, -1)
    a = head.anchor_norm(head.context_proj(anchor) * valid[:, None])
    expanded = a[:, None, None].expand_as(x)
    factors = F.silu(head.factor_input_proj(torch.cat((x, expanded, x * expanded), -1)))
    incoming = F.normalize(head.in_head(factors), dim=-1, eps=head.vector_eps)
    outgoing = F.normalize(head.out_head(factors), dim=-1, eps=head.vector_eps)
    a = F.normalize(head.anchor_out_head(a), dim=-1, eps=head.vector_eps)
    start = (a[:, None] * incoming[:, 0]).sum(-1)
    pairs = outgoing[:, :-1] @ incoming[:, 1:].transpose(-1, -2)
    return (
        head.logit_scale
        * torch.cat((start[:, None, None].expand(-1, 1, top_k, -1), pairs), 1).float()
    )


@pytest.mark.parametrize("head_width", [8, 16])
@pytest.mark.parametrize("slots", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lilicorr_matches_exported_head(head_width, slots, dtype):
    torch.manual_seed(7)
    with set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            compilation_config=CompilationConfig(mode=0),
        )
    ):
        head = LiLiCorrHead(
            model_hidden_size=16,
            block_size=slots + 1,
            rms_norm_eps=1e-6,
            config=_config(hidden_size=head_width),
        ).to(dtype)
    # vLLM linear parameters are initialized by the checkpoint loader.
    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, ReplicatedLinear):
                for parameter in module.parameters():
                    parameter.weight_loader(
                        parameter, torch.randn_like(parameter) * 0.1
                    )
        head.relative_slot_bias.normal_(std=0.1)
        head.same_slot_bias.normal_(std=0.1)
        head.slot_embedding.normal_(std=0.1)
        head.rank_embedding.normal_(std=0.1)
        head.materialize_inference_buffers(torch.device("cpu"), dtype)
        inputs = (
            torch.randn(2, slots, 4, 16, dtype=dtype),
            torch.randn(2, slots, 23).log_softmax(-1).topk(4).values,
            torch.randn(2, slots, 16, dtype=dtype),
            torch.randn(2, 16, dtype=dtype),
            torch.tensor([True, False]),
        )
        actual = head(*inputs)
        expected = _reference_scores(head, *inputs)
    tolerance = 0.04 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_candidates_use_global_partition_and_exclude_padding(monkeypatch, tp_size):
    """Candidate features must include mass outside top-k and on other TP ranks."""
    shards = [
        torch.tensor([[1.0, 3.0, 2.0, 100.0], [4.0, 1.0, 2.0, 100.0]]),
        torch.tensor([[6.0, 5.0, 0.0, 100.0], [3.0, 6.0, 5.0, 100.0]]),
    ][:tp_size]
    for shard in shards:
        shard[:, -1] = -float("inf")
    for rank in range(tp_size):
        fake = SimpleNamespace(
            scale=1.0,
            soft_cap=None,
            _apply_head=lambda *args, rank=rank: shards[rank].clone(),
        )
        lm_head = SimpleNamespace(
            tp_size=tp_size,
            shard_indices=SimpleNamespace(
                num_org_vocab_padding=1, org_vocab_start_index=rank * 3
            ),
        )
        gathered = [
            torch.cat([s.logsumexp(-1, keepdim=True) for s in shards], -1),
            torch.cat([s.topk(2).values for s in shards], -1),
            torch.cat([s.topk(2).indices + r * 3 for r, s in enumerate(shards)], -1),
        ]
        calls = iter(gathered)
        monkeypatch.setattr(
            logits_processor,
            "tensor_model_parallel_all_gather",
            lambda *args, calls=calls, **kwargs: next(calls),
        )
        monkeypatch.setattr(logits_processor, "_topk", lambda x, k: x.topk(k, dim=-1))
        ids, values = logits_processor.LogitsProcessor.get_top_k_tokens(
            fake, lm_head, torch.empty(2, 1), 2, return_log_probs=True
        )
        expected = torch.cat([s[:, :3] for s in shards], -1).log_softmax(-1).topk(2)
        torch.testing.assert_close(ids, expected.indices)
        torch.testing.assert_close(values, expected.values)


def test_context_anchor_is_last_committed_normalized_feature():
    hidden = torch.arange(1, 25, dtype=torch.float32).view(6, 4)
    norm = nn.RMSNorm(4)
    spec = SimpleNamespace(
        hidden_states=hidden,
        model=SimpleNamespace(model=SimpleNamespace(hidden_norm=norm)),
        anchor_hidden=torch.full((4, 4), 123.0),
        anchor_valid=torch.ones(4, dtype=torch.bool),
    )
    batch = SimpleNamespace(num_reqs=2, query_start_loc=torch.tensor([0, 2, 6]))
    LiLiCorrSpeculator.prepare_context_anchor(spec, batch, torch.tensor([0, 2]))
    torch.testing.assert_close(spec.anchor_hidden[:2], norm(hidden[[1, 3]]))
    assert spec.anchor_valid.tolist() == [True, True, False, False]
    assert not spec.anchor_hidden[2:].any()
    # A smaller reordered batch must not retain an old request's anchor.
    batch = SimpleNamespace(num_reqs=1, query_start_loc=torch.tensor([0, 3]))
    LiLiCorrSpeculator.prepare_context_anchor(spec, batch, torch.tensor([1]))
    torch.testing.assert_close(spec.anchor_hidden[0], norm(hidden[1]))
    assert not spec.anchor_hidden[1:].any()


@pytest.mark.parametrize(
    "key,value",
    [
        ("candidate_topk", 3),
        ("vector_eps", 0),
        ("logit_scale", float("nan")),
        ("hidden_size", -1),
    ],
)
def test_invalid_head_geometry_is_rejected(key, value):
    values = {f"lilicorr_{k}": v for k, v in vars(_config()).items()}
    values[f"lilicorr_{key}"] = value
    with pytest.raises(ValueError):
        LiLiCorrConfig.from_dict(values)


def test_lilicorr_architecture_survives_dflash_config_wrapping():
    from transformers import Qwen3Config

    config = Qwen3Config(architectures=["LiLiCorrDraftModel"])
    assert EAGLEConfig(config, method="dflash").architectures == ["LiLiCorrDraftModel"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("probabilistic", [False, True])
def test_candidate_walk_preserves_conditional_scores_and_padding(probabilistic):
    """Check the shared sampler's actual q, stale support, and graph padding."""
    device = torch.device("cuda")
    batch, steps, top_k, vocab = 3, 3, 4, 32
    sampler = CandidateSampler(batch, steps, top_k, device)
    candidates = (
        torch.arange(steps * top_k, device=device)
        .view(1, steps, top_k)
        .repeat(batch, 1, 1)
    )
    scores = torch.randn(batch, steps, top_k, top_k, device=device)
    tokens = torch.full((batch, steps), -1, dtype=torch.long, device=device)
    cache = (
        torch.full((batch, steps, vocab), -float("inf"), device=device)
        if probabilistic
        else None
    )
    positions = torch.arange(batch * steps * 2, device=device)[::2]
    mappings = torch.tensor(
        [2] * steps + [0] * steps + [-1] * steps, device=device
    ).repeat_interleave(2)[::2]
    temperature = torch.tensor([0.0, 1.0, 0.8], device=device)
    seeds = torch.tensor([11, 22, 33], device=device)

    def run():
        sampler.sample(
            candidates,
            scores,
            batch,
            positions,
            mappings,
            temperature,
            seeds,
            tokens,
            cache,
            False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    old_candidates = candidates.clone()
    candidates.add_(12)
    graph.replay()
    for row, state in ((0, 2), (1, 0)):
        previous = 0
        for step in range(steps):
            realized = scores[row, step, previous]
            torch.testing.assert_close(sampler.scores[row, step], realized)
            index = (candidates[row, step] == tokens[row, step]).nonzero().item()
            if not probabilistic or temperature[state] == 0:
                assert index == realized.argmax().item()
            if cache is not None:
                torch.testing.assert_close(
                    cache[state, step, candidates[row, step]], realized
                )
                assert cache[state, step, old_candidates[row, step]].isneginf().all()
            previous = index
    assert (tokens[2] == -1).all()
    if cache is not None:
        assert cache[1].isneginf().all()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("convolution", [False, True])
@pytest.mark.parametrize(
    "mismatch", [None, "missing_head", "extra_head", "missing_conv", "extra_conv"]
)
def test_checkpoint_coverage_rejects_incomplete_or_wrong_heads(
    monkeypatch, convolution, mismatch, quantized
):
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
    from vllm.model_executor.models.qwen3_lilicorr import LiLiCorrQwen3ForCausalLM

    wrapper = LiLiCorrQwen3ForCausalLM.__new__(LiLiCorrQwen3ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = nn.Module()
    with set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            compilation_config=CompilationConfig(mode=0),
        )
    ):
        wrapper.model.lilicorr = LiLiCorrHead(
            model_hidden_size=16, block_size=4, rms_norm_eps=1e-6, config=_config()
        )
    layer = nn.Module()
    if convolution:
        layer.attention_conv = nn.Linear(2, 2, bias=False)
    wrapper.model.layers = nn.ModuleList([layer])
    if quantized:
        wrapper.model.quant_config = object()
        # Simulate a quantized projection whose scale is synthesized at load.
        projection = wrapper.model.lilicorr.pass_hidden_proj
        projection.quant_method = object()
        projection.register_parameter("weight_scale", nn.Parameter(torch.ones(1)))
    weights = {
        name.replace(".attn.in_proj.weight", ".attn.in_proj_weight").replace(
            ".attn.in_proj.bias", ".attn.in_proj_bias"
        ): value
        for name, value in wrapper.model.state_dict().items()
    }
    if quantized:
        del weights["lilicorr.pass_hidden_proj.weight_scale"]
    if mismatch == "missing_head":
        del weights["lilicorr.slot_embedding"]
    elif mismatch == "extra_head":
        weights["lilicorr.untrained.weight"] = torch.zeros(1)
    elif mismatch == "missing_conv":
        if not convolution:
            pytest.skip("requires a convolutional model")
        del weights["layers.0.attention_conv.weight"]
    elif mismatch == "extra_conv":
        weights["layers.0.mlp_conv.untrained"] = torch.zeros(1)
    # Isolate the coverage contract from the backbone's GPU weight loaders.
    monkeypatch.setattr(
        DFlashQwen3ForCausalLM, "load_weights", lambda self, weights: list(weights)
    )
    supplied = [("model." + name, value) for name, value in weights.items()]
    if mismatch:
        with pytest.raises(ValueError, match="coverage mismatch"):
            wrapper.load_weights(supplied)
    else:
        wrapper.load_weights(supplied)
        assert wrapper.model.lilicorr._attn_bias is not None


@pytest.mark.parametrize("head_width", [8, 16])
def test_quantized_head_calls_methods_without_reading_packed_weights(
    monkeypatch, head_width
):
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    calls = []
    configured = []
    quant_config = object()

    class PackedMethod(UnquantizedLinearMethod):
        def apply(self, layer, x, bias=None):
            calls.append(layer.prefix)
            # Packed storage has a different name/dtype from the activations.
            assert x.dtype == torch.float32
            return F.linear(x, layer.packed_weight.float() * layer.weight_scale, bias)

    def resolve(config, layer, prefix):
        assert config is quant_config
        configured.append(prefix)
        return PackedMethod()

    monkeypatch.setattr(linear, "resolve_quant_method", resolve)
    with set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            compilation_config=CompilationConfig(mode=0),
        )
    ):
        head = LiLiCorrHead(
            model_hidden_size=16,
            block_size=4,
            rms_norm_eps=1e-6,
            config=_config(hidden_size=head_width),
            quant_config=quant_config,
            prefix="model.lilicorr",
        )
    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, ReplicatedLinear):
                module.register_parameter(
                    "packed_weight",
                    nn.Parameter(
                        torch.randint(-4, 5, module.weight.shape, dtype=torch.int8),
                        requires_grad=False,
                    ),
                )
                module.register_parameter(
                    "weight_scale",
                    nn.Parameter(
                        torch.tensor(0.05),
                        requires_grad=False,
                    ),
                )
                del module.weight
                module.bias.zero_()
        head.materialize_inference_buffers(torch.device("cpu"), torch.float32)
        result = head(
            torch.randn(2, 3, 4, 16),
            torch.randn(2, 3, 4).log_softmax(-1),
            torch.randn(2, 3, 16),
            torch.randn(2, 16),
            torch.tensor([True, False]),
        )
    assert result.shape == (2, 3, 4, 4)
    assert torch.isfinite(result).all()
    assert set(calls) == set(configured)
    assert len(configured) == len(set(configured))
    for name, module in head.named_modules():
        if isinstance(module, ReplicatedLinear):
            assert module.prefix == f"model.lilicorr.{name}"
