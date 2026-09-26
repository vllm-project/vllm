# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.model_executor.layers import logits_processor
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.lilicorr import LiLiCorrHead
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator


@pytest.fixture(autouse=True)
def head_tp_metadata(monkeypatch):
    # Single-rank unit tests only need TP metadata.
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
    return {f"lilicorr_{key}": value for key, value in config.items()}


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
    features = head.feature_norm(features.to(x.dtype))
    x = x + head.feature_mlp.down_proj(F.silu(head.feature_mlp.up_proj(features)[0]))[0]
    x = x + head.slot_embedding[:, 0, :slots] + head.rank_embedding[:, 0]
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
        attention.load_state_dict(layer.attn.state_dict())
        normalized = layer.attn_norm(x)
        x = (
            x
            + attention(
                normalized, normalized, normalized, attn_mask=bias, need_weights=False
            )[0]
        )
        x = x + layer.mlp.down_proj(F.silu(layer.mlp.up_proj(layer.mlp_norm(x))[0]))[0]
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")
@pytest.mark.parametrize(
    "head_width,slots,dtype",
    [
        pytest.param(8, 3, torch.float32, id="projected-multi-slot"),
        pytest.param(16, 1, torch.float32, id="identity-shortened-single-slot"),
        pytest.param(8, 2, torch.bfloat16, id="bf16-shortened-multi-slot"),
    ],
)
def test_lilicorr_matches_exported_head(head_width, slots, dtype):
    device = torch.device("cuda")
    torch.manual_seed(7)
    with (
        torch.device(device),
        set_current_vllm_config(
            VllmConfig(
                device_config=DeviceConfig("cuda"),
                compilation_config=CompilationConfig(mode=0),
            )
        ),
    ):
        head = LiLiCorrHead(
            model_hidden_size=16,
            block_size=4,
            rms_norm_eps=1e-6,
            config=_config(hidden_size=head_width),
        ).to(dtype)
    # vLLM linear parameters are initialized by the checkpoint loader.
    with torch.no_grad(), torch.device(device):
        for module in head.modules():
            if isinstance(module, LinearBase):
                for parameter in module.parameters():
                    parameter.weight_loader(
                        parameter, torch.randn_like(parameter) * 0.1
                    )
        for layer in head.layers:
            layer.attn.in_proj_weight.normal_(std=0.1)
            layer.attn.in_proj_bias.normal_(std=0.1)
        head.relative_slot_bias.normal_(std=0.1)
        head.same_slot_bias.normal_(std=0.1)
        head.slot_embedding.normal_(std=0.1)
        head.rank_embedding.normal_(std=0.1)
        head.materialize_inference_buffers(device, dtype)
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


def test_runtime_length_keeps_trained_checkpoint_geometry(monkeypatch):
    from vllm.model_executor.models.lilicorr import LiLiCorr
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    def init_backbone(self, **kwargs):
        nn.Module.__init__(self)
        self.quant_config = None

    monkeypatch.setattr(DFlashQwen3Model, "__init__", init_backbone)
    spec = SimpleNamespace(
        num_speculative_tokens=1,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=16,
                rms_norm_eps=1e-6,
                dflash_config=_config() | {"block_size": 4},
            )
        ),
    )
    with set_current_vllm_config(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            compilation_config=CompilationConfig(mode=0),
        )
    ):
        for slots in (1, 3):
            spec.num_speculative_tokens = slots
            head = LiLiCorr(
                vllm_config=SimpleNamespace(speculative_config=spec)
            ).lilicorr
            assert head.slot_embedding.shape == (1, 1, 3, 1, 8)
            assert head.relative_slot_bias.shape == (2, 7)
        spec.draft_model_config.hf_config.dflash_config["lilicorr_enabled"] = False
        with pytest.raises(ValueError, match="lilicorr_enabled"):
            LiLiCorr(vllm_config=SimpleNamespace(speculative_config=spec))
        spec.draft_model_config.hf_config.dflash_config["lilicorr_enabled"] = True
        spec.num_speculative_tokens = 4
        with pytest.raises(ValueError, match="trained block_size"):
            LiLiCorr(vllm_config=SimpleNamespace(speculative_config=spec))


@pytest.mark.parametrize("tp_size", [1, 2])
def test_candidates_use_global_partition_and_exclude_padding(monkeypatch, tp_size):
    """Candidate features must include mass outside top-k and on other TP ranks."""

    class Gather:
        def __init__(self, calls):
            self.calls = calls
            self.index = 0

        def __call__(self, value, dim=-1):
            expected_input, result = self.calls[self.index]
            assert dim == -1
            torch.testing.assert_close(value, expected_input)
            self.index += 1
            return result

    shards = [
        torch.tensor([[1.0, 3.0, 2.0, 100.0], [4.0, 1.0, 2.0, 100.0]]),
        torch.tensor([[6.0, 5.0, 0.0, 100.0], [3.0, 6.0, 5.0, 100.0]]),
    ][:tp_size]
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
        valid_shards = [shard[:, :3] for shard in shards]
        local_topk = [shard.topk(2) for shard in valid_shards]
        gather_calls = [
            (
                valid_shards[rank].logsumexp(-1, keepdim=True),
                torch.cat(
                    [shard.logsumexp(-1, keepdim=True) for shard in valid_shards],
                    -1,
                ),
            ),
            (
                local_topk[rank].values,
                torch.cat([topk.values for topk in local_topk], -1),
            ),
            (
                local_topk[rank].indices + rank * 3,
                torch.cat(
                    [
                        topk.indices + shard_rank * 3
                        for shard_rank, topk in enumerate(local_topk)
                    ],
                    -1,
                ),
            ),
        ]
        all_gather = Gather(gather_calls)

        monkeypatch.setattr(
            logits_processor,
            "tensor_model_parallel_all_gather",
            all_gather,
        )
        monkeypatch.setattr(logits_processor, "_topk", lambda x, k: x.topk(k, dim=-1))
        ids, values = logits_processor.LogitsProcessor.get_top_k_tokens(
            fake, lm_head, torch.empty(2, 1), 2, return_log_probs=True
        )
        expected = torch.cat(valid_shards, -1).log_softmax(-1).topk(2)
        torch.testing.assert_close(ids, expected.indices)
        torch.testing.assert_close(values, expected.values)
        assert all_gather.index == (3 if tp_size > 1 else 0)


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


@pytest.mark.parametrize("lilicorr", [False, True])
def test_candidate_generation_routes_scores_and_adaptive_inputs(monkeypatch, lilicorr):
    """Both heads share the walk, including a padded request and adaptive scores."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    key = "lilicorr_candidate_topk" if lilicorr else "selector_top_k"

    def init_base(self, config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=4, dflash_config={key: 4})
        )
        self.max_num_reqs = 3
        self.num_speculative_steps = 2
        self.num_query_per_req = 3
        self.dtype = torch.float32

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)
    cls = LiLiCorrSpeculator if lilicorr else DFlash2Speculator
    spec = cls(None, torch.device("cpu"))
    hidden = torch.arange(36).float().view(9, 4)
    spec._run_model = lambda *args: hidden
    spec.sample_indices = torch.tensor([1, 2, 4, 5, 7, 8])
    spec.sample_pos = torch.arange(6)
    spec.sample_idx_mapping = torch.tensor([1, 1, 0, 0, -1, -1])
    spec.sample_col = torch.tensor([0, 1, 0, 1, 0, 1])
    spec.temperature = torch.ones(3)
    spec.seeds = torch.arange(3)
    spec.draft_tokens = torch.empty(3, 2, dtype=torch.long)
    spec.draft_logits = None
    spec.use_fp64_gumbel = False
    spec.enable_adaptive_verification = True
    spec.input_buffers = SimpleNamespace(input_ids=torch.arange(9))
    candidates = torch.arange(24).view(6, 4)
    log_probs = candidates.float().log_softmax(-1)
    scores = torch.randn(3, 2, 4, 4)
    scoring_inputs: list[torch.Tensor] = []
    sampling_inputs: list[Any] = []
    adaptive_inputs: list[torch.Tensor] = []

    def score(*args):
        scoring_inputs.extend(args)
        return scores

    def sample(*args):
        sampling_inputs.extend(args)
        spec.candidate_sampler.scores.copy_(scores[:, :, 0])

    spec.model = SimpleNamespace(
        compute_candidates=lambda h: (candidates, log_probs),
        model=SimpleNamespace(lilicorr=score, candidate_selector=score),
    )
    spec.target_embeddings = lambda ids: ids.float().unsqueeze(-1).expand(*ids.shape, 4)
    if lilicorr:
        spec.anchor_hidden.copy_(torch.arange(12).view(3, 4))
        spec.anchor_valid[:2] = True
    spec.candidate_sampler.sample = sample
    spec._maybe_predict_acceptance = lambda *args: adaptive_inputs.extend(args)
    spec._generate_draft(3, 9, None, None, None)

    expected_ids = candidates.view(3, 2, 4)
    expected_first = spec.target_embeddings(expected_ids) if lilicorr else expected_ids
    torch.testing.assert_close(scoring_inputs[0], expected_first)
    torch.testing.assert_close(scoring_inputs[1], log_probs.view(3, 2, 4))
    torch.testing.assert_close(
        scoring_inputs[2], hidden[spec.sample_indices].view(3, 2, 4)
    )
    if lilicorr:
        torch.testing.assert_close(scoring_inputs[3], spec.anchor_hidden)
        torch.testing.assert_close(scoring_inputs[4], spec.anchor_valid)
    else:
        torch.testing.assert_close(scoring_inputs[3], torch.tensor([0, 3, 6]))
    torch.testing.assert_close(sampling_inputs[0], expected_ids)
    assert sampling_inputs[1] is scores
    assert sampling_inputs[4] is spec.sample_idx_mapping
    torch.testing.assert_close(adaptive_inputs[0], scores[:, :, 0].flatten(0, 1))
    torch.testing.assert_close(adaptive_inputs[1], spec.sample_idx_mapping)
    torch.testing.assert_close(adaptive_inputs[2], spec.sample_col)


def test_lilicorr_config_preserves_architecture():
    from transformers import Qwen3Config

    config = Qwen3Config(architectures=["LiLiCorrDraftModel"])
    wrapped = EAGLEConfig(config, method="dflash")
    assert wrapped.architectures == ["LiLiCorrDraftModel"]


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


@pytest.mark.parametrize(
    ("convolution", "mismatch", "quantized"),
    [
        pytest.param(False, None, False, id="valid-plain"),
        pytest.param(True, None, False, id="valid-convolution"),
        pytest.param(False, "missing_head", False, id="missing-head"),
        pytest.param(False, "extra_head", False, id="extra-head"),
        pytest.param(True, "missing_conv", False, id="missing-convolution"),
        pytest.param(False, "extra_conv", False, id="unexpected-convolution"),
        pytest.param(False, None, True, id="optional-w4a16-input-scale"),
        pytest.param(False, "missing_weight", True, id="missing-quantized-weight"),
        pytest.param(False, "missing_bias", True, id="missing-quantized-bias"),
        pytest.param(False, "missing_weight_scale", True, id="missing-block-scale"),
        pytest.param(False, "missing_weight_scale_2", True, id="missing-global-scale"),
    ],
)
def test_checkpoint_coverage_rejects_incomplete_or_wrong_heads(
    monkeypatch, convolution, mismatch, quantized
):
    from vllm.model_executor.models.lilicorr import LiLiCorr, LiLiCorrForCausalLM

    wrapper = LiLiCorrForCausalLM.__new__(LiLiCorrForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = LiLiCorr.__new__(LiLiCorr)
    nn.Module.__init__(wrapper.model)
    wrapper.has_own_lm_head = False
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
        from vllm.config.quantization import QuantSpec
        from vllm.model_executor.layers.quantization.modelopt import (
            CkptCtx,
            ModelOptLinearMethod,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kNvfp4Static,
        )

        # Match the W4A16 NVFP4 checkpoint contract without GPU kernel setup.
        projection = wrapper.model.lilicorr.pass_hidden_proj
        projection.quant_method = ModelOptLinearMethod(
            QuantSpec(weight=kNvfp4Static, activation=None), CkptCtx(group_size=16)
        )
        projection.weight = nn.Parameter(
            torch.zeros(8, 8, dtype=torch.uint8), requires_grad=False
        )
        for name, value in (
            ("weight_scale", torch.ones(8, 1).to(torch.float8_e4m3fn)),
            ("weight_scale_2", torch.ones(1)),
            ("input_scale", torch.full((1,), torch.nan)),
        ):
            projection.register_parameter(
                name, nn.Parameter(value, requires_grad=False)
            )
    weights = dict(wrapper.model.state_dict())
    if quantized:
        del weights["lilicorr.pass_hidden_proj.input_scale"]
    if mismatch == "missing_head":
        del weights["lilicorr.slot_embedding"]
    elif mismatch in (
        "missing_weight",
        "missing_bias",
        "missing_weight_scale",
        "missing_weight_scale_2",
    ):
        del weights[f"lilicorr.pass_hidden_proj.{mismatch.removeprefix('missing_')}"]
    elif mismatch == "extra_head":
        weights["lilicorr.untrained.weight"] = torch.zeros(1)
    elif mismatch == "missing_conv":
        del weights["layers.0.attention_conv.weight"]
    elif mismatch == "extra_conv":
        weights["layers.0.mlp_conv.untrained"] = torch.zeros(1)

    # Keep production loading and mapping, bypass only backbone GPU KV setup.
    wrapper.model.use_aux_hidden_state = True
    wrapper.model.has_separate_mask_embedding = False
    monkeypatch.setattr(wrapper, "_read_mask_embedding", lambda: None)
    monkeypatch.setattr(
        wrapper.model, "_build_fused_kv_buffers", lambda: None, raising=False
    )
    supplied = [("model." + name, value) for name, value in weights.items()]
    if mismatch:
        message = (
            "no module or parameter"
            if mismatch.startswith("extra")
            else "coverage mismatch"
        )
        with pytest.raises(ValueError, match=message):
            wrapper.load_weights(supplied)
    else:
        wrapper.load_weights(supplied)
        assert wrapper.model.lilicorr._attn_bias is not None


@pytest.fixture
def owned_head_model(monkeypatch):
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models import qwen3_dflash
    from vllm.model_executor.models.lilicorr import LiLiCorrForCausalLM

    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    monkeypatch.setattr(qwen3_dflash, "get_draft_quant_config", lambda _: quant_config)

    class Backbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.quant_config = quant_config
            self.use_aux_hidden_state = True
            self.has_separate_mask_embedding = False
            self.lilicorr = LiLiCorrHead(
                model_hidden_size=16, block_size=4, rms_norm_eps=1e-6, config=_config()
            )

        def _build_fused_kv_buffers(self):
            pass

    monkeypatch.setattr(LiLiCorrForCausalLM, "model_cls", Backbone)
    monkeypatch.setattr(LiLiCorrForCausalLM, "_read_mask_embedding", lambda _: None)

    def build(owned=True):
        config = SimpleNamespace(
            vocab_size=64,
            draft_vocab_size=64,
            hidden_size=16,
            has_own_lm_head=owned,
            num_hidden_layers=0,
            dflash_config={},
        )
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(hf_config=config),
                draft_load_config=None,
                attention_backend=None,
                kv_cache_dtype=None,
            ),
            model_config=SimpleNamespace(
                get_total_num_hidden_layers=lambda: 0,
                get_vocab_size=lambda: 64,
            ),
            attention_config=SimpleNamespace(),
            cache_config=SimpleNamespace(cache_dtype="auto"),
            load_config=LoadConfig(),
        )
        with set_current_vllm_config(
            VllmConfig(
                device_config=DeviceConfig("cpu"),
                compilation_config=CompilationConfig(mode=0),
            )
        ):
            model = LiLiCorrForCausalLM(vllm_config=vllm_config)
        return model, vllm_config

    return build


@pytest.mark.parametrize("missing", [None, "weight", "weight_scale", "weight_scale_2"])
def test_owned_nvfp4_lm_head_loads_required_checkpoint_tensors(
    owned_head_model, missing
):
    model, _ = owned_head_model()
    assert model.lm_head.weight.dtype == torch.uint8
    assert model.lm_head.weight.shape == (64, 8)
    weights = {
        name: torch.ones_like(value) for name, value in model.state_dict().items()
    }
    del weights["lm_head.input_scale"]  # Deprecated and unused for W4A16.
    if missing:
        del weights[f"lm_head.{missing}"]
        with pytest.raises(ValueError, match=f"lm_head.{missing}"):
            model.load_weights(weights.items())
    else:
        model.load_weights(weights.items())
        assert model.has_own_lm_head
        for name in ("weight", "weight_scale", "weight_scale_2"):
            torch.testing.assert_close(
                getattr(model.lm_head, name), weights[f"lm_head.{name}"]
            )


@pytest.mark.parametrize("owned", [False, True])
def test_selected_lm_head_survives_dflash_and_lilicorr_sharing(
    monkeypatch, owned_head_model, owned
):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    model, config = owned_head_model(owned)
    own_head = model.lm_head
    target_head = owned_head_model(owned)[0].lm_head
    # Identical packed bytes do not imply identical quantized heads.
    with torch.no_grad():
        own_head.weight.fill_(1)
        target_head.weight.fill_(1)
        if owned:
            own_head.weight_scale_2.fill_(1)
            target_head.weight_scale_2.fill_(2)
    embedding = VocabParallelEmbedding(64, 16)
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=embedding), lm_head=target_head
    )
    monkeypatch.setattr(
        eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        dflash_utils, "replace", lambda obj, **kw: SimpleNamespace(**(vars(obj) | kw))
    )
    monkeypatch.setattr(dflash_utils, "get_pp_safe_draft_load_config", lambda c: c)
    monkeypatch.setattr(dflash_utils, "get_model", lambda **kw: model)
    if owned:

        def no_weight_comparison(*args):
            raise AssertionError("Explicit ownership must bypass raw-weight comparison")

        monkeypatch.setattr(dflash_utils, "_should_share", no_weight_comparison)
    spec = LiLiCorrSpeculator.__new__(LiLiCorrSpeculator)
    spec.vllm_config = config
    assert spec.load_draft_model(target, set()) is model
    assert model.lm_head is (own_head if owned else target_head)
    assert spec.target_embeddings is embedding
    if owned:
        target.lm_head = None
        assert spec.load_draft_model(target, set()).lm_head is own_head


def test_quantized_head_calls_methods_without_reading_packed_weights(monkeypatch):
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
            config=_config(hidden_size=8),
            quant_config=quant_config,
            prefix="model.lilicorr",
        )
    with torch.no_grad():
        for module in head.modules():
            if isinstance(module, LinearBase):
                if module.quant_config is None:
                    module.weight.normal_(std=0.1)
                    module.bias.zero_()
                    continue
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
        for layer in head.layers:
            layer.attn.in_proj_weight.normal_(std=0.1)
            layer.attn.in_proj_bias.zero_()
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
    for name in ("factor_input_proj", "out_head", "in_head"):
        assert getattr(head, name).quant_config is None
        assert f"model.lilicorr.{name}" not in configured
    assert set(calls) == set(configured)
    assert len(configured) == len(set(configured))
    for name, module in head.named_modules():
        if isinstance(module, LinearBase):
            assert module.prefix == f"model.lilicorr.{name}"
