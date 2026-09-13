# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits, *, adaptive=False):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.speculative_config = SimpleNamespace(enable_adaptive_verification=adaptive)
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.parametrize("adaptive", [False, True])
def test_confidence_buffer_is_only_allocated_for_adaptive(monkeypatch, adaptive):
    _stub_base(monkeypatch, None, adaptive=adaptive)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    confidence = speculator.draft_token_confidence_probs
    if adaptive:
        assert confidence.shape == speculator.draft_tokens.shape
        assert confidence.dtype == torch.float32
    else:
        assert confidence is None


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("has_head", [False, True])
def test_adaptive_requires_loaded_confidence_weights(monkeypatch, adaptive, has_head):
    _stub_base(monkeypatch, None, adaptive=adaptive)
    selector = SimpleNamespace(
        confidence_head=torch.nn.Linear(2, 1) if has_head else None
    )
    model = SimpleNamespace(model=SimpleNamespace(candidate_selector=selector))
    monkeypatch.setattr(DFlashSpeculator, "load_draft_model", lambda *args: model)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    if adaptive and not has_head:
        with pytest.raises(ValueError, match="confidence head"):
            speculator.load_draft_model(None, set())
    else:
        assert speculator.load_draft_model(None, set()) is model


@pytest.mark.parametrize("compute_confidence", [False, True])
def test_selector_confidence_is_opt_in_and_does_not_change_scores(
    monkeypatch, compute_confidence
):
    """A serialized head must not add work or change proposals in fixed mode."""
    from vllm.model_executor.models import qwen3_dflash2 as module

    predecessors = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    candidates = torch.tensor([[[1, 2], [2, 3]]])
    hidden = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
    anchors = torch.tensor([3])
    unary = torch.zeros(1, 2, 2)
    head = torch.nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[2.0, -1.0]]))
        head.bias.fill_(0.5)
    selector = SimpleNamespace(
        hidden_projection=torch.nn.Identity(),
        predecessor_codebook=predecessors,
        successor_codebook=predecessors,
        confidence_head=head,
        top_k=2,
    )
    if not compute_confidence:

        def unexpected_confidence(*args, **kwargs):
            pytest.fail("fixed-budget drafting must not compute confidence")

        monkeypatch.setattr(module, "_confidence_rows", unexpected_confidence)

    scores, confidence = module.CandidateSelector.forward(
        selector,
        candidates,
        unary,
        hidden,
        anchors,
        compute_confidence=compute_confidence,
    )
    torch.testing.assert_close(
        scores,
        _score_edges(predecessors, predecessors, candidates, unary, hidden, anchors, 2),
        rtol=0,
        atol=0,
    )
    if compute_confidence:
        # The anchor conditions every row at p0; p1 uses p0's candidates.
        torch.testing.assert_close(
            confidence, torch.tensor([[[2.5, 2.5], [-3.5, 2.5]]])
        )
    else:
        assert confidence is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("adaptive", [False, True])
def test_selector_walk_confidence_follows_selected_predecessors(monkeypatch, adaptive):
    _stub_base(monkeypatch, None, adaptive=adaptive)
    speculator = DFlash2Speculator(None, torch.device("cuda"))
    speculator.num_speculative_steps = 3
    speculator.selector_top_k = 2
    speculator.use_fp64_gumbel = False
    speculator.sample_pos = torch.arange(3, device="cuda").view(1, 3)
    speculator.sample_idx_mapping = torch.zeros(
        (1, 3), dtype=torch.int32, device="cuda"
    )
    speculator.temperature = torch.zeros(1, device="cuda")
    speculator.seeds = torch.zeros(1, dtype=torch.int64, device="cuda")
    speculator.draft_tokens = torch.empty((1, 3), dtype=torch.int64, device="cuda")
    speculator._selector_scores = torch.empty((1, 3, 2), device="cuda")
    speculator.draft_token_confidence_probs = (
        torch.full((1, 3), -1.0, device="cuda") if adaptive else None
    )
    candidates = torch.tensor([[[10, 11], [20, 21], [30, 31]]], device="cuda")
    scores = torch.tensor(
        [
            [
                [[0.0, 2.0], [0.0, 2.0]],
                [[0.0, 3.0], [4.0, 0.0]],
                [[0.0, 5.0], [6.0, 0.0]],
            ]
        ],
        device="cuda",
    )
    confidence = torch.tensor([[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]], device="cuda")
    speculator._sample_path(candidates, scores, 1, confidence if adaptive else None)
    assert speculator.draft_tokens.tolist() == [[11, 20, 31]]
    if adaptive:
        torch.testing.assert_close(
            speculator.draft_token_confidence_probs,
            torch.tensor([[0.1, 0.8, 0.3]], device="cuda").sigmoid(),
        )


@pytest.mark.parametrize("adaptive", [False, True])
def test_generate_draft_uses_selector_call_with_opt_in_confidence(
    monkeypatch, adaptive
):
    """Both modes must use __call__, which owns the selector's compile wrapper."""
    _stub_base(monkeypatch, None, adaptive=adaptive)
    speculator = DFlash2Speculator(None, torch.device("cpu"))
    hidden = torch.zeros(5, 2)
    candidates = torch.arange(12).view(4, 3)
    scores = torch.zeros(1, 4, 3, 3)
    confidence = torch.zeros(1, 4, 3) if adaptive else None

    class Selector:
        def __call__(self, ids, unary, states, anchors, *, compute_confidence):
            assert compute_confidence is adaptive
            return scores, confidence

    speculator.model = SimpleNamespace(
        compute_candidates=lambda _: (candidates, torch.zeros(4, 3)),
        model=SimpleNamespace(candidate_selector=Selector()),
    )
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(5, dtype=torch.long)
    )
    speculator.sample_indices = torch.arange(4)
    speculator._run_model = lambda *args: hidden
    sampled = []
    speculator._sample_path = lambda *args: sampled.append(args)
    speculator._generate_draft(1, 5, None, None, None)
    assert len(sampled) == 1
    ids, actual_scores, num_reqs, actual_confidence = sampled[0]
    torch.testing.assert_close(ids, candidates.view(1, 4, 3))
    assert actual_scores is scores
    assert num_reqs == 1
    assert actual_confidence is confidence


@pytest.mark.skip_global_cleanup
def test_dflash2_model_decoder_layer_cls(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    # 1. Mock get_current_vllm_config and TP groups
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    # 2. Mock vllm_config
    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 3,
            "conv_group_size": 2,
            "use_aux_hidden_state": False,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config

    # 3. Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(mock_current_vllm_config), torch.device("meta"):
        model = DFlash2Qwen3Model(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], DFlash2Qwen3DecoderLayer)
