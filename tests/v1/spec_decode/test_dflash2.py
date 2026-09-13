# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
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


def test_dflash2_moe_builds_sparse_ffn(monkeypatch):
    """A MoE draft config reaches the HF config root and selects a sparse FFN."""
    from vllm.model_executor.models import qwen3_dflash2
    from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

    speculators_config = {
        "speculators_model_type": "dflash2",
        "aux_hidden_state_layer_ids": [2, 5],
        "mask_token_id": 31,
        "conv_kernel_size": 2,
        "conv_group_size": 4,
        "selector_rank": 8,
        "selector_top_k": 4,
        "transformer_layer_config": {"model_type": "qwen3", "hidden_size": 16},
        "draft_ffn_type": "moe",
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 12,
        "shared_expert_intermediate_size": 10,
        "norm_topk_prob": False,
    }

    config_dict = SpeculatorsConfig.extract_transformers_pre_trained_config(
        deepcopy(speculators_config)
    )

    # MoE geometry lands at the draft config root, where the FFN reads it -- not
    # inside `dflash_config`, which only carries DFlash runtime settings.
    assert config_dict["draft_ffn_type"] == "moe"
    assert config_dict["num_experts"] == 8
    assert config_dict["num_experts_per_tok"] == 2
    assert config_dict["moe_intermediate_size"] == 12
    assert config_dict["shared_expert_intermediate_size"] == 10
    assert config_dict["norm_topk_prob"] is False
    assert "draft_ffn_type" not in config_dict["dflash_config"]

    # An incomplete MoE config fails at load time rather than at layer build.
    partial = deepcopy(speculators_config)
    del partial["num_experts_per_tok"]
    with pytest.raises(ValueError, match="missing.*num_experts_per_tok"):
        SpeculatorsConfig.extract_transformers_pre_trained_config(partial)

    captured: dict[str, object] = {}

    class FakeSparseMoe(torch.nn.Module):
        def __init__(self, vllm_config, config, quant_config, prefix):
            super().__init__()
            captured.update(
                vllm_config=vllm_config,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            )

    monkeypatch.setattr(qwen3_dflash2, "Qwen3NextSparseMoeBlock", FakeSparseMoe)
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_eplb=False))
    quant_config = object()
    config = SimpleNamespace(**config_dict)

    mlp = qwen3_dflash2.DFlash2Qwen3DecoderLayer.build_mlp(
        None,
        vllm_config=vllm_config,
        config=config,
        quant_config=quant_config,
        prefix="model.layers.0.mlp",
    )

    assert isinstance(mlp, FakeSparseMoe)
    assert captured == {
        "vllm_config": vllm_config,
        "config": config,
        "quant_config": quant_config,
        "prefix": "model.layers.0.mlp",
    }

    # The drafter does not implement the `MixtureOfExperts` lifecycle, so it must
    # refuse to inherit the target's EPLB setting instead of running half-managed.
    vllm_config.parallel_config.enable_eplb = True
    with pytest.raises(NotImplementedError, match="EPLB is not supported"):
        qwen3_dflash2.DFlash2Qwen3DecoderLayer.build_mlp(
            None,
            vllm_config=vllm_config,
            config=config,
            quant_config=quant_config,
            prefix="model.layers.0.mlp",
        )


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
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


@pytest.mark.skip_global_cleanup
def test_dflash2_model_decoder_layer_cls(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.qwen2 import Qwen2MLP
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
    assert isinstance(model.layers[0].mlp, Qwen2MLP)
