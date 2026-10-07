# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-only DFlash behavior.

``dflash_has_any_non_causal`` decides pre-build whether the draft needs a
non-causal-capable backend, so its branch table (explicit override, SWA-derived
per-layer causality, and the no-``layer_types`` fallback) is worth pinning.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.qwen3_dflash import (
    _dflash_layer_causal,
    _get_dflash_fc_input_size,
    dflash_has_any_non_causal,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    apply_eagle3_aux_layer_scale,
    get_eagle3_aux_layers_from_config,
    set_eagle3_aux_hidden_state_layers,
)


def _config(num_hidden_layers, layer_types=None, causal_override=None, is_causal=None):
    dflash_config = None if causal_override is None else {"causal": causal_override}
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        # Override forces causality on every layer, ignoring layer_types.
        (_config(2, layer_types=["full_attention"] * 2, causal_override=True), False),
        # Override forces non-causal on every layer.
        (
            _config(2, layer_types=["sliding_attention"] * 2, causal_override=False),
            True,
        ),
        # DFlash2 stores the explicit attention semantics at the top level.
        (
            _config(
                2,
                layer_types=["sliding_attention"] * 2,
                is_causal=False,
            ),
            True,
        ),
        (
            _config(2, layer_types=["full_attention"] * 2, is_causal=True),
            False,
        ),
        # SWA-derived: full-attention layers are non-causal.
        (_config(2, layer_types=["sliding_attention", "full_attention"]), True),
        # SWA-derived: all-sliding is fully causal.
        (_config(2, layer_types=["sliding_attention", "sliding_attention"]), False),
        # No layer_types -> non-causal fallback.
        (_config(2, layer_types=None), True),
        (_config(2, layer_types=[]), True),
    ],
)
def test_dflash_has_any_non_causal(config, expected):
    assert dflash_has_any_non_causal(config) is expected


def test_dflash_layer_causal_is_per_layer():
    config = _config(2, layer_types=["sliding_attention", "full_attention"])
    assert _dflash_layer_causal(config, 0) is True
    assert _dflash_layer_causal(config, 1) is False


def test_dflash_layer_causal_honors_top_level_override():
    config = _config(
        2,
        layer_types=["sliding_attention", "full_attention"],
        is_causal=False,
    )
    assert _dflash_layer_causal(config, 0) is False
    assert _dflash_layer_causal(config, 1) is False


def _vllm_config(**draft_config):
    config = SimpleNamespace(**draft_config)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )


def test_dflash_fc_uses_aux_layer_count():
    vllm_config = _vllm_config(
        num_hidden_layers=5,
        hidden_size=4096,
        target_hidden_size=None,
        target_layer_ids=[1, 17, 32],
    )

    assert _get_dflash_fc_input_size(vllm_config) == 3 * 4096


class _Eagle3Target(SupportsEagle3):
    def __init__(self, config):
        self.config = config
        self.applied = None

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.applied = layers


def _edge_spec_config():
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                dflash_config={"target_layer_ids": [1, 7, 13, 19, 25]}
            )
        )
    )


def test_cosmos3_edge_declares_eagle3_on_split_backbone():
    from vllm.model_executor.models.cosmos3_edge import (
        Cosmos3EdgeForConditionalGeneration,
        Cosmos3EdgeTextModel,
    )
    from vllm.transformers_utils.configs.cosmos3_edge import Cosmos3EdgeConfig

    assert SupportsEagle3 in Cosmos3EdgeForConditionalGeneration.__mro__
    assert issubclass(Cosmos3EdgeTextModel, EagleModelMixin)
    config = Cosmos3EdgeConfig()
    assert config.image_token_index == config.image_token_id == 19
    assert config.eagle_aux_hidden_state_layer_scale == 2


def test_set_eagle3_aux_layers_scales_from_target_config():
    spec_config = _edge_spec_config()
    edge = _Eagle3Target(SimpleNamespace(eagle_aux_hidden_state_layer_scale=2))
    qwen = _Eagle3Target(SimpleNamespace(model_type="qwen3_vl"))
    named_only = _Eagle3Target(SimpleNamespace(model_type="cosmos3_edge"))

    set_eagle3_aux_hidden_state_layers(edge, spec_config)
    set_eagle3_aux_hidden_state_layers(qwen, spec_config)
    set_eagle3_aux_hidden_state_layers(named_only, spec_config)

    assert edge.applied == (4, 16, 28, 40, 52)
    assert qwen.applied == (2, 8, 14, 20, 26)
    assert named_only.applied == (2, 8, 14, 20, 26)


def test_cosmos3_edge_aux_layers_use_split_block_capture():
    """Edge HF block ids stay in the draft config; runtime ids are 2*(i+1)."""
    from vllm.transformers_utils.configs.cosmos3_edge import Cosmos3EdgeConfig

    target_layer_ids = [1, 7, 13, 19, 25]
    vllm_config = _vllm_config(dflash_config={"target_layer_ids": target_layer_ids})
    qwen_ids = get_eagle3_aux_layers_from_config(vllm_config.speculative_config)
    assert qwen_ids == (2, 8, 14, 20, 26)
    assert (
        apply_eagle3_aux_layer_scale(qwen_ids, SimpleNamespace(model_type="qwen3_vl"))
        == qwen_ids
    )
    assert (
        apply_eagle3_aux_layer_scale(
            qwen_ids, SimpleNamespace(model_type="cosmos3_edge")
        )
        == qwen_ids
    )
    assert apply_eagle3_aux_layer_scale(qwen_ids, Cosmos3EdgeConfig()) == (
        4,
        16,
        28,
        40,
        52,
    )


def test_dflash_target_layers_map_to_post_layer_boundaries():
    vllm_config = _vllm_config(
        dflash_config={"target_layer_ids": [0, 16, 31]},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == (
        1,
        17,
        32,
    )


@pytest.mark.parametrize("config_name", ["dflash_config", "eagle_config"])
def test_eagle_aux_layers_preserves_legacy_layer_ids(config_name):
    layer_ids = [1, 17, 32]
    vllm_config = _vllm_config(
        **{config_name: {"layer_ids": layer_ids}},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == tuple(
        layer_ids
    )
