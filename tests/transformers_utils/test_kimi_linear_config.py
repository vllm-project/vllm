# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

NUM_HIDDEN_LAYERS = 93
KDA_HEAD_DIMS = {
    "num_heads": 96,
    "head_dim": 128,
    "use_full_rank_gate": True,
    "short_conv_kernel_size": 4,
}


def _layer_layout(num_hidden_layers):
    """Reproduce the real Kimi-Linear / Kimi-K3 hybrid layout.

    Full-attention layers are the multiples of ``4`` starting at ``4`` plus the
    final layer when it is not a multiple of four; every other layer is KDA.
    This matches the released ``Kimi-Linear-48B-A3B`` (27 layers) and Kimi-K3
    (93 layers) configs.
    """
    full_attn_layers = list(range(4, num_hidden_layers + 1, 4))
    if num_hidden_layers % 4 != 0:
        full_attn_layers.append(num_hidden_layers)
    kda_layers = sorted(set(range(1, num_hidden_layers + 1)) - set(full_attn_layers))
    return kda_layers, full_attn_layers


def _make_config(num_hidden_layers=NUM_HIDDEN_LAYERS, **overrides):
    kda_layers, full_attn_layers = _layer_layout(num_hidden_layers)
    kwargs = {
        "num_hidden_layers": num_hidden_layers,
        "linear_attn_config": {
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            **KDA_HEAD_DIMS,
        },
    }
    kwargs.update(overrides)
    return KimiLinearConfig(**kwargs)


def test_kimi_linear_valid_real_layouts_accepted():
    # Kimi-K3 (93 layers) and Kimi-Linear-48B-A3B (27 layers) layouts construct.
    for num_hidden_layers in (93, 27):
        config = _make_config(num_hidden_layers=num_hidden_layers)
        kda, full = _layer_layout(num_hidden_layers)
        assert config.linear_attn_config["kda_layers"] == kda
        assert config.linear_attn_config["full_attn_layers"] == full


def test_kimi_linear_overlapping_layers_rejected():
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    # Layer 4 is full attention; also list it as KDA.
    assert 4 in full
    with pytest.raises(ValueError, match="overlap"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda + [4],
                "full_attn_layers": full,
                **KDA_HEAD_DIMS,
            }
        )


def test_kimi_linear_uncovered_layer_rejected():
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    full = [i for i in full if i != NUM_HIDDEN_LAYERS]  # drop final layer
    with pytest.raises(ValueError, match="do not cover"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda,
                "full_attn_layers": full,
                **KDA_HEAD_DIMS,
            }
        )


@pytest.mark.parametrize("bad_layers", [[0], [94], [-1], [1, 94]])
def test_kimi_linear_out_of_range_layer_index_rejected(bad_layers):
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    with pytest.raises(ValueError, match="outside the 1-based range"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda + bad_layers,
                "full_attn_layers": full,
                **KDA_HEAD_DIMS,
            }
        )


@pytest.mark.parametrize("bad_layers", [[1.5], ["3"]])
def test_kimi_linear_non_int_layer_rejected(bad_layers):
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    with pytest.raises(ValueError, match="invalid entry"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda + bad_layers,
                "full_attn_layers": full,
                **KDA_HEAD_DIMS,
            }
        )


@pytest.mark.parametrize("bad_layers", [[True], [False], [1, True]])
def test_kimi_linear_boolean_layer_rejected(bad_layers):
    # bool is an int subclass, so it must be rejected explicitly; otherwise
    # True would be silently accepted as layer 1.
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    with pytest.raises(ValueError, match="invalid entry"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda + bad_layers,
                "full_attn_layers": full,
                **KDA_HEAD_DIMS,
            }
        )


def test_kimi_linear_none_layer_value_rejected():
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    with pytest.raises(ValueError, match="invalid entry"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda[:1] + [None],
                "full_attn_layers": full,
                **KDA_HEAD_DIMS,
            }
        )


@pytest.mark.parametrize("container", [{1, 2}, "1,2", 42])
def test_kimi_linear_invalid_container_type_rejected(container):
    kda, _ = _layer_layout(NUM_HIDDEN_LAYERS)
    with pytest.raises(ValueError, match="must be a list or tuple"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda,
                "full_attn_layers": container,
                **KDA_HEAD_DIMS,
            }
        )


def test_kimi_linear_duplicate_kda_layer_rejected():
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    # Layer 1 appears both at the front and the end of the KDA list.
    with pytest.raises(ValueError, match="duplicate"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda + [1],
                "full_attn_layers": full,
                **KDA_HEAD_DIMS,
            }
        )


def test_kimi_linear_duplicate_full_attn_layer_rejected():
    kda, full = _layer_layout(NUM_HIDDEN_LAYERS)
    with pytest.raises(ValueError, match="duplicate"):
        _make_config(
            linear_attn_config={
                "kda_layers": kda,
                "full_attn_layers": full + [4],
                **KDA_HEAD_DIMS,
            }
        )


def test_kimi_linear_missing_keys_rejected():
    with pytest.raises(ValueError, match="kda_layers"):
        _make_config(
            linear_attn_config={
                "full_attn_layers": _layer_layout(NUM_HIDDEN_LAYERS)[1],
                **KDA_HEAD_DIMS,
            }
        )
    with pytest.raises(ValueError, match="full_attn_layers"):
        _make_config(linear_attn_config={**KDA_HEAD_DIMS})


def test_kimi_linear_without_linear_attn_config_is_unaffected():
    config = KimiLinearConfig(num_hidden_layers=8)
    assert config.linear_attn_config is None


def test_kimi_linear_all_full_attention_accepted():
    config = _make_config(
        linear_attn_config={
            "kda_layers": [],
            "full_attn_layers": list(range(1, NUM_HIDDEN_LAYERS + 1)),
            **KDA_HEAD_DIMS,
        }
    )
    assert config.is_linear_attn is False


def test_kimi_linear_all_kda_accepted():
    # All KDA, no full-attention layers, remains a linear-attention model.
    config = _make_config(
        linear_attn_config={
            "kda_layers": list(range(1, NUM_HIDDEN_LAYERS + 1)),
            "full_attn_layers": [],
            **KDA_HEAD_DIMS,
        }
    )
    assert config.is_linear_attn is True
