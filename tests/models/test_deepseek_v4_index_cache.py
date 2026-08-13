# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexCache layer selection for DeepSeek V4.

Only C4A layers run an indexer, so ``index_topk_freq`` /
``index_topk_pattern`` must be applied over those layers rather than over all
layers, and the first C4A layer must always compute its own top-k.
"""

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v4.attention import (
    _resolve_skip_topk,
    _validate_index_cache_ubatching,
)

# DeepSeek-V4-Flash: 43 layers (C4A on even ids 2..42, C128A between them) plus
# trailing SWA-only slots. V4-Flash ships one trailing entry, V4-Flash-0731
# ships three; both must resolve to the same 21 C4A layers.
NUM_HIDDEN_LAYERS = 43
FLASH_COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0]
FLASH_0731_COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0, 0, 0]
LAYOUTS = {
    "v4-flash": FLASH_COMPRESS_RATIOS,
    "v4-flash-0731": FLASH_0731_COMPRESS_RATIOS,
}
C4A_LAYERS = [i for i, ratio in enumerate(FLASH_COMPRESS_RATIOS) if ratio == 4]


def _config(ratios=None, **kwargs) -> SimpleNamespace:
    ratios = FLASH_COMPRESS_RATIOS if ratios is None else ratios
    return SimpleNamespace(
        compress_ratios=ratios, num_hidden_layers=NUM_HIDDEN_LAYERS, **kwargs
    )


def _skipped_layers(config: SimpleNamespace, **kwargs) -> list[int]:
    # A pipeline rank only builds its own layers, so only those are resolved.
    start = kwargs.get("local_start_layer", 0)
    end = kwargs.get("local_end_layer") or len(config.compress_ratios)
    return [
        layer_id
        for layer_id in range(start, end)
        if _resolve_skip_topk(config, layer_id, **kwargs)
    ]


@pytest.mark.parametrize("ratios", LAYOUTS.values(), ids=list(LAYOUTS))
def test_both_checkpoints_have_the_same_21_c4a_layers(ratios):
    config = _config(ratios, use_index_cache=True, index_topk_freq=2)

    assert [i for i, r in enumerate(ratios[:NUM_HIDDEN_LAYERS]) if r == 4] == C4A_LAYERS
    # The trailing MTP/draft slots must not shift the selection.
    assert _skipped_layers(config) == _skipped_layers(
        _config(FLASH_COMPRESS_RATIOS, use_index_cache=True, index_topk_freq=2)
    )


def test_flash_layout_has_21_c4a_layers():
    assert list(range(2, 43, 2)) == C4A_LAYERS


def test_disabled_without_use_index_cache():
    assert _skipped_layers(_config(index_topk_freq=2)) == []


@pytest.mark.parametrize("freq", [2, 3, 4])
def test_freq_counts_c4a_layers_not_all_layers(freq):
    config = _config(use_index_cache=True, index_topk_freq=freq)

    expected = [
        layer_id for c4a_idx, layer_id in enumerate(C4A_LAYERS) if c4a_idx % freq != 0
    ]
    assert _skipped_layers(config) == expected
    # The naive V3.2 formula over absolute layer ids would skip every C4A layer
    # (they are all even), leaving nothing to populate the shared buffer.
    assert C4A_LAYERS[0] not in expected
    assert len(expected) < len(C4A_LAYERS)


def test_freq_one_is_a_no_op():
    assert _skipped_layers(_config(use_index_cache=True, index_topk_freq=1)) == []


def test_pattern_maps_to_c4a_layers_in_order():
    pattern = "FSS" + "F" * (len(C4A_LAYERS) - 3)
    config = _config(use_index_cache=True, index_topk_pattern=pattern)

    assert _skipped_layers(config) == [C4A_LAYERS[1], C4A_LAYERS[2]]


def test_pattern_overrides_freq():
    config = _config(
        use_index_cache=True,
        index_topk_freq=2,
        index_topk_pattern="F" * len(C4A_LAYERS),
    )

    assert _skipped_layers(config) == []


def test_v32_length_pattern_is_rejected():
    # The 61-character DeepSeek-V3.2 example from the IndexCache docs.
    config = _config(use_index_cache=True, index_topk_pattern="F" * 61)

    with pytest.raises(ValueError, match="C4A layers"):
        _resolve_skip_topk(config, C4A_LAYERS[0])


def test_pattern_cannot_share_on_first_c4a_layer():
    config = _config(
        use_index_cache=True,
        index_topk_pattern="S" + "F" * (len(C4A_LAYERS) - 1),
    )

    with pytest.raises(ValueError, match="no previous"):
        _resolve_skip_topk(config, C4A_LAYERS[0])


def test_pattern_rejects_characters_other_than_f_and_s():
    config = _config(use_index_cache=True, index_topk_pattern="FX" + "F" * 19)

    with pytest.raises(ValueError, match="only accepts"):
        _resolve_skip_topk(config, C4A_LAYERS[1])


def test_each_pipeline_stage_computes_its_own_first_c4a_layer():
    # Two PP stages over 43 layers, split so stage 1 starts on layer 24 — a C4A
    # layer that freq=2 would otherwise mark shared.
    config = _config(use_index_cache=True, index_topk_freq=2)

    assert 24 in _skipped_layers(config)  # shared when the model is one stage

    stage0 = _skipped_layers(config, local_start_layer=0, local_end_layer=24)
    stage1 = _skipped_layers(config, local_start_layer=24, local_end_layer=43)

    # topk_indices_buffer is rank-local: reusing here would read a buffer that
    # only the other rank ever wrote.
    assert 24 not in stage1
    assert stage1 == [28, 32, 36, 40]
    assert C4A_LAYERS[0] not in stage0
    assert stage0 == [4, 8, 12, 16, 20]


def test_reuse_is_rejected_under_ubatching():
    # Micro-batches share one topk_indices_buffer, so a skipped layer would read
    # whichever micro-batch wrote last.
    with pytest.raises(NotImplementedError, match="DBO/ubatching"):
        _validate_index_cache_ubatching(skip_topk=True, use_ubatching=True)


@pytest.mark.parametrize(
    "skip_topk,use_ubatching", [(False, True), (True, False), (False, False)]
)
def test_ubatching_guard_allows_everything_else(skip_topk, use_ubatching):
    _validate_index_cache_ubatching(skip_topk, use_ubatching)
