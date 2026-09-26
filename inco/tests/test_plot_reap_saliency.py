# SPDX-License-Identifier: Apache-2.0
"""The saliency plot's real logic is the normalization and the pruning cut it
draws: a layer whose one huge expert would swamp max-normalization must still
show structure, and the retained mask has to match what `reap.prune` deletes
(the `int(n * ratio)` lowest-saliency experts, independently per layer)."""

import numpy as np
import pytest
from scripts.plot_reap_saliency import (
    experts_pruned,
    load_saliency,
    relative_to_layer_median,
    retained_mask,
    sort_within_layer,
)


def test_median_normalization_is_per_layer_so_layer_scale_cancels():
    # Same shape, 100x apart in absolute scale, as layer 0 vs layer 47 are.
    scaled = relative_to_layer_median(
        np.array([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]])
    )

    assert scaled[0] == pytest.approx(scaled[1])
    assert scaled[0] == pytest.approx([0.5, 1.0, 1.5])


def test_median_survives_the_outlier_expert_that_would_flatten_max_scaling():
    """One expert at 80x the median must not push the rest toward zero."""
    layer = np.array([[1.0, 1.0, 1.0, 1.0, 80.0]])

    scaled = relative_to_layer_median(layer)

    assert scaled[0, :4] == pytest.approx(1.0)
    assert scaled[0, 4] == pytest.approx(80.0)


def test_all_zero_layer_normalizes_to_zero_not_nan():
    scaled = relative_to_layer_median(np.array([[0.0, 0.0], [1.0, 3.0]]))

    assert np.isfinite(scaled).all()
    assert scaled[0] == pytest.approx([0.0, 0.0])


def test_sorting_is_descending_and_per_row():
    sorted_rows = sort_within_layer(np.array([[1.0, 3.0, 2.0], [9.0, 0.0, 5.0]]))

    assert sorted_rows.tolist() == [[3.0, 2.0, 1.0], [9.0, 5.0, 0.0]]


@pytest.mark.parametrize(
    ("num_experts", "ratio", "expected"),
    [(128, 0.5, 64), (128, 0.38, 48), (128, 0.0, 0), (64, 0.25, 16), (3, 0.5, 1)],
)
def test_prune_count_truncates_like_reap(num_experts, ratio, expected):
    assert experts_pruned(num_experts, ratio) == expected


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
def test_out_of_range_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="ratio"):
        experts_pruned(128, ratio)


def test_retained_mask_keeps_the_top_half_of_each_layer_independently():
    saliency = np.array([[4.0, 1.0, 3.0, 2.0], [1.0, 4.0, 2.0, 3.0]])

    mask = retained_mask(saliency, 0.5)

    assert mask.tolist() == [[True, False, True, False], [False, True, False, True]]


def test_retained_mask_ranks_on_saliency_not_layer_scale():
    """A whole layer of small values keeps its top half, not nothing."""
    saliency = np.array([[0.01, 0.02, 0.03, 0.04], [10.0, 20.0, 30.0, 40.0]])

    assert retained_mask(saliency, 0.5).sum(axis=1).tolist() == [2, 2]


def test_zero_ratio_retains_everything():
    assert retained_mask(np.zeros((3, 8)), 0.0).all()


def test_load_saliency_stacks_layers_in_order(tmp_path):
    torch = pytest.importorskip("torch")
    dump = tmp_path / "observations.pt"
    torch.save(
        {
            1: {"reap": torch.tensor([3.0, 4.0]), "ean_sum": torch.tensor([9.0, 9.0])},
            0: {"reap": torch.tensor([1.0, 2.0]), "ean_sum": torch.tensor([8.0, 8.0])},
        },
        dump,
    )

    assert load_saliency(dump).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert load_saliency(dump, "ean_sum").tolist() == [[8.0, 8.0], [9.0, 9.0]]


def test_load_saliency_names_the_available_metrics_when_one_is_missing(tmp_path):
    torch = pytest.importorskip("torch")
    dump = tmp_path / "observations.pt"
    torch.save({0: {"reap": torch.tensor([1.0])}}, dump)

    with pytest.raises(KeyError, match="reap_l2"):
        load_saliency(dump, "reap_l2")


def test_figure_renders_from_a_dump_shaped_like_the_real_one(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.plot_reap_saliency import main

    rng = np.random.default_rng(0)
    dump = tmp_path / "observations.pt"
    torch.save(
        {
            layer: {"reap": torch.tensor(rng.random(16) * (layer + 1))}
            for layer in range(4)
        },
        dump,
    )
    out = tmp_path / "heatmap.png"

    assert main(dump, 0.5, out) == out
    assert out.stat().st_size > 0
