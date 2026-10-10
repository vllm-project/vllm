# SPDX-License-Identifier: Apache-2.0
"""Joining the two phases of a speculative-decoding A/B.

The failure mode these tests guard is a *ratio between unrelated points*: the
drafted and undrafted phases are separate server runs, so nothing but this join
stops a c=64 drafted number being divided by a c=32 undrafted one and reported
as a speedup.
"""

import json

import pytest
from bench.plot_specdec import (
    MODEL_COLORS,
    MalformedResults,
    comparison_rows,
    draw_frontier,
    frontiers,
    main,
    merge,
    overlap_disagreement,
    plot,
    plot_frontier,
    slug,
)


def point(concurrency, gpu, user, accept_len=None):
    measured = {
        "concurrency": concurrency,
        "tok_s_gpu": gpu,
        "tok_s_user": user,
    }
    if accept_len is not None:
        measured["accept_len"] = accept_len
    return measured


def results(drafted, plain):
    return {
        "prompts": 128,
        "sources": {"evalplus/humanevalplus": 64, "evalplus/mbppplus": 64},
        "max_tokens": 512,
        "phases": {
            "draft": {"audit": {}, "points": drafted},
            "no-draft": {"audit": {}, "points": plain},
        },
    }


def test_rows_pair_each_concurrency_and_divide_within_it():
    rows = comparison_rows(
        results(
            [point(1, 351.2, 374.4, 3.893), point(64, 4036.0, 78.2, 3.914)],
            [point(1, 144.1, 143.9), point(64, 2721.0, 53.3)],
        )
    )
    assert [r.concurrency for r in rows] == [1, 64]
    assert rows[0].speedup == pytest.approx(2.438, abs=1e-3)
    assert rows[1].speedup == pytest.approx(1.483, abs=1e-3)
    assert rows[1].accept_len == 3.914


def test_rows_are_sorted_even_when_the_phases_were_written_out_of_order():
    rows = comparison_rows(
        results(
            [point(64, 4036.0, 78.2), point(1, 351.2, 374.4)],
            [point(1, 144.1, 143.9), point(64, 2721.0, 53.3)],
        )
    )
    assert [r.concurrency for r in rows] == [1, 64]
    assert rows[0].draft_gpu == 351.2


def test_mismatched_concurrencies_raise_rather_than_silently_ratio():
    with pytest.raises(MalformedResults, match="disagree on concurrency"):
        comparison_rows(
            results([point(32, 4622.4, 169.6)], [point(64, 2721.0, 53.3)])
        )


def test_a_run_with_skip_baseline_is_refused():
    only_drafted = results([point(1, 351.2, 374.4)], [])
    del only_drafted["phases"]["no-draft"]
    with pytest.raises(MalformedResults, match="no-draft"):
        comparison_rows(only_drafted)


def test_plot_writes_a_figure_and_main_reports_the_path(tmp_path, capsys):
    path = tmp_path / "nested" / "specdec.png"
    payload = results(
        [point(1, 351.2, 374.4, 3.893), point(64, 4036.0, 78.2, 3.914)],
        [point(1, 144.1, 143.9), point(64, 2721.0, 53.3)],
    )
    assert plot({"REAP-50%": payload}, path) == path
    assert path.stat().st_size > 0

    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(payload))
    assert main([str(results_file)]) == 0
    out = capsys.readouterr().out
    assert "2.44x" in out
    assert main([str(results_file), "--labels", "a,b"]) == 2
    assert str(tmp_path / "specdec-comparison.png") in out


def test_merge_concatenates_runs_and_later_points_win():
    first = results([point(1, 351.2, 374.4, 3.893)], [point(1, 144.1, 143.9)])
    second = results(
        [point(64, 4036.0, 78.2, 3.914), point(96, 3800.0, 52.0, 3.9)],
        [point(64, 2721.0, 53.3), point(96, 3400.0, 35.0)],
    )
    rows = comparison_rows(merge([first, second]))
    assert [r.concurrency for r in rows] == [1, 64, 96]
    assert rows[2].speedup == pytest.approx(3800.0 / 3400.0)


def test_overlap_compares_a_phase_against_itself_across_runs():
    first = results([point(64, 4000.0, 78.2)], [point(64, 2700.0, 53.3)])
    repeat = results([point(64, 4100.0, 78.0)], [point(64, 2754.0, 53.0)])
    gaps = overlap_disagreement([first, repeat])

    assert gaps[("draft", 64)] == pytest.approx(0.025, abs=1e-3)
    assert gaps[("no-draft", 64)] == pytest.approx(0.02, abs=1e-3)


def test_overlap_does_not_mistake_the_speedup_for_a_disagreement():
    """draft 4036 vs no-draft 2721 at c=64 is the result, not a discrepancy."""
    single = results([point(64, 4036.0, 78.2)], [point(64, 2721.0, 53.3)])
    assert overlap_disagreement([single]) == {}


def test_merge_refuses_an_empty_list():
    with pytest.raises(MalformedResults, match="no results"):
        merge([])


def _axis():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots()


def _labels(ax):
    """Every annotation on the axis as (text, horizontal, vertical) alignment."""
    return [(t.get_text(), t.get_ha(), t.get_va()) for t in ax.texts]


SWEEP = results(
    [
        point(1, 400.3, 429.8, 3.894),
        point(2, 664.1, 355.7, 3.926),
        point(96, 11579.9, 137.7, 3.922),
    ],
    [point(1, 168.1, 167.7), point(2, 281.6, 141.0), point(96, 5003.8, 55.4)],
)


def test_every_point_is_labelled_on_a_single_run_frontier():
    """The whole reason the panel is drawn alone: c= on all of them."""
    fig, ax = _axis()
    draw_frontier(ax, {"REAP-50%": comparison_rows(SWEEP)}, label_every=True)

    assert sorted(text for text, _, _ in _labels(ax)) == [
        "c=1", "c=1", "c=2", "c=2", "c=96", "c=96",
    ]


def test_the_shared_figure_labels_only_the_ends():
    """Three c= labels per run: both drafted ends, and the undrafted top end
    only -- undrafted c=1 is where every model lands in the same corner."""
    fig, ax = _axis()
    draw_frontier(ax, {"REAP-50%": comparison_rows(SWEEP)}, label_every=False)

    assert sorted(text for text, _, _ in _labels(ax)) == ["c=1", "c=96", "c=96"]
    assert {(ha, va) for _, ha, va in _labels(ax)} == {("left", "baseline")}


def test_a_label_moves_beside_its_point_where_the_curve_is_vertical():
    """Above/below would land on the next marker once throughput saturates."""
    fig, ax = _axis()
    draw_frontier(ax, {"REAP-50%": comparison_rows(SWEEP)}, label_every=True)

    placed = set(_labels(ax))

    # c=96 undrafted is the top of the saturated, near-vertical tail.
    assert ("c=96", "right", "center") in placed
    # c=1 drafted sits on the shallow tail and keeps its label above.
    assert ("c=1", "center", "baseline") in placed


def test_a_run_keeps_its_colour_when_drawn_without_the_other():
    """Hue is the model, so the second run must not become the first's."""
    fig, ax = _axis()
    draw_frontier(
        ax, {"unpruned": comparison_rows(SWEEP)}, True, indices={"unpruned": 1}
    )

    assert {line.get_color() for line in ax.lines} == {MODEL_COLORS[1]}


def test_frontiers_writes_one_file_per_run_plus_a_combined_one(tmp_path):
    written = frontiers(
        {"REAP-50%": SWEEP, "unpruned": SWEEP}, tmp_path / "specdec-v5.png"
    )

    assert [p.name for p in written] == [
        "specdec-v5-frontier-reap-50.png",
        "specdec-v5-frontier-unpruned.png",
        "specdec-v5-frontier.png",
    ]
    assert all(p.stat().st_size > 0 for p in written)


def test_a_lone_run_gets_no_duplicate_combined_frontier(tmp_path):
    written = frontiers({"REAP-50%": SWEEP}, tmp_path / "specdec-v5.png")

    assert [p.name for p in written] == ["specdec-v5-frontier-reap-50.png"]


def test_plot_frontier_writes_a_single_panel_figure(tmp_path):
    path = tmp_path / "nested" / "frontier.png"

    assert plot_frontier({"REAP-50%": SWEEP}, path) == path
    assert path.stat().st_size > 0


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("REAP-50%", "reap-50"),
        ("unpruned", "unpruned"),
        ("Qwen3 30B / A3B", "qwen3-30b-a3b"),
        ("%%%", "run"),
    ],
)
def test_slug_is_filename_safe(label, expected):
    assert slug(label) == expected
