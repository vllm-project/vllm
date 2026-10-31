# SPDX-License-Identifier: Apache-2.0
"""Quality-delta reporting for a compressed checkpoint.

The failure mode these tests guard is a *wrong claim in the report*: quoting a
1pp delta as a regression when the per-task stderr is 2pp, or reading `acc`
instead of `acc_norm` and inventing a collapse that byte-length normalization
explains away.
"""

import json

import pytest
from bench.eval_compare import (
    MalformedEval,
    Score,
    compare,
    combined_stderr,
    load_eval_dir,
    main,
    mean_delta,
    parse_task_metrics,
    verdict,
)


def mc(acc, acc_norm, acc_se=0.02, norm_se=0.022):
    """A multiple-choice metric block as lm-eval writes it."""
    return {
        "acc,none": acc,
        "acc_stderr,none": acc_se,
        "acc_norm,none": acc_norm,
        "acc_norm_stderr,none": norm_se,
    }


def write_eval(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(payload))
    return directory


@pytest.fixture
def baseline_dir(tmp_path):
    return write_eval(
        tmp_path / "base",
        "bundle",
        {
            "openbookqa": mc(0.324, 0.448),
            "rte": {"acc,none": 0.812, "acc_stderr,none": 0.023},
            "humaneval": {"pass@1,create_test": 0.600, "pass@1_stderr,create_test": 0.038},
        },
    )


@pytest.fixture
def pruned_dir(tmp_path):
    return write_eval(
        tmp_path / "pruned",
        "bundle",
        {
            "openbookqa": mc(0.212, 0.314),
            "rte": {"acc,none": 0.805, "acc_stderr,none": 0.024},
            "humaneval": {"pass@1,create_test": 0.400, "pass@1_stderr,create_test": 0.038},
        },
    )


class TestMetricSelection:
    def test_acc_norm_is_preferred_over_acc(self):
        """The REAP paper reports acc_norm; acc is length-biased."""
        score = parse_task_metrics(mc(0.324, 0.448), "openbookqa")
        assert score.metric == "acc_norm"
        assert score.value == 0.448
        assert score.stderr == 0.022

    def test_acc_is_used_when_no_acc_norm_exists(self):
        score = parse_task_metrics({"acc,none": 0.812, "acc_stderr,none": 0.023}, "rte")
        assert (score.metric, score.value) == ("acc", 0.812)

    def test_pass_at_1_is_picked_up_for_generative_tasks(self):
        score = parse_task_metrics({"pass@1,create_test": 0.6}, "humaneval")
        assert (score.metric, score.value) == ("pass@1", 0.6)
        assert score.stderr is None

    def test_a_lone_unknown_metric_still_surfaces(self):
        """A new benchmark should appear in the table, not vanish."""
        score = parse_task_metrics({"bleu,none": 0.42}, "sometask")
        assert (score.metric, score.value) == ("bleu", 0.42)

    def test_ambiguous_unknown_metrics_are_skipped(self):
        assert parse_task_metrics({"bleu,none": 0.4, "rouge,none": 0.5}, "t") is None

    def test_non_numeric_and_boolean_values_are_ignored(self):
        """`higher_is_better` is a bool; bools are ints in Python."""
        score = parse_task_metrics(
            {"alias": "openbookqa", "higher_is_better": True, "acc,none": 0.3},
            "openbookqa",
        )
        assert (score.metric, score.value) == ("acc", 0.3)

    def test_empty_metric_block_yields_nothing(self):
        assert parse_task_metrics({}, "openbookqa") is None


class TestLoadEvalDir:
    def test_metrics_from_several_files_merge(self, tmp_path):
        d = write_eval(tmp_path / "m", "a", {"rte": {"acc,none": 0.8}})
        write_eval(d, "b", {"openbookqa": mc(0.3, 0.45)})
        assert set(load_eval_dir(d)) == {"rte", "openbookqa"}

    def test_later_file_overrides_an_earlier_rerun(self, tmp_path):
        """Re-running one task must win over the bundled value."""
        d = write_eval(tmp_path / "m", "a_bundle", {"rte": {"acc,none": 0.80}})
        write_eval(d, "z_rerun", {"rte": {"acc,none": 0.85}})
        assert load_eval_dir(d)["rte"].value == 0.85

    def test_directory_without_json_is_an_error(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            load_eval_dir(tmp_path / "empty")

    def test_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eval_dir(tmp_path / "nope")

    def test_malformed_json_is_reported_with_the_filename(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "x.json").write_text("{not json")
        with pytest.raises(MalformedEval, match="x.json"):
            load_eval_dir(d)

    def test_json_that_is_not_an_object_is_rejected(self, tmp_path):
        d = tmp_path / "arr"
        d.mkdir()
        (d / "x.json").write_text("[1, 2]")
        with pytest.raises(MalformedEval, match="expected a JSON object"):
            load_eval_dir(d)

    def test_non_dict_task_entries_are_skipped(self, tmp_path):
        d = write_eval(
            tmp_path / "m", "a", {"config": "junk", "rte": {"acc,none": 0.8}}
        )
        assert set(load_eval_dir(d)) == {"rte"}

    def test_json_with_no_usable_metric_is_an_error(self, tmp_path):
        d = write_eval(tmp_path / "m", "a", {"rte": {"alias": "rte"}})
        with pytest.raises(MalformedEval, match="no usable metrics"):
            load_eval_dir(d)


class TestSignificance:
    def test_combined_stderr_adds_in_quadrature(self):
        a = Score("t", "acc", 0.5, 0.03)
        b = Score("t", "acc", 0.4, 0.04)
        assert combined_stderr(a, b) == pytest.approx(0.05)

    def test_combined_stderr_is_none_when_either_side_lacks_one(self):
        a = Score("t", "acc", 0.5, 0.03)
        assert combined_stderr(a, Score("t", "acc", 0.4)) is None
        assert combined_stderr(Score("t", "acc", 0.5), a) is None

    def test_small_delta_inside_the_error_bars_is_noise(self):
        """RTE moving 0.812 -> 0.805 at ~2.3pp stderr proves nothing."""
        base = Score("rte", "acc", 0.812, 0.023)
        new = Score("rte", "acc", 0.805, 0.024)
        assert verdict(base, new) == "noise"

    def test_large_delta_outside_the_error_bars_is_significant(self):
        base = Score("openbookqa", "acc_norm", 0.448, 0.022)
        new = Score("openbookqa", "acc_norm", 0.314, 0.021)
        assert verdict(base, new) == "significant"

    def test_missing_stderr_refuses_to_judge(self):
        base = Score("humaneval", "pass@1", 0.6)
        assert verdict(base, Score("humaneval", "pass@1", 0.4)) == "no stderr"

    def test_zero_stderr_distinguishes_identical_from_moved(self):
        base = Score("t", "acc", 0.5, 0.0)
        assert verdict(base, Score("t", "acc", 0.5, 0.0)) == "same"
        assert verdict(base, Score("t", "acc", 0.6, 0.0)) == "significant"


class TestAtChance:
    def test_a_four_way_task_at_the_floor_is_flagged(self):
        assert Score("openbookqa", "acc_norm", 0.25, 0.0).at_chance

    def test_the_error_bar_counts_toward_the_floor(self):
        """0.28 with 2pp stderr is not distinguishable from guessing."""
        assert Score("openbookqa", "acc_norm", 0.28, 0.02).at_chance

    def test_a_healthy_score_is_not_flagged(self):
        assert not Score("openbookqa", "acc_norm", 0.448, 0.022).at_chance

    def test_binary_tasks_use_a_half_floor(self):
        assert Score("rte", "acc", 0.50, 0.0).at_chance
        assert not Score("rte", "acc", 0.80, 0.02).at_chance

    def test_generative_tasks_have_no_chance_level(self):
        assert not Score("humaneval", "pass@1", 0.0, 0.0).at_chance


class TestCompare:
    def test_regression_is_signed_and_relativized(self, baseline_dir, pruned_dir):
        table = compare(load_eval_dir(baseline_dir), load_eval_dir(pruned_dir))
        assert "-0.1340" in table  # 0.448 -> 0.314 acc_norm
        assert "-29.9%" in table

    def test_noise_and_significance_are_labelled_per_task(
        self, baseline_dir, pruned_dir
    ):
        table = compare(load_eval_dir(baseline_dir), load_eval_dir(pruned_dir))
        rows = {line.split("|")[1].strip(): line for line in table.splitlines()}
        assert "significant" in rows["openbookqa"]
        assert "noise" in rows["rte"]

    def test_a_task_without_stderr_is_not_judged(self, tmp_path):
        base = write_eval(tmp_path / "b", "x", {"humaneval": {"pass@1,none": 0.6}})
        new = write_eval(tmp_path / "p", "x", {"humaneval": {"pass@1,none": 0.4}})
        table = compare(load_eval_dir(base), load_eval_dir(new))
        assert "no stderr" in table

    def test_labels_appear_in_the_header(self, baseline_dir, pruned_dir):
        table = compare(
            load_eval_dir(baseline_dir), load_eval_dir(pruned_dir), "dense", "reap50"
        )
        assert "| dense |" in table and "| reap50 |" in table

    def test_task_missing_on_one_side_is_kept_and_marked(self, tmp_path):
        """Dropping the row would silently shrink the comparison."""
        base = write_eval(tmp_path / "b", "x", {"rte": {"acc,none": 0.8}})
        new = write_eval(
            tmp_path / "p", "x", {"rte": {"acc,none": 0.8}, "mmlu": {"acc,none": 0.7}}
        )
        table = compare(load_eval_dir(base), load_eval_dir(new))
        assert "mmlu" in table and "missing" in table

    def test_falling_to_chance_is_called_out(self, tmp_path):
        base = write_eval(tmp_path / "b", "x", {"openbookqa": mc(0.40, 0.448)})
        new = write_eval(tmp_path / "p", "x", {"openbookqa": mc(0.20, 0.250)})
        table = compare(load_eval_dir(base), load_eval_dir(new))
        assert "at chance" in table

    def test_zero_baseline_has_no_relative_delta(self, tmp_path):
        base = write_eval(tmp_path / "b", "x", {"humaneval": {"pass@1,none": 0.0}})
        new = write_eval(tmp_path / "p", "x", {"humaneval": {"pass@1,none": 0.1}})
        table = compare(load_eval_dir(base), load_eval_dir(new))
        assert "+0.1000" in table

    def test_differing_metrics_are_shown_as_a_pair(self, tmp_path):
        """Never compare acc against acc_norm without saying so."""
        base = write_eval(tmp_path / "b", "x", {"t": mc(0.3, 0.45)})
        new = write_eval(tmp_path / "p", "x", {"t": {"acc,none": 0.3}})
        table = compare(load_eval_dir(base), load_eval_dir(new))
        assert "acc_norm/acc" in table


class TestMeanDelta:
    def test_mean_is_over_shared_tasks_only(self, baseline_dir, pruned_dir):
        avg = mean_delta(load_eval_dir(baseline_dir), load_eval_dir(pruned_dir))
        # acc_norm -0.134, rte -0.007, humaneval -0.200
        assert avg == pytest.approx((-0.134 - 0.007 - 0.200) / 3, abs=1e-6)

    def test_no_shared_tasks_yields_none(self, tmp_path):
        base = write_eval(tmp_path / "b", "x", {"rte": {"acc,none": 0.8}})
        new = write_eval(tmp_path / "p", "x", {"mmlu": {"acc,none": 0.7}})
        assert mean_delta(load_eval_dir(base), load_eval_dir(new)) is None


class TestMain:
    def test_directories_are_compared_and_summarized(
        self, baseline_dir, pruned_dir, capsys
    ):
        assert main([str(baseline_dir), str(pruned_dir)]) == 0
        out = capsys.readouterr().out
        assert "openbookqa" in out
        assert "mean headline delta over 3 shared task(s)" in out

    def test_labels_resolve_under_the_eval_root(self, tmp_path, capsys):
        root = tmp_path / "evals"
        write_eval(root / "dense", "x", {"rte": {"acc,none": 0.8, "acc_stderr,none": 0.02}})
        write_eval(root / "reap", "x", {"rte": {"acc,none": 0.7, "acc_stderr,none": 0.02}})
        assert main(["dense", "reap", "--eval-root", str(root)]) == 0
        assert "rte" in capsys.readouterr().out

    def test_a_label_beats_a_same_named_directory_in_cwd(
        self, tmp_path, monkeypatch, capsys
    ):
        """`inco/reap/` is the REAP source clone, not an eval result dir."""
        root = tmp_path / "evals"
        for label in ("dense", "reap"):
            write_eval(root / label, "x", {"rte": {"acc,none": 0.8}})
        decoy = tmp_path / "cwd"
        (decoy / "reap").mkdir(parents=True)
        monkeypatch.chdir(decoy)
        assert main(["dense", "reap", "--eval-root", str(root)]) == 0
        assert "rte" in capsys.readouterr().out

    def test_missing_directory_exits_nonzero(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope"), str(tmp_path / "also")]) == 1
        assert "error:" in capsys.readouterr().err

    def test_malformed_input_exits_nonzero(self, tmp_path, baseline_dir, capsys):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "x.json").write_text("{")
        assert main([str(baseline_dir), str(bad)]) == 1
        assert "error:" in capsys.readouterr().err
