# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 exit-gate integration tests exercising all three plugins end-to-end."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from tests.byog.contract_helpers import assert_plugin_contract
from tests.byog.fake_launcher import FakeLauncher
from tools.byog.bundle import validate_bundle
from tools.byog.catalog import Catalog
from tools.byog.executor.coordinator import Coordinator
from tools.byog.executor.interfaces import FixtureAssetProvider
from tools.byog.schemas.catalog import EvidenceSource
from tools.byog.schemas.result import ExecutionStatus, QualityState
from tools.byog.schemas.spec import RunSpec
from tools.byog.spec import SpecResolutionError, load_spec_file, resolve_single_job

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


def _gsm8k_assets(tmp_path: Path) -> FixtureAssetProvider:
    return FixtureAssetProvider(
        tmp_path,
        overrides={
            "gsm8k_train": FIXTURES_DIR / "gsm8k_mini_train.jsonl",
            "gsm8k_test": FIXTURES_DIR / "gsm8k_mini_test.jsonl",
        },
    )


def test_all_three_plugins_satisfy_strict_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-never-leak-to-direct")
    cat = Catalog.discover()

    # 1. HTTP benchmark smoke
    res_http, _ = assert_plugin_contract(
        workload_name="http_smoke",
        tmp_path=tmp_path / "http",
        catalog=cat,
        workload_overrides={"concurrency": [1, 2], "num_requests": 4},
    )
    assert len(res_http.measurements) == 2
    assert res_http.quality_state == QualityState.NOT_EVALUATED

    # 2. Direct benchmark smoke (no server started, candidate evidence, no HF_TOKEN)
    direct_launcher = FakeLauncher()
    res_direct, _ = assert_plugin_contract(
        workload_name="direct_smoke",
        tmp_path=tmp_path / "direct",
        catalog=cat,
        launcher=direct_launcher,
    )
    assert direct_launcher.starts == 0
    assert res_direct.measurements[0].evidence_source == EvidenceSource.CANDIDATE
    assert res_direct.measurements[0].metrics["hf_token_leaked"] == 0

    # 3. GSM8K eval with threshold -> PASSED
    res_gsm, _ = assert_plugin_contract(
        workload_name="gsm8k",
        tmp_path=tmp_path / "gsm8k",
        catalog=cat,
        assets=_gsm8k_assets(tmp_path),
        workload_overrides={"num_questions": 6, "concurrency": [2]},
        min_accuracy=0.8,
    )
    assert res_gsm.measurements[0].metrics["accuracy"] == 1.0
    assert res_gsm.quality_state == QualityState.PASSED


def test_gsm8k_unthresholded_failing_and_infra_error_semantics(
    tmp_path: Path,
) -> None:
    cat = Catalog.discover()
    assets = _gsm8k_assets(tmp_path)

    # 1. Spec cannot lower catalog threshold (0.3) with min_accuracy=0.0.
    with pytest.raises(SpecResolutionError, match="cannot lower catalog threshold"):
        assert_plugin_contract(
            workload_name="gsm8k",
            tmp_path=tmp_path / "lower",
            catalog=cat,
            assets=assets,
            workload_overrides={"num_questions": 4},
            min_accuracy=0.0,
        )

    # 1b. Unthresholded eval -> INCONCLUSIVE even if spec sets min_accuracy=0.0.
    unth_w = cat.get_workload("gsm8k").model_copy(
        update={"quality_thresholds": [], "required_for_quality": False}
    )
    unth_cat = Catalog(workloads={"gsm8k": unth_w}, suites={})
    res_un, _ = assert_plugin_contract(
        workload_name="gsm8k",
        tmp_path=tmp_path / "unth",
        catalog=unth_cat,
        assets=assets,
        workload_overrides={"num_questions": 4},
        min_accuracy=0.0,
    )
    assert res_un.quality_state == QualityState.INCONCLUSIVE

    # 2. Wrong answers with min_accuracy=0.5 -> FAILED quality_state.
    res_bad, _ = assert_plugin_contract(
        workload_name="gsm8k",
        tmp_path=tmp_path / "bad",
        catalog=cat,
        assets=assets,
        launcher=FakeLauncher(wrong_answer_mode=True),
        workload_overrides={"num_questions": 4},
        min_accuracy=0.5,
    )
    assert res_bad.quality_state == QualityState.FAILED

    # 3. Transient HTTP 500 on call #2 -> infra_errors=1, score=None on errored row,
    # invalid_rate=0.0, and quality_state=INCONCLUSIVE.
    res_err, tar_err = assert_plugin_contract(
        workload_name="gsm8k",
        tmp_path=tmp_path / "err",
        catalog=cat,
        assets=assets,
        launcher=FakeLauncher(fail_on_calls={2}),
        workload_overrides={"num_questions": 4, "concurrency": [1]},
        min_accuracy=0.5,
    )
    m = res_err.measurements[0]
    assert m.infra_errors == 1
    assert m.attempted == 4 and m.completed == 3
    assert m.metrics["invalid_rate"] == 0.0
    assert res_err.quality_state == QualityState.INCONCLUSIVE

    with tarfile.open(tar_err, "r:gz") as tar:
        blob = tar.extractfile("evals/gsm8k/responses.jsonl").read().decode("utf-8")  # type: ignore[union-attr]
        rows = [json.loads(ln) for ln in blob.splitlines() if ln.strip()]
        errored = [r for r in rows if r["error_kind"] == "infra"]
        assert len(errored) == 1
        assert errored[0]["score"] is None
        assert errored[0]["correct"] is None


def test_combined_gpu_smoke_spec_end_to_end_on_fake_launcher(tmp_path: Path) -> None:
    cat = Catalog.discover()
    spec_path = FIXTURES_DIR / "smoke_gpu_spec.yaml"
    spec = load_spec_file(spec_path)
    job = resolve_single_job(spec, cat, attempt_id="att-smoke-e2e")

    launcher = FakeLauncher(strict_model=True)
    coord = Coordinator(
        launcher=launcher,
        assets=_gsm8k_assets(tmp_path),
        image_ref="vllm/byog:test",
        vllm_commit=VALID_SHA,
    )
    res, archive = coord.run(job, tmp_path / "e2e", source_spec_path=spec_path)
    assert res.execution_status == ExecutionStatus.COMPLETED
    assert res.quality_state == QualityState.PASSED
    assert launcher.stops == launcher.starts
    # All 3 workloads present with distinct per-measurement compatibility_keys.
    workloads_seen = {m.workload for m in res.measurements}
    assert workloads_seen == {"http_smoke", "direct_smoke", "gsm8k"}
    keys = [m.compatibility_key for m in res.measurements]
    assert len(set(keys)) == len(keys)

    manifest, validated = validate_bundle(archive)
    assert manifest.job_id == "default"
    assert validated.quality_state == QualityState.PASSED

    # Replayable spec.resolved.yaml inside bundle parses cleanly as RunSpec.
    replayed = load_spec_file(archive.parent / "spec.resolved.yaml")
    assert isinstance(replayed, RunSpec)
