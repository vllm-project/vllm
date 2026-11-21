"""Run every predictor through the same small synthetic batch fixture.

AISimulate uses the existing B300/SOL test configuration. Synthetic latencies
only support contract tests; they are not accuracy references.
"""

import gzip
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from vllm_simulator.simulation.types import SchedulerConfig
from vllm_simulator.spec import AcceleratorInfo
from vllm_simulator.time_predictor.aisimulate import AISimulateTimePredictor
from vllm_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from vllm_simulator.time_predictor.linear import LinearTimePredictor
from vllm_simulator.time_predictor.ml import MLTimePredictor
from vllm_simulator.time_predictor.replay import ReplayTimePredictor

PREDICTORS = ("aisimulate", "linear", "ml", "replay")


@pytest.fixture(scope="module", autouse=True)
def single_threaded_predictors():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def batch_rows():
    # Synthetic rows matching the collector's rank*.schedule_batch.jsonl exports.
    # Real batch data can be obtained by running the collector.
    return [
        {
            "requests": [["a", 64, 0, 0]],
            "iter_latency": 0.012,
            "sample_tokens_latency": 0.001,
        },
        {
            "requests": [["a", 16, 64, 0], ["b", 1, 128, 1]],
            "iter_latency": 0.008,
            "sample_tokens_latency": 0.0008,
        },
        {
            "requests": [["a", 1, 128, 1], ["b", 1, 64, 1]],
            "iter_latency": 0.002,
            "sample_tokens_latency": 0.0002,
        },
        {
            "requests": [["b", 1, 64, 2], ["a", 1, 128, 2]],
            "iter_latency": 0.004,
            "sample_tokens_latency": 0.0004,
        },
    ]


@pytest.fixture(scope="module")
def schedule_batch_path(batch_rows, tmp_path_factory):
    path = tmp_path_factory.mktemp("batches") / "schedule_batch.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in batch_rows))
    return str(path)


@pytest.fixture(scope="module")
def batches(batch_rows):
    return [
        (ScheduleBatch([ScheduleRequest(e, p) for _, e, p, _ in row["requests"]]), row)
        for row in batch_rows
    ]


@pytest.fixture(scope="module")
def predictors(tmp_path_factory, schedule_batch_path):
    scheduler = SchedulerConfig(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        backend_name="vllm",
        backend_version="0.24.0",
    )
    hw = AcceleratorInfo(name="b300_sxm", hbm_capacity_gb=80000)
    model = SimpleNamespace(model="Qwen/Qwen3-8B")
    directory = tmp_path_factory.mktemp("predictors")
    model_path = str(directory / "model.pkl")
    table_path = str(directory / "table.json")
    bundle = MLTimePredictor.build_model_from_schedule_batch(
        schedule_batch_path, model_path
    )
    assert bundle["features"] == MLTimePredictor.FEATURE_NAMES
    assert bundle["prefill"] is not bundle["decode"]
    ReplayTimePredictor.build_table_from_schedule_batch(schedule_batch_path, table_path)
    return [
        AISimulateTimePredictor(model, hw, scheduler, database_mode="SOL"),
        LinearTimePredictor(model, hw, scheduler, base_ms=1, per_token_ms=0.01),
        MLTimePredictor(model, hw, scheduler, model_path),
        ReplayTimePredictor(model, hw, scheduler, table_path, miss_strategy="knn"),
    ]


@pytest.fixture(scope="module", params=range(len(PREDICTORS)), ids=PREDICTORS)
def predictor(request, predictors):
    return predictors[request.param]


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_predictors_on_batches(predictor, batches, phase, record_property):
    """All backends must produce finite nonnegative RPC-1/RPC-2 seconds."""
    samples = [
        batch for batch, _ in batches if batch.is_decode() == (phase == "decode")
    ]
    assert samples
    predictions = []
    for batch in samples:
        result = predictor.predict_step_time(batch)
        assert isinstance(result, tuple) and len(result) == 2
        assert all(math.isfinite(value) and value >= 0 for value in result)
        assert result[0] > 0
        predictions.append(result)
    record_property("batches", len(samples))
    record_property(
        "mean_forward_ms", float(np.mean([p[0] for p in predictions]) * 1000)
    )
    record_property(
        "mean_sample_ms", float(np.mean([p[1] for p in predictions]) * 1000)
    )


@pytest.mark.parametrize("phase", ("prefill", "decode"))
def test_ml_training_merges_plain_and_gzip_with_missing_phase(tmp_path, batches, phase):
    """A one-phase export must supply a fallback model for the missing phase."""
    samples = [
        row for batch, row in batches if batch.is_decode() == (phase == "decode")
    ]
    paths = [tmp_path / "first.jsonl", tmp_path / "second.jsonl.gz"]
    for path, opener, row in zip(paths, (open, gzip.open), samples):
        with opener(path, "wt") as output:
            output.write("\n" + json.dumps({"requests": []}) + "\n")
            output.write(json.dumps(row) + "\n")
    bundle = MLTimePredictor.build_model_from_schedule_batch([str(p) for p in paths])
    assert bundle["prefill"] is bundle["decode"]
