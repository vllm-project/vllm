import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict

import numpy
import pytest
import torch
import yaml

from tests.nm_utils.server import ServerContext
from tests.nm_utils.utils_skip import should_skip_test_group

if should_skip_test_group(group_name="TEST_ACCURACY"):
    pytest.skip("TEST_ACCURACY=DISABLE, skipping accuracy test group",
                allow_module_level=True)

if TYPE_CHECKING:
    import lm_eval as lm_eval_t

# requires a particular lm-evaluation-harness
# pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@9516087b81a61d0e220b22cc1b75be76de23bc10
lm_eval: "lm_eval_t" = pytest.importorskip("lm_eval",
                                           reason="lm_eval required")


class Metric(TypedDict):
    name: str
    value: float


class Task(TypedDict):
    name: str
    metrics: List[Metric]


# to support python3.8 typing prior to adding `Required`/`NotRequired`, this
# class stores the optional keys and the `EvalTaskDefinition` subclass inherits
# those alongside the required keys it defines.
class EvalTaskDefinitionOpts(TypedDict, total=False):
    enable_tensor_parallel: bool
    extra_args: Dict[str, Any]
    rtol: float


class EvalTaskDefinition(EvalTaskDefinitionOpts):
    model_name: str
    tasks: List[Task]


TEST_DATA_FILE = os.environ.get("LM_EVAL_TEST_DATA_FILE", None)
if TEST_DATA_FILE is None:
    raise ValueError("LM_EVAL_TEST_DATA_FILE env variable is not set.")
TEST_DATA_FILE = Path(TEST_DATA_FILE)

TEST_DATA: List[EvalTaskDefinition] = [
    pytest.param(eval_def, id=eval_def["model_name"])
    for eval_def in yaml.safe_load(TEST_DATA_FILE.read_text(encoding="utf-8"))
]
DEFAULT_RTOL = 0.05


@pytest.mark.parametrize("eval_data", TEST_DATA)
def test_lm_eval_correctness(
    eval_data: EvalTaskDefinition,
    logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    model_name = eval_data["model_name"]
    logger.info("building server startup args")
    vllm_args = {
        "--model": model_name,
        "--disable-log-requests": None,
        "--max-model-len": 4096,
    }

    if eval_data.get("enable_tensor_parallel") is True:
        tp = torch.cuda.device_count()
        logger.info("Enabling tensor parallelism with %d devices", tp)
        vllm_args["--tensor-parallel-size"] = tp

    if extra_args := eval_data.get("extra_args"):
        vllm_args.update(extra_args)

    openai_args = ",".join([
        f"model={model_name}",
        "tokenizer_backend=huggingface",
        "base_url=http://localhost:8000/v1",
    ])

    logger.info("launching server")
    with ServerContext(vllm_args, logger=logger) as _:
        task_names = [task["name"] for task in eval_data["tasks"]]
        limit = eval_data["limit"]
        new_fewshot = eval_data["num_fewshot"]
        logger.info("getting results for task_names=%s", task_names)
        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=openai_args,
            tasks=task_names,
            batch_size=32,
            num_fewshot=new_fewshot,
            limit=limit,
        )

    logger.info("clearing torch cache")
    lm_eval.models.utils.clear_torch_cache()

    rtol = eval_data.get("rtol", DEFAULT_RTOL)
    for task in eval_data["tasks"]:
        logger.info("checking metrics for task=%s", task["name"])
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            logger.info(
                "%s %s:\nground_truth=%s measured_value=%s",
                task["name"],
                metric["name"],
                ground_truth,
                measured_value,
            )

            assert numpy.isclose(ground_truth, measured_value, rtol=rtol)
