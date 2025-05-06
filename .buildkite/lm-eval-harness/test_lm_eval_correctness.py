# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import lm_eval
import numpy as np
import pytest
import yaml

RTOL = 0.08


def pytest_addoption(parser):
    parser.addoption(
        "--config-list-file",
        action="store",
        help="Path to the file listing model config YAMLs (one per line)")
    parser.addoption("--tp-size",
                     action="store",
                     default="1",
                     help="Tensor parallel size to use for evaluation")


@pytest.fixture(scope="session")
def config_dir():
    # Directory containing this script
    return Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def config_list_file(pytestconfig, config_dir):
    # Relative to the script's directory
    rel_path = pytestconfig.getoption("--config-list-file")
    return config_dir / rel_path


@pytest.fixture(scope="session")
def tp_size(pytestconfig):
    return pytestconfig.getoption("--tp-size")


def get_model_configs(config_list_file):
    """Read the list of model config YAML filenames."""
    with open(config_list_file, encoding="utf-8") as f:
        configs = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
    return configs


def launch_lm_eval(eval_config, tp_size):
    trust_remote_code = eval_config.get('trust_remote_code', False)
    model_args = f"pretrained={eval_config['model_name']}," \
                 f"tensor_parallel_size={tp_size}," \
                 f"enforce_eager=true," \
                 f"add_bos_token=true," \
                 f"trust_remote_code={trust_remote_code}"
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[task["name"] for task in eval_config["tasks"]],
        num_fewshot=eval_config["num_fewshot"],
        limit=eval_config["limit"],
        batch_size="auto")
    return results


def pytest_generate_tests(metafunc):
    if "config_filename" in metafunc.fixturenames:
        config_list_file = metafunc.config._store.get("config_list_file")
        if config_list_file is None:
            # fallback to fixture
            config_list_file = metafunc.config._conftest.getfixturevalue(
                "config_list_file")
        configs = get_model_configs(config_list_file)
        metafunc.parametrize("config_filename", configs)


def test_lm_eval_correctness_param(config_filename, config_dir, tp_size):
    config_path = config_dir / config_filename
    eval_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    results = launch_lm_eval(eval_config, tp_size)

    success = True
    for task in eval_config["tasks"]:
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            print(f'{task["name"]} | {metric["name"]}: '
                  f'ground_truth={ground_truth} | measured={measured_value}')
            success = success and np.isclose(
                ground_truth, measured_value, rtol=RTOL)

    assert success
