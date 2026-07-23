# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, ModelRunnerOutput


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def test_detach_zero_copy_from_model_runner_output_copies_only_numpy_views():
    cu_num_generated_tokens = [0, 2]
    prompt_logprobs = LogprobsTensors.empty_cpu(1, 2)
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        logprobs=LogprobsLists(
            logprob_token_ids=_make_readonly(
                np.array([[1, 2], [3, 4]], dtype=np.int32)
            ),
            logprobs=_make_readonly(
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            ),
            sampled_token_ranks=_make_readonly(np.array([1, 2], dtype=np.int32)),
            cu_num_generated_tokens=cu_num_generated_tokens,
        ),
        prompt_logprobs_dict={"req-0": prompt_logprobs},
    )

    original_logprobs = output.logprobs
    assert original_logprobs is not None

    detach_zero_copy_from_model_runner_output(output)

    detached_logprobs = output.logprobs
    assert detached_logprobs is not None
    assert detached_logprobs is not original_logprobs
    assert (
        detached_logprobs.logprob_token_ids is not original_logprobs.logprob_token_ids
    )
    assert detached_logprobs.logprobs is not original_logprobs.logprobs
    assert (
        detached_logprobs.sampled_token_ranks
        is not original_logprobs.sampled_token_ranks
    )
    assert detached_logprobs.logprob_token_ids.flags.writeable
    assert detached_logprobs.logprobs.flags.writeable
    assert detached_logprobs.sampled_token_ranks.flags.writeable
    assert detached_logprobs.cu_num_generated_tokens is cu_num_generated_tokens
    assert output.prompt_logprobs_dict["req-0"] is prompt_logprobs


def test_pg_strategy_env_var_default(monkeypatch):
    """VLLM_RAY_PG_STRATEGY defaults to PACK."""
    monkeypatch.delenv("VLLM_RAY_PG_STRATEGY", raising=False)
    # Access the raw dictionary entry rather than the cached attribute:
    # __getattr__ caches the first lookup so monkeypatch cannot override
    # `envs.VLLM_RAY_PG_STRATEGY` after the module has been imported.
    from vllm.envs import environment_variables

    assert environment_variables["VLLM_RAY_PG_STRATEGY"]() == "PACK"


@pytest.mark.parametrize(
    "strategy",
    ["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"],
)
def test_pg_strategy_env_var_all_valid_choices(monkeypatch, strategy):
    """All four Ray PG strategies are accepted without error."""
    monkeypatch.setenv("VLLM_RAY_PG_STRATEGY", strategy)
    from vllm.envs import environment_variables

    assert environment_variables["VLLM_RAY_PG_STRATEGY"]() == strategy


def test_pg_strategy_env_var_invalid_raises(monkeypatch):
    """Invalid PG strategy raises ValueError via env_with_choices."""
    monkeypatch.setenv("VLLM_RAY_PG_STRATEGY", "INVALID")
    # See test_pg_strategy_env_var_default for why we access the dict
    # directly rather than using the cached module attribute.
    from vllm.envs import environment_variables

    validator = environment_variables["VLLM_RAY_PG_STRATEGY"]
    with pytest.raises(ValueError, match="VLLM_RAY_PG_STRATEGY"):
        validator()


def test_initialize_ray_cluster_passes_strategy_to_ray(monkeypatch):
    """initialize_ray_cluster passes VLLM_RAY_PG_STRATEGY to
    ray.util.placement_group(strategy=...)."""
    monkeypatch.setenv("VLLM_RAY_PG_STRATEGY", "SPREAD")

    mock_ray = MagicMock()
    mock_ray.is_initialized.return_value = True
    mock_ray.util.get_current_placement_group.return_value = None
    mock_ray.cluster_resources.return_value = {"CPU": 8}
    mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node1"

    from vllm.platforms import current_platform
    from vllm.v1.executor import ray_utils

    # Platform detection is incomplete when vllm is not installed as a
    # package (editable install).  Provide minimal attributes so
    # initialize_ray_cluster can determine the device string.
    monkeypatch.setattr(current_platform, "device_name", "cpu")
    monkeypatch.setattr(current_platform, "ray_device_key", "CPU")

    # Ray is not installed here, so `available_resources_per_node` was
    # never set by the module-level import block. Assign it directly.
    # Save the original (possibly None) to restore after the test so
    # other tests in the same process are not affected.
    _saved_available_resources_per_node = getattr(
        ray_utils, "available_resources_per_node", None
    )
    ray_utils.available_resources_per_node = lambda: {"node1": {"CPU": 8}}
    try:
        with (
            patch.object(ray_utils, "ray", mock_ray),
            patch.object(ray_utils, "get_ip", return_value="1.2.3.4"),
            patch.object(ray_utils, "_wait_until_pg_ready"),
            patch.object(ray_utils, "_verify_bundles"),
        ):
            parallel_config = MagicMock()
            parallel_config.world_size = 4
            parallel_config.placement_group = None
            parallel_config.ray_runtime_env = None

            ray_utils.initialize_ray_cluster(
                parallel_config, require_gpu_on_driver=False
            )

            mock_ray.util.placement_group.assert_called_once()
            _, kwargs = mock_ray.util.placement_group.call_args
            assert kwargs["strategy"] == "SPREAD"
    finally:
        if _saved_available_resources_per_node is not None:
            ray_utils.available_resources_per_node = _saved_available_resources_per_node
        else:
            del ray_utils.available_resources_per_node
