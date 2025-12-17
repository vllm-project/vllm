# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import tempfile

import numpy as np
import pytest
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.tpu import TPUModelLoader


def _setup_environment(model):
    engine_args = EngineArgs(
        model=model,
    )
    vllm_config = engine_args.create_engine_config()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        # Under single worker mode, full model is init first and then
        # partitioned using GSPMD.
        ensure_model_parallel_initialized(1, 1)
    return vllm_config


MESH = None


def _get_spmd_mesh():
    global MESH
    if MESH is None:
        xr.use_spmd()
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (num_devices, 1)
        device_ids = np.array(range(num_devices))
        MESH = xs.Mesh(device_ids, mesh_shape, ("x", "y"))
    return MESH


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen2-1.5B-Instruct",
        # Skip large models due to CI runner disk space limitations
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
    ],
)
def test_tpu_model_loader(model):
    # Skip the 70B test if there are less than 8 chips
    # TODO: Query using torch xla API, the query API is not working
    # with SPMD now. However, This test is running under SPMD mode.
    if "70B" in model and xr.global_runtime_device_count() < 8:
        pytest.skip(
            "Skipping 70B model if the TPU VM has less than 8 chips to \
                     avoid OOM."
        )

    vllm_config = _setup_environment(model)
    loader = TPUModelLoader(load_config=vllm_config.load_config)
    mesh = _get_spmd_mesh()
    model = loader.load_model(vllm_config, vllm_config.model_config, mesh)
    del model
    gc.collect()
