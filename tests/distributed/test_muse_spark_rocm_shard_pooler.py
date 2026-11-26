# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check column order on replicated inputs and full projection on distinct rows."""

from unittest.mock import patch

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import ensure_model_parallel_initialized
from vllm.platforms import current_platform

from ..utils import multi_gpu_test
from .eplb_utils import distributed_run, set_env_vars_and_device


def projection_round_trip(env: dict[str, str], world_size: int) -> None:
    set_env_vars_and_device(env)
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.models.muse_spark.amd.model import TokenShard
    from vllm.models.muse_spark.amd.ops import skinny_gemm as projection

    config = VllmConfig()
    with set_current_vllm_config(config):
        ensure_model_parallel_initialized(world_size, 1)
    with (
        set_current_vllm_config(config),
        torch.device("cuda"),
        torch.inference_mode(),
    ):
        for columns, inner in ((512, 8192), (1024, 4096)):
            out_features = columns * world_size
            layer = ReplicatedLinear(
                inner, out_features, bias=False, params_dtype=torch.bfloat16
            )
            # Copy one input channel to each output. Every result is exactly
            # representable and rank blocks have distinct column order.
            layer.weight.zero_()
            idx = torch.arange(out_features)
            layer.weight[idx, (idx * 13) % inner] = 1
            x = (torch.arange(7 * inner).view(7, inner) % 127).bfloat16()
            shard = TokenShard(x.shape[0])
            local = shard.local(x)
            with (
                patch.object(
                    projection, "skinny_gemm", wraps=projection.skinny_gemm
                ) as gemm,
                patch.object(
                    projection,
                    "tensor_model_parallel_all_gather",
                    wraps=projection.tensor_model_parallel_all_gather,
                ) as gather,
            ):
                assert torch.equal(
                    projection.project_columns(layer, x, True, None), layer(x)[0]
                )
                gemm.assert_called_once()
                gather.assert_called_once()

                gemm.reset_mock()
                gather.reset_mock()
                assert torch.equal(
                    projection.project_columns(layer, local, True, shard),
                    layer(local)[0],
                )
                gemm.assert_not_called()
                gather.assert_not_called()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Muse Spark ROCm model path")
@multi_gpu_test(num_gpus=2)
def test_sharded_pooler_matches_replicated() -> None:
    distributed_run(projection_round_trip, 2)
