# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared config builders for EC connector unit tests."""

import uuid
from unittest.mock import Mock

import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.config.ec_transfer import ECRole, ECTransferConfig
from vllm.config.parallel import ParallelConfig

# The scheduler derives its per-load ack count from the PCP size only under the
# `mp` backend -- which also keeps the world size off the local GPU count, so
# TP > 1 stays expressible on a single-GPU host.
_EXECUTOR_BACKEND = "mp"
_PCP_SIZE = 1


def create_ec_vllm_config(
    *,
    ec_role: ECRole = "ec_both",
    tensor_parallel_size: int = 1,
    rank: int = 0,
    dtype: torch.dtype = torch.float16,
) -> Mock:
    """Build a `VllmConfig` stand-in for EC connector unit tests.

    `ec_transfer_config` and `parallel_config` are the real config objects, so
    role derivation and the rank arithmetic the connector depends on behave as
    they do in production.

    `model_config` is a stub because nothing under test needs a real one: the
    connector reads only `dtype` here, and the paths that inspect the model
    itself are not exercised. Building a real `ModelConfig` would resolve an HF
    model and inspect its architecture.

    Args:
        ec_role: EC role this instance plays. `is_ec_producer` and
            `is_ec_consumer` are derived from it, as in production.
        tensor_parallel_size: TP degree.
        rank: Global rank of this worker.
        dtype: Encoder cache dtype.

    Returns:
        A `VllmConfig`-specced mock carrying the configs above.
    """
    parallel_config = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=_PCP_SIZE,
        distributed_executor_backend=_EXECUTOR_BACKEND,
        rank=rank,
    )
    ec_transfer_config = ECTransferConfig(
        # The role properties report False unless a connector is named.
        ec_connector="ECCPUConnector",
        ec_role=ec_role,
        engine_id=str(uuid.uuid4()),
    )

    model_config = Mock(spec=ModelConfig)
    model_config.dtype = dtype

    vllm_config = Mock(spec=VllmConfig)
    vllm_config.ec_transfer_config = ec_transfer_config
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = model_config
    vllm_config.instance_id = f"ec-test-{uuid.uuid4()}"
    return vllm_config
