# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC connector helper utilities."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase

logger = init_logger(__name__)


class ECOutputAggregator:
    """Utility class to aggregate the EC connector output of all workers
    into a single output corresponding to `output_rank` for the scheduler.

    Mirrors KVOutputAggregator's role for KV connectors: only one worker's
    ModelRunnerOutput (`output_rank`, e.g. the last pipeline-parallel rank)
    reaches the scheduler, but the EC connector's real work may happen on a
    different rank (e.g. the first PP rank, where the multimodal encoder
    runs) -- this merges that rank's ec_connector_output onto the selected
    output before it's returned.
    """

    def __init__(self, world_size: int):
        self._world_size = world_size

    @classmethod
    def from_connector(cls, connector: "ECConnectorBase", world_size: int):
        return cls(world_size)

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        if not outputs[output_rank]:
            return None

        finished_sending = set[str]()
        finished_recving = set[str]()
        aggregated_ec_connector_worker_meta = None
        for model_runner_output in outputs:
            assert model_runner_output is not None
            ec_output = model_runner_output.ec_connector_output
            if not ec_output:
                continue

            finished_sending |= ec_output.finished_sending or set()
            finished_recving |= ec_output.finished_recving or set()

            # Aggregate ec_connector_worker_meta from all workers.
            if aggregated_ec_connector_worker_meta is None:
                # Use the first worker's ec_connector_worker_meta as accumulator.
                aggregated_ec_connector_worker_meta = ec_output.ec_connector_worker_meta
            elif ec_connector_worker_meta := ec_output.ec_connector_worker_meta:
                aggregated_ec_connector_worker_meta = (
                    aggregated_ec_connector_worker_meta.aggregate(
                        ec_connector_worker_meta
                    )
                )

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        assert output is not None
        output.ec_connector_output = ECConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            ec_connector_worker_meta=aggregated_ec_connector_worker_meta,
        )
        return output
