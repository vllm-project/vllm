# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EC connector helper utilities."""

from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput


class ECOutputAggregator:
    """Merge the EC connector output of all workers into the one output that
    reaches the scheduler.

    Mirrors KVOutputAggregator's role for KV connectors: only one worker's
    ModelRunnerOutput (`output_rank`, e.g. the last pipeline-parallel rank)
    reaches the scheduler, while the EC connector runs on the first pipeline
    rank, where the multimodal encoder lives.

    Finished sending/recving ids are unioned, so a connector that reports an
    id from more than one worker must tolerate the scheduler acting on the
    first report.
    """

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        output = outputs[output_rank]
        if not output:
            return None

        finished_sending = set[str]()
        finished_recving = set[str]()
        worker_meta = None
        for model_runner_output in outputs:
            assert model_runner_output is not None
            ec_output = model_runner_output.ec_connector_output
            if not ec_output:
                continue

            finished_sending |= ec_output.finished_sending or set()
            finished_recving |= ec_output.finished_recving or set()

            if worker_meta is None:
                worker_meta = ec_output.ec_connector_worker_meta
            elif other := ec_output.ec_connector_worker_meta:
                worker_meta = worker_meta.aggregate(other)

        aggregated = ECConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            ec_connector_worker_meta=worker_meta,
        )
        output.ec_connector_output = None if aggregated.is_empty() else aggregated
        return output
