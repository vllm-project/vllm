# SPDX-License-Identifier: Apache-2.0
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch


class EagleProposer:

    def generate_draft_token_ids(
            self, input_batch: InputBatch, sampled_token_ids: list[list[int]],
            sampling_metadata: SamplingMetadata) -> list[list[int]]:
        raise NotImplementedError("This method is not implemented yet.")
