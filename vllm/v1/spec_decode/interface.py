# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Union

from vllm.config import SpeculativeConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle_proposer import EagleProposer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_input_batch import InputBatch


class ProposerInterface(ABC):
    """Abstract base class for speculative proposers."""

    @abstractmethod
    def generate_draft_token_ids(
            self, input_batch: InputBatch, sampled_token_ids: list[list[int]],
            sampling_metadata: SamplingMetadata) -> list[list[int]]:
        """Generates draft tokens using speculative proposal strategy.
        NOTE: This function will change the input_batch by writing 
        proposed tokens to token_ids_cpu.
        Args:
            input_batch: Contains input data and sequences metadata
            sampled_token_ids: Already sampled tokens from previous steps
            sampling_metadata: Additional sampling parameters and 
                            constraints
            
        Returns:
            List of draft token IDs for each sequence in
            the input batch.
        """
        raise NotImplementedError


def create_proposer(
    speculative_config: SpeculativeConfig
) -> Union[NgramProposer, EagleProposer]:
    """Factory function for creating proposer instances."""

    if speculative_config.type == "ngram":
        return NgramProposer(n=speculative_config.ngram_n,
                             k=speculative_config.ngram_k)

    elif speculative_config.type == "eagle":
        return EagleProposer()

    else:
        raise ValueError(
            f"Unsupported proposer type: {speculative_config.type}"
            "Valid types: 'ngram', 'eagle'")
