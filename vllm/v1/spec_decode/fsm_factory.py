# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for creating FSM-specific components."""

from vllm.config import VllmConfig
from vllm.config.model import LogprobsMode
from vllm.custom_fsm import CustomFSM


class FSMComponentFactory:
    """Factory for creating FSM-specific sampler, proposer, and rejection sampler."""

    @staticmethod
    def create_sampler(fsm_path: str, logprobs_mode: LogprobsMode):
        """Create FSM sampler."""
        from vllm.v1.sample.fsm_sampler import FSMSampler

        return FSMSampler(fsm_path=fsm_path, logprobs_mode=logprobs_mode)

    @staticmethod
    def create_proposer(vllm_config: VllmConfig):
        """Create FSM proposer."""
        from vllm.v1.spec_decode.fsm_proposer import FSMProposer

        assert vllm_config.speculative_config is not None
        fsm_path = vllm_config.speculative_config.fsm_path
        assert fsm_path is not None, "fsm_path must be set for fsm method"
        fsm = CustomFSM.from_prebuilt(fsm_path)
        return FSMProposer(vllm_config, fsm)

    @staticmethod
    def create_rejection_sampler(sampler):
        """Create FSM rejection sampler."""
        from vllm.v1.sample.fsm_rejection_sampler import FSMRejectionSampler

        return FSMRejectionSampler(sampler)

    @staticmethod
    def needs_output_token_ids(speculative_config) -> bool:
        """Check if FSM needs output token IDs."""
        return (
            speculative_config
            and getattr(speculative_config, "fsm_path", None) is not None
        )
