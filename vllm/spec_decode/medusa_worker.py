import weakref
from typing import List, Optional, Set, Tuple

import torch

from vllm.model_executor import SamplingMetadata
from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.proposer_worker_base import NonLLMProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker import Worker


class MedusaWorker(NonLLMProposerWorkerBase, Worker):
    """Worker for Medusa.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lazy initialization list.
        self._proposer: Top1Proposer

    def init_device(self):
        super().init_device()

        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )

    def set_include_gpu_probs_tensor(self):
        pass

    def set_should_modify_greedy_probs_inplace(self):
        pass

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        # Unused parameter.
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass to generate sample_len future tokens.
        Returns the list of sampler output, one per layer, along with indicator
        of whether torch tensor in sampler output need to be transposed in
        latter sampler_output_to_torch logic.

        For medusa worker, this indicator shall be False.
        """
        self._raise_if_unsupported(execute_model_req)

        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        seq_lens, query_lens = self._prepare_input_tensors(
            seq_group_metadata_list)

        generators = self.model_runner.get_generators(
            execute_model_req.finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list, seq_lens, query_lens, self.device,
            self.model_runner.pin_memory, generators)

        model_outputs = self.model_runner.model.generate_proposals(
            previous_hidden_states=execute_model_req.previous_hidden_states.
            hidden_states,
            sampling_metadata=sampling_metadata)

        return model_outputs, False

    def _prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[List[int], List[int]]:
        if not seq_group_metadata_list:
            return [], []

        seq_lens: List[int] = []
        query_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            is_prompt = seq_group_metadata.is_prompt

            for seq_data in seq_group_metadata.seq_data.values():
                seq_data_len = seq_data.get_len()
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                    seq_len = min(
                        seq_data_len,
                        context_len + seq_group_metadata.token_chunk_size)
                    seq_lens.append(seq_len)
                    query_lens.append(seq_len - context_len)
                else:
                    seq_lens.append(seq_data_len)
                    query_lens.append(1)

        return seq_lens, query_lens

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """

        return self._proposer.get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """MedusaWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "MedusaWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "MedusaWorker does not support beam search.")
