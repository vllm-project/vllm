from typing import List, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.util import nvtx_range
from vllm.worker.worker_base import WorkerBase

SeqId = int
TargetSeqId = int
TokenId = int


class MultiQueryTop1Scorer(SpeculativeScorer):
    """Implements a speculative scorer that uses batch expansion to get
    probabilities of speculative tokens according to the scoring model.
    Batch expansion converts a list of sequences and multiple query positions
    to a new batch of sequences, each with a single query position. This allows
    for MQA-like scoring in speculative decoding without requiring an MQA
    kernel.
    It is strictly less efficient than MQA scoring.
    It only supports scoring the top1 proposal tokens of the proposer, instead
    of topk/tree.
    """

    def __init__(self, scorer_worker: WorkerBase, device: str,
                 vocab_size: int):
        self._scorer_worker = scorer_worker
        self._device = device
        self._vocab_size = vocab_size

    @nvtx_range("BatchExpansionTop1Scorer.score_proposals")
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        """Score the proposed tokens via the scorer model.
        This converts each input sequence to a set of k+1 target sequences. The
        target sequences have the unique continuations to be scored and a
        unique sequence ID that is different from all input sequence ids.
        If a speculative sequence length would exceed the max model length, then
        no speculation is produced for that sequence.
        Args:
            execute_model_req: The execution request.
            proposals: The speculative proposals to score.
        Returns:
            SpeculativeScores: The scores of each speculative token, along with
                which sequences were ignored during scoring.
        """

        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_token_ids_list = proposals.proposal_token_ids.tolist()

        for seq_group_metadata, proposal_token_ids in zip(
                execute_model_req.seq_group_metadata_list,
                proposal_token_ids_list,
        ):
            seq_id, seq_data = next(iter(seq_group_metadata.seq_data.items()))
            if proposal_token_ids:
                seq_data.update_num_computed_tokens(
                    (seq_data.get_len() - 1) -
                    seq_data.get_num_computed_tokens())
                for token in proposal_token_ids:
                    seq_data._output_token_ids.append(token)
                    seq_data._cached_all_token_ids.append(token)
                # use the prompt mode for multi-query sampling
                seq_group_metadata.token_chunk_size += len(proposal_token_ids)

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req.clone(
                seq_group_metadata_list=execute_model_req.
                seq_group_metadata_list))
        assert len(target_sampler_output) == 1, "expected single-step output"
        target_sampler_output = target_sampler_output[0]

        for seq_group_metadata, proposal_token_ids in zip(
                execute_model_req.seq_group_metadata_list,
                proposal_token_ids_list,
        ):
            seq_id, seq_data = next(iter(seq_group_metadata.seq_data.items()))
            if proposal_token_ids:
                for token in proposal_token_ids:
                    seq_data._output_token_ids.pop()
                    seq_data._cached_all_token_ids.pop()
                seq_group_metadata.token_chunk_size -= len(proposal_token_ids)

        all_tokens, all_probs, spec_logprobs = self._contract_batch(
            contracted_bs=len(execute_model_req.seq_group_metadata_list),
            target_sampler_output=target_sampler_output,
            proposals=proposals,
            k=execute_model_req.num_lookahead_slots,
        )

        return SpeculativeScores(
            probs=all_probs,
            token_ids=all_tokens,
            logprobs=spec_logprobs,
            hidden_states=target_sampler_output.hidden_states,
        )

    def _contract_batch(
            self, contracted_bs: int, target_sampler_output: SamplerOutput,
            proposals: SpeculativeProposals,
            k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contract the expanded batch back into its original size.
        This maps the scores of speculative tokens back to their original
        sequences.
        contracted_bs is the original batch size, and the batch size that the
        target_sampler_output will be contracted to.
        """
        target_token_ids = target_sampler_output.sampled_token_ids
        target_probs = target_sampler_output.sampled_token_probs
        target_logprobs = target_sampler_output.logprobs

        all_tokens = target_token_ids.new_full(size=(contracted_bs, k + 1),
                                               fill_value=-1)
        all_probs = target_probs.new_zeros(*all_tokens.shape, self._vocab_size)
        all_logprobs = target_logprobs.new_full(size=all_probs.shape,
                                                fill_value=-float("inf"))

        seq_indices: List[int] = []
        rank_indices: List[int] = []
        for i, output in enumerate(target_sampler_output.outputs):
            num_tokens = len(output.samples)
            seq_indices.extend([i] * num_tokens)
            rank_indices.extend(range(num_tokens))

        seq_indices = torch.tensor(seq_indices, device=self._device)
        rank_indices = torch.tensor(rank_indices, device=self._device)
        all_tokens[seq_indices, rank_indices] = target_token_ids.flatten()
        all_probs[seq_indices,
                  rank_indices] = target_probs.view(-1, self._vocab_size)
        all_logprobs[seq_indices, rank_indices] = target_logprobs.view(
            -1, self._vocab_size)

        return all_tokens, all_probs, all_logprobs
