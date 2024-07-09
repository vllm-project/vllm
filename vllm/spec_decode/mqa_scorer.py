from itertools import count
from typing import Iterator, List

from vllm.sequence import (ExecuteModelRequest, SequenceData,
                           SequenceGroupMetadata, get_all_seq_ids)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
from vllm.spec_decode.util import nvtx_range
from vllm.worker.worker_base import WorkerBase

SeqId = int
TargetSeqId = int


class MQAScorer(SpeculativeScorer):

    def __init__(self, scorer_worker: WorkerBase, device: str,
                 vocab_size: int):
        self._scorer_worker = scorer_worker
        self._device = device
        self._vocab_size = vocab_size

    @nvtx_range("FlashinferScorer.score_proposals")
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        target_seq_group_metadata_list = []
        target_seq_ids_iter = self._create_target_seq_id_iterator(
            seq_ids=get_all_seq_ids(execute_model_req.seq_group_metadata_list))
        for i, seq_group_metadata in enumerate(
                execute_model_req.seq_group_metadata_list):
            seq_data_dict = seq_group_metadata.seq_data
            seq_id = next(iter(seq_data_dict.keys()))

            seq_data: SequenceData = seq_data_dict[seq_id]
            prompt_token_ids = seq_data.get_prompt_token_ids()
            output_token_ids = seq_data.get_output_token_ids()
            proposal_token_ids = proposals.proposal_token_ids.tolist()[i]
            # print("propoese token ids", proposal_token_ids)
            new_output_token_ids = [*output_token_ids, *proposal_token_ids]

            target_seq_id = next(target_seq_ids_iter)
            new_seq_data = SequenceData(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=new_output_token_ids,
            )
            assert len(output_token_ids) - 1 >= 0
            new_seq_data.update_num_computed_tokens(
                len(prompt_token_ids) + len(output_token_ids) - 1)
            new_seq_data_dict = {target_seq_id: new_seq_data}

            new_seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group_metadata.request_id,
                is_prompt=seq_group_metadata.is_prompt,
                seq_data=new_seq_data_dict,
                sampling_params=seq_group_metadata.sampling_params,
                block_tables={
                    target_seq_id: seq_group_metadata.block_tables[seq_id],
                },
                lora_request=None,
                token_chunk_size=1,
            )
            target_seq_group_metadata_list.append(new_seq_group_metadata)

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req.clone(
                seq_group_metadata_list=target_seq_group_metadata_list))

        target_sampler_output = target_sampler_output[0]

        bs, k = proposals.proposal_token_ids.shape
        all_tokens = target_sampler_output.sampled_token_ids.reshape(bs, k + 1)

        all_probs = target_sampler_output.sampled_token_probs.reshape(
            bs, k + 1, self._vocab_size)
        all_logprobs = target_sampler_output.logprobs.reshape(
            bs, k + 1, self._vocab_size)

        return SpeculativeScores(probs=all_probs,
                                 token_ids=all_tokens,
                                 logprobs=all_logprobs)

    def _create_target_seq_id_iterator(
            self, seq_ids: List[SeqId]) -> Iterator[TargetSeqId]:
        """Create an iterator for creating target sequence ids.
        Target sequence ids are distinct from sequence ids because we create a
        distinct target sequence id for each proposal token to be scored.

        This implementation increments a counter starting at 1 + max of all
        provided input sequence ids.
        """
        return count(start=max(seq_ids) + 1)
