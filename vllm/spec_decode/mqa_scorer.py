# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.sequence import (ExecuteModelRequest, SequenceData,
                           SequenceGroupMetadata, get_all_seq_ids)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)

SeqId = int
TargetSeqId = int


class MQAScorer(SpeculativeScorer):

    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        target_seq_group_metadata_list = []
        target_seq_id_start = max(
            get_all_seq_ids(execute_model_req.seq_group_metadata_list)) + 1
        all_proposal_tokens = proposals.proposal_token_ids.tolist()
        all_proposal_lengths = proposals.proposal_lens.tolist()
        for i, seq_group_metadata in enumerate(
                execute_model_req.seq_group_metadata_list):
            if all_proposal_lengths[i] == 0:
                # Keep prompt seqs untouched (keep computed_tokens for chunks).
                target_seq_group_metadata_list.append(seq_group_metadata)
                continue

            seq_data_dict = seq_group_metadata.seq_data
            assert len(seq_data_dict) == 1
            seq_id = next(iter(seq_data_dict.keys()))

            seq_data: SequenceData = seq_data_dict[seq_id]
            prompt_token_ids = seq_data.get_prompt_token_ids()
            output_token_ids = seq_data.get_output_token_ids()
            proposal_token_ids = all_proposal_tokens[
                i][:all_proposal_lengths[i]]
            new_output_token_ids = [*output_token_ids, *proposal_token_ids]

            target_seq_id = target_seq_id_start + i
            new_seq_data = SequenceData.from_seqs(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=new_output_token_ids,
            )
            new_seq_data.update_num_computed_tokens(
                len(prompt_token_ids) + len(output_token_ids) - 1)

            # Ensure that the new decode sequence has at least one token.
            assert len(output_token_ids) >= 1
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
            )
            target_seq_group_metadata_list.append(new_seq_group_metadata)

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req.clone(
                seq_group_metadata_list=target_seq_group_metadata_list))

        target_sampler_output = target_sampler_output[0]

        k = execute_model_req.num_lookahead_slots
        bs = len(execute_model_req.seq_group_metadata_list)
        target_token_ids = target_sampler_output.sampled_token_ids
        target_probs = target_sampler_output.sampled_token_probs
        target_logprobs = target_sampler_output.logprobs
        prompt_logprobs = None

        # If all requests have the same number of query tokens, we can avoid
        # the for loop to build output for better performance.
        if min(all_proposal_lengths) == k:
            # Regular decodes only.
            assert all(not sg.is_prompt
                       for sg in target_seq_group_metadata_list
                       if sg.is_prompt)
            bs, _ = proposals.proposal_token_ids.shape
            all_tokens = target_token_ids.reshape(bs, k + 1)
            all_probs = target_probs.reshape(bs, k + 1, self._vocab_size)
            all_logprobs = target_logprobs.reshape(bs, k + 1, self._vocab_size)
        else:
            # We either have decodes with different lens or prefill+decodes.
            all_tokens = target_token_ids.new_full(size=(bs, k + 1),
                                                   fill_value=-1)
            all_probs = target_probs.new_zeros(*all_tokens.shape,
                                               self._vocab_size)
            all_logprobs = target_logprobs.new_full(size=all_probs.shape,
                                                    fill_value=-float("inf"))
            target_token_ids = target_token_ids.flatten()

            # When prompt logprobs is enabled, lens of returned tensors go from
            # n_sampled (requests with do_sample=True) to n_prompt+n_prefills.
            # We adjust stride accordingly to get the generated tokens and
            # their probs, but pass on prompt_logprobs as is, since it may be
            # that n_prompts >> K.
            has_prompt_log = any((sg.sampling_params.prompt_logprobs
                                  and sg.sampling_params.prompt_logprobs > 0)
                                 for sg in target_seq_group_metadata_list)
            # TODO (NickLucche) we should surface `disable_logprobs` as to not
            # break abstraction to get its value.
            if (not self._scorer_worker.model_runner.disable_logprobs\
                and has_prompt_log):
                prompt_logprobs = [
                    o.prompt_logprobs for o in target_sampler_output.outputs
                ]

            # Split loop into prefill|decode for readability.
            start_loc, i = 0, 0
            while i < len(target_seq_group_metadata_list
                          ) and target_seq_group_metadata_list[i].is_prompt:
                seq_meta = target_seq_group_metadata_list[i]
                end_loc = start_loc
                if has_prompt_log:
                    end_loc += seq_meta.token_chunk_size
                elif seq_meta.do_sample:
                    end_loc += 1

                # Skip chunks with no output tokens.
                if seq_meta.do_sample:
                    # Get sampled token (last position in chunk) and its prob.
                    all_tokens[i, 0] = target_token_ids[end_loc - 1]
                    all_probs[i, 0] = target_probs[end_loc - 1]
                    all_logprobs[i, 0] = target_logprobs[end_loc - 1]

                i += 1
                start_loc = end_loc
            # Decodes.
            while i < len(target_seq_group_metadata_list):
                proposed_len, seq_meta = all_proposal_lengths[
                    i], target_seq_group_metadata_list[i]
                output_len = proposed_len + 1
                end_loc = start_loc + output_len
                all_tokens[
                    i, :output_len] = target_token_ids[start_loc:end_loc]
                all_probs[i, :output_len] = target_probs[start_loc:end_loc]
                all_logprobs[
                    i, :output_len] = target_logprobs[start_loc:end_loc]
                start_loc = end_loc
                i += 1

        hidden_states = None
        if target_sampler_output.hidden_states is not None:
            hidden_states = target_sampler_output.hidden_states.reshape(
                bs, (k + 1), -1)

        return SpeculativeScores(probs=all_probs,
                                 token_ids=all_tokens,
                                 logprobs=all_logprobs,
                                 hidden_states=hidden_states,
                                 prompt_logprobs=prompt_logprobs)
