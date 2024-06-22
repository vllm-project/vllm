from typing import List, Optional, Tuple

import torch

from vllm.model_executor import SamplingMetadata
from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.proposer_worker_base import NonLLMProposerWorkerBase


class MLPSpeculatorWorker(NonLLMProposerWorkerBase, MultiStepWorker):
    """Worker for MLPSpeculator models.

    Not currently compatible with LoRA or chunked prefill.
    """

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass to generate sample_len future tokens.
        Returns the list of sampler output, one per layer, along with indicator
        of whether torch tensor in sampler output need to be transposed in
        latter sampler_output_to_torch logic.

        For mlp spec worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)

        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        (input_tokens, seq_lens,
         query_lens) = self._prepare_input_tensors(seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list, seq_lens, query_lens, self.device,
            self.model_runner.pin_memory)

        model_outputs = self.model_runner.model.generate_proposals(
            input_ids=input_tokens,
            previous_hidden_states=execute_model_req.previous_hidden_states.
            hidden_states,
            num_predict_tokens=sample_len,
            sampling_metadata=sampling_metadata)

        assert len(model_outputs) == sample_len

        return model_outputs, True

    def _prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        if not seq_group_metadata_list:
            return torch.empty(0, device=self.device), [], []

        input_tokens: List[int] = []
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
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                    seq_lens.append(seq_len)
                    input_tokens.extend(tokens)
                    query_lens.append(seq_len - context_len)
                else:
                    seq_lens.append(seq_data_len)
                    input_tokens.append(seq_data.get_last_token_id())
                    query_lens.append(1)

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        return input_tokens_tensor, seq_lens, query_lens
