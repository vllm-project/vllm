import weakref
from typing import List, Optional, Tuple

import torch

from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker_base import LoraNotSupportedWorkerBase


class NGramWorker(LoraNotSupportedWorkerBase):
    """NGramWorker provides a light drafter without need for model.

    Current NGramWorker only implement prompt lookup decoding,
    and in future we may also do RAG type drafter and other scenerios
    which don't rely on LLM model to give proposals.
    """

    def __init__(self, *args, **kwargs):
        # Get local_rank/vocab_size from kwargs attribute
        self.local_rank = kwargs["local_rank"]
        self.vocab_size = kwargs["model_config"].get_vocab_size()

        # Lazy initialization list.
        self._proposer: Top1Proposer

    def set_ngram_window_size(self, ngram_prompt_lookup_min: int,
                              ngram_prompt_lookup_max: int):
        # Search valid candidate window between
        # ngram_prompt_lookup_min/ngram_prompt_lookup_max
        self.ngram_prompt_lookup_max = ngram_prompt_lookup_max
        self.ngram_prompt_lookup_min = ngram_prompt_lookup_min

    def init_device(self):
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.load_model = lambda *args, **kwargs: None

        # Current only support Top1Proposer
        self._proposer = Top1Proposer(
            weakref.proxy(self),
            device=self.device,
            vocab_size=self.vocab_size,
        )

    def set_include_gpu_probs_tensor(self):
        # NGram don't need gpu sampler
        pass

    def execute_model(self, execute_model_req: ExecuteModelRequest) -> None:
        """NGram doesn't depend on model execution, just pass this function"""
        pass

    def determine_num_available_blocks(self) -> None:
        """NGram doesn't depend on model execution, no need to check blocks"""
        pass

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """As there is no cache need to handle, just pass this function"""
        pass

    def get_cache_block_size_bytes(self):
        """Return the size of a cache block in bytes."""
        return 0

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[Optional[List[SamplerOutput]], bool]:
        """NGram match algo to pick proposal candidate. Returns the list of
        sampler output, one per SequenceGroupMetadata.

        For ngram worker, we already done needed transposed internal, so the
        indicator pass to sampler_output_to_torch shall be False.
        """
        self._raise_if_unsupported(execute_model_req)

        arr = []
        has_spec_out = False
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            seq_data = next(iter(seq_group_metadata.seq_data.values()))

            input_ids = torch.as_tensor(seq_data.get_token_ids(),
                                        dtype=torch.long,
                                        device=self.device)
            input_length = seq_data.get_len()

            for ngram_size in range(
                    min(self.ngram_prompt_lookup_max, input_length - 1),
                    self.ngram_prompt_lookup_min,
                    -1,
            ):
                ngram_tensor = input_ids[-1 * ngram_size:]
                windows = input_ids.unfold(dimension=0,
                                           size=ngram_size,
                                           step=1)
                matches = (windows == ngram_tensor).all(dim=1)
                match_indices = matches.nonzero(as_tuple=True)[0]
                if match_indices.size()[0] > 1:
                    has_spec_out = True
                    res = seq_data.get_token_ids()
                    res = res[match_indices[0] + ngram_size:match_indices[0] +
                              ngram_size + sample_len]
                    res_len = len(res)
                    # pad 0 towards output as sample_len tokens required
                    res += [0] * (sample_len - res_len)

                    break
            else:
                # if no candidate found, fill with 0
                res = [0] * sample_len

            arr.append(res)

        if not has_spec_out:
            return None, False

        outputs = []
        token_ids = torch.as_tensor(arr, dtype=torch.long, device=self.device)
        indices = token_ids.unsqueeze(2)

        token_probs = torch.zeros(
            (len(execute_model_req.seq_group_metadata_list), sample_len,
             self.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        token_probs.scatter_(2, indices, 1)
        token_logprobs = torch.zeros(
            (len(execute_model_req.seq_group_metadata_list), sample_len,
             self.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(len(execute_model_req.seq_group_metadata_list)):
            outputs.append(
                SamplerOutput(
                    outputs=None,
                    sampled_token_probs=token_probs[i],
                    logprobs=token_logprobs[i],
                    sampled_token_ids=token_ids[i],
                ))
        return outputs, False

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """

        return self._proposer.get_proposals(execute_model_req)

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """NGramWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "NGramWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "NGramWorker does not support beam search.")
