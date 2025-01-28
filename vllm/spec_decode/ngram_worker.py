import weakref
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.proposer_worker_base import NonLLMProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer

if current_platform.is_cuda_alike():
    DEVICE_TYPE = "cuda"
elif current_platform.is_neuron():
    DEVICE_TYPE = "neuron"
elif current_platform.is_hpu():
    DEVICE_TYPE = "hpu"
elif current_platform.is_openvino():
    DEVICE_TYPE = "openvino"
elif current_platform.is_cpu():
    DEVICE_TYPE = "cpu"
elif current_platform.is_tpu():
    DEVICE_TYPE = "tpu"
elif current_platform.is_xpu():
    DEVICE_TYPE = "xpu"
else:
    raise ValueError(f"Unsupported platform: {current_platform}")


class _DummyModel(nn.Module):
    pass


class NGramWorker(NonLLMProposerWorkerBase):
    """NGramWorker provides a light drafter without need for model.

    Current NGramWorker only implements prompt lookup decoding,
    and in future we may also do RAG type drafter and other scenarios
    which don't rely on LLM model to give proposals.
    """

    def __init__(self, *args, **kwargs):
        # Get local_rank/vocab_size from kwargs attribute
        self.local_rank = kwargs["local_rank"]
        self.vocab_size = kwargs["vllm_config"].model_config.get_vocab_size()
        self.device_type = kwargs.get("device_type", "cuda")

        # Lazy initialization list.
        self._proposer: Top1Proposer

    def set_ngram_window_size(self, ngram_prompt_lookup_min: int,
                              ngram_prompt_lookup_max: int):
        # Search valid candidate window between
        # ngram_prompt_lookup_min/ngram_prompt_lookup_max
        self.ngram_prompt_lookup_max = ngram_prompt_lookup_max
        self.ngram_prompt_lookup_min = ngram_prompt_lookup_min

    def init_device(self):
        self.device = torch.device(f"{self.device_type}:{self.local_rank}")

        # Current NGramWorker only supports Top1Proposer
        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            device=self.device,
            vocab_size=self.vocab_size,
        )

    def load_model(self) -> None:
        pass  # Dummy

    def get_model(self) -> nn.Module:
        return _DummyModel()

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        # Unused parameter. NGramWorker does not use the KV Cache and
        # therefore does not need this parameter.
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[Optional[List[Optional[SamplerOutput]]], bool]:
        """NGram match algo to pick proposal candidate. Returns the list of
        sampler output, one per SequenceGroupMetadata.

        For ngram worker, we already done needed transposed internal, so the
        indicator pass to sampler_output_to_torch shall be False.
        """
        self._raise_if_unsupported(execute_model_req)

        has_spec_out = False
        token_id_list: List[Optional[torch.Tensor]] = []
        token_prob_list: List[Optional[torch.Tensor]] = []
        for idx, seq_group_metadata in enumerate(
                execute_model_req.seq_group_metadata_list):
            seq_data = next(iter(seq_group_metadata.seq_data.values()))

            seq_len = seq_data.get_len()
            # When seq_len is less than 3072 (3K), we use CPU to perform
            # the ngram match. Otherwise, we use the device specified in
            # the model config (normally GPU). 3072 is a rough threshold
            # based on profiling on H100, and it can be adjusted based
            # on the actual performance on different hardware.
            cur_device = "cpu" if seq_len < 3072 else self.device
            input_ids = torch.as_tensor(seq_data.get_token_ids(),
                                        dtype=torch.long,
                                        device=cur_device)
            input_length = seq_data.get_len()

            for ngram_size in range(
                    min(self.ngram_prompt_lookup_max, input_length - 1),
                    self.ngram_prompt_lookup_min - 1,
                    -1,
            ):
                ngram_tensor = input_ids[-ngram_size:]
                if ngram_size == 1:
                    # Do not match itself and do not use unfold and all
                    matches = (input_ids[:-1] == ngram_tensor)
                else:
                    windows = input_ids.unfold(dimension=0,
                                               size=ngram_size,
                                               step=1)
                    # Do not match itself
                    matches = (windows[:-1] == ngram_tensor).all(dim=-1)

                # first_match includes "values" (bool), indicating whether
                # the match is found, and "indices", indicating the index
                # of the first match.
                first_match = matches.max(dim=-1)
                if first_match.values.item():
                    proposal_start_idx = first_match.indices.add_(ngram_size)
                    spec_indices = (
                        proposal_start_idx).repeat(sample_len) + torch.arange(
                            sample_len, device=cur_device)
                    spec_indices.clamp_(max=input_ids.shape[-1] - 1)
                    res = input_ids.gather(dim=-1,
                                           index=spec_indices).to(self.device)
                    token_id_list.append(res)
                    token_prob_list.append(
                        torch.nn.functional.one_hot(
                            res,
                            num_classes=self.vocab_size).to(torch.float32))
                    has_spec_out = True
                    break
            else:
                token_id_list.append(None)
                token_prob_list.append(None)

        if not has_spec_out:
            return None, False

        outputs: List[Optional[SamplerOutput]] = []
        for idx in range(len(execute_model_req.seq_group_metadata_list)):
            if token_id_list[idx] is None:
                outputs.append(None)
            else:
                outputs.append(
                    SamplerOutput(
                        outputs=None,
                        sampled_token_probs=token_prob_list[idx],
                        logprobs=torch.zeros((sample_len, self.vocab_size),
                                             dtype=torch.float32,
                                             device=self.device),
                        sampled_token_ids=token_id_list[idx],
                    ))

        return outputs, False

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        # Unused parameter. NGramWorker does not use the KV Cache and
        # therefore does not need this parameter.
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
