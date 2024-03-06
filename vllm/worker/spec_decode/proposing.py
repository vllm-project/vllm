from typing import Iterator, List, Tuple, Optional, Union, Dict
from itertools import chain, count
from functools import cached_property
import logging
import time
from dataclasses import dataclass

#import msgspec
import torch
import traceback

from vllm.worker.spec_decode.metrics import SpecDecodeWorkerMetrics, AsyncMetricsCollector
#from vllm.anyscale.shm.msgspec_shm import SharedMsgspecBufferWithEvent
#from vllm.sequence import (SampleLogprob, SamplerOutput, SequenceGroupMetadata,
#                           ExecuteModelData, SequenceOutputs, SequenceData,
#                           SequenceGroupOutputs, SpecDecodeWorkerMetrics)
from vllm.sequence import (SamplerOutput, SequenceGroupMetadata, SequenceData,
                           SequenceGroupOutput, SequenceOutput)
from vllm.worker.worker import Worker
#from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
#from vllm.worker.prompt_lookup_worker import PromptLookupWorker
#from vllm.worker.single_tp_worker import SingleTpWorker
#from vllm.model_executor.layers.sampler import sampler_output_to_torch
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group
from vllm.config import CacheConfig
#from vllm.worker.base_worker import BaseWorker
#from vllm.model_executor.layers.sampler import RawSamplerOutput
from vllm.utils import in_wsl
from vllm.worker.spec_decode.util import nvtx_range, sampler_output_to_torch, SpeculativeProposals, get_all_seq_ids

SeqId = int
TargetSeqId = int
TokenId = int

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod


class SpeculativeProposer(ABC):
    @abstractmethod
    def get_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        max_proposal_len: int,
    ) -> SpeculativeProposals:
        raise NotImplementedError

class DraftModelTop1Proposer(SpeculativeProposer):
    def __init__(
        self,
        draft_worker: "MultiStepWorker",
        device: str,
        max_model_len: int,
        vocab_size: int,
    ):
        self._draft_worker = draft_worker
        self._device = device
        self._max_model_len = max_model_len
        self._vocab_size = vocab_size

    def get_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        max_proposal_len: int,
    ) -> SpeculativeProposals:
        """
        - create proposal lens tensor
            - determine which seqs are over len
            - set k = 0
        - create new batch that ignores k=0
        - do normal fwd pass
        - construct output
            - inject empty rows for token ids/probs tensor
        """
        
        max_model_len = self._max_model_len

        proposal_lens: List[int] = []
        nonzero_proposal_len_seqs: List[SequenceGroupMetadata] = []
        nonzero_proposal_len_indices: List[int] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            seq_len = seq_data.get_len()

            if seq_len + max_proposal_len < max_model_len:
                proposal_lens.append(max_proposal_len)
                nonzero_proposal_len_seqs.append(seq_group_metadata)
                nonzero_proposal_len_indices.append(i)
            else:
                proposal_lens.append(0)
        
        # run fwd pass

        if nonzero_proposal_len_seqs:
            sampler_output = self._draft_worker.execute_model_multi_step(
                seq_group_metadata_list=nonzero_proposal_len_seqs,
                blocks_to_swap_in=blocks_to_swap_in,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                num_steps=max_proposal_len,
            )
            proposal_tokens, proposal_probs = sampler_output_to_torch(sampler_output)

            # Now, reformat the output GPU tensors such that each sequence has
            # a proposal. the proposal can be empty, e.g. [-1, -1, -1]
            
            batch_size = len(seq_group_metadata_list)

            entire_proposal_tokens = torch.ones(batch_size, *proposal_tokens.shape[1:], dtype=torch.long, device=self._device) * -1
            entire_proposal_tokens[nonzero_proposal_len_indices] = proposal_tokens
            entire_proposal_probs = torch.zeros(batch_size, *proposal_probs.shape[1:], dtype=torch.float32, device=self._device)
            entire_proposal_probs[nonzero_proposal_len_indices] = proposal_probs

            proposal_tokens, proposal_probs = entire_proposal_tokens, entire_proposal_probs
            
            proposal_lens = torch.zeros(batch_size, dtype=torch.long, device=self._device)
            proposal_lens[nonzero_proposal_len_indices] = max_proposal_len
        else:
            proposal_tokens = torch.zeros(0, max_proposal_len, dtype=torch.long, device=self._device)
            proposal_probs = torch.zeros(0, max_proposal_len, self._vocab_size, dtype=torch.float32, device=self._device)
            proposal_lens = torch.zeros(len(proposal_lens), dtype=torch.long, device=self._device)

        proposals = SpeculativeProposals(
            proposal_token_ids=proposal_tokens,
            proposal_probs=proposal_probs,
            proposal_lens=proposal_lens,
        )

        return proposals
