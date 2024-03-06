from typing import Iterator, List, Tuple, Optional, Union, Dict
from itertools import chain, count
from functools import cached_property
import logging
import time
from dataclasses import dataclass

import torch
import traceback

from vllm.worker.spec_decode.metrics import SpecDecodeWorkerMetrics, AsyncMetricsCollector
from vllm.sequence import (SamplerOutput, SequenceGroupMetadata, SequenceData,
                           SequenceGroupOutput, SequenceOutput)
from vllm.worker.worker import Worker
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group
from vllm.config import CacheConfig
from vllm.utils import in_wsl
from vllm.worker.spec_decode.util import nvtx_range, sampler_output_to_torch, SpeculativeProposals, get_all_seq_ids


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


class SpeculativeScorer(ABC):
    @abstractmethod
    def score_proposals(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata], blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        k: int,
        proposals: SpeculativeProposals,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
