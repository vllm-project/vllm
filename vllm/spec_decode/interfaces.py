from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from vllm.sequence import SequenceGroupMetadata


@dataclass
class SpeculativeProposals:
    """Datastructure used to represent proposal tokens from some proposer. It
    also tracks how many speculative tokens each sequence has.
    """

    # Speculative proposal tokens.
    proposal_token_ids: torch.Tensor

    # Probabilities of the proposal tokens according to the proposer.
    proposal_probs: torch.Tensor

    # The valid length of each proposal; can be zero.
    proposal_lens: torch.Tensor

    def __repr__(self):
        return (f"SpeculativeProposals("
                f"proposal_token_ids={self.proposal_token_ids.shape}, "
                f"proposal_probs={self.proposal_probs.shape}, "
                f"proposal_lens={self.proposal_lens.shape})")


@dataclass
class SpeculativeScores:
    """Datastructure used to represent the scores of speculative tokens
    according to the scoring model.
    """

    # Probabilities of the speculative tokens according to the scoring model.
    probs: torch.Tensor

    # Token ids sampled from the scoring model. Used for speculative bonus
    # tokens and also non-speculative normal decoding.
    token_ids: torch.Tensor

    def __repr__(self):
        return (f"SpeculativeScores("
                f"probs={self.probs.shape}, "
                f"token_ids={self.token_ids.shape})")


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
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Optional[Dict[int, int]],
        blocks_to_swap_out: Optional[Dict[int, int]],
        blocks_to_copy: Optional[Dict[int, List[int]]],
        k: int,
        proposals: SpeculativeProposals,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
