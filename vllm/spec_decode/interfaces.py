from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set

import torch

from vllm.sequence import ExecuteModelRequest


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

    # A flag to mark that there's no available proposals
    no_proposals: bool = False

    def __repr__(self):
        return (f"SpeculativeProposals("
                f"proposal_token_ids={self.proposal_token_ids}, "
                f"proposal_probs={self.proposal_probs.shape}, "
                f"proposal_lens={self.proposal_lens})")


@dataclass
class SpeculativeScores:
    """Datastructure used to represent the scores of speculative tokens
    according to the scoring model.
    """

    # Probabilities of the speculative tokens according to the scoring model.
    probs: torch.Tensor

    # Log-probabilities of the speculative tokens according to the scoring
    # model. These values can be used to generate Logprob objects that are
    # returned to the user.
    logprobs: torch.Tensor

    # Token ids sampled from the scoring model. Used for speculative bonus
    # tokens and also non-speculative normal decoding.
    token_ids: torch.Tensor

    # Optional last hidden states from the scoring model.
    hidden_states: Optional[torch.Tensor] = None

    def __repr__(self):
        return (f"SpeculativeScores("
                f"probs={self.probs.shape}, "
                f"token_ids={self.token_ids.shape})")


class SpeculativeProposer(ABC):

    @abstractmethod
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        # If set, this contains all sequence IDs that were assigned
        # bonus tokens in their last forward pass.
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        raise NotImplementedError


class SpeculativeScorer(ABC):

    @abstractmethod
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        raise NotImplementedError
