from dataclasses import dataclass
from vllm.sequence import SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores


@dataclass
class WorkerResponse:
    sampler_outputs: list[SamplerOutput]


class SpeculativeWorkerResponse(WorkerResponse):
    speculative_proposals: SpeculativeProposals
    speculative_proposal_scores: SpeculativeScores
