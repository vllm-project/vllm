from collections.abc import AsyncGenerator
from typing import Union

from vllm.beam.debug import BeamDebugInfo
from vllm.beam.penalty import PenaltyComputer
import torch
from vllm.beam.ranking import RankingComputer
from vllm.entrypoints.openai.protocol import CompletionResponse, ErrorResponse
from vllm.logger import init_logger

logger = init_logger(__name__)


class BeamScorer:
    def __init__(self, classi_idx):
        self.penalty_computer = PenaltyComputer(classi_idx)
        self.ranking_computer = RankingComputer(classi_idx)

    async def pick_best_beam(self, responses: list[
        Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]]) -> Union[
        AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
        debug_info = [BeamDebugInfo() for _ in responses]

        scores = torch.zeros(len(responses), dtype=torch.float)

        heads = [response.choices[0].additional_heads[0] for response in responses]
        heads_tensor = torch.tensor(heads, dtype=torch.float)
        if len(heads_tensor) > 0:
            penalties = self.penalty_computer.compute(heads_tensor, debug_info)
            scores -= penalties

            ranking_scores = self.ranking_computer.compute(
                heads_tensor, debug_info
            )
            scores += ranking_scores

        for i in range(len(responses)):
            debug_info[i].final_score = scores[i]
            debug_info[i].content = responses[i].choices[0].text

        logger.debug('debug_info: %s', debug_info)

        best_idx = torch.argmax(scores).item()
        return responses[best_idx]
