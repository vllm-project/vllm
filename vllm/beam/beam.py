from collections.abc import AsyncGenerator
from vllm.beam.debug import BeamDebugInfo
from vllm.beam.penalty import PenaltyComputer
import torch
from vllm.beam.ranking import RankingComputer


class BeamScorer:
    def __init__(self, classi_idx):
        self.penalty_computer = PenaltyComputer(classi_idx)
        self.ranking_computer = RankingComputer(classi_idx)

    async def collapse_beams(self, responses: list[AsyncGenerator], chunk_num = 0, max_chunks = 4):
            debug_info = [BeamDebugInfo() for _ in responses]
            
            scores = torch.zeros(len(responses), dtype=torch.float)
 
            has_additional_heads = torch.tensor([response.choices[0].additional_heads is not None for response in responses], dtype=torch.bool)
            if has_additional_heads.any():
                heads = [response.choices[0].additional_heads[0] for response in responses]
                heads_tensor = torch.tensor(heads, dtype=torch.float)
                penalties = self.penalty_computer.compute(heads_tensor, debug_info)
                scores -= penalties

                ranking_scores = self.ranking_computer.compute(
                heads_tensor, debug_info
               )
                scores *= ranking_scores

            for i in range(len(responses)):
                debug_info[i].final_score = scores[i]
                debug_info[i].content = responses[i].choices[0].text

            print('debug_info', debug_info)

            best_idx = torch.argmax(scores).item()
            return responses[best_idx]
    