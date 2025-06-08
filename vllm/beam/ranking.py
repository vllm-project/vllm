from vllm.beam.debug import BeamDebugInfo
from vllm.beam.utils import filter_missing_classis
import torch

MODEL_SERVER_RANKING = [
        {"name": "chosen_after_swipe_crowd_preference", "pow": 0.5, "negation": False},
        {"name": "chosen_after_swipe_preference", "pow": 0.5, "negation": False},
]


class RankingComputer:
    def __init__(self, classi_idx):
        self.classi_idx = classi_idx

        ranking_params = filter_missing_classis(MODEL_SERVER_RANKING, classi_idx, warn=True)
        self.ranking_params = ranking_params
        pnames = [p["name"] for p in ranking_params]
        self.classi_indices = [self.classi_idx[p["name"]] for p in ranking_params]

        if not self.classi_indices:
            print(f"No ranking classifiers {pnames} found. Candidates will not be ranked.")

        self.dtype = torch.float32

        def _tensor(k, dtype):
            data = [p[k] for p in ranking_params]
            return torch.tensor(data, dtype=dtype, device="cpu")

        self.pow_R = _tensor("pow", torch.float32)          
        self.negation_R = _tensor("negation", torch.bool)

    def compute(self, logit_GC, debug_infos_G: list[BeamDebugInfo] = None):
        if not self.classi_indices:
            return torch.zeros_like(logit_GC[:, 0])

        probs_GC = torch.sigmoid(logit_GC[:, self.classi_indices])
        ranking_GC = torch.where(self.negation_R, 1.0 - probs_GC, probs_GC).pow(self.pow_R)
        return ranking_GC.prod(dim=-1)                  
