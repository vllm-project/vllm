from typing import Dict, List

import torch


def logit_bias_logits_processor(logit_bias: Dict[str,
                                                 float], token_ids: List[int],
                                logits: torch.Tensor) -> torch.Tensor:
    for token_id, bias in logit_bias.items():
        logits[token_id] += bias
    return logits
