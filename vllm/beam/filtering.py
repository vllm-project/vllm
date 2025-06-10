import asyncio
from collections.abc import AsyncGenerator
from typing import Callable, Optional
from urllib.request import Request
from vllm.beam.debug import BeamDebugInfo
import torch
from vllm.entrypoints.openai.protocol import CompletionRequest


def format_filter(filter_params_list):
    keys = ["name", "threshold"]
    if any([len(p) != len(keys) for p in filter_params_list]):
        raise ValueError(f"Expect 2 keys, seeing {filter_params_list=}")
    return [dict(zip(keys, vals)) for vals in filter_params_list]

DEFAULT_CHAR_SERVER_FILTER = format_filter(
        [
            ("annotations_porn", 0.1098),
            ("annotations_racist", 0.2814),
            ("annotations_disturbing", 0.1827),
            ("annotations_harmful_promotes_selfharm", 0.0749),
            ("annotations_harmful_promotes_terrorism", 0.1129),
        ]
        )
    
MAX_GENERATIONS = 10
_CHUNK_SIZE = 16

class BeamValidator:
    def __init__(self, classi_idx, classifier_names):
        self.classi_idx = classi_idx
        self.classifier_names = classifier_names

    async def get_n_valid_beams(self, create_completion: Callable, request: CompletionRequest, raw_request: Optional[Request] = None):
        request.stream = False
        n = request.n
        request.n = 1
        request.max_tokens = _CHUNK_SIZE
        request.echo = True
        tasks = []
        for _ in range(n):
            request = request
            tasks.append(create_completion(
                request,
            ))
        res = await asyncio.gather(*tasks)
        request.n = n
        beam_validator_res = self.validate(res)
        filtered_res = [r for r, valid in zip(res, beam_validator_res) if valid]
        if len(filtered_res) == 0:
            return res
        
        return filtered_res
    
    def validate(self, responses: list[AsyncGenerator], debug_infos_G: list[BeamDebugInfo] = None):
        #TODO(@tanuj) - share this with the beam scorer
        heads = [response.choices[0].additional_heads[0] for response in responses]
        heads_tensor = torch.tensor(heads, dtype=torch.float)
        prob_GC = torch.sigmoid(heads_tensor)
        valid_G = torch.ones(prob_GC.shape[0], dtype=torch.bool)
        
        for g in range(heads_tensor.shape[0]):
            filtered = self.get_filtered_classifiers(prob_GC[g], DEFAULT_CHAR_SERVER_FILTER)
            if debug_infos_G is not None:
                debug_infos_G[g].filtered_classifiers = filtered
            
            if filtered:
                valid_G[g] = False
                for choice in responses[g].choices:
                    choice.is_filtered = True

        return valid_G
    
    def get_filtered_classifiers(self, prob_C, filter_params) -> list[str]:
        relevant_filters = [
            (p["name"], self.classi_idx[p["name"]], p["threshold"])
            for p in filter_params
            if p["name"] in self.classifier_names
        ]

        if not relevant_filters:
            return []

        ret = []
        for name, idx, threshold in relevant_filters:
            if prob_C[idx] > threshold:
                ret.append(name)

        return ret