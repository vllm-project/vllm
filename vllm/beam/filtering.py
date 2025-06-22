import asyncio
from collections.abc import AsyncGenerator
from typing import Callable, Optional, Union
from urllib.request import Request
from vllm.beam.debug import BeamDebugInfo
import torch
from starlette.datastructures import MutableHeaders

from vllm.entrypoints.openai.protocol import CompletionRequest, CompletionResponse, \
    ErrorResponse
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


def format_filter(filter_params_list):
    keys = ["name", "threshold"]
    if any([len(p) != len(keys) for p in filter_params_list]):
        raise ValueError(f"Expect 2 keys, seeing {filter_params_list=}")
    return [dict(zip(keys, vals)) for vals in filter_params_list]


DEFAULT_CHAR_SERVER_FILTER = format_filter(
    [{"name": "annotations_porn", "threshold": 0.5156}, {"name": "annotations_racist", "threshold": 0.9763}, {"name": "annotations_disturbing", "threshold": 0.5472}, {"name": "annotations_harmful_promotes_selfharm", "threshold": 0.0657}]
)

MAX_GENERATIONS = 10
_CHUNK_SIZE = 16
_DEFAULT_BEAM_SIZE = 3

class BeamValidator:
    def __init__(self, classi_idx, classifier_names):
        self.classi_idx = classi_idx
        self.classifier_names = classifier_names

    async def get_n_valid_beams(self, create_completion: Callable,
                                request: CompletionRequest,
                                chunk_num: int,
                                raw_request: Optional[Request] = None) -> list[
        Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]]:
        request.stream = False
        n = request.n if request.n > 1 else _DEFAULT_BEAM_SIZE
        request.n = 1
        # TODO(@tanuj): accept max tokens as a parameter
        request.max_tokens = _CHUNK_SIZE
        request.echo = True
        original_request_id = None
        if raw_request is not None:
            original_request_id = raw_request.headers.get("X-Request-Id", None)
        
        tasks = []
        # TODO(@tanuj): deep copy request and raw_request?
        for _ in range(n):
            if original_request_id is not None:
                mh = MutableHeaders(scope=raw_request.scope)
                del mh["x-request-id"]
                if hasattr(raw_request, "_headers"):
                    delattr(raw_request, "_headers")

            tasks.append(create_completion(
                request,
                raw_request=raw_request,
            ))
        res = await asyncio.gather(*tasks)
        request.n = n
        beam_validator_res = self.validate(res)
        if isinstance(beam_validator_res, ErrorResponse):
            return beam_validator_res
        
        filtered_res = [r for r, valid in zip(res, beam_validator_res) if valid]
        logger.debug("Filtered count: %d", len(filtered_res))
        if len(filtered_res) == 0:
            return res

        return filtered_res

    def validate(self, responses: list[AsyncGenerator],
                 debug_infos_G: list[BeamDebugInfo] = None):
        error_responses = [r for r in responses if isinstance(r, ErrorResponse)]
        print(f"error_responses: {error_responses}")
        if len(error_responses) > 0:
            combined_message = "; ".join(er.message for er in error_responses)
            return ErrorResponse(
                message=combined_message,
                type=error_responses[0].type,
                code=error_responses[0].code
            )

        # TODO(@tanuj) - share this with the beam scorer
        heads = [response.choices[0].additional_heads[0] for response in responses]
        heads_tensor = torch.tensor(heads, dtype=torch.float)
        prob_GC = torch.sigmoid(heads_tensor)
        valid_G = torch.ones(prob_GC.shape[0], dtype=torch.bool)

        for g in range(heads_tensor.shape[0]):
            filtered = self.get_filtered_classifiers(prob_GC[g],
                                                     DEFAULT_CHAR_SERVER_FILTER)
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
