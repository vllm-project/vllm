# SPDX-License-Identifier: Apache-2.0

import enum
import json
from typing import Union

import torch

from vllm.config import VllmConfig
from vllm.engine.metrics import Stats
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest
from vllm.version import __version__ as VLLM_VERSION
from vllm.worker.worker_base import ModelExecutionV0Error

logger = init_logger(__name__)


def prepare_object_to_dump(obj) -> str:
    if isinstance(obj, str):
        return "'{obj}'"  # Double quotes
    elif isinstance(obj, dict):
        dict_str = ', '.join({f'{str(k)}: {prepare_object_to_dump(v)}' \
            for k, v in obj.items()})
        return f'{{{dict_str}}}'
    elif isinstance(obj, list):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, set):
        return f"[{', '.join([prepare_object_to_dump(v) for v in list(obj)])}]"
        # return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    elif isinstance(obj, torch.Tensor):
        # We only print the 'draft' of the tensor to not expose sensitive data
        # and to get some metadata in case of CUDA runtime crashed
        return (f"Tensor(shape={obj.shape}, "
                f"device={obj.device},"
                f"dtype={obj.dtype})")
    elif hasattr(obj, 'anon_repr'):
        return obj.anon_repr()
    elif hasattr(obj, '__dict__'):
        items = obj.__dict__.items()
        dict_str = ','.join([f'{str(k)}={prepare_object_to_dump(v)}' \
            for k, v in items])
        return (f"{type(obj).__name__}({dict_str})")
    else:
        # Hacky way to make sure we can serialize the object in JSON format
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError):
            return repr(obj)


def dump_engine_exception(err: BaseException, config: VllmConfig):

    logger.error("Dumping input data")

    logger.error(
        "V1 LLM engine (v%s) with config: %s, ",
        VLLM_VERSION,
        config,
    )

    # TODO: Have stats for V1

    from vllm.v1.engine.core import ModelExecutionError
    if isinstance(err, ModelExecutionError):
        try:
            dump_obj = prepare_object_to_dump(err.scheduler_output)
            logger.error("Dumping scheduler output for model execution:")
            logger.error(dump_obj)
        except BaseException as exception:
            logger.error("Error preparing object to dump")
            logger.error(repr(exception))


# TODO: Remove this when V1 is default
def dump_engine_exception_v0(err: BaseException,
                             config: VllmConfig,
                             stats: Union[Stats, None] = None,
                             use_cached_outputs: Union[bool, None] = None,
                             execute_model_req: Union[ExecuteModelRequest,
                                                      None] = None):

    logger.error("Dumping input data")

    logger.error(
        "V0 LLM engine (v%s) with config: %s, "
        "use_cached_outputs=%s, ",
        VLLM_VERSION,
        config,
        use_cached_outputs,
    )

    # For V0
    if isinstance(err, ModelExecutionV0Error):
        try:
            dump_obj = prepare_object_to_dump(err.model_input)
            logger.error("Dumping model input for execution:")
            logger.error(dump_obj)
        except BaseException as exception:
            logger.error("Error preparing object to dump")
            logger.error(repr(exception))

    # In case we do not have a ModelExecutionV0Error, which is only present if
    # the engine raise an error, we still can dump the information from the
    # batch
    if execute_model_req is not None:
        batch = execute_model_req.seq_group_metadata_list
        requests_count = len(batch)

        requests_prompt_token_ids_lenghts = [{
            k: len(v.prompt_token_ids)
            for (k, v) in r.seq_data.items()
        } for r in batch]

        requests_ids = ', '.join([str(r.request_id) for r in batch])
        logger.error(
            "Batch info: requests_count=%s, "
            "requests_prompt_token_ids_lenghts=(%s), "
            "requests_ids=(%s)", requests_count,
            requests_prompt_token_ids_lenghts, requests_ids)

        for idx, r in enumerate(batch):
            logger.error(
                "Errored Batch request #%s: request_id=%s "
                "prompt_token_ids_lengths=%s, "
                "params=%s, "
                "lora_request=%s, prompt_adapter_request=%s ", idx,
                r.request_id, str(len(r.seq_data[idx].prompt_token_ids)),
                r.sampling_params, r.lora_request, r.prompt_adapter_request)

    if stats is not None:
        logger.error("System stats:")
        logger.error(stats)
