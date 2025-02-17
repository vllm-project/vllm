# SPDX-License-Identifier: Apache-2.0
import enum
import json
from typing import Union

import torch

from vllm.config import VllmConfig
from vllm.engine.metrics import Stats
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SequenceData
from vllm.v1.core.scheduler_output import NewRequestData
from vllm.version import __version__ as VLLM_VERSION
from vllm.worker.worker_base import ModelExecutionError

logger = init_logger(__name__)


# Hacky way to make sure we can serialize the object in JSON format
def is_json_serializable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def prepare_object_to_dump(obj):
    if isinstance(obj, dict):
        return {k: prepare_object_to_dump(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [prepare_object_to_dump(v) for v in obj]
    elif isinstance(obj, set):
        return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return [prepare_object_to_dump(v) for v in obj]
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    elif isinstance(obj, SequenceData):
        # Custom representation (based on SequenceData.__repr__)
        # to obfuscate some parameters
        return {
            "class": "SequenceData",
            "prompt_token_ids_len": len(obj._prompt_token_ids),
            "output_token_ids_len": len(obj.output_token_ids),
            "cumulative_logprob": obj.cumulative_logprob,
            "get_num_computed_tokens": obj.get_num_computed_tokens()
        }

    elif isinstance(obj, NewRequestData):
        obj_dict = {'class': type(obj).__name__}
        for k, v in obj.__dict__.items():
            if k == 'prompt_token_ids':
                obj_dict['prompt_token_ids_len'] = len(v)
            elif k == 'prompt':
                obj_dict['prompt'] = ""
            else:
                obj_dict[k] = prepare_object_to_dump(v)

        return obj_dict
    elif isinstance(obj, torch.Tensor):
        # We only print the 'draft'of the tensor to not expose sensitive data
        # and to get some metadata in case of CUDA illegal memory access
        return (f"Tensor(shape={obj.shape}, "
                f"device={obj.device},"
                f"dtype={obj.dtype})")
    elif hasattr(obj, '__dict__'):
        obj_dict = {'class': type(obj).__name__}
        obj_dict.update(obj.__dict__)
        return prepare_object_to_dump(obj_dict)
    else:
        # Try to make sure we can serialize the object
        # to avoid exception
        if is_json_serializable(obj):
            return obj
        else:
            return repr(obj)


def dump_engine_exception(err: BaseException,
                          config: VllmConfig,
                          engine_version: int,
                          stats: Stats = None,
                          use_cached_outputs: Union[bool, None] = None,
                          execute_model_req: ExecuteModelRequest = None):

    assert engine_version == 0 or engine_version == 1

    logger.error("Engine crashed, dumping input data")

    if engine_version == 1:
        logger.error(
            "V1 LLM engine (v%s) with config: %s, ",
            VLLM_VERSION,
            config,
        )
    else:
        logger.error(
            "V0 LLM engine (v%s) with config: %s, "
            "use_cached_outputs=%s, ",
            VLLM_VERSION,
            config,
            use_cached_outputs,
        )

    # For V0
    if isinstance(err, ModelExecutionError):
        try:
            err_json = prepare_object_to_dump(err.model_input)
            logger.error("Model input for execution as JSON:")
            logger.error(json.dumps(err_json))
        except BaseException as err:
            logger.error("Error preparing object to dump")
            logger.error(repr(err))

    # In case we do not have a ModelExecutionError we still can
    # get information from the batch
    if execute_model_req is not None:
        batch = execute_model_req.seq_group_metadata_list
        requests_count = len(batch)
        requests_prompt_token_ids_lenghts = ', '.join([
            str(len(r.seq_data[idx].prompt_token_ids))
            for idx, r in enumerate(batch)
        ])
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

    # TODO: Have stats for V1
    if stats is not None:
        logger.error("System stats:")
        logger.error(stats)

    if engine_version == 1:
        from vllm.v1.engine.core import ModelExecutionV1Error
        if isinstance(err, ModelExecutionV1Error):
            try:
                err_json = prepare_object_to_dump(err.scheduler_output)
                logger.error("Scheduler output for model execution as JSON:")
                logger.error(json.dumps(err_json))
            except BaseException as err:
                logger.error("Error preparing object to dump")
                logger.error(repr(err))
