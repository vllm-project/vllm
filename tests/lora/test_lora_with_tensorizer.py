# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig, tensorize_lora_adapter, tensorize_vllm_model)


@pytest.mark.parametrize(
    ("model_name", "lora_uri"),
    [("meta-llama/Llama-2-7b-hf", "yard1/llama-2-7b-sql-lora-test"),
     ("baichuan-inc/Baichuan-7B", "jeeejeee/baichuan7b-text2sql-spider")])
def test_serialize_and_deserialize_lora(tmp_path, model_name, lora_uri):

    model_ref = model_name
    lora_path = lora_uri

    model_uri = tmp_path / (model_ref + ".tensors")
    tensorizer_config = TensorizerConfig(tensorizer_uri=str(model_uri))
    tensorizer_config.lora_dir = tensorizer_config.tensorizer_dir

    # trust_remote_code=True for Baichuan-7B
    args = EngineArgs(model=model_ref, trust_remote_code=True)

    tensorize_lora_adapter(lora_path, tensorizer_config)
    tensorize_vllm_model(args, tensorizer_config)

    gc.collect()
    torch.cuda.empty_cache()

    tensorizer_config_dict = TensorizerConfig.as_dict(str(model_uri))

    loaded_vllm_model = LLM(model=model_ref,
                            load_format="tensorizer",
                            trust_remote_code=True,
                            model_loader_extra_config=tensorizer_config,
                            enable_lora=True,
                            max_lora_rank=128)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
    ]

    loaded_vllm_model.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(
            "sql-lora",
            1,
            lora_path,
            tensorizer_config_dict=tensorizer_config_dict))
