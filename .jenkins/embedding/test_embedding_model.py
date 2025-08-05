# SPDX-License-Identifier: Apache-2.0
import atexit
import os
from pathlib import Path

import torch
import yaml

from vllm import LLM

TEST_DATA_FILE = os.environ.get(
    "TEST_DATA_FILE", ".jenkins/embedding/configs/e5-mistral-7b-instruct.yaml")

TP_SIZE = int(os.environ.get("TP_SIZE", 1))


def fail_on_exit():
    os._exit(1)


def launch_embedding_model(config):
    model_name = config.get('model_name')
    dtype = config.get('dtype', 'bfloat16')
    tensor_parallel_size = TP_SIZE
    llm = LLM(
        model=model_name,
        task="embed",
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=False,
    )
    return llm


def get_input():
    # Sample 1k prompts
    with open('data/prompts.txt') as file:
        # Read the entire content of the file
        content = file.read()

    prompts = content.split('\n')

    return prompts


def get_current_gaudi_platform():

    #Inspired by: https://github.com/HabanaAI/Model-References/blob/a87c21f14f13b70ffc77617b9e80d1ec989a3442/PyTorch/computer_vision/classification/torchvision/utils.py#L274

    import habana_frameworks.torch.utils.experimental as htexp

    device_type = htexp._get_device_type()

    if device_type == htexp.synDeviceType.synDeviceGaudi:
        return "Gaudi1"
    elif device_type == htexp.synDeviceType.synDeviceGaudi2:
        return "Gaudi2"
    elif device_type == htexp.synDeviceType.synDeviceGaudi3:
        return "Gaudi3"
    else:
        raise ValueError(
            f"Unsupported device: the device type is {device_type}.")


def test_embedding_model(record_xml_attribute, record_property):
    try:
        config = yaml.safe_load(
            Path(TEST_DATA_FILE).read_text(encoding="utf-8"))
        # Record JUnitXML test name
        platform = get_current_gaudi_platform()
        testname = (f'test_{Path(TEST_DATA_FILE).stem}_{platform}_'
                    f'tp{TP_SIZE}')
        record_xml_attribute("name", testname)

        llm = launch_embedding_model(config)

        # Generate embedding. The output is a list of EmbeddingRequestOutputs.
        prompts = get_input()
        outputs = llm.embed(prompts)
        torch.hpu.synchronize()

        # Print the outputs.
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            embeds = output.outputs.embedding
            embeds_trimmed = ((str(embeds[:16])[:-1] +
                               ", ...]") if len(embeds) > 16 else embeds)
            print(f"Prompt {i+1}: {prompt!r} | "
                  f"Embeddings: {embeds_trimmed} (size={len(embeds)})")
        os._exit(0)

    except Exception as exc:
        atexit.register(fail_on_exit)
        raise exc
