import pytest
import torch.nn.functional as F

from ....conftest import IMAGE_ASSETS
from ..utils import check_embeddings_close

HF_TEXT_PROMPTS = [
    # T -> X
    "Find me an everyday image that matches the given caption: The label of the object is stop sign",  # noqa: E501
    # T -> X
    "Retrieve an image of this caption: cherry blossom",
]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    # T + I -> X
    "stop_sign":
    "<|image_1|> Select the portion of the image that isolates the object of the given label: The label of the object is stop sign",  # noqa: E501
    # I -> X
    "cherry_blossom":
    "<|image_1|> Represent the given image for classification",  # noqa: E501
})

MODELS = ["TIGER-Lab/VLM2Vec-Full"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [
        *((text, None) for text in HF_TEXT_PROMPTS),
        *((text, image)
          for text, image in zip(HF_IMAGE_PROMPTS, image_assets)),
    ]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(model, task="embedding", dtype=dtype,
                     enforce_eager=True) as vllm_model:
        vllm_outputs = vllm_model.encode(input_texts, images=input_images)

    with hf_runner(model, dtype=dtype) as hf_model:
        all_inputs = hf_model.get_inputs(input_texts, images=input_images)

        all_outputs = []
        for inputs in all_inputs:
            # Based on: https://github.com/TIGER-AI-Lab/VLM2Vec/blob/db3b951bccabba220c1f53ab46a734e50dd2fc08/src/model.py
            outputs = hf_model.model(
                **hf_model.wrap_device(inputs,
                                       device=hf_model.model.device.type),
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1][0]
            reps = last_hidden_state[inputs.attention_mask[0].sum() - 1]
            pooled_output = F.normalize(reps, p=2, dim=-1)

            all_outputs.append(pooled_output.tolist())

        hf_outputs = all_outputs

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
