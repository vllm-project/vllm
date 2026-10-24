# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

import tests.ci_envs as ci_envs
from tests.models.utils import (
    GenerateModelInfo,
    TokensTextLogprobsPromptLogprobs,
    get_vllm_extra_kwargs,
)
from vllm.logprobs import Logprob

PPL_TOL = 0.01


def _answer_span(token_ids: list[int], answer_ids: list[int]) -> range:
    """Positions of the reference answer within the prompt tokens.

    Only answer tokens are scored: they depend on the image and the question,
    while system-prompt, question and chat-template tokens have perplexities in
    the thousands that dominate the average and mostly reflect numerical noise.
    """
    n = len(answer_ids)
    for start in range(len(token_ids) - n, -1, -1):
        if token_ids[start : start + n] == answer_ids:
            return range(start, start + n)
    raise AssertionError("Reference answer tokens not found in the prompt")


@torch.inference_mode
def vqa_ppl_test(
    hf_runner,
    vllm_runner,
    model_info: GenerateModelInfo,
    mm_processor_kwargs=None,
    vllm_extra_kwargs=None,
    tol=PPL_TOL,
):
    dataset = load_dataset("lmms-lab-encoder/llava-bench-in-the-wild", split="train")

    vllm_extra_kwargs = get_vllm_extra_kwargs(model_info, vllm_extra_kwargs)

    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}

    processor = AutoProcessor.from_pretrained(model_info.name)

    images = []
    prompts = []
    answers_ids = []
    for row in dataset:
        image = row["image"]
        images.append(image)

        question = row["question"]
        answer = row["gpt_answer"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        prompts.append(prompt)
        answers_ids.append(
            processor.tokenizer(answer, add_special_tokens=False).input_ids
        )

    with vllm_runner(
        model_info.name,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        max_model_len=4096,
        mm_processor_kwargs=mm_processor_kwargs,
        **vllm_extra_kwargs,
    ) as vllm_model:
        # Use max_num_seqs=1 to avoid OOM,
        # and avoid batch different requests together.

        model_config = vllm_model.llm.llm_engine.model_config
        mm_device_do_normalize = model_config.multimodal_config.mm_device_do_normalize

        # Confirm whether vllm is using the correct architecture
        if model_info.architecture:
            assert model_info.architecture in model_config.architectures

        nll_sum = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        n_tokens = 0

        outputs = vllm_model.generate_greedy_logprobs(
            prompts=prompts,
            images=images,
            max_tokens=1,
            num_logprobs=None,
            num_prompt_logprobs=0,
            use_tqdm=False,
        )

        for output, answer_ids in zip(outputs, answers_ids):
            output = cast(TokensTextLogprobsPromptLogprobs, output)
            token_datas = cast(list[dict[int, Logprob] | None], output[3])

            assert token_datas[0] is None
            # The first prompt token has no logprob and is never an answer token.
            token_ids = [-1]
            token_log_probs = [0.0]
            for token_data in token_datas[1:]:
                assert token_data is not None
                assert len(token_data) == 1
                token_id, token_log_prob = next(iter(token_data.items()))
                token_ids.append(token_id)
                token_log_probs.append(token_log_prob.logprob)

            span = _answer_span(token_ids, answer_ids)
            neg_log_likelihood = -torch.tensor(
                token_log_probs[span.start : span.stop],
                dtype=torch.float32,
                device="cpu",
            ).sum()
            nll_sum += neg_log_likelihood
            n_tokens += len(span)
        vllm_ppl = float(torch.exp(nll_sum / n_tokens))
        vllm_dtype = model_config.dtype
        head_dtype = model_config.head_dtype

    # Accelerate ppl test by setting Transformers ppl score to a constant
    if model_info.hf_ppl is None:
        with hf_runner(
            model_info.name,
            dtype=ci_envs.VLLM_CI_HF_DTYPE or model_info.hf_dtype,
            auto_cls=AutoModelForImageTextToText,
        ) as hf_model:
            nll_sum = torch.tensor(0.0, dtype=torch.float32, device="cpu")
            n_tokens = 0

            for prompt, image, answer_ids in zip(prompts, images, answers_ids):
                inputs = processor(
                    text=[prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                    **mm_processor_kwargs,
                )
                inputs = inputs.to("cuda")
                input_ids = inputs["input_ids"]

                span = _answer_span(input_ids[0].tolist(), answer_ids)
                labels = torch.full_like(input_ids, -100)
                labels[:, span.start : span.stop] = input_ids[:, span.start : span.stop]
                outputs = hf_model.model(**inputs, labels=labels)
                neg_log_likelihood = outputs.loss
                neg_log_likelihood = neg_log_likelihood.to(torch.float32).cpu()

                num_loss_tokens = int((labels[:, 1:] != -100).sum())
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

            hf_ppl = float(torch.exp(nll_sum / n_tokens))
            hf_dtype = next(hf_model.model.parameters()).dtype
    else:
        hf_ppl = model_info.hf_ppl
        hf_dtype = "Constant"

    differ = (vllm_ppl - hf_ppl) / hf_ppl
    print("Model:", model_info.name)
    print("mm_device_do_normalize:", mm_device_do_normalize)
    print("VLLM:", f"dtype:{vllm_dtype}", f"head_dtype:{head_dtype}", vllm_ppl)
    print("Transformers:", hf_dtype, hf_ppl)
    print("Difference (%):", differ * 100)

    assert abs(differ) < tol
    return vllm_ppl
