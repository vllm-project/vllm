# SPDX-License-Identifier: Apache-2.0
"""Core test implementation to be shared across modalities."""
from typing import Any, Callable, Optional, Union

import torch
from PIL.Image import Image
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.config import TaskOption
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .....conftest import HfRunner, VllmRunner
from ....registry import HF_EXAMPLE_MODELS
from .types import RunnerOutput


def run_test(
    *,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: list[tuple[list[str], list[Union[list[Image], Image]]]],
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    enforce_eager: bool,
    max_model_len: int,
    max_num_seqs: int,
    hf_output_post_proc: Optional[Callable[[RunnerOutput, str], Any]],
    vllm_output_post_proc: Optional[Callable[[RunnerOutput, str], Any]],
    auto_cls: type[_BaseAutoModelClass],
    use_tokenizer_eos: bool,
    comparator: Callable[..., None],
    get_stop_token_ids: Optional[Callable[[AnyTokenizer], list[int]]],
    stop_str: Optional[list[str]],
    limit_mm_per_prompt: dict[str, int],
    vllm_runner_kwargs: Optional[dict[str, Any]],
    hf_model_kwargs: Optional[dict[str, Any]],
    patch_hf_runner: Optional[Callable[[HfRunner], HfRunner]],
    task: TaskOption = "auto",
    runner_mm_key: str = "images",
    distributed_executor_backend: Optional[str] = None,
    tensor_parallel_size: int = 1,
    vllm_embeddings: Optional[torch.Tensor] = None,
):
    """Modality agnostic test test executor for comparing HF/vLLM outputs."""
    # In the case of embeddings, vLLM takes separate input tensors
    vllm_inputs = vllm_embeddings if vllm_embeddings is not None else inputs

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    limit_mm_per_prompt = default_limits | limit_mm_per_prompt

    vllm_outputs_per_mm = []
    hf_outputs_per_mm = []

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    vllm_runner_kwargs_: dict[str, Any] = {
        "disable_mm_preprocessor_cache": True,
    }
    if model_info.tokenizer:
        vllm_runner_kwargs_["tokenizer_name"] = model_info.tokenizer
    if model_info.tokenizer_mode:
        vllm_runner_kwargs_["tokenizer_mode"] = model_info.tokenizer_mode
    if model_info.hf_overrides:
        vllm_runner_kwargs_["hf_overrides"] = model_info.hf_overrides

    if vllm_runner_kwargs:
        vllm_runner_kwargs_.update(vllm_runner_kwargs)

    with vllm_runner(model,
                     max_model_len=max_model_len,
                     max_num_seqs=max_num_seqs,
                     dtype=dtype,
                     limit_mm_per_prompt=limit_mm_per_prompt,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=enforce_eager,
                     task=task,
                     **vllm_runner_kwargs_) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()

        vllm_kwargs: dict[str, Any] = {}
        if get_stop_token_ids is not None:
            vllm_kwargs["stop_token_ids"] = get_stop_token_ids(tokenizer)
        if stop_str:
            vllm_kwargs["stop"] = stop_str

        for prompts, media in vllm_inputs:
            vllm_kwargs[runner_mm_key] = media
            vllm_output = vllm_model.generate_greedy_logprobs(
                prompts, max_tokens, num_logprobs=num_logprobs, **vllm_kwargs)
            vllm_outputs_per_mm.append(vllm_output)

    hf_model = hf_runner(model,
                         dtype=dtype,
                         auto_cls=auto_cls,
                         model_kwargs=hf_model_kwargs)

    # Some models need to patch things like the model processor, e.g., internvl
    if patch_hf_runner is not None:
        hf_model = patch_hf_runner(hf_model)

    with hf_model, torch.no_grad():
        tokenizer = hf_model.tokenizer

        # Some models need to explicitly pass the eos_token_id off the tokenizer
        # or processor for a good comparison;
        # currently assume processor/tokenizer agree on the EOS, and pull it off
        # the tokenizer if requested.
        hf_kwargs = {}
        if use_tokenizer_eos:
            hf_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if stop_str:
            hf_kwargs["stop_strings"] = stop_str

        for prompts, media in inputs:
            hf_kwargs[runner_mm_key] = media
            hf_output = hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                tokenizer=tokenizer,
                **hf_kwargs)
            hf_outputs_per_mm.append(hf_output)

    # Apply output processing / sanitation to the vLLM and HF runner results
    hf_outputs_per_mm, vllm_outputs_per_mm = process_runner_outputs(
        model,
        first_runner_outputs=hf_outputs_per_mm,
        second_runner_outputs=vllm_outputs_per_mm,
        first_runner_processor=hf_output_post_proc,
        second_runner_processor=vllm_output_post_proc,
    )

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_mm,
                                        vllm_outputs_per_mm):
        # This is usually check_logprobs_close, but it's passed through to
        # allow things like check_outputs_equal where needed
        comparator(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


def process_runner_outputs(
    model,
    first_runner_outputs,
    second_runner_outputs,
    first_runner_processor=None,
    second_runner_processor=None,
):
    """Applies the runner processor(s) to the runner outputs, if any."""
    if first_runner_processor is not None:
        first_runner_outputs = process_outputs(first_runner_processor, model,
                                               first_runner_outputs)
    if second_runner_processor is not None:
        second_runner_outputs = process_outputs(second_runner_processor, model,
                                                second_runner_outputs)
    return first_runner_outputs, second_runner_outputs


def process_outputs(output_processor, model, outputs_per_image):
    """Applies a model specific post-processor function to a runner's output"""
    return [[output_processor(res, model) for res in outputs]
            for outputs in outputs_per_image]
