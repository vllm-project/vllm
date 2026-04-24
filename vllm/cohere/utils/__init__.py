# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import traceback

from vllm.cohere.guided_decoding.cohere_constants import (
    END_THINKING_TOKEN,
    START_THINKING_TOKEN,
)
from vllm.config import ModelConfig, VllmConfig
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.v1.outputs import LogprobsLists


def print_stack_trace():
    # Get the stack trace as a list of strings
    stack_trace = traceback.format_stack()

    # Join the stack trace into a single string
    stack_trace_string = "".join(stack_trace)

    # Print or use the stack trace string
    print(stack_trace_string)


def get_cohere_thinking_token_ids(vllm_config: VllmConfig) -> tuple[int, int]:
    tokenizer = init_tokenizer_from_configs(
        model_config=vllm_config.model_config,
    )
    end_thinking_token_id = tokenizer.encode(
        END_THINKING_TOKEN, add_special_tokens=True
    )[1]
    start_thinking_token_id = tokenizer.encode(
        START_THINKING_TOKEN, add_special_tokens=True
    )[1]
    return start_thinking_token_id, end_thinking_token_id


def get_current_usage(output_tokens, start_thinking_token_index):
    if start_thinking_token_index is None:
        current_usage = len(output_tokens)
    else:
        current_usage = len(output_tokens) - start_thinking_token_index
    return current_usage


def get_tokenizer(vllm_config: VllmConfig):
    """
    Initializes the tokenizer based on the provided VLLM configuration.
    """
    tokenizer = init_tokenizer_from_configs(
        model_config=vllm_config.model_config,
    )

    if not tokenizer:
        raise ValueError("Tokenizer initialization failed.")

    return tokenizer


def get_text_model_name(model_config: ModelConfig) -> str:
    if model_config.hf_text_config is not None:
        # get text model arch for vision models
        return model_config.hf_text_config.architectures[0]
    return model_config.architectures[0]


def handle_thinking_tokens(
    *,
    request,
    new_token_ids,
    start_thinking_token_id,
    end_thinking_token_id,
    requests_to_start_thinking_idx,
    requests_with_remaining_budget,
    logger,
):
    """
    Handle START_THINKING and END_THINKING tokens for a request.

    """

    if start_thinking_token_id in new_token_ids:
        # start thinking token is present in this step
        logger.info(
            "Request %s has generated <|START_THINKING|> token.", request.request_id
        )

        start_thinking_index = new_token_ids.index(start_thinking_token_id)
        previous_length = len(request.output_token_ids) - len(new_token_ids)

        requests_to_start_thinking_idx[request.request_id] = (
            start_thinking_index + 1 + previous_length
        )

    if end_thinking_token_id in new_token_ids:
        # end thinking token is present, finish thinking
        del requests_to_start_thinking_idx[request.request_id]

        if request.request_id in requests_with_remaining_budget:
            del requests_with_remaining_budget[request.request_id]
        else:
            logger.warning(
                "Request %s has generated "
                "<|END_THINKING|> before <|START_THINKING|>. "
                "These are the previous output tokens: %s",
                request.request_id,
                request.output_token_ids,
            )


def _adjust_logprobs_for_force_end_thinking(
    logprobs_lists: LogprobsLists,
    token_adjustments: list[dict],
    valid_sampled_token_ids: list[list[int]],
    end_thinking_token_id: int | None = None,
) -> LogprobsLists:
    """Adjust logprobs structure to match force end thinking token modifications."""
    if not token_adjustments and end_thinking_token_id is None:
        return logprobs_lists

    import numpy as np

    orig = logprobs_lists
    new_cu_tokens = np.cumsum([len(tokens) for tokens in valid_sampled_token_ids])
    adj_map = (
        {adj["req_index"]: adj for adj in token_adjustments}
        if token_adjustments
        else {}
    )
    truncated_requests = [
        adj["req_index"] for adj in token_adjustments if adj["action"] == "truncate"
    ]

    # Copy original arrays to modify
    new_token_ids = orig.logprob_token_ids.copy()
    new_logprobs = orig.logprobs.copy()
    new_ranks = orig.sampled_token_ranks.copy()
    # Collect indices to remove (process from right to left to avoid index shifting)
    indices_to_remove: list[int] = []

    # Calculate flat indices for each request
    flat_idx = 0
    for req_idx, tokens in enumerate(valid_sampled_token_ids):
        if req_idx in truncated_requests:
            original_length = adj_map[req_idx]["original_length"]
            final_length = adj_map[req_idx]["final_length"]
            remove_count = original_length - final_length

            if remove_count > 0:
                # Add indices to remove (the truncated tokens)
                start_remove = flat_idx + final_length
                end_remove = flat_idx + original_length
                indices_to_remove.extend(range(start_remove, end_remove))

            flat_idx += original_length
        else:
            flat_idx += len(tokens)

    # Remove indices in reverse order to avoid index shifting issues
    for idx in sorted(indices_to_remove, reverse=True):
        new_token_ids = np.delete(new_token_ids, idx, axis=0)
        new_logprobs = np.delete(new_logprobs, idx, axis=0)
        new_ranks = np.delete(new_ranks, idx)
    # Handle end thinking token adjustment (after truncation)
    if end_thinking_token_id is not None:
        flat_idx = 0
        for req_idx, tokens in enumerate(valid_sampled_token_ids):
            for i, token_id in enumerate(tokens):
                if token_id == end_thinking_token_id and flat_idx + i < len(
                    new_logprobs
                ):
                    # Set logprob to 0.0 for end thinking token
                    new_token_ids[flat_idx + i] = end_thinking_token_id
                    new_logprobs[flat_idx + i] = 0.0
            flat_idx += len(tokens)

    return LogprobsLists(
        logprob_token_ids=new_token_ids,
        logprobs=new_logprobs,
        sampled_token_ranks=new_ranks,
        cu_num_generated_tokens=new_cu_tokens.tolist(),
    )
