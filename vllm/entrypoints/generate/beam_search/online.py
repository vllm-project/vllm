# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from abc import ABC
from collections.abc import AsyncGenerator, Mapping

import numpy as np

from vllm import CompletionOutput, RequestOutput
from vllm.engine.protocol import EngineClient
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.renderers import BaseRenderer
from vllm.sampling_params import (
    MAX_LOGPROB_TOKEN_IDS,
    BeamSearchParams,
    SamplingParams,
)
from vllm.utils import random_uuid
from vllm.utils.async_utils import collect_from_async_generator

from .utils import BeamSearchSequence, create_sort_beams_key_function


class BeamSearchOnlineMixin(ABC):
    """online serving for beam search"""

    renderer: BaseRenderer
    engine_client: EngineClient

    async def beam_search(
        self,
        prompt: EngineInput,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output

        tokenizer = self.renderer.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        if prompt["type"] == "embeds":
            raise NotImplementedError("Embedding prompt not supported for beam search")

        # Extract prompt tokens and text based on model type
        decoder_prompt = (
            prompt if prompt["type"] != "enc_dec" else prompt["decoder_prompt"]
        )
        prompt_text = decoder_prompt.get("prompt")
        prompt_token_ids = decoder_prompt["prompt_token_ids"]

        tokenized_length = len(prompt_token_ids)

        allowed_token_ids = params.allowed_token_ids
        allowed_token_ids_set = (
            set(allowed_token_ids) if allowed_token_ids is not None else None
        )
        logprobs_num = 2 * beam_width
        if (
            allowed_token_ids is not None
            and len(allowed_token_ids) <= MAX_LOGPROB_TOKEN_IDS
        ):
            sampling_params = SamplingParams(
                max_tokens=1,
                temperature=temperature,
                detokenize=False,
                allowed_token_ids=allowed_token_ids,
                logprob_token_ids=allowed_token_ids,
            )
        else:
            sampling_params = SamplingParams(
                logprobs=logprobs_num,
                max_tokens=1,
                temperature=temperature,
                detokenize=False,
                allowed_token_ids=allowed_token_ids,
            )
        all_beams = [
            BeamSearchSequence(
                orig_prompt=prompt,
                tokens=prompt_token_ids,
                cum_logprob=0,
                logprobs=[],
                lora_request=lora_request,
            )
        ]
        completed = []

        for _ in range(max_tokens):
            tasks = []
            request_id_batch = f"{request_id}-{random_uuid()}"

            for i, beam in enumerate(all_beams):
                prompt_item = beam.get_prompt()
                lora_request_item = beam.lora_request
                request_id_item = f"{request_id_batch}-beam-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.engine_client.generate(
                            prompt_item,
                            sampling_params,
                            request_id_item,
                            lora_request=lora_request_item,
                            trace_headers=trace_headers,
                        )
                    )
                )
                tasks.append(task)

            output = [x[0] for x in await asyncio.gather(*tasks)]

            candidates = []
            # Iterate through all beam inference results
            for i, result in enumerate(output):
                current_beam = all_beams[i]

                # check for error finish reason and abort beam search
                if result.outputs[0].finish_reason == "error":
                    # yield error output and terminate beam search
                    yield RequestOutput(
                        request_id=request_id,
                        prompt=prompt_text,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="",
                                token_ids=[],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="error",
                            )
                        ],
                        finished=True,
                        prompt_token_ids=prompt_token_ids,
                        prompt_logprobs=None,
                    )
                    return

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        if (
                            allowed_token_ids_set is not None
                            and token_id not in allowed_token_ids_set
                        ):
                            continue
                        candidate_logprob = (
                            current_beam.cum_logprob + logprob_obj.logprob
                        )
                        if token_id == eos_token_id and not ignore_eos:
                            completed.append(
                                BeamSearchSequence(
                                    orig_prompt=prompt,
                                    tokens=current_beam.tokens + [eos_token_id]
                                    if include_stop_str_in_output
                                    else current_beam.tokens,
                                    logprobs=current_beam.logprobs + [logprobs],
                                    cum_logprob=candidate_logprob,
                                    finish_reason="stop",
                                    stop_reason=eos_token_id,
                                )
                            )
                        else:
                            candidates.append(
                                (
                                    candidate_logprob,
                                    int(token_id),
                                    current_beam,
                                    logprobs,
                                )
                            )

            # Processing non-EOS tokens
            candidate_logprobs = np.fromiter(
                (candidate[0] for candidate in candidates),
                dtype=np.float64,
                count=len(candidates),
            )
            if len(candidates) <= beam_width:
                topn_idx = np.argsort(-candidate_logprobs)
            else:
                topn_idx = np.argpartition(
                    -candidate_logprobs,
                    beam_width - 1,
                )[:beam_width]
                topn_idx = topn_idx[np.argsort(-candidate_logprobs[topn_idx])]

            new_beams = []
            for idx in topn_idx:
                cum_logprob, token_id, current_beam, logprobs = candidates[int(idx)]
                new_beams.append(
                    BeamSearchSequence(
                        orig_prompt=prompt,
                        tokens=current_beam.tokens + [token_id],
                        logprobs=current_beam.logprobs + [logprobs],
                        lora_request=current_beam.lora_request,
                        cum_logprob=cum_logprob,
                    )
                )

            all_beams = new_beams
            if not all_beams:
                break

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            if beam.tokens[-1] == eos_token_id and not ignore_eos:
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(tokens)

        yield RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=[
                CompletionOutput(
                    text=beam.text,  # type: ignore
                    cumulative_logprob=beam.cum_logprob,
                    token_ids=beam.tokens[tokenized_length:],
                    index=i,
                    logprobs=beam.logprobs,
                    finish_reason=beam.finish_reason
                    if beam.finish_reason is not None
                    else "length",
                    stop_reason=beam.stop_reason,
                )
                for (i, beam) in enumerate(best_beams)
            ],
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
        )
