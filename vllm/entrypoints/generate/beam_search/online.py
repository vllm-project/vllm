# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from abc import ABC
from collections.abc import AsyncGenerator, Mapping

import numpy as np

from vllm import CompletionOutput, RequestOutput
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.beam_search_utils import (
    get_beam_allowed_token_ids,
    get_trie_allowed_token_ids,
    init_beam_search_so_backend,
)
from vllm.entrypoints.choice_trie import ChoiceTrie
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.renderers import BaseRenderer
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid
from vllm.utils.async_utils import collect_from_async_generator
from vllm.v1.structured_output.backend_types import StructuredOutputBackend

from .utils import BeamSearchSequence, create_sort_beams_key_function

# Engine-side cap on `SamplingParams.allowed_token_ids`; keep in sync with
# MAX_NUM_ALLOWED_TOKEN_IDS in vllm/v1/worker/gpu/sample/logit_bias.py.
_MAX_NUM_ALLOWED_TOKEN_IDS = 1024


class BeamSearchOnlineMixin(ABC):
    """online serving for beam search."""

    renderer: BaseRenderer
    engine_client: EngineClient

    async def beam_search(
        self,
        prompt: EngineInput,
        request_id: str,
        params: BeamSearchParams,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        session_id: str | None = None,
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

        logprobs_num = 2 * beam_width
        sampling_params = SamplingParams(
            logprobs=logprobs_num,
            max_tokens=1,
            temperature=temperature,
            detokenize=False,
        )

        so_backend: StructuredOutputBackend | None = None
        so_key: tuple | None = None
        so_bitmask = None
        so_trie: ChoiceTrie | None = None
        vocab_size = 0
        if params.structured_outputs is not None:
            vllm_config = self.engine_client.vllm_config
            vocab_size = vllm_config.model_config.get_vocab_size()
            so_backend, so_key, so_bitmask, so_trie = init_beam_search_so_backend(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                structured_outputs=params.structured_outputs,
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

        try:
            for _ in range(max_tokens):
                if so_backend is not None or so_trie is not None:
                    (
                        active_beams,
                        beam_params_list,
                        allowed_sets,
                        newly_completed,
                    ) = self._build_online_so_params(
                        all_beams,
                        logprobs_num,
                        temperature,
                        so_backend,
                        so_key,
                        so_bitmask,
                        so_trie,
                        vocab_size,
                    )
                    completed.extend(newly_completed)
                    if not active_beams:
                        all_beams = []
                        break
                else:
                    active_beams = all_beams
                    beam_params_list = [sampling_params] * len(all_beams)
                    allowed_sets = [None] * len(all_beams)

                tasks = []
                request_id_batch = f"{request_id}-{random_uuid()}"

                for i, beam in enumerate(active_beams):
                    prompt_item = beam.get_prompt()
                    lora_request_item = beam.lora_request
                    request_id_item = f"{request_id_batch}-beam-{i}"
                    task = asyncio.create_task(
                        collect_from_async_generator(
                            self.engine_client.generate(
                                prompt_item,
                                beam_params_list[i],
                                request_id_item,
                                lora_request=lora_request_item,
                                trace_headers=trace_headers,
                                session_id=session_id,
                            )
                        )
                    )
                    tasks.append(task)

                output = [x[0] for x in await asyncio.gather(*tasks)]

                for result in output:
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

                if any(result.outputs[0].finish_reason == "abort" for result in output):
                    for beam in active_beams:
                        beam.finish_reason = "abort"
                        completed.append(beam)
                    all_beams = []
                    break

                candidates = []
                for i, result in enumerate(output):
                    current_beam = active_beams[i]

                    if result.outputs[0].logprobs is not None:
                        logprobs = result.outputs[0].logprobs[0]
                        allowed = allowed_sets[i]
                        for token_id, logprob_obj in logprobs.items():
                            if allowed is not None and token_id not in allowed:
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
        finally:
            if so_backend is not None:
                so_backend.destroy()

        completed.extend(all_beams)
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        for beam in best_beams:
            if beam.tokens[-1] == eos_token_id and not ignore_eos:
                # Skip the eos token in the text.
                tokens = beam.tokens[tokenized_length:-1]
            else:
                tokens = beam.tokens[tokenized_length:]
            beam.text = tokenizer.decode(
                tokens, skip_special_tokens=params.skip_special_tokens
            )

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

    def _build_online_so_params(
        self,
        beams: list[BeamSearchSequence],
        logprobs_num: int,
        temperature: float,
        so_backend: StructuredOutputBackend | None,
        so_key: tuple | None,
        so_bitmask,
        so_trie: ChoiceTrie | None,
        vocab_size: int,
    ) -> tuple[
        list[BeamSearchSequence],
        list[SamplingParams],
        list[set[int] | None],
        list[BeamSearchSequence],
    ]:
        """Build per-beam params and allowed sets under structured output.

        Returns ``(active_beams, params, allowed_sets, completed)``. Beams
        whose grammar has terminated (or whose trie node is terminal/off-trie)
        are dropped from ``active_beams`` and returned in ``completed`` when
        they have generated at least one token. The trie path is
        O(generation_length) per beam and independent of the choice count, so
        it stays fast even for very large choice sets.
        """
        active_beams: list[BeamSearchSequence] = []
        params: list[SamplingParams] = []
        allowed_sets: list[set[int] | None] = []
        completed: list[BeamSearchSequence] = []
        for beam in beams:
            if so_trie is not None:
                allowed_ids = get_trie_allowed_token_ids(beam, so_trie)
            else:
                assert so_backend is not None and so_key is not None
                allowed_ids = get_beam_allowed_token_ids(
                    beam, so_backend, so_key, so_bitmask, vocab_size
                )
            if not allowed_ids:
                # Grammar/trie reached a terminal state (no EOS emitted), so
                # this beam is a completed valid output, not a truncation.
                if beam.logprobs:
                    beam.finish_reason = "stop"
                    completed.append(beam)
                continue
            # The engine caps the size of allowed_token_ids. When the allowed
            # set exceeds the cap (e.g. a trie root over thousands of choices),
            # skip the engine-side constraint and rely on the logprobs
            # filtering via allowed_sets instead.
            beam_params = SamplingParams(
                logprobs=logprobs_num,
                max_tokens=1,
                temperature=temperature,
                detokenize=False,
                allowed_token_ids=(
                    allowed_ids
                    if len(allowed_ids) <= _MAX_NUM_ALLOWED_TOKEN_IDS
                    else None
                ),
            )
            active_beams.append(beam)
            params.append(beam_params)
            allowed_sets.append(set(allowed_ids))
        return active_beams, params, allowed_sets, completed
