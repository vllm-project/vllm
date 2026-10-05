# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
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
from vllm.inputs import (
    EncoderDecoderInput,
    EngineInput,
    MultiModalInput,
    TokensInput,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.renderers import BaseRenderer
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.utils import random_uuid
from vllm.utils.async_utils import collect_from_async_generator
from vllm.v1.structured_output.backend_types import StructuredOutputBackend

from .choice_trie import ChoiceTrie
from .utils import BeamSearchSequence, create_sort_beams_key_function

logger = init_logger(__name__)

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
        self.engine_client.input_processor.resolve_watermarking(params)

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
            watermarking=False,
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
        completed: list[BeamSearchSequence] = []

        try:
            for _ in range(max_tokens):
                all_beams, error_output = await self._beam_search_step(
                    all_beams=all_beams,
                    completed=completed,
                    prompt=prompt,
                    prompt_text=prompt_text,
                    prompt_token_ids=prompt_token_ids,
                    request_id=request_id,
                    sampling_params=sampling_params,
                    logprobs_num=logprobs_num,
                    beam_width=beam_width,
                    temperature=temperature,
                    eos_token_id=eos_token_id,
                    ignore_eos=ignore_eos,
                    include_stop_str_in_output=include_stop_str_in_output,
                    so_backend=so_backend,
                    so_key=so_key,
                    so_bitmask=so_bitmask,
                    so_trie=so_trie,
                    vocab_size=vocab_size,
                    trace_headers=trace_headers,
                    session_id=session_id,
                )
                if error_output is not None:
                    yield error_output
                    return
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

    async def _beam_search_step(
        self,
        *,
        all_beams: list[BeamSearchSequence],
        completed: list[BeamSearchSequence],
        prompt: TokensInput | MultiModalInput | EncoderDecoderInput,
        prompt_text: str | None,
        prompt_token_ids: list[int],
        request_id: str,
        sampling_params: SamplingParams,
        logprobs_num: int,
        beam_width: int,
        temperature: float,
        eos_token_id: int | None,
        ignore_eos: bool,
        include_stop_str_in_output: bool,
        so_backend: StructuredOutputBackend | None,
        so_key: tuple | None,
        so_bitmask,
        so_trie: ChoiceTrie | None,
        vocab_size: int,
        trace_headers: Mapping[str, str] | None,
        session_id: str | None,
    ) -> tuple[list[BeamSearchSequence], RequestOutput | None]:
        """Advance beam search by one token step.

        Finished beams are appended to ``completed`` in place. Returns
        ``(next_beams, error_output)``: a non-``None`` ``error_output`` means
        the engine aborted with an error and the caller must yield it and
        stop; an empty ``next_beams`` means the search is complete.
        """
        if so_backend is not None or so_trie is not None:
            # Grammar compile/accept/fill_bitmask (and the trie walk) are
            # CPU-bound and run once per beam per step. Offload to a worker
            # thread so they do not block the API server's asyncio event loop
            # and stall other concurrent requests, mirroring
            # StructuredOutputManager's executor offload.
            #
            # Keep a handle so cancellation of this coroutine does not race the
            # caller's `finally: so_backend.destroy()`. A detached thread keeps
            # dereferencing `so_backend`; destroying it mid-run raises
            # AttributeError. Shield the task and, on cancellation, wait for the
            # thread to finish before propagating so the backend is only
            # destroyed once nothing touches it.
            so_build_task = asyncio.ensure_future(
                asyncio.to_thread(
                    self._build_online_so_params,
                    all_beams,
                    logprobs_num,
                    temperature,
                    so_backend,
                    so_key,
                    so_bitmask,
                    so_trie,
                    vocab_size,
                )
            )
            try:
                (
                    active_beams,
                    beam_params_list,
                    allowed_sets,
                    newly_completed,
                ) = await asyncio.shield(so_build_task)
            except asyncio.CancelledError:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(so_build_task)
                raise
            completed.extend(newly_completed)
            if not active_beams:
                return [], None
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
                # signal the caller to yield an error output and terminate
                error_output = RequestOutput(
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
                return all_beams, error_output

        if any(result.outputs[0].finish_reason == "abort" for result in output):
            for beam in active_beams:
                beam.finish_reason = "abort"
                completed.append(beam)
            return [], None

        candidates = []
        for i, result in enumerate(output):
            current_beam = active_beams[i]

            if result.outputs[0].logprobs is not None:
                logprobs = result.outputs[0].logprobs[0]
                allowed = allowed_sets[i]
                beam_produced = False
                for token_id, logprob_obj in logprobs.items():
                    if allowed is not None and token_id not in allowed:
                        continue
                    candidate_logprob = current_beam.cum_logprob + logprob_obj.logprob
                    if token_id == eos_token_id and not ignore_eos:
                        beam_produced = True
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
                        beam_produced = True
                        candidates.append(
                            (
                                candidate_logprob,
                                int(token_id),
                                current_beam,
                                logprobs,
                            )
                        )
                if (
                    not beam_produced
                    and allowed is not None
                    and len(allowed) > _MAX_NUM_ALLOWED_TOKEN_IDS
                ):
                    # Over-cap regime (see _build_online_so_params): the engine
                    # sampled unconstrained and none of its top logprobs fell
                    # in the allowed set, so the grammar was not enforced for
                    # this beam and the beam is silently dropped. Surface it.
                    logger.warning(
                        "Beam search (request %s): structured-output allowed "
                        "set has %d tokens (> cap %d), so engine-side "
                        "constraint was disabled and none of the sampled "
                        "top-%d tokens were valid. Dropping this beam; the "
                        "request may finish with fewer outputs.",
                        request_id,
                        len(allowed),
                        _MAX_NUM_ALLOWED_TOKEN_IDS,
                        logprobs_num,
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

        return new_beams, None

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
            #
            # Limitation: in this over-cap regime the engine samples
            # unconstrained and returns only 2*beam_width logprobs, which are
            # then filtered against allowed_sets. If none of those top tokens
            # are in the allowed set, the beam yields no candidates and is
            # dropped, so a request can finish early with fewer (possibly zero)
            # outputs and finish_reason="length" rather than an error. The
            # allowed set is near-full in this regime, so the model's natural
            # top tokens almost always fall inside it; engine-side masking for
            # arbitrary-size allowed sets is left as a follow-up. This mirrors
            # the pre-existing offline limitation.
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
