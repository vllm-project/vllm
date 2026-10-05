# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import json
from collections.abc import Callable, Sequence

import torch
from tqdm import tqdm

from vllm import RequestOutput, TextPrompt, TokensPrompt
from vllm.entrypoints.beam_search_utils import (
    BeamSearchSOState,
    bitmask_to_token_ids,
    get_trie_allowed_token_ids,
    validate_and_resolve_beam_search_so,
)
from vllm.entrypoints.offline_utils import OfflineInferenceMixin
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import (
    BeamSearchParams,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key

from .choice_trie import ChoiceTrie
from .utils import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
)

logger = init_logger(__name__)

# Engine-side cap on `SamplingParams.allowed_token_ids`; keep in sync with
# MAX_NUM_ALLOWED_TOKEN_IDS in vllm/v1/worker/gpu/sample/logit_bias.py.
_MAX_NUM_ALLOWED_TOKEN_IDS = 1024


class BeamSearchOfflineMixin(OfflineInferenceMixin):
    """Offline inference for beam search."""

    def beam_search(
        self,
        prompts: list[TokensPrompt | TextPrompt],
        params: BeamSearchParams,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list[BeamSearchOutput]:
        """Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            concurrency_limit: The maximum number of concurrent requests.
                If None, the number of concurrent requests is unlimited.

        """
        # TODO: how does beam search work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty
        self.llm_engine.input_processor.resolve_watermarking(params)

        tokenizer = self.renderer.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        sort_beams_key = create_sort_beams_key_function(eos_token_id, length_penalty)

        engine_inputs = self._preprocess_cmpl(prompts)
        lora_requests = self._lora_request_to_seq(lora_request, len(engine_inputs))

        if use_tqdm and concurrency_limit is not None:
            logger.warning(
                "Progress bar is not supported when using concurrency_limit. "
                "Disabling progress bar."
            )
            use_tqdm = False

        if concurrency_limit is None:
            concurrency_limit = len(engine_inputs)

        structured_output_backend: StructuredOutputBackend | None = None
        structured_output_key = None
        structured_output_bitmask = None
        structured_output_trie: ChoiceTrie | None = None
        if params.structured_outputs is not None:
            (
                structured_output_backend,
                structured_output_key,
                structured_output_bitmask,
                structured_output_trie,
            ) = self._init_beam_search_structured_output(
                params.structured_outputs, tokenizer
            )

        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        base_sampling_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
            watermarking=False,
            detokenize=False,
            skip_clone=True,  # Internal beam search, safe to skip clone
        )
        instances: list[BeamSearchInstance] = []

        for lora_req, prompt in zip(lora_requests, engine_inputs):
            if prompt["type"] == "embeds":
                raise NotImplementedError(
                    "Embedding prompt not supported for beam search"
                )

            instances.append(
                BeamSearchInstance(
                    prompt,
                    lora_request=lora_req,
                    logprobs=None,
                ),
            )

        try:
            for prompt_start in range(0, len(instances), concurrency_limit):
                instances_batch = instances[
                    prompt_start : prompt_start + concurrency_limit
                ]

                token_iter = range(max_tokens)
                if use_tqdm:
                    token_iter = tqdm(
                        token_iter,
                        desc="Beam search",
                        unit="token",
                        unit_scale=False,
                    )
                    logger.warning(
                        "The progress bar shows the upper bound on token "
                        "steps and may finish early due to stopping "
                        "conditions. It does not reflect instance-level "
                        "progress."
                    )
                for _ in token_iter:
                    should_stop = self._beam_search_step(
                        instances_batch=instances_batch,
                        base_sampling_params=base_sampling_params,
                        eos_token_id=eos_token_id,
                        ignore_eos=ignore_eos,
                        beam_width=beam_width,
                        sort_beams_key=sort_beams_key,
                        structured_output_backend=structured_output_backend,
                        structured_output_key=structured_output_key,
                        structured_output_bitmask=structured_output_bitmask,
                        structured_output_trie=structured_output_trie,
                    )
                    if should_stop:
                        break
        finally:
            if structured_output_backend is not None:
                structured_output_backend.destroy()

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(
                instance.completed, key=sort_beams_key, reverse=True
            )
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(
                    beam.tokens, skip_special_tokens=params.skip_special_tokens
                )

            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def _beam_search_step(
        self,
        instances_batch: list[BeamSearchInstance],
        base_sampling_params: SamplingParams,
        eos_token_id: int | None,
        ignore_eos: bool,
        beam_width: int,
        sort_beams_key: Callable,
        structured_output_backend: StructuredOutputBackend | None,
        structured_output_key: tuple | None,
        structured_output_bitmask: torch.Tensor | None,
        structured_output_trie: ChoiceTrie | None,
    ) -> bool:
        """Run one token step of beam search across a batch of instances.

        Returns True if all beams are exhausted and search should stop.
        """
        all_beams: list[BeamSearchSequence] = list(
            itertools.chain.from_iterable(
                instance.beams for instance in instances_batch
            )
        )
        pos = [0] + list(
            itertools.accumulate(len(instance.beams) for instance in instances_batch)
        )
        instance_start_and_end: list[tuple[int, int]] = list(zip(pos[:-1], pos[1:]))

        if len(all_beams) == 0:
            return True

        beam_entries: list[tuple[SamplingParams, list[int]] | None] | None = None
        if structured_output_trie is not None:
            beam_entries = self._build_trie_sampling_params(
                all_beams,
                base_sampling_params,
                structured_output_trie,
            )
        elif structured_output_backend is not None:
            assert (
                structured_output_key is not None
                and structured_output_bitmask is not None
            )
            beam_entries = self._build_beam_sampling_params(
                all_beams,
                base_sampling_params,
                structured_output_backend,
                structured_output_key,
                structured_output_bitmask,
            )

        if beam_entries is not None:
            active_indices = [
                i for i, entry in enumerate(beam_entries) if entry is not None
            ]
            for i, entry in enumerate(beam_entries):
                if entry is None:
                    beam = all_beams[i]
                    assert beam.orig_prompt["type"] != "enc_dec"
                    prompt_len = len(beam.orig_prompt["prompt_token_ids"])
                    if len(beam.tokens) > prompt_len:
                        # Grammar/trie reached a terminal state (no EOS
                        # emitted); this beam is a completed valid output.
                        beam.finish_reason = "stop"
                        for (s, e), inst in zip(
                            instance_start_and_end,
                            instances_batch,
                        ):
                            if s <= i < e:
                                inst.completed.append(beam)
                                break

            if not active_indices:
                return True

            active_beams = [all_beams[i] for i in active_indices]
            active_params: Sequence[SamplingParams | PoolingParams] = [
                beam_entries[i][0]  # type: ignore[index]
                for i in active_indices
            ]
        else:
            active_indices = list(range(len(all_beams)))
            active_beams = all_beams
            active_params = self._params_to_seq(  # type: ignore[assignment]
                base_sampling_params, len(all_beams)
            )

        # only runs for one step
        # we don't need to use tqdm here
        active_output = self._render_and_run_requests(
            prompts=(beam.get_prompt() for beam in active_beams),
            params=active_params,
            output_type=RequestOutput,
            lora_requests=[beam.lora_request for beam in active_beams],
            use_tqdm=False,
        )

        output: list[RequestOutput | None] = [None] * len(all_beams)
        for idx, active_idx in enumerate(active_indices):
            output[active_idx] = active_output[idx]

        # Logprobs are computed from raw logits before
        # allowed_token_ids masking, so they may contain
        # tokens outside the grammar's allowed set. This filtering is also
        # the only grammar enforcement for beams whose allowed set exceeds
        # the engine-side allowed_token_ids cap.
        allowed_sets: list[set[int] | None] = [None] * len(all_beams)
        if beam_entries is not None:
            for i, entry in enumerate(beam_entries):
                if entry is not None:
                    allowed_sets[i] = set(entry[1])

        for (start, end), instance in zip(instance_start_and_end, instances_batch):
            instance_output = output[start:end]
            if any(
                result is not None and result.outputs[0].finish_reason == "abort"
                for result in instance_output
            ):
                for beam, result in zip(all_beams[start:end], instance_output):
                    if result is not None:
                        beam.finish_reason = "abort"
                        instance.completed.append(beam)
                instance.beams = []
                continue

            instance_new_beams = []
            for i in range(start, end):
                current_beam = all_beams[i]
                result = output[i]

                if result is None:
                    continue

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    allowed = allowed_sets[i]
                    for token_id, logprob_obj in logprobs.items():
                        if allowed is not None and token_id not in allowed:
                            continue
                        new_beam = BeamSearchSequence(
                            current_beam.orig_prompt,
                            tokens=current_beam.tokens + [token_id],
                            logprobs=current_beam.logprobs + [logprobs],
                            lora_request=current_beam.lora_request,
                            cum_logprob=current_beam.cum_logprob + logprob_obj.logprob,
                        )

                        if token_id == eos_token_id and not ignore_eos:
                            new_beam.finish_reason = "stop"
                            new_beam.stop_reason = eos_token_id
                            instance.completed.append(new_beam)
                        else:
                            instance_new_beams.append(new_beam)
            sorted_beams = sorted(
                instance_new_beams,
                key=sort_beams_key,
                reverse=True,
            )
            instance.beams = sorted_beams[:beam_width]

        return False

    def _init_beam_search_structured_output(
        self,
        structured_outputs: StructuredOutputsParams,
        tokenizer: TokenizerLike,
    ) -> BeamSearchSOState:
        """Initialize the structured output backend for beam search.

        For CHOICE requests a token-level prefix trie is built and returned
        instead of a grammar backend; it is choice-count independent and much
        faster for large choice sets. Returns a :class:`BeamSearchSOState`
        where exactly one of ``backend`` or ``trie`` is set.
        """
        # Capture the request type BEFORE validation: the shared validator
        # rewrites a CHOICE request to an equivalent grammar in place, which
        # would otherwise hide the CHOICE type and make this fast path
        # unreachable for the default "auto"/"xgrammar" backend.
        vllm_config = self.llm_engine.vllm_config
        key = get_structured_output_key(structured_outputs)
        validate_and_resolve_beam_search_so(structured_outputs, vllm_config, tokenizer)

        if key[0] == StructuredOutputOptions.CHOICE:
            choices = json.loads(key[1])
            trie = ChoiceTrie.build(
                choices, tokenizer, eos_token_id=tokenizer.eos_token_id
            )
            return BeamSearchSOState(None, None, None, trie)

        so_config = vllm_config.structured_outputs_config
        if so_config is None:
            raise ValueError(
                "structured_outputs_config is required for beam search "
                "with structured outputs"
            )

        # `_backend` is now a concrete backend resolved by the validator above.
        backend_name = structured_outputs._backend
        vocab_size = self.model_config.get_vocab_size()

        backend: StructuredOutputBackend
        if backend_name == "xgrammar":
            from vllm.v1.structured_output.backend_xgrammar import (
                XgrammarBackend,
            )

            backend = XgrammarBackend(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
            )
        elif backend_name == "guidance":
            from vllm.v1.structured_output.backend_guidance import (
                GuidanceBackend,
            )

            backend = GuidanceBackend(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
            )
        elif backend_name == "outlines":
            from vllm.v1.structured_output.backend_outlines import (
                OutlinesBackend,
            )

            backend = OutlinesBackend(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
            )
        elif backend_name == "lm-format-enforcer":
            from vllm.v1.structured_output.backend_lm_format_enforcer import (
                LMFormatEnforcerBackend,
            )

            backend = LMFormatEnforcerBackend(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
            )
        else:
            raise ValueError(f"Unsupported structured output backend: {backend_name}")

        bitmask = backend.allocate_token_bitmask(1)

        return BeamSearchSOState(backend, key, bitmask, None)

    def _build_beam_sampling_params(
        self,
        beams: list[BeamSearchSequence],
        base_params: SamplingParams,
        backend: StructuredOutputBackend,
        structured_output_key: tuple,
        bitmask: torch.Tensor,
    ) -> list[tuple[SamplingParams, list[int]] | None]:
        """Build per-beam SamplingParams and allowed token IDs from grammar.

        Returns None for beams where the grammar has terminated.
        """
        vocab_size = self.model_config.get_vocab_size()
        request_type, grammar_spec = structured_output_key
        result: list[tuple[SamplingParams, list[int]] | None] = []

        for beam in beams:
            # Fresh grammar per beam, replaying generated tokens.
            # Backends don't support cloning grammar state, so
            # replay is needed to reconstruct the FSM position.
            grammar = backend.compile_grammar(request_type, grammar_spec)
            assert beam.orig_prompt["type"] != "enc_dec"
            prompt_len = len(beam.orig_prompt["prompt_token_ids"])
            generated_tokens = beam.tokens[prompt_len:]

            if generated_tokens:
                grammar.accept_tokens("beam", generated_tokens)

            if grammar.is_terminated():
                result.append(None)
                continue

            grammar.fill_bitmask(bitmask, 0)
            allowed_ids = bitmask_to_token_ids(bitmask[0], vocab_size)

            if not allowed_ids:
                result.append(None)
                continue

            # The engine caps the size of allowed_token_ids. While the
            # grammar still allows more tokens than the cap (e.g. inside
            # free-form strings), skip the engine-side constraint and rely
            # on the logprobs filtering in _beam_search_step instead.
            beam_params = SamplingParams(
                logprobs=base_params.logprobs,
                max_tokens=1,
                temperature=base_params.temperature,
                watermarking=base_params.watermarking,
                detokenize=False,
                allowed_token_ids=(
                    allowed_ids
                    if len(allowed_ids) <= _MAX_NUM_ALLOWED_TOKEN_IDS
                    else None
                ),
                skip_clone=True,
            )
            result.append((beam_params, allowed_ids))

        return result

    def _build_trie_sampling_params(
        self,
        beams: list[BeamSearchSequence],
        base_params: SamplingParams,
        trie: ChoiceTrie,
    ) -> list[tuple[SamplingParams, list[int]] | None]:
        """Build per-beam SamplingParams and allowed token IDs from a trie.

        Returns None for beams that have reached a trie terminal or gone
        off-trie; both are dropped by the caller. The trie walk is
        O(generation_length) per beam and independent of the number of
        choices, so this stays fast even for very large choice sets.
        """
        result: list[tuple[SamplingParams, list[int]] | None] = []
        for beam in beams:
            allowed_ids = get_trie_allowed_token_ids(beam, trie)
            if not allowed_ids:
                result.append(None)
                continue

            # The engine caps the size of allowed_token_ids. When a trie node
            # branches wider than the cap (e.g. the root over thousands of
            # choices), skip the engine-side constraint and rely on the
            # logprobs filtering in _beam_search_step instead.
            beam_params = SamplingParams(
                logprobs=base_params.logprobs,
                max_tokens=1,
                temperature=base_params.temperature,
                detokenize=False,
                allowed_token_ids=(
                    allowed_ids
                    if len(allowed_ids) <= _MAX_NUM_ALLOWED_TOKEN_IDS
                    else None
                ),
                skip_clone=True,
            )
            result.append((beam_params, allowed_ids))

        return result
