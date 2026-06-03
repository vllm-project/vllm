# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from typing import Any

from tqdm import tqdm

from vllm import RequestOutput, TextPrompt, TokensPrompt
from vllm.entrypoints.beam_search_utils import (
    get_beam_allowed_token_ids,
    init_beam_search_so_backend,
)
from vllm.entrypoints.offline_utils import OfflineInferenceMixin
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import (
    BeamSearchParams,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.structured_output.backend_types import StructuredOutputBackend

from .utils import (
    BeamSearchInstance,
    BeamSearchOutput,
    BeamSearchSequence,
    create_sort_beams_key_function,
)

logger = init_logger(__name__)


class BeamSearchOfflineMixin(OfflineInferenceMixin):
    """Offline inference for beam search"""

    def beam_search(
        self,
        prompts: list[TokensPrompt | TextPrompt],
        params: BeamSearchParams,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        use_tqdm: bool = False,
        concurrency_limit: int | None = None,
    ) -> list[BeamSearchOutput]:
        """
        Generate sequences using beam search.

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

        # Initialize structured output backend if requested.
        so_backend: StructuredOutputBackend | None = None
        so_key: tuple | None = None
        so_bitmask: Any = None
        if params.structured_outputs is not None:
            so_backend, so_key, so_bitmask = self._init_beam_search_so_backend(
                params.structured_outputs
            )

        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        sampling_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
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

        for prompt_start in range(0, len(instances), concurrency_limit):
            instances_batch = instances[prompt_start : prompt_start + concurrency_limit]

            token_iter = range(max_tokens)
            if use_tqdm:
                token_iter = tqdm(
                    token_iter, desc="Beam search", unit="token", unit_scale=False
                )
                logger.warning(
                    "The progress bar shows the upper bound on token steps and "
                    "may finish early due to stopping conditions. It does not "
                    "reflect instance-level progress."
                )
            for _ in token_iter:
                all_beams: list[BeamSearchSequence] = list(
                    sum((instance.beams for instance in instances_batch), [])
                )
                pos = [0] + list(
                    itertools.accumulate(
                        len(instance.beams) for instance in instances_batch
                    )
                )
                instance_start_and_end: list[tuple[int, int]] = list(
                    zip(pos[:-1], pos[1:])
                )

                if len(all_beams) == 0:
                    break

                if so_backend is not None:
                    assert so_key is not None and so_bitmask is not None
                    vocab_size = self.model_config.get_vocab_size()
                    active_beams: list[BeamSearchSequence] = []
                    active_params: list[SamplingParams] = []
                    active_indices: list[int] = []
                    for i, beam in enumerate(all_beams):
                        allowed_ids = get_beam_allowed_token_ids(
                            beam, so_backend, so_key, so_bitmask, vocab_size
                        )
                        if allowed_ids is None:
                            # Grammar terminated — mark beam completed.
                            for (s, e), inst in zip(
                                instance_start_and_end, instances_batch
                            ):
                                if s <= i < e:
                                    inst.completed.append(beam)
                                    break
                        else:
                            active_beams.append(beam)
                            active_params.append(
                                SamplingParams(
                                    logprobs=2 * beam_width,
                                    max_tokens=1,
                                    temperature=temperature,
                                    allowed_token_ids=allowed_ids,
                                    skip_clone=True,
                                )
                            )
                            active_indices.append(i)
                    if not active_beams:
                        break
                else:
                    active_beams = all_beams
                    active_indices = list(range(len(all_beams)))
                    active_params = self._params_to_seq(  # type: ignore[assignment]
                        sampling_params, len(all_beams)
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

                # Build per-beam allowed-id sets for logprob filtering.
                allowed_sets: list[set[int] | None] = [None] * len(all_beams)
                if so_backend is not None:
                    for idx, p in zip(active_indices, active_params):
                        if p.allowed_token_ids:
                            allowed_sets[idx] = set(p.allowed_token_ids)

                for (start, end), instance in zip(
                    instance_start_and_end, instances_batch
                ):
                    instance_new_beams = []
                    for i in range(start, end):
                        current_beam = all_beams[i]
                        result = output[i]

                        if result is None:
                            continue

                        if result.outputs[0].logprobs is not None:
                            # if `result.outputs[0].logprobs` is None, it means
                            # the sequence is completed because of the
                            # max-model-len or abortion. we don't need to add
                            # it to the new beams.
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
                                    cum_logprob=current_beam.cum_logprob
                                    + logprob_obj.logprob,
                                )

                                if token_id == eos_token_id and not ignore_eos:
                                    instance.completed.append(new_beam)
                                else:
                                    instance_new_beams.append(new_beam)
                    sorted_beams = sorted(
                        instance_new_beams, key=sort_beams_key, reverse=True
                    )
                    instance.beams = sorted_beams[:beam_width]

        if so_backend is not None:
            so_backend.destroy()

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(
                instance.completed, key=sort_beams_key, reverse=True
            )
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)

            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def _init_beam_search_so_backend(
        self,
        structured_outputs: StructuredOutputsParams,
    ) -> tuple[StructuredOutputBackend, tuple, Any]:
        """Initialize the structured output backend for beam search."""
        return init_beam_search_so_backend(
            vllm_config=self.llm_engine.vllm_config,
            tokenizer=self.renderer.get_tokenizer(),
            vocab_size=self.model_config.get_vocab_size(),
            structured_outputs=structured_outputs,
        )
