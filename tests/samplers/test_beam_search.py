# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the outputs of HF and vLLM when using beam search.

Run `pytest tests/samplers/test_beam_search.py`.
"""

import pytest
from transformers import AutoModelForSeq2SeqLM

from vllm.assets.audio import AudioAsset

# FIXME(zhuohan): The test can not pass if we:
#   1. Increase max_tokens to 256.
#   2. Increase beam_width to 8.
#   3. Use the model "huggyllama/llama-7b".
MAX_TOKENS = [64]
BEAM_WIDTHS = [4]
MM_BEAM_WIDTHS = [2]
MODELS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]


@pytest.mark.skip_v1  # V1 engine does not yet support beam search
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_single_input(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    example_prompts = example_prompts[:1]
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_beam_search(
            example_prompts, beam_width, max_tokens
        )

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_beam_search(
            example_prompts, beam_width, max_tokens
        )

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_texts = hf_outputs[i]
        vllm_output_ids, vllm_output_texts = vllm_outputs[i]
        for j, (hf_text, vllm_text) in enumerate(
            zip(hf_output_texts, vllm_output_texts)
        ):
            print(f">>>{j}-th hf output:")
            print(hf_text)
            print(f">>>{j}-th vllm output:")
            print(vllm_text)
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == vllm_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}"
            )


@pytest.mark.skip_v1  # V1 engine does not yet support beam search
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_with_concurrency_limit(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    # example_prompts[1]&[3]&[7] fails due to unknown reason even without
    # concurrency limit. skip them for now.
    example_prompts = example_prompts[:8]
    concurrency_limit = 2
    assert len(example_prompts) > concurrency_limit
    with vllm_runner(model, dtype=dtype) as vllm_model:
        outputs_with_limit = vllm_model.generate_beam_search(
            example_prompts, beam_width, max_tokens, concurrency_limit=concurrency_limit
        )
        outputs_without_limit = []

        for i in range(0, len(example_prompts), concurrency_limit):
            outputs_without_limit.extend(
                vllm_model.generate_beam_search(
                    example_prompts[i : i + concurrency_limit], beam_width, max_tokens
                )
            )

    correct = True
    for i in range(len(example_prompts)):
        output_ids_with_limit, output_texts_with_limit = outputs_with_limit[i]
        output_ids_without_limit, output_texts_without_limit = outputs_without_limit[i]
        for j, (text_with_limit, text_without_limit) in enumerate(
            zip(output_texts_with_limit, output_texts_without_limit)
        ):
            print(f">>>{j}-th with limit output:")
            print(text_with_limit)
            print(f">>>{j}-th without limit output:")
            print(text_without_limit)
        assert len(output_ids_with_limit) == len(output_ids_without_limit)
        for j in range(len(output_ids_with_limit)):
            if output_ids_with_limit[j] != output_ids_without_limit[j]:
                print(
                    f"Test{i} output{j}:\n+limit: {output_ids_with_limit}\n"
                    f"-limit: {output_ids_without_limit}"
                )
                correct = False
    assert correct


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", MM_BEAM_WIDTHS)
def test_beam_search_passes_multimodal_data(
    hf_runner,
    vllm_runner,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    """Ensure that beam search passes multimodal data through correctly."""
    # NOTE - this test is primarily to check that mm data is passed to beams
    # correctly. As such, we just need to check one extra modality to make
    # sure things pass through properly.
    audios = [AudioAsset("mary_had_lamb").audio_and_sample_rate]
    model = "Qwen/Qwen2-Audio-7B-Instruct"
    audio_seq = "<|audio_bos|><|AUDIO|><|audio_eos|>"
    prompts = [
        f"<|im_start|>user\n{audio_seq}Can you transcribe this?<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
    ]

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSeq2SeqLM) as hf_model:
        audio_token_id = hf_model.config.audio_token_index
        eos_token_id = hf_model.tokenizer.eos_token_id  # <|im_end|>
        hf_outputs = hf_model.generate_beam_search(
            prompts,
            beam_width=beam_width,
            max_tokens=max_tokens,
            audios=audios,
        )

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_beam_search(
            prompts,
            beam_width=beam_width,
            max_tokens=max_tokens,
            audios=audios,
        )

    seq_with_no_audio_toks = lambda seq: [tok for tok in seq if tok != audio_token_id]

    for i in range(len(prompts)):
        hf_output_ids, hf_output_texts = hf_outputs[i]
        vllm_output_ids, vllm_output_texts = vllm_outputs[i]

        for j, (hf_text, vllm_text) in enumerate(
            zip(hf_output_texts, vllm_output_texts)
        ):
            print(f">>>{j}-th hf output [NOTE: special tokens are filtered]:")
            print(hf_text)
            print(f">>>{j}-th vllm output:")
            print(vllm_text)
        assert len(hf_output_ids) == len(vllm_output_ids)

        for j in range(len(hf_output_ids)):
            # Compare everything except for the audio tokens; we do this since
            # the IDs returned from the transformers helper expands the audio
            # token to match features, while the vLLM helper maintains the
            # single audio token in the input text
            filtered_hf_output_ids = seq_with_no_audio_toks(hf_output_ids[j])
            filtered_vllm_output_ids = seq_with_no_audio_toks(vllm_output_ids[j])

            # HF output IDs may contain the end of sequence
            if len(filtered_hf_output_ids) == len(filtered_vllm_output_ids) + 1:
                assert filtered_hf_output_ids[-1] == eos_token_id
                filtered_hf_output_ids = filtered_hf_output_ids[:-1]

            assert filtered_hf_output_ids == filtered_vllm_output_ids


# Our results are very similar to hf, but not exactly the same as. so we chose
# Levenshtein distance as a simple test metric. When `dist <= 10`, we thought
# the result is correct.
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_by_levenshtein(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    import importlib
    from collections.abc import Hashable, Sequence

    try:
        importlib.import_module("rapidfuzz")
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        return

    def levenshtein_dist(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
        """
        Levenshtein distance of two token lists.
        """

        if a is b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        return Levenshtein.distance(a, b)

    dist_threshold = 10
    example_prompts = example_prompts[:1]
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_beam_search(
            example_prompts, beam_width, max_tokens
        )

    with vllm_runner(model, dtype=dtype, enable_prefix_caching=False) as vllm_model:
        vllm_outputs = vllm_model.generate_beam_search(
            example_prompts, beam_width, max_tokens
        )

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_texts = hf_outputs[i]
        vllm_output_ids, vllm_output_texts = vllm_outputs[i]
        for j, (hf_text, vllm_text) in enumerate(
            zip(hf_output_texts, vllm_output_texts)
        ):
            print(f">>>{j}-th hf output:")
            print(hf_text)
            print(f">>>{j}-th vllm output:")
            print(vllm_text)
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j, (hf_ids, vllm_ids) in enumerate(zip(hf_output_ids, vllm_output_ids)):
            assert len(hf_ids) == len(vllm_ids)
            assert levenshtein_dist(hf_ids, vllm_ids) <= dist_threshold, (
                f"Test{i} output{j}:\nHF: {hf_ids}\nvLLM: {vllm_ids}"
            )  # use levenshtein distance to check
