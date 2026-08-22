# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch
from transformers import SiglipModel

from vllm.model_executor.models.siglip import SiglipEmbeddingModel

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ...utils import check_embeddings_close

HF_TEXT_PROMPTS = [
    "a photo of a stop sign",
    "a photo of a cherry blossom",
]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": "",
        "cherry_blossom": "",
    }
)

MODELS = [
    "google/siglip-base-patch16-224",
    "google/siglip2-base-patch16-224",
    # Different image embedding dim than text_config.hidden_size
    "google/siglip2-giant-opt-patch16-384",
]


def _reference_flip_sequences_by_position_ids(
    features: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    if len(features) <= 1:
        return features

    boundary_mask = position_ids[1:] <= position_ids[:-1]
    boundary_mid = torch.where(boundary_mask)[0] + 1
    boundary_indices = torch.cat(
        [
            torch.zeros(1, dtype=boundary_mid.dtype, device=features.device),
            boundary_mid,
            torch.full(
                (1,),
                len(features),
                dtype=boundary_mid.dtype,
                device=features.device,
            ),
        ]
    )

    lengths = boundary_indices[1:] - boundary_indices[:-1]
    starts = boundary_indices[:-1]
    ends = boundary_indices[1:]
    sequence_ids = torch.arange(
        len(lengths), dtype=boundary_mid.dtype, device=features.device
    ).repeat_interleave(lengths)
    current_positions = torch.arange(
        len(features), dtype=boundary_mid.dtype, device=features.device
    )
    flip_indices = starts[sequence_ids] + ends[sequence_ids] - (1 + current_positions)
    return features[flip_indices]


def _flip_sequences_by_position_ids(
    features: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    return SiglipEmbeddingModel._flip_sequences_by_position_ids(
        None, features, position_ids
    )


@pytest.mark.parametrize(
    "position_ids",
    [
        [],
        [0],
        [0, 1, 2, 3],
        [0, 1, 0, 1, 2],
        [3, 2, 1, 0, 2, 1, 0],
        [0, 0, 0],
    ],
)
@pytest.mark.skip_global_cleanup
def test_flip_sequences_by_position_ids_matches_reference(
    position_ids: list[int],
) -> None:
    position_ids_tensor = torch.tensor(position_ids, dtype=torch.long)
    features = torch.arange(len(position_ids) * 3, dtype=torch.float32).reshape(
        len(position_ids), 3
    )

    actual = _flip_sequences_by_position_ids(features, position_ids_tensor)
    expected = _reference_flip_sequences_by_position_ids(features, position_ids_tensor)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skip_global_cleanup
def test_flip_sequences_by_position_ids_matches_reference_randomized() -> None:
    generator = torch.Generator().manual_seed(0)

    for _ in range(100):
        remaining_tokens = 64
        sequence_lengths = []
        while remaining_tokens:
            sequence_length = int(
                torch.randint(
                    1,
                    min(remaining_tokens, 8) + 1,
                    (1,),
                    generator=generator,
                )
            )
            sequence_lengths.append(sequence_length)
            remaining_tokens -= sequence_length

        position_ids = torch.cat([torch.arange(length) for length in sequence_lengths])
        features = torch.randn(position_ids.numel(), 5, generator=generator)

        actual = _flip_sequences_by_position_ids(features, position_ids)
        expected = _reference_flip_sequences_by_position_ids(features, position_ids)

        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="requires torch.compile")
@pytest.mark.skip_global_cleanup
def test_flip_sequences_by_position_ids_supports_torch_compile() -> None:
    position_ids = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)
    features = torch.randn(position_ids.numel(), 4)

    compiled_flip = torch.compile(_flip_sequences_by_position_ids, fullgraph=True)

    actual = compiled_flip(features, position_ids)
    expected = _reference_flip_sequences_by_position_ids(features, position_ids)

    torch.testing.assert_close(actual, expected)


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_cases: list[tuple[list[str], PromptImageInput, dict[str, Any]]],
    model: str,
    *,
    dtype: str,
) -> None:
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        enforce_eager=True,
        max_model_len=64,
        gpu_memory_utilization=0.7,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.embed(
                input_texts,
                images=input_images,
                tokenization_kwargs=tokenization_kwargs,
            )
            for input_texts, input_images, tokenization_kwargs in input_cases
        ]

        texts = [HF_TEXT_PROMPTS[0]]
        images = [input_cases[1][1][0]]
        with pytest.raises(ValueError, match="not both"):
            vllm_model.embed(texts, images=images)

        vllm_model.embed(texts)
        vllm_model.embed([""], images=images)

    with hf_runner(model, dtype=dtype, auto_cls=SiglipModel) as hf_model:
        hf_outputs_per_case = []
        for input_texts, input_images, tokenization_kwargs in input_cases:
            all_inputs = hf_model.get_inputs(
                input_texts,
                images=input_images,
                tokenization_kwargs=tokenization_kwargs,
            )

            hf_outputs = []
            for inputs in all_inputs:
                inputs = hf_model.wrap_device(inputs)

                if "pixel_values" in inputs:
                    pooled_output = hf_model.model.get_image_features(
                        pixel_values=inputs.pixel_values,
                    )
                else:
                    pooled_output = hf_model.model.get_text_features(
                        input_ids=inputs.input_ids,
                    )

                if not isinstance(pooled_output, torch.Tensor):
                    pooled_output = pooled_output.pooler_output
                pooled_output = pooled_output.squeeze(0)
                hf_outputs.append(pooled_output.tolist())

            hf_outputs_per_case.append(hf_outputs)

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_embeddings_close(
            embeddings_0_lst=hf_outputs,
            embeddings_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
def test_models(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    text_images = [None] * len(HF_TEXT_PROMPTS)
    images = [asset.pil_image for asset in image_assets]
    input_cases = [
        (
            HF_TEXT_PROMPTS,
            text_images,
            {
                "padding": "max_length",
                "max_length": 64,
            },
        ),
        (HF_IMAGE_PROMPTS, images, {}),
    ]

    _run_test(
        hf_runner,
        vllm_runner,
        input_cases,  # type: ignore[arg-type]
        model,
        dtype=dtype,
    )
