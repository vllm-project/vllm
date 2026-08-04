# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio helpers shared by multimodal model implementations.

Kept free of heavy CUDA dependencies so it stays testable without a full
vLLM build.
"""

import torch


def batch_audio_features(
    input_features: torch.Tensor | list[torch.Tensor],
    input_features_mask: torch.Tensor | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mel features and their validity mask as batched tensors.

    Audio features are unpadded per item so that a multimodal cache entry does
    not depend on the batch it was first processed in.
    [`MultiModalFieldConfig.batched`][vllm.multimodal.inputs.MultiModalFieldConfig.batched]
    stacks items only when their shapes agree, so a batch of clips with
    differing durations reaches the model as a list and must be re-padded here.

    Padded frames are zero-filled and marked invalid. Audio towers consume the
    mask, and callers keep only masked-in positions, so padding never reaches
    the language model.

    Args:
        input_features: `(bn, s, f)` tensor, or a list of `(s_i, f)` tensors
            when clip durations differ.
        input_features_mask: Matching `(bn, s)` tensor, or list of `(s_i,)`
            tensors. `True` marks a valid frame.

    Returns:
        The `(bn, s_max, f)` features and their `(bn, s_max)` mask.
    """
    if isinstance(input_features, torch.Tensor):
        return input_features.squeeze(1), input_features_mask.squeeze(1)

    max_len = max(features.shape[0] for features in input_features)
    batched_features = input_features[0].new_zeros(
        (len(input_features), max_len, input_features[0].shape[-1])
    )
    batched_mask = input_features_mask[0].new_zeros(
        (len(input_features_mask), max_len), dtype=torch.bool
    )
    for i, (features, mask) in enumerate(
        zip(input_features, input_features_mask, strict=True)
    ):
        batched_features[i, : features.shape[0]] = features
        batched_mask[i, : mask.shape[0]] = mask
    return batched_features, batched_mask
