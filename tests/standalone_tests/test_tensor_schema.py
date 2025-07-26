# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.phi3v import Phi3VImagePixelInputs


def test_tensor_schema_valid_tensor():
    Phi3VImagePixelInputs(
        data=torch.randn(16, 64, 3, 32, 32),
        image_sizes=torch.randint(0, 256, (16, 2)),
    )


def test_tensor_schema_optional_fields():
    Phi3VImagePixelInputs(
        data=torch.randn(16, 64, 3, 32, 32),
        image_sizes=None,
    )

    Phi3VImagePixelInputs(data=torch.randn(16, 64, 3, 32, 32), )


def test_tensor_schema_constant_dim_failure():
    with pytest.raises(ValueError, match="dim\\[2\\] expected 3, got 4"):
        Phi3VImagePixelInputs(
            data=torch.randn(16, 64, 4, 32, 32),  # dim[2] = 4
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


def test_tensor_schema_symbolic_dim_mismatch():
    with pytest.raises(ValueError, match="expected 'bn'=12, got 16"):
        Phi3VImagePixelInputs(
            data=torch.randn(12, 64, 3, 32, 32),
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


def test_tensor_schema_list_tensor_valid():
    Phi3VImagePixelInputs(
        data=[torch.randn(64, 3, 32, 32) for _ in range(16)],
        image_sizes=torch.randint(0, 256, (16, 2)),
    )


def test_tensor_schema_variable_patch_counts_valid():
    # Each image has a different number of patches (p)
    # Each tensor has shape (p, 3, 32, 32)
    data = [
        torch.randn(16, 3, 32, 32),  # p = 16
        torch.randn(32, 3, 32, 32),  # p = 32
        torch.randn(64, 3, 32, 32),  # p = 64
    ]
    image_sizes = torch.randint(0, 256, (3, 2))  # bn = 3
    Phi3VImagePixelInputs(
        data=data,
        image_sizes=image_sizes,
    )


def test_tensor_schema_tuple_tensor_valid():
    Phi3VImagePixelInputs(
        data=tuple(torch.randn(64, 3, 32, 32) for _ in range(16)),
        image_sizes=torch.randint(0, 256, (16, 2)),
    )


def test_tensor_schema_inconsistent_shapes_in_list():
    with pytest.raises(ValueError, match="contains inconsistent shapes"):
        Phi3VImagePixelInputs(
            data=[torch.randn(64, 3, 32, 32),
                  torch.randn(64, 3, 16, 16)] +
            [torch.randn(64, 3, 32, 32) for _ in range(14)],
            image_sizes=torch.randint(0, 256, (16, 2)),
        )


def test_tensor_schema_empty_list():
    with pytest.raises(ValueError, match="is an empty list"):
        Phi3VImagePixelInputs(
            data=[],
            image_sizes=torch.randint(0, 256, (0, 2)),
        )


def test_tensor_schema_validation_disabled_skips_shape_check():
    # This should NOT raise, because validation is turned off
    # This would normally fail (dim[2] should be 3, not 4)
    Phi3VImagePixelInputs(
        data=torch.randn(16, 64, 4, 32, 32),
        image_sizes=torch.randint(0, 256, (16, 2)),
        validate=False,
    )


def test_tensor_schema_with_valid_resolve_binding_dims():
    data = torch.randn(16, 64, 3, 336, 336)  # h=336, w=336
    image_sizes = torch.randint(0, 256, (16, 2))

    Phi3VImagePixelInputs(
        data=data,
        image_sizes=image_sizes,
        resolve_bindings={
            "h": 336,
            "w": 336
        },
    )


def test_tensor_schema_with_invalid_resolve_binding_dims():
    data = torch.randn(16, 64, 3, 36, 36)  # h=36, w=36
    image_sizes = torch.randint(0, 256, (16, 2))

    # Should raise because 'h' and 'w' don't match resolve bindings
    with pytest.raises(ValueError, match="dim\\[3\\] expected 336, got 36"):
        Phi3VImagePixelInputs(
            data=data,
            image_sizes=image_sizes,
            resolve_bindings={
                "h": 336,
                "w": 336
            },
        )
