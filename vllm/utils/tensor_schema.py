# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import torch
from typing import get_type_hints, get_args, get_origin, Optional, Union, Annotated


class TensorShape:
    def __init__(self, *dims):
        self.dims = dims
    
    def __repr__(self):
        return f"TensorShape{self.dims}"


class TensorSchema:
    def __init__(self, validate: bool = True, **kwargs):
        """Initialize the schema with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

        if validate:
            self.validate()

    
    def validate(self):
        type_hints = get_type_hints(self.__class__, include_extras=True)
        shape_env = {}  # optional, used later for symbolic matching

        for field_name, field_type in type_hints.items():
            # Check if the field was provided
            if not hasattr(self, field_name) or getattr(self, field_name) is None:
                # Field is missing - check if it's optional
                # Handle Annotated types by extracting the actual type
                actual_type = field_type
                if get_origin(field_type) is Annotated:
                    args = get_args(field_type)
                    actual_type = args[0]  # First argument is the actual type
                
                # Check if the actual type is Optional (Union with None)
                if get_origin(actual_type) is Union:
                    args = get_args(actual_type)
                    if type(None) in args:  # Optional field
                        continue  # Skip validation for missing optional fields
                # If not optional, raise error
                raise ValueError(f"Required field '{field_name}' is missing")
            
            # Field exists, proceed with validation
            value = getattr(self, field_name)

            if get_origin(field_type) is not None:
                args = get_args(field_type)

                for arg in args:
                    if isinstance(arg, TensorShape):
                        expected_shape = arg.dims
                        if isinstance(value, (list, tuple)):
                            if not value:
                                raise ValueError(f"{field_name} is an empty list")

                            # Ensure all tensors in the list have the same shape
                            first = value[0]
                            for i, v in enumerate(value):
                                if not isinstance(v, torch.Tensor):
                                    raise ValueError(f"{field_name}[{i}] is not a torch.Tensor")
                                if v.shape != first.shape:
                                    raise ValueError(
                                        f"{field_name} contains inconsistent "
                                        f"shapes: {first.shape} vs {v.shape} "
                                        f"at index {i}"
                                    )

                            # Treat the list as a stacked tensor: shape = (len(list), *tensor.shape)
                            actual_shape = (len(value),) + first.shape

                        elif isinstance(value, torch.Tensor):
                            actual_shape = value.shape

                        else:
                            type_names = []
                            for arg in args:
                                if hasattr(arg, '__name__'):
                                    type_names.append(str(arg.__name__))
                                else:
                                    type_names.append(str(arg))
                                    
                            expected_types = ", ".join(type_names)
                            raise ValueError(f"{field_name} is not one of the expected types: {expected_types}")
                            
                        if len(actual_shape) != len(expected_shape):
                            raise ValueError(f"{field_name} has rank {len(actual_shape)} but expected {len(expected_shape)}")

                        for i, dim in enumerate(expected_shape):
                            if isinstance(dim, int):
                                if actual_shape[i] != dim:
                                    raise ValueError(
                                        f"{field_name} dim[{i}] expected "
                                        f"{dim}, got {actual_shape[i]}"
                                    )
                            elif isinstance(dim, str):
                                if dim in shape_env:
                                    if actual_shape[i] != shape_env[dim]:
                                        raise ValueError(
                                            f"{field_name} dim[{i}] expected "
                                            f"'{dim}'={shape_env[dim]}, got "
                                            f"{actual_shape[i]}"
                                        )
                                else:
                                    shape_env[dim] = actual_shape[i]
                            else:
                                raise TypeError(f"{field_name} dim[{i}] has unsupported type: {type(dim)}")

    
    def print_shapes(self):
        """Print TensorShape annotations for debugging."""
        print(f"Shapes in {self.__class__.__name__}:")
        type_hints = get_type_hints(self.__class__, include_extras=True)
        
        for field_name, field_type in type_hints.items():
            if get_origin(field_type) is not None:
                args = get_args(field_type)
                for arg in args:
                    if isinstance(arg, TensorShape):
                        print(f"  {field_name}: {arg}")
