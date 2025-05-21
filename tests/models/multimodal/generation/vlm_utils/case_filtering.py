# SPDX-License-Identifier: Apache-2.0
"""Utils for determining which subset of model tests belong to a specific
modality, getting all combinations (similar to pytest's parametrization),
handling multimodal placeholder substitution, and so on.
"""
import itertools
from collections import OrderedDict
from collections.abc import Iterable

import pytest

from .types import (EMBEDDING_SIZE_FACTORS, ExpandableVLMTestArgs,
                    ImageSizeWrapper, SizeType, VLMTestInfo, VLMTestType)


def get_filtered_test_settings(
        test_settings: dict[str, VLMTestInfo], test_type: VLMTestType,
        new_proc_per_test: bool) -> dict[str, VLMTestInfo]:
    """Given the dict of potential test settings to run, return a subdict
    of tests who have the current test type enabled with the matching val for
    fork_per_test.
    """

    def matches_test_type(test_info: VLMTestInfo, test_type: VLMTestType):
        return test_info.test_type == test_type or (
            isinstance(test_info.test_type, Iterable)
            and test_type in test_info.test_type)

    matching_tests = {}
    for test_name, test_info in test_settings.items():
        # Otherwise check if the test has the right type & keep if it does
        if matches_test_type(test_info, test_type):
            # Embedding tests need to have a conversion func in their test info
            if matches_test_type(test_info, VLMTestType.EMBEDDING):
                assert test_info.convert_assets_to_embeddings is not None
            # Custom test inputs need to explicitly define the mm limit/inputs
            if matches_test_type(test_info, VLMTestType.CUSTOM_INPUTS):
                assert (test_info.custom_test_opts is not None
                        and isinstance(test_info.custom_test_opts, Iterable))
            # For all types besides custom inputs, we need a prompt formatter
            else:
                assert test_info.prompt_formatter is not None

            # Everything looks okay; keep if this is has correct proc handling
            if (test_info.distributed_executor_backend
                    is not None) == new_proc_per_test:
                matching_tests[test_name] = test_info

    return matching_tests


def get_parametrized_options(test_settings: dict[str, VLMTestInfo],
                             test_type: VLMTestType,
                             create_new_process_for_each_test: bool):
    """Converts all of our VLMTestInfo into an expanded list of parameters.
    This is similar to nesting pytest parametrize calls, but done directly
    through an itertools product so that each test can set things like
    size factors etc, while still running in isolated test cases.
    """
    matching_tests = get_filtered_test_settings(
        test_settings, test_type, create_new_process_for_each_test)

    # Ensure that something is wrapped as an iterable it's not already
    ensure_wrapped = lambda e: e if isinstance(e, (list, tuple)) else (e, )

    def get_model_type_cases(model_type: str, test_info: VLMTestInfo):
        # This is essentially the same as nesting a bunch of mark.parametrize
        # decorators, but we do it programmatically to allow overrides for on
        # a per-model basis, while still being able to execute each of these
        # as individual test cases in pytest.
        iter_kwargs = OrderedDict([
            ("model", ensure_wrapped(test_info.models)),
            ("max_tokens", ensure_wrapped(test_info.max_tokens)),
            ("num_logprobs", ensure_wrapped(test_info.num_logprobs)),
            ("dtype", ensure_wrapped(test_info.dtype)),
            ("distributed_executor_backend",
             ensure_wrapped(test_info.distributed_executor_backend)),
        ])

        # num_frames is video only
        if test_type == VLMTestType.VIDEO:
            iter_kwargs["num_video_frames"] = ensure_wrapped(
                test_info.num_video_frames)

        # No sizes passed for custom inputs, since inputs are directly provided
        if test_type not in (VLMTestType.CUSTOM_INPUTS, VLMTestType.AUDIO):
            wrapped_sizes = get_wrapped_test_sizes(test_info, test_type)
            if wrapped_sizes is None:
                raise ValueError(
                    f"Sizes must be set for test type {test_type}")
            iter_kwargs["size_wrapper"] = wrapped_sizes

        #Otherwise expand the custom test options instead
        elif test_type == VLMTestType.CUSTOM_INPUTS:
            if test_info.custom_test_opts is None:
                raise ValueError("Test has type CUSTOM_INPUTS, but none given")
            iter_kwargs["custom_test_opts"] = test_info.custom_test_opts

        # yapf: disable
        # Wrap all model cases in a pytest parameter & pass marks through
        return [
            pytest.param(
                model_type,
                ExpandableVLMTestArgs(
                    **{k: v for k, v in zip(iter_kwargs.keys(), case)}
                ),
                marks=test_info.marks if test_info.marks is not None else []
            ) for case in list(itertools.product(*iter_kwargs.values()))
        ]
        # yapf: enable

    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    cases_by_model_type = [
        get_model_type_cases(model_type, test_info)
        for model_type, test_info in matching_tests.items()
    ]
    return list(itertools.chain(*cases_by_model_type))


def get_wrapped_test_sizes(
        test_info: VLMTestInfo,
        test_type: VLMTestType) -> tuple[ImageSizeWrapper, ...]:
    """Given a test info which may have size factors or fixed sizes, wrap them
    and combine them into an iterable, each of which will be used in parameter
    expansion.

    Args:
        test_info: Test configuration to be expanded.
        test_type: The type of test being filtered for.
    """
    # If it is an embedding test, we always use the EMBEDDING_SIZE_FACTORS
    if test_type == VLMTestType.EMBEDDING:
        return tuple([
            ImageSizeWrapper(type=SizeType.SIZE_FACTOR, data=factor)
            for factor in EMBEDDING_SIZE_FACTORS
        ])
    # Audio and Custom inputs have preprocessed inputs
    elif test_type in (VLMTestType.AUDIO, VLMTestType.CUSTOM_INPUTS):
        return tuple()

    size_factors = test_info.image_size_factors \
        if test_info.image_size_factors else []
    fixed_sizes = test_info.image_sizes \
        if test_info.image_sizes else []

    wrapped_factors = [
        ImageSizeWrapper(type=SizeType.SIZE_FACTOR, data=factor)
        for factor in size_factors
    ]

    wrapped_sizes = [
        ImageSizeWrapper(type=SizeType.FIXED_SIZE, data=size)
        for size in fixed_sizes
    ]

    return tuple(wrapped_factors + wrapped_sizes)
