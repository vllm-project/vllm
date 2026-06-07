# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common tests for testing .generate() functionality for single / multiple
image, embedding, and video support for different VLMs in vLLM.
"""

import math
from collections import defaultdict
from pathlib import PosixPath

import pytest
from packaging.version import Version
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTextToWaveform,
)
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.platforms import current_platform
from vllm.utils.func_utils import identity

from ....conftest import (
    IMAGE_ASSETS,
    AudioTestAssets,
    HfRunner,
    ImageTestAssets,
    VideoTestAssets,
    VllmRunner,
)
from ....utils import create_new_process_for_each_test, large_gpu_mark, multi_gpu_marks
from ...utils import check_outputs_equal
from .vlm_utils import custom_inputs, model_utils, runners
from .vlm_utils.case_filtering import get_parametrized_options
from .vlm_utils.types import (
    CustomTestOptions,
    ExpandableVLMTestArgs,
    VLMTestInfo,
    VLMTestType,
)

COMMON_BROADCAST_SETTINGS = {
    "test_type": VLMTestType.IMAGE,
    "dtype": "half",
    "max_tokens": 5,
    "tensor_parallel_size": 2,
    "hf_model_kwargs": {"device_map": "auto"},
    "image_size_factors": [(0.25, 0.5, 1.0)],
    "distributed_executor_backend": (
        "ray",
        "mp",
    ),
}

### Test configuration for specific models
# NOTE: The convention of the test settings below is to lead each test key
# with the name of the model arch used in the test, using underscores in place
# of hyphens; this makes it more convenient to filter tests for a specific kind
# of model. For example....
#
# To run all test types for a specific key:
#     use the k flag to substring match with a leading square bracket; if the
#     model arch happens 

# Add a skip condition for the InternVL2 model
@pytest.mark.skipif(TRANSFORMERS_VERSION.startswith("5"), reason="InternVL2 model is not compatible with Transformers v5")
def test_single_image_models_intern_vl_test_case25():
    # existing test code here
    pass