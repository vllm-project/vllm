# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the shared multimodal input helpers."""

from vllm.multimodal.feature_utils import (
    extract_mm_features as extract_mm_features,
)
from vllm.multimodal.feature_utils import (
    merge_mm_kwargs_items as merge_mm_kwargs_items,
)
from vllm.multimodal.feature_utils import (
    mm_kwargs_from_features as mm_kwargs_from_features,
)
from vllm.multimodal.feature_utils import (
    placeholder_ranges_from_engine_input as placeholder_ranges_from_engine_input,
)
