# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Backwards compatibility - prefer importing from vllm_kernels.scalar_type
import vllm_kernels.scalar_type as scalar_type_impl

# Explicit re-exports for backwards compatibility
ScalarType = scalar_type_impl.ScalarType
NanRepr = scalar_type_impl.NanRepr
scalar_types = scalar_type_impl.scalar_types

# Re-export the ID mapping dict for any code that may use it
_SCALAR_TYPES_ID_MAP = scalar_type_impl._SCALAR_TYPES_ID_MAP
