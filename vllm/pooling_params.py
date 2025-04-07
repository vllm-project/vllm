# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import msgspec

from vllm.sampling_params import RequestOutputKind


class PoolingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """API parameters for pooling models. This is currently a placeholder.

    Attributes:
        additional_data: Any additional data needed for pooling.
    """
    additional_data: Optional[Any] = None
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return PoolingParams(additional_data=self.additional_data)

    def __repr__(self) -> str:
        return (f"PoolingParams("
                f"additional_metadata={self.additional_data})")

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY,\
            "For pooling output_kind has to be FINAL_ONLY"
