# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.platforms import current_platform


def is_compute_capability_supported(
    min_capability: int,
    compute_capability: int | None = None,
    max_capability: int | None = None,
) -> tuple[bool, str | None]:
    """Utility function to check if the current platform's compute capability
    meets the minimum required capability.

    Args:
        min_capability (int): The minimum required compute capability.
        compute_capability (int | None): The compute capability to check.
            If None, it will be obtained from the current platform.
        max_capability (int | None): The maximum allowed compute capability.
            If None, no upper bound check is performed.

    Returns:
        tuple[bool, str | None]: A tuple where the first element indicates
        whether the capability is supported, and the second element is an
        optional error message if not supported.
    """
    if compute_capability is None:
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc.major * 10 + _cc.minor

    if compute_capability is not None and compute_capability < min_capability:
        return (
            False,
            f"requires capability >= {min_capability}, got {compute_capability}",
        )

    if (
        compute_capability is not None
        and max_capability is not None
        and compute_capability > max_capability
    ):
        return (
            False,
            f"requires capability <= {max_capability}, got {compute_capability}",
        )

    return True, None
