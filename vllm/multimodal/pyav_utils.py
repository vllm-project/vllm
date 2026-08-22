# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

LibAVFormatVersion = tuple[int, int, int]


class UnsafePyAVError(ValueError):
    """Raised when PyAV uses an FFmpeg build with unsafe IAMF parsing."""


def is_safe_iamf_libavformat_version(version: LibAVFormatVersion) -> bool:
    """Return whether libavformat contains the IAMF scalable-layer fix."""
    major, minor, micro = version
    if major < 61:
        return True
    if major == 61:
        return version >= (61, 7, 102)
    if major == 62:
        return (minor == 3 and micro >= 102) or version >= (62, 6, 103)
    return True


def require_safe_pyav_stack(
    library_versions: Mapping[str, LibAVFormatVersion],
) -> None:
    """Reject PyAV container parsing on FFmpeg builds with unsafe IAMF parsing."""
    libavformat_version = library_versions.get("libavformat")
    if libavformat_version is not None and is_safe_iamf_libavformat_version(
        libavformat_version
    ):
        return

    raise UnsafePyAVError(
        "PyAV container decoding requires an FFmpeg build with the IAMF parser fix. "
        "Upgrade PyAV to >= 17.1.0 or rebuild it against FFmpeg 7.1.4, 8.0.2, "
        "or a later fixed release."
    )
