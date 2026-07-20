# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any


def _commit_from_version_tuple(version_tuple: tuple[Any, ...]) -> str:
    """Extract a git commit hash from a setuptools-scm version tuple.

    Dev builds typically end with a local segment like ``gabcdef`` or
    ``gabcdef.dYYYYMMDD``. Return the short hash without the leading ``g``.
    """
    if not version_tuple:
        return ""
    local = version_tuple[-1]
    if not isinstance(local, str) or not local.startswith("g"):
        return ""
    # Strip the leading 'g' and any trailing local date segment.
    body = local[1:]
    if "." in body:
        body = body.split(".", 1)[0]
    # setuptools-scm short hashes are hex; reject empty/malformed leftovers.
    if not body or any(c not in "0123456789abcdef" for c in body.lower()):
        return ""
    return body


try:
    from ._version import __version__, __version_tuple__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}", RuntimeWarning, stacklevel=2)

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)

# Public commit id for fix-containment checks (empty when unavailable).
__commit__ = _commit_from_version_tuple(__version_tuple__)


def _prev_minor_version_was(version_str):
    """Check whether a given version matches the previous minor version.

    Return True if version_str matches the previous minor version.

    For example - return True if the current version if 0.7.4 and the
    supplied version_str is '0.6'.

    Used for --show-hidden-metrics-for-version.
    """
    # Match anything if this is a dev tree
    if __version_tuple__[0:2] == (0, 0):
        return True

    # Note - this won't do the right thing when we release 1.0!
    assert __version_tuple__[0] == 0
    assert isinstance(__version_tuple__[1], int)
    return version_str == f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"


def _prev_minor_version():
    """For the purpose of testing, return a previous minor version number."""
    # In dev tree, this will return "0.-1", but that will work fine"
    assert isinstance(__version_tuple__[1], int)
    return f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"
