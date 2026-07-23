# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path


def _load_upstream_version_metadata() -> dict[str, str | None]:
    metadata_path = Path(__file__).resolve().parents[1] / "upstream_version.json"
    if not metadata_path.exists():
        return {
            "upstream_version": None,
            "upstream_commit": None,
        }

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "upstream_version": None,
            "upstream_commit": None,
        }

    if not isinstance(payload, dict):
        return {
            "upstream_version": None,
            "upstream_commit": None,
        }

    return {
        "upstream_version": payload.get("upstream_version"),
        "upstream_commit": payload.get("upstream_commit"),
    }


try:
    from . import _version as _generated_version

    _upstream_metadata = _load_upstream_version_metadata()
    __version__ = getattr(_generated_version, "__version__", "dev")
    __version_tuple__ = getattr(
        _generated_version,
        "__version_tuple__",
        (0, 0, __version__),
    )
    __upstream_version__ = getattr(
        _generated_version,
        "__upstream_version__",
        _upstream_metadata["upstream_version"],
    )
    __upstream_commit__ = getattr(
        _generated_version,
        "__upstream_commit__",
        _upstream_metadata["upstream_commit"],
    )
    __commit_id__ = getattr(_generated_version, "__commit_id__", None)
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}", RuntimeWarning, stacklevel=2)

    _upstream_metadata = _load_upstream_version_metadata()
    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
    __upstream_version__ = _upstream_metadata["upstream_version"]
    __upstream_commit__ = _upstream_metadata["upstream_commit"]
    __commit_id__ = None


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
