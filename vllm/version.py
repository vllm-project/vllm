# SPDX-License-Identifier: Apache-2.0

try:
    from ._version import __version__, __version_tuple__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)

    __version__ = "0.7.2.dev"
    __version_tuple__ = (0, 0, __version__)
