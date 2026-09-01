# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard access to the optional Mooncake TransferEngine dependency.

Keeping the import check here lets metadata and configuration modules remain
importable in environments that do not install Mooncake.
"""

_MOONCAKE_IMPORT_ERROR: ImportError | None
try:
    from mooncake.engine import TransferEngine as _TransferEngine  # noqa: F401
except ImportError as e:
    _MOONCAKE_IMPORT_ERROR = e
else:
    _MOONCAKE_IMPORT_ERROR = None


def ensure_mooncake_available() -> None:
    """Raise a user-facing error when Mooncake is unavailable.

    Raises:
        ImportError: If ``mooncake-transfer-engine`` cannot be imported.
    """
    if _MOONCAKE_IMPORT_ERROR is not None:
        raise ImportError(
            "Install mooncake-transfer-engine (see "
            "https://github.com/kvcache-ai/Mooncake ) to use ECMooncakeConnector."
        ) from _MOONCAKE_IMPORT_ERROR
