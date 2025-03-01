# SPDX-License-Identifier: Apache-2.0
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""
    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""
    pass
