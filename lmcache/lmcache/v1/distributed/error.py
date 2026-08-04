# SPDX-License-Identifier: Apache-2.0

"""
Definition of errors for class APIs.
"""

# Standard
import enum


class L1Error(enum.Enum):
    """Errors for L1Manager class APIs."""

    SUCCESS = enum.auto()
    """ Operation succeeded. """

    KEY_NOT_EXIST = enum.auto()
    """ The specified key does not exist. """

    KEY_NOT_READABLE = enum.auto()
    """ The specified key exists but cannot be read """

    KEY_NOT_WRITABLE = enum.auto()
    """ The specified key exists but cannot be written """

    KEY_IN_WRONG_STATE = enum.auto()
    """ The specified key is in the wrong state for the operation. """

    KEY_IS_LOCKED = enum.auto()
    """ The specified key is locked and cannot perform the operation. """

    OUT_OF_MEMORY = enum.auto()
    """ Not enough memory to complete the operation. """


ErrorType = L1Error


def strerror(error: ErrorType) -> str:
    """Convert error code to human-readable string.

    Args:
        error (ErrorType): The error code.

    Returns:
        str: The human-readable string.
    """
    if isinstance(error, L1Error):
        if error == L1Error.SUCCESS:
            return "Operation succeeded."
        elif error == L1Error.KEY_NOT_EXIST:
            return "The specified key does not exist."
        elif error == L1Error.KEY_NOT_READABLE:
            return "The specified key exists but cannot be read."
        elif error == L1Error.KEY_NOT_WRITABLE:
            return "The specified key exists but cannot be written."
        elif error == L1Error.KEY_IN_WRONG_STATE:
            return "The specified key is in the wrong state for the operation."
        elif error == L1Error.KEY_IS_LOCKED:
            return "The specified key is locked and cannot perform the operation."
        elif error == L1Error.OUT_OF_MEMORY:
            return "Not enough memory to complete the operation."

    return "Unknown error."
