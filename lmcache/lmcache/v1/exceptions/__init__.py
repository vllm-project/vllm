# SPDX-License-Identifier: Apache-2.0
"""
Custom exceptions for LMCache v1.
"""


class IrrecoverableException(Exception):
    """
    Exception raised when an irrecoverable error occurs.

    This exception indicates that the system has encountered an error
    that cannot be recovered from, and the health monitor should stop
    checking and propagate the error up.
    """

    pass
