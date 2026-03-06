# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


class LoRAAdapterNotFoundError(Exception):
    """Exception raised when a LoRA adapter is not found.

    This exception is thrown when a requested LoRA adapter does not exist
    in the system.

    Attributes:
        message: The error message string describing the exception
    """

    message: str

    def __init__(
        self,
        message: str,
    ) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
