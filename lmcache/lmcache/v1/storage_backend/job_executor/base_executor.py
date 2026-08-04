# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Awaitable, Callable
import abc


class BaseJobExecutor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def submit_job(
        self,
        fn: Callable[..., Awaitable[Any]] | Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Submit a job to the executor.

        :param fn: The function to execute.
        :param args: The positional arguments to pass to the function.
        :param kwargs: The keyword arguments to pass to the function (e.g., priority).

        :return: Return type aligned with the function being executed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """
        Clean up the executor, optionally waiting for currently running jobs to finish.

        :param wait: If True, wait for currently running jobs to finish before
        returning.
        """
        raise NotImplementedError
