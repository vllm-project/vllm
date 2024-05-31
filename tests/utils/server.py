import logging
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import ray
import requests
import torch

from tests.utils.logging import log_banner

MAX_SERVER_START_WAIT = 15 * 60  # time (seconds) to wait for server to start


@ray.remote(num_gpus=torch.cuda.device_count())
class ServerRunner:

    def __init__(self,
                 args: List[str],
                 *,
                 logger: Optional[logging.Logger] = None):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.startup_command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            *args,
        ]

        if logger:
            log_banner(
                logger,
                "server startup command",
                shlex.join(self.startup_command),
                logging.DEBUG,
            )

        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                *args
            ],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server()

    def ready(self):
        return True

    def _wait_for_server(self):
        # run health check
        start = time.time()
        while True:
            try:
                if requests.get(
                        "http://localhost:8000/health").status_code == 200:
                    break
            except Exception as err:
                if self.proc.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > MAX_SERVER_START_WAIT:
                    raise RuntimeError(
                        "Server failed to start in time.") from err

    def __del__(self):
        if hasattr(self, "proc"):
            self.proc.terminate()


class ServerContext:
    """
    Context manager for the lifecycle of a vLLM server, wrapping `ServerRunner`.
    """

    def __init__(self, args: Dict[str, str], *,
                 logger: logging.Logger) -> None:
        """Initialize a vLLM server

        :param args: dictionary of flags/values to pass to the server command
        :param logger: logging.Logger instance to use for logging
        :param port: port the server is running on
        """
        self._args = self._args_to_list(args)
        self._logger = logger
        self.server_runner = None

    def __enter__(self):
        """Executes the server process and waits for it to become ready."""
        ray.init(ignore_reinit_error=True)
        log_banner(self._logger, "server startup command args",
                   shlex.join(self._args))

        try:
            self.server_runner = ServerRunner.remote(self._args,
                                                     logger=self._logger)
            ray.get(self.server_runner.ready.remote())
            return self.server_runner
        except Exception as e:
            self.__exit__(*sys.exc_info())
            raise e

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Stops the server if it's still running.
        """
        if self.server_runner is not None:
            del self.server_runner
        ray.shutdown()

    def _args_to_list(self, args: Dict[str, Any]) -> List[str]:
        """
        Convert a dict mapping of CLI args to a list. All values must be
        string-able.

        :param args: `dict` containing CLI flags and their values
        :return: flattened list to pass to a CLI
        """

        arg_list: List[str] = []
        for flag, value in args.items():
            # minimal error-checking: flag names must be strings
            if not isinstance(flag, str):
                error = f"all flags must be strings, got {type(flag)} ({flag})"
                raise ValueError(error)

            arg_list.append(flag)
            if value is not None:
                arg_list.append(str(value))

        return arg_list
