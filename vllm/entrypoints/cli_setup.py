# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
from logging import Logger

VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)


def is_cli_subcommand(command_name: str) -> bool:
    return (
        next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
        == command_name
    )


def cli_env_setup(logger: Logger) -> None:
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using vLLM as a library. `spawn` will cause
    # unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
