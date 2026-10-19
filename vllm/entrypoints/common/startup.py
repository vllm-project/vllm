# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import os
from argparse import Namespace
from logging import Logger
from string import Template
from typing import Any

from vllm import EngineArgs, envs
from vllm.logger import current_formatter_type, init_logger

logger = init_logger(__name__)


def log_version_and_model(lgr: Logger, version: str, model_name: str) -> None:
    if envs.VLLM_DISABLE_LOG_LOGO or (formatter := current_formatter_type(lgr)) is None:
        message = "vLLM server version %s, serving model %s"
    else:
        logo_template = Template(
            "\n       ${w}█     █     █▄   ▄█${r}\n"
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}  version ${w}%s${r}\n"
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}  model   ${w}%s${r}\n"
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}\n"
        )
        colors = {
            "w": "\033[1m",  # bold, default foreground
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
        }
        if formatter != "color":
            # monochrome logo (no ansi escape codes)
            colors = dict.fromkeys(colors, "")

        message = logo_template.substitute(colors)

    lgr.info(message, version, model_name)


VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)


# Fields whose values must never be logged verbatim.
_SENSITIVE_ARG_FIELDS = frozenset({"api_key", "hf_token", "watermark_config"})


def redact_sensitive_args(args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `args` with sensitive values redacted for logging."""
    if not any(key in _SENSITIVE_ARG_FIELDS for key in args):
        return args
    return {
        key: ("***" if key in _SENSITIVE_ARG_FIELDS else value)
        for key, value in args.items()
    }


def get_non_default_args(args: Namespace | EngineArgs) -> dict[str, Any]:
    from vllm.entrypoints.common.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    non_default_args = {}

    # Handle Namespace
    if isinstance(args, Namespace):
        parser = make_arg_parser(FlexibleArgumentParser())
        for arg, default in vars(parser.parse_args([])).items():
            if default != getattr(args, arg):
                non_default_args[arg] = getattr(args, arg)

    # Handle EngineArgs instance
    elif isinstance(args, EngineArgs):
        default_args = EngineArgs(model=args.model)  # Create default instance
        for field in dataclasses.fields(args):
            current_val = getattr(args, field.name)
            default_val = getattr(default_args, field.name)
            if current_val != default_val:
                non_default_args[field.name] = current_val
        if default_args.model != EngineArgs.model:
            non_default_args["model"] = default_args.model
    else:
        raise TypeError(
            "Unsupported argument type. Must be Namespace or EngineArgs instance."
        )

    return non_default_args


def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", redact_sensitive_args(non_default_args))


def _jsonify_arg_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            key: _jsonify_arg_value(val)
            for key, val in dataclasses.asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _jsonify_arg_value(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonify_arg_value(item) for item in value]
    if (model_dump := getattr(value, "model_dump", None)) is not None:
        return _jsonify_arg_value(model_dump(mode="json"))
    if (to_dict := getattr(value, "dict", None)) is not None:
        return _jsonify_arg_value(to_dict())
    return repr(value)


def jsonify_non_default_args(
    args: Namespace | EngineArgs,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    non_default_args = get_non_default_args(args)
    if exclude is not None:
        for key in exclude:
            non_default_args.pop(key, None)

    return {key: _jsonify_arg_value(value) for key, value in non_default_args.items()}


def cli_env_setup():
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
