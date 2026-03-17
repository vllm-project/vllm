# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING
from pathlib import Path

from vllm.entrypoints.cli.local_backends import (
    build_doctor_report,
    get_runtime_profile,
)
from vllm.entrypoints.cli.local_runtime import (
    ensure_model_available,
    find_service,
    format_service_rows,
    get_pulled_model,
    iter_known_aliases,
    load_models_registry,
    print_kv,
    print_table,
    remove_pulled_model,
    resolve_model_reference,
    stop_service,
    tail_file,
)
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


logger = init_logger(__name__)


def _add_model_argument(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("model", type=str, help="Model alias, HF repo, or local path.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision when using Hugging Face repositories.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache override used when pulling models.",
    )
    return parser


def _sampling_params_from_args(args: argparse.Namespace):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def _prompt_from_args_or_stdin(args: argparse.Namespace) -> str | None:
    if args.prompt:
        return args.prompt
    if sys.stdin.isatty():
        return None
    piped = sys.stdin.read().strip()
    return piped or None


def _args_has_option(args: argparse.Namespace, option: str) -> bool:
    argv = getattr(args, "_argv", [])
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _apply_profile_defaults(args: argparse.Namespace) -> dict[str, object]:
    profile = get_runtime_profile(args.profile)
    applied: dict[str, object] = {}
    if (
        hasattr(args, "gpu_memory_utilization")
        and profile.gpu_memory_utilization is not None
        and not _args_has_option(args, "--gpu-memory-utilization")
    ):
        args.gpu_memory_utilization = profile.gpu_memory_utilization
        applied["gpu_memory_utilization"] = profile.gpu_memory_utilization
    if (
        hasattr(args, "enable_prefix_caching")
        and profile.enable_prefix_caching is not None
        and not _args_has_option(args, "--enable-prefix-caching")
        and not _args_has_option(args, "--no-enable-prefix-caching")
    ):
        args.enable_prefix_caching = profile.enable_prefix_caching
        applied["enable_prefix_caching"] = profile.enable_prefix_caching
    if (
        hasattr(args, "enforce_eager")
        and profile.enforce_eager is not None
        and not _args_has_option(args, "--enforce-eager")
    ):
        args.enforce_eager = profile.enforce_eager
        applied["enforce_eager"] = profile.enforce_eager
    return applied


def _format_doctor_kv(report) -> dict[str, object]:
    payload: dict[str, object] = {
        "os": report.os,
        "architecture": report.architecture,
        "current_platform": report.current_platform,
        "current_device_type": report.current_device_type,
        "current_device_name": report.current_device_name,
        "requested_backend": report.requested_backend,
        "selected_backend": report.selected_backend,
        "selection_reason": report.selection_reason,
    }
    if report.fallback_reason:
        payload["fallback_reason"] = report.fallback_reason
    if report.model:
        payload["model"] = report.model
        payload["profile"] = report.profile
    return payload


def _print_backend_table(report) -> None:
    rows = []
    for backend in report.backends:
        rows.append(
            {
                "backend": backend.name,
                "available": "yes" if backend.available else "no",
                "selected": "yes" if backend.selected else "",
                "tier": backend.performance_tier,
                "source": backend.source,
                "reason": backend.reason or "",
            }
        )
    print_table(
        rows,
        [
            ("backend", "BACKEND"),
            ("available", "AVAILABLE"),
            ("selected", "SELECTED"),
            ("tier", "TIER"),
            ("source", "SOURCE"),
            ("reason", "REASON"),
        ],
    )


def _run_chat(args: argparse.Namespace, llm) -> None:
    conversation = []
    if args.system_prompt:
        conversation.append({"role": "system", "content": args.system_prompt})

    prompt = _prompt_from_args_or_stdin(args)
    if prompt:
        conversation.append({"role": "user", "content": prompt})
        output = llm.chat(conversation, _sampling_params_from_args(args), use_tqdm=False)
        print(output[0].outputs[0].text)
        return

    if not args.interactive:
        raise ValueError("No prompt provided. Use `--prompt` or `--interactive`.")

    while True:
        try:
            user_prompt = input("> ")
        except EOFError:
            break
        if not user_prompt.strip():
            continue
        conversation.append({"role": "user", "content": user_prompt})
        output = llm.chat(conversation, _sampling_params_from_args(args), use_tqdm=False)
        assistant_message = output[0].outputs[0].text
        print(assistant_message)
        conversation.append({"role": "assistant", "content": assistant_message})


def _run_complete(args: argparse.Namespace, llm) -> None:
    prompt = _prompt_from_args_or_stdin(args)
    if prompt is None:
        raise ValueError("`vllm run --complete` requires `--prompt` or piped stdin.")
    outputs = llm.generate(prompt, _sampling_params_from_args(args), use_tqdm=False)
    print(outputs[0].outputs[0].text)


class PullCommand(CLISubcommand):
    name = "pull"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        resolved = resolve_model_reference(args.model, revision=args.revision)
        record = ensure_model_available(resolved, download_dir=args.download_dir)
        print_kv(
            {
                "requested": resolved.requested,
                "resolved": resolved.model,
                "source": resolved.source,
                "local_path": record["local_path"],
            }
        )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Resolve a model alias or HF repo and download it locally.",
        )
        return _add_model_argument(parser)


class RunCommand(CLISubcommand):
    name = "run"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm import LLM

        report = build_doctor_report(
            requested_backend=args.backend,
            model=args.model,
            dtype=args.dtype,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            profile=args.profile,
        )
        if args.backend != "auto" and report.selected_backend != args.backend:
            raise ValueError(
                f"Requested backend `{args.backend}` is unavailable. "
                f"{report.fallback_reason or report.selection_reason}"
            )
        profile_applied = _apply_profile_defaults(args)
        resolved = resolve_model_reference(args.model, revision=args.revision)
        record = ensure_model_available(resolved, download_dir=args.download_dir)

        logger.info(
            "Local run backend=%s profile=%s model=%s applied_defaults=%s",
            report.selected_backend,
            args.profile,
            resolved.model,
            profile_applied or "none",
        )
        if report.fallback_reason:
            logger.info("Backend fallback reason: %s", report.fallback_reason)
        if report.preflight is not None:
            logger.info("Preflight: %s", report.preflight.summary)

        llm = LLM(
            model=record["local_path"],
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
            max_model_len=args.max_model_len,
            download_dir=args.download_dir,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
        )
        if args.complete:
            _run_complete(args, llm)
        else:
            _run_chat(args, llm)

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        _add_model_argument(parser)
        parser.add_argument(
            "-p",
            "--prompt",
            type=str,
            default=None,
            help="Single prompt to run. Without this, `vllm run` opens a shell chat.",
        )
        parser.add_argument(
            "--system-prompt",
            type=str,
            default=None,
            help="Optional system prompt for chat mode.",
        )
        parser.add_argument(
            "--interactive",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Open an interactive shell chat when no prompt is provided.",
        )
        parser.add_argument(
            "--chat",
            action="store_true",
            default=False,
            help="Force chat mode explicitly.",
        )
        parser.add_argument(
            "--complete",
            action="store_true",
            default=False,
            help="Use text completion instead of chat mode.",
        )
        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=1,
            help="Number of devices to use for tensor parallelism.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="auto",
            help="Model dtype to use when loading the model.",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=None,
            help="Optional quantization hint used for load and preflight diagnostics.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="auto",
            choices=["auto", "cuda", "rocm", "xpu", "apple-metal", "cpu"],
            help="Backend preference for local runtime selection.",
        )
        parser.add_argument(
            "--profile",
            type=str,
            default="balanced",
            choices=["balanced", "throughput", "low-memory"],
            help="Local performance profile used to choose sensible defaults.",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=0.9,
            help="Fraction of GPU memory reserved by vLLM.",
        )
        parser.add_argument(
            "--enable-prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Override prefix caching behavior for local runs.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            default=False,
            help="Force eager execution and disable graph-backed execution paths.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Optional maximum context length override.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=False,
            help="Trust model-provided custom code when loading from HF.",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=512,
            help="Maximum number of output tokens to generate.",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Sampling temperature.",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=0.95,
            help="Top-p sampling cutoff.",
        )
        return parser

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Run a model directly in the terminal.",
        )
        return self.add_cli_args(parser)


class ListCommand(CLISubcommand):
    name = "ls"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        registry = load_models_registry()
        rows = []
        for model in sorted(registry["models"], key=lambda item: item["requested"]):
            rows.append(
                {
                    "requested": model["requested"],
                    "resolved": model["model"],
                    "source": model["source"],
                    "local_path": model["local_path"],
                }
            )

        if not rows:
            print("No pulled models found. Use `vllm pull <model>` first.")
            return

        print_table(
            rows,
            [
                ("requested", "REQUESTED"),
                ("resolved", "RESOLVED"),
                ("source", "SOURCE"),
                ("local_path", "LOCAL_PATH"),
            ],
        )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        return subparsers.add_parser(self.name, help="List pulled local models.")


class ListAliasCommand(ListCommand):
    name = "list"

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        return subparsers.add_parser(
            self.name,
            help="List pulled local models. Alias for `vllm ls`.",
        )


class AliasesCommand(CLISubcommand):
    name = "aliases"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        rows = []
        for alias, value in sorted(iter_known_aliases().items()):
            rows.append(
                {
                    "alias": alias,
                    "resolved": value["model"],
                    "description": value.get("description", ""),
                }
            )
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
            return
        print_table(
            rows,
            [
                ("alias", "ALIAS"),
                ("resolved", "RESOLVED"),
                ("description", "DESCRIPTION"),
            ],
        )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="List built-in and user-defined model aliases.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Print machine-readable JSON output.",
        )
        return parser


class InspectCommand(CLISubcommand):
    name = "inspect"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        resolved = resolve_model_reference(args.model, revision=args.revision)
        record = get_pulled_model(resolved)
        report = build_doctor_report(
            requested_backend=args.backend,
            model=args.model,
            dtype=args.dtype,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            profile=args.profile,
        )
        payload = {
            "requested": resolved.requested,
            "resolved": resolved.model,
            "source": resolved.source,
            "alias": resolved.alias,
            "revision": resolved.revision,
            "local_path": record["local_path"] if record else None,
            "pulled": record is not None,
            "backend": report.selected_backend,
            "selection_reason": report.selection_reason,
            "fallback_reason": report.fallback_reason,
            "profile": args.profile,
            "preflight": report.preflight.to_dict() if report.preflight else None,
            "trtllm": report.trtllm.to_dict() if report.trtllm else None,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        print_kv(payload)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Inspect model resolution and local cache metadata.",
        )
        _add_model_argument(parser)
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Print machine-readable JSON output.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="auto",
            choices=["auto", "cuda", "rocm", "xpu", "apple-metal", "cpu"],
            help="Backend preference for inspection and preflight.",
        )
        parser.add_argument(
            "--profile",
            type=str,
            default="balanced",
            choices=["balanced", "throughput", "low-memory"],
            help="Local profile used when generating diagnostics.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="auto",
            help="Dtype hint used for diagnostics.",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=None,
            help="Quantization hint used for diagnostics.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Context length hint used for preflight diagnostics.",
        )
        return parser


class DoctorCommand(CLISubcommand):
    name = "doctor"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        report = build_doctor_report(
            requested_backend=args.backend,
            model=args.model,
            dtype=args.dtype,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            profile=args.profile,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
            return

        print_kv(_format_doctor_kv(report))
        print()
        _print_backend_table(report)
        if report.available_plugins:
            print()
            print("PLUGINS")
            print_table(
                report.available_plugins,
                [("name", "NAME"), ("value", "ENTRYPOINT")],
            )
        if report.preflight is not None:
            print()
            print("PREFLIGHT")
            print_kv(
                {
                    "summary": report.preflight.summary,
                    "fit": report.preflight.fit,
                    "estimated_weight_bytes": report.preflight.estimated_weight_bytes,
                    "estimated_kv_cache_bytes": report.preflight.estimated_kv_cache_bytes,
                    "estimated_runtime_overhead_bytes": (
                        report.preflight.estimated_runtime_overhead_bytes
                    ),
                    "estimated_total_bytes": report.preflight.estimated_total_bytes,
                    "available_memory_bytes": report.preflight.available_memory_bytes,
                }
            )
        if report.trtllm is not None:
            print()
            print("TENSORRT-LLM")
            print_kv(
                {
                    "eligible": report.trtllm.eligible,
                    "environment_supported": report.trtllm.environment_supported,
                    "model_supported": report.trtllm.model_supported,
                    "flashinfer_available": report.trtllm.flashinfer_available,
                    "flashinfer_trtllm_moe_available": (
                        report.trtllm.flashinfer_trtllm_moe_available
                    ),
                    "sink_attention_supported": report.trtllm.sink_attention_supported,
                    "ragged_mla_supported": report.trtllm.ragged_mla_supported,
                    "reasons": "; ".join(report.trtllm.reasons),
                }
            )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Show backend selection, capability, and compatibility diagnostics.",
        )
        parser.add_argument(
            "model",
            nargs="?",
            default=None,
            help="Optional model alias, HF repo, or local path for preflight checks.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="auto",
            choices=["auto", "cuda", "rocm", "xpu", "apple-metal", "cpu"],
            help="Backend preference for diagnostics.",
        )
        parser.add_argument(
            "--profile",
            type=str,
            default="balanced",
            choices=["balanced", "throughput", "low-memory"],
            help="Local profile used when generating diagnostics.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="auto",
            help="Dtype hint used for model preflight.",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=None,
            help="Quantization hint used for model preflight.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Context length hint used for model preflight.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Print machine-readable JSON output.",
        )
        return parser


class StatusCommand(DoctorCommand):
    name = "status"

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = super().subparser_init(subparsers)
        parser.prog = parser.prog.replace("doctor", "status")
        return parser


class PreflightCommand(CLISubcommand):
    name = "preflight"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        report = build_doctor_report(
            requested_backend=args.backend,
            model=args.model,
            dtype=args.dtype,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            profile=args.profile,
        )
        if report.preflight is None:
            raise ValueError("`vllm preflight` requires a model reference.")
        if args.json:
            print(json.dumps(report.preflight.to_dict(), indent=2, sort_keys=True))
            return
        print_kv(
            {
                "selected_backend": report.selected_backend,
                "selection_reason": report.selection_reason,
                "summary": report.preflight.summary,
                "fit": report.preflight.fit,
                "estimated_total_bytes": report.preflight.estimated_total_bytes,
                "available_memory_bytes": report.preflight.available_memory_bytes,
            }
        )
        if report.fallback_reason:
            print(f"\nFallback: {report.fallback_reason}")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Estimate whether a model is likely to fit on the selected backend.",
        )
        _add_model_argument(parser)
        parser.add_argument(
            "--backend",
            type=str,
            default="auto",
            choices=["auto", "cuda", "rocm", "xpu", "apple-metal", "cpu"],
            help="Backend preference for diagnostics.",
        )
        parser.add_argument(
            "--profile",
            type=str,
            default="balanced",
            choices=["balanced", "throughput", "low-memory"],
            help="Local profile used when generating diagnostics.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="auto",
            help="Dtype hint used for preflight.",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=None,
            help="Quantization hint used for preflight.",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=None,
            help="Context length hint used for fit estimation.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            help="Print machine-readable JSON output.",
        )
        return parser


class PsCommand(CLISubcommand):
    name = "ps"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        rows = format_service_rows()
        if not rows:
            print("No running vLLM local services.")
            return
        print_table(
            rows,
            [
                ("name", "NAME"),
                ("pid", "PID"),
                ("port", "PORT"),
                ("status", "STATUS"),
                ("uptime_s", "UPTIME_S"),
                ("model", "MODEL"),
            ],
        )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        return subparsers.add_parser(self.name, help="List managed local services.")


class StopCommand(CLISubcommand):
    name = "stop"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        service = stop_service(args.service, force=args.force)
        print(f"Stopped {service['name']} (pid {service['pid']}).")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(self.name, help="Stop a managed local service.")
        parser.add_argument("service", type=str, help="Service name or PID.")
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Send SIGKILL instead of SIGTERM.",
        )
        return parser


class LogsCommand(CLISubcommand):
    name = "logs"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        service = find_service(args.service)
        if service is None:
            raise ValueError(f"Unknown service: {args.service}")
        tail_file(Path(service["log_path"]), follow=args.follow, lines=args.lines)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(self.name, help="Show service logs.")
        parser.add_argument("service", type=str, help="Service name or PID.")
        parser.add_argument(
            "--follow",
            action="store_true",
            default=False,
            help="Follow the log output.",
        )
        parser.add_argument(
            "--lines",
            type=int,
            default=40,
            help="Number of trailing log lines to print.",
        )
        return parser


class RemoveCommand(CLISubcommand):
    name = "rm"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        resolved = resolve_model_reference(args.model, revision=args.revision)
        removed, purged_path = remove_pulled_model(resolved, purge_cache=args.purge_cache)
        if not removed:
            print("Model metadata not found.")
            return
        if purged_path:
            print(f"Removed metadata and deleted {purged_path}.")
            return
        print("Removed local metadata entry.")

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(self.name, help="Remove pulled model metadata.")
        _add_model_argument(parser)
        parser.add_argument(
            "--purge-cache",
            action="store_true",
            default=False,
            help="Also delete the pulled snapshot path for HF-backed models.",
        )
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [
        PullCommand(),
        RunCommand(),
        ListCommand(),
        ListAliasCommand(),
        AliasesCommand(),
        InspectCommand(),
        DoctorCommand(),
        StatusCommand(),
        PreflightCommand(),
        PsCommand(),
        StopCommand(),
        LogsCommand(),
        RemoveCommand(),
    ]
