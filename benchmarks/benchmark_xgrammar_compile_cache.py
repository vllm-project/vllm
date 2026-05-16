# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro benchmark for xgrammar compiled-grammar cache reuse.

This measures the time spent in ``XgrammarBackend.compile_grammar`` for the
same structured-output specification under two conditions:

- cold compile: clear the backend cache before every call
- hot compile: reuse the same cached compiled grammar across calls

Usage:
    python benchmarks/benchmark_xgrammar_compile_cache.py \
        --tokenizer meta-llama/Llama-3.2-1B-Instruct \
        --iterations 200
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from vllm.tokenizers import get_tokenizer
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend


DEFAULT_GRAMMAR = """
root ::= select_statement
select_statement ::= "SELECT " column " FROM " table " WHERE " condition
column ::= "col_1" | "col_2"
table ::= "table_1" | "table_2"
condition ::= column " = " number
number ::= "1" | "2"
""".strip()

DEFAULT_REGEX = r"\w+@\w+\.com\n"


def _build_backend(tokenizer_name: str, trust_remote_code: bool) -> XgrammarBackend:
    tokenizer = get_tokenizer(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    vllm_config = SimpleNamespace(
        structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
        speculative_config=None,
    )
    return XgrammarBackend(
        vllm_config=vllm_config,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
    )


def _resolve_spec(args: argparse.Namespace) -> tuple[StructuredOutputOptions, str]:
    if args.request_type == "json":
        schema_path = Path(args.schema_path)
        return StructuredOutputOptions.JSON, schema_path.read_text()
    if args.request_type == "json-object":
        return StructuredOutputOptions.JSON_OBJECT, ""
    if args.request_type == "grammar":
        return StructuredOutputOptions.GRAMMAR, DEFAULT_GRAMMAR
    if args.request_type == "regex":
        return StructuredOutputOptions.REGEX, DEFAULT_REGEX
    raise ValueError(f"Unsupported request type: {args.request_type}")


def _resolve_specs(
    args: argparse.Namespace,
) -> tuple[StructuredOutputOptions, list[str]]:
    request_type, grammar_spec = _resolve_spec(args)
    if args.scenario == "reused":
        return request_type, [grammar_spec] * args.iterations

    if request_type != StructuredOutputOptions.JSON:
        raise ValueError(
            "The unique scenario is currently supported only for JSON schemas."
        )

    base_schema = json.loads(grammar_spec)
    specs: list[str] = []
    for index in range(args.iterations):
        schema = copy.deepcopy(base_schema)
        properties = schema.setdefault("properties", {})
        properties[f"__unique_optional_field_{index}"] = {
            "type": "string",
            "description": "Unique field added for cache-miss benchmarking.",
        }
        specs.append(json.dumps(schema))
    return request_type, specs


def _measure_compile_time(
    backend: XgrammarBackend,
    request_type: StructuredOutputOptions,
    grammar_specs: list[str],
    *,
    use_cache: bool,
) -> list[float]:
    times: list[float] = []

    if use_cache:
        backend.clear_compiled_grammar_cache()
        backend.compile_grammar(request_type, grammar_specs[0])
        backend.compiled_grammar_cache_stats(delta=True)

    for grammar_spec in grammar_specs:
        if not use_cache:
            backend.clear_compiled_grammar_cache()

        start = time.perf_counter()
        backend.compile_grammar(request_type, grammar_spec)
        end = time.perf_counter()
        times.append(end - start)

    return times


def _format_stats(times: list[float]) -> tuple[float, float]:
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or local path used to initialize xgrammar.",
    )
    parser.add_argument(
        "--request-type",
        choices=("json", "json-object", "grammar", "regex"),
        default="json",
        help="Structured-output request type to benchmark.",
    )
    parser.add_argument(
        "--scenario",
        choices=("reused", "unique"),
        default="reused",
        help=(
            "Benchmark repeated use of one schema or low-reuse unique schemas. "
            "The unique scenario is currently supported only for JSON schemas."
        ),
    )
    parser.add_argument(
        "--schema-path",
        default=str(
            Path(__file__).resolve().parent
            / "structured_schemas"
            / "structured_schema_1.json"
        ),
        help="Path to the JSON schema used when --request-type=json.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of measured iterations for each condition.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    args = parser.parse_args()

    backend = _build_backend(args.tokenizer, args.trust_remote_code)
    request_type, grammar_specs = _resolve_specs(args)

    cold_times = _measure_compile_time(
        backend,
        request_type,
        grammar_specs,
        use_cache=False,
    )
    backend.clear_compiled_grammar_cache()

    hot_times = _measure_compile_time(
        backend,
        request_type,
        grammar_specs,
        use_cache=True,
    )
    hot_stats = backend.compiled_grammar_cache_stats(delta=True)

    cold_mean, cold_std = _format_stats(cold_times)
    hot_mean, hot_std = _format_stats(hot_times)
    speedup = cold_mean / hot_mean if hot_mean > 0 else float("inf")

    print("=" * 60)
    print("XGRAMMAR COMPILED-GRAMMAR CACHE MICRO BENCHMARK")
    print("=" * 60)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Request type: {args.request_type}")
    print(f"Scenario: {args.scenario}")
    print(f"Iterations per condition: {args.iterations}")
    if request_type == StructuredOutputOptions.JSON:
        schema = json.loads(grammar_specs[0])
        print(f"JSON schema top-level keys: {sorted(schema.keys())}")
    print("=" * 60)
    print(f"Cold compile : {cold_mean * 1e6:8.2f} ± {cold_std * 1e6:6.2f} us")
    print(f"Hot compile  : {hot_mean * 1e6:8.2f} ± {hot_std * 1e6:6.2f} us")
    print(f"Speedup      : {speedup:8.2f}x")
    print(
        "Hot cache    : "
        f"hits={hot_stats.hits} total={hot_stats.total} hit_ratio={hot_stats.hit_ratio:.4f}"
    )

    backend.destroy()


if __name__ == "__main__":
    main()