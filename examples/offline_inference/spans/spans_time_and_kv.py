# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt

# =============================================================================
# Configuration
# =============================================================================

BLOCK_SIZE = 16
CPU_CACHE_BYTES = 8_000_000_000

# Model selection
MODEL_NAMES = [
    "ldsjmdy/Tulu3-Block-FT",  # Finetuned to handle block-attention
    "ldsjmdy/Tulu3-RAG",  # Baseline
    "ibm-granite/granite-3.1-8b-instruct",  # IBMchuk
]
SELECTED_MODEL_INDEX = 2

# Block-attention tokens
PAD_TOKEN = 27  # "<"
SPAN_TOKEN_PLUS = 10  # "+"
SPAN_TOKEN_CROSS = 31  # "@"

# Paths
TEXT_FILE_1 = "example_text.txt"
TEXT_FILE_2 = "example_text_2.txt"
NUM_SEGMENTS = 4

# LLM configuration
MAX_TOKENS = 10_000
MAX_GENERATED_TOKENS = 128
GPU_MEMORY_UTIL = 0.9


# =============================================================================
# Environment Setup
# =============================================================================


def setup_environment():
    """Configure environment variables for deterministic behavior and spans."""
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # Enable block attention
    os.environ["VLLM_V1_SPANS_ENABLED"] = "True"
    os.environ["VLLM_V1_SPANS_TOKEN_PLUS"] = str(SPAN_TOKEN_PLUS)
    os.environ["VLLM_V1_SPANS_TOKEN_CROSS"] = str(SPAN_TOKEN_CROSS)

    # Debug and configuration
    os.environ["VLLM_V1_SPANS_DEBUG"] = "True"
    os.environ["VLLM_V1_SPANS_DISABLE_REPOSITION"] = "True"


def disable_spans():
    """Disable spans for baseline test."""
    os.environ["VLLM_V1_SPANS_ENABLED"] = "False"


# =============================================================================
# Utility Functions
# =============================================================================


def pad_tokens(toklist: list[int], padtok: int) -> list[int]:
    """Pad token list to block size boundary."""
    return (
        toklist[:-1]
        + [padtok] * ((BLOCK_SIZE - len(toklist)) % BLOCK_SIZE)
        + toklist[-1:]
    )


def wrap_prompt(
    prompt: list[int] | list[list[int]],
) -> TokensPrompt | list[TokensPrompt]:
    """Wrap prompt(s) in TokensPrompt objects."""
    if isinstance(prompt[0], list):
        return [TokensPrompt(prompt_token_ids=p) for p in prompt]
    return TokensPrompt(prompt_token_ids=prompt)


def load_and_segment_text(filename: str, num_segments: int = 4) -> list[str]:
    """Load a text file and segment it into multiple documents."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {filename}")

    text = path.read_text(encoding="utf-8")
    text = " ".join(text.split())  # Normalize whitespace
    text = text.replace('"', '\\"')

    segment_length = len(text) // num_segments
    segments = []
    start = 0

    for i in range(num_segments):
        if i == num_segments - 1:
            segment = text[start:]
        else:
            end = start + segment_length
            while end < len(text) and text[end] != " ":
                end += 1
            segment = text[start:end]
            start = end + 1

        segments.append(segment.strip())

    return segments


# =============================================================================
# KV Cache Dumping
# =============================================================================


def dump_kv_cache_to_file(llm, filename: str, test_name: str) -> None:
    """Dump KV cache blocks to a text file for comparison."""
    try:
        layer_idx = 19
        results = llm.llm_engine.engine_core.collective_rpc(
            lambda w: _get_kv_cache_info_from_worker(w, layer_idx)
        )

        if not results:
            print(f"  No results from collective_rpc for {test_name}")
            return

        kv_cache_info = results[0]
        if isinstance(kv_cache_info, dict) and "error" in kv_cache_info:
            print(f"  Error getting KV cache for {test_name}: {kv_cache_info['error']}")
            return

        _write_kv_cache_file(filename, test_name, kv_cache_info)
        print(f"  ✓ KV cache dumped to: {filename}")

    except Exception as e:
        print(f"  ✗ Error dumping KV cache for {test_name}: {e}")
        import traceback

        traceback.print_exc()


def _get_kv_cache_info_from_worker(worker_self, layer_idx: int) -> dict:
    """Extract KV cache info from a worker."""
    import torch

    if worker_self is None or not hasattr(worker_self, "model_runner"):
        return {"error": "No model_runner found"}

    model_runner = worker_self.model_runner
    if not hasattr(model_runner, "kv_caches"):
        return {"error": "No kv_caches found"}

    kv_cache = model_runner.kv_caches[layer_idx]
    tensor_cpu = kv_cache.detach().cpu()
    if tensor_cpu.dtype == torch.bfloat16:
        tensor_cpu = tensor_cpu.to(torch.float32)

    shape = tensor_cpu.shape
    num_blocks = shape[0] if len(shape) > 0 else 1

    block_hashes = []
    for block_idx in range(num_blocks):
        block_tensor = tensor_cpu[block_idx]
        block_bytes = block_tensor.numpy().tobytes()
        block_hash = hashlib.sha256(block_bytes).hexdigest()
        block_hashes.append(block_hash)

    layer_name = f"layer_{layer_idx}"
    return {
        layer_name: {
            "shape": str(kv_cache.shape),
            "dtype": str(kv_cache.dtype),
            "device": str(kv_cache.device),
            "num_blocks": num_blocks,
            "block_hashes": block_hashes,
        }
    }


def _write_kv_cache_file(filename: str, test_name: str, kv_cache_info: dict) -> None:
    """Write KV cache info to a file."""
    with open(filename, "w") as f:
        f.write(f"=== KV Cache Dump for {test_name} ===\n\n")
        for layer_name, layer_info in kv_cache_info.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Layer: {layer_name}\n")
            f.write(f"{'=' * 80}\n")

            if "block_hashes" in layer_info:
                f.write(f"Shape: {layer_info['shape']}\n")
                f.write(f"Dtype: {layer_info['dtype']}\n")
                f.write(f"Device: {layer_info['device']}\n")
                f.write(f"Number of KV blocks: {layer_info['num_blocks']}\n\n")
                f.write("KV Block Hashes (SHA256):\n")
                for block_idx, block_hash in enumerate(layer_info["block_hashes"]):
                    f.write(f"Block {block_idx}: {block_hash}\n")


# =============================================================================
# LLM Factory
# =============================================================================


class KVConfigBuilder:
    """Builds KVTransferConfig instances for different connector types."""

    BASE_CONFIG = {
        "kv_role": "kv_both",
    }

    @classmethod
    def segmented_prefill_offload(cls) -> KVTransferConfig:
        return KVTransferConfig(
            kv_connector="SegmentedPrefillOffloadConnector",
            **cls.BASE_CONFIG,
            kv_connector_extra_config={"cpu_bytes_to_use": CPU_CACHE_BYTES, "gap_length": 64},
            kv_connector_module_path="segmented_prefill_example_connector",
        )

    @classmethod
    def offloading(cls) -> KVTransferConfig:
        return KVTransferConfig(
            kv_connector="OffloadingConnector",
            **cls.BASE_CONFIG,
            kv_connector_extra_config={"cpu_bytes_to_use": CPU_CACHE_BYTES},
        )

    @classmethod
    def segmented_prefill_simple(cls) -> KVTransferConfig:
        return KVTransferConfig(
            kv_connector="SegmentedPrefillExampleConnector",
            **cls.BASE_CONFIG,
            kv_connector_extra_config={},
            kv_connector_module_path="segmented_prefill_example_connector_2",
        )

    @classmethod
    def example(cls) -> KVTransferConfig:
        return KVTransferConfig(
            kv_connector="ExampleConnector",
            **cls.BASE_CONFIG,
        )


@dataclass
class LLMInstance:
    """Container for LLM instance and related configuration."""

    llm: LLM
    tokenizer: Callable[[str], list[int]]
    sampling_params_preload: SamplingParams
    sampling_params_generate: SamplingParams

    @classmethod
    def create(
        cls,
        model_name: str,
        kv_config: KVTransferConfig | None = None,
        max_generated: int = MAX_GENERATED_TOKENS,
    ) -> "LLMInstance":
        """Create a new LLM instance with the given configuration."""
        samp_preload = SamplingParams(temperature=0, max_tokens=1)
        samp_generate = SamplingParams(temperature=0, max_tokens=max_generated)

        llm = LLM(
            model=model_name,
            gpu_memory_utilization=GPU_MEMORY_UTIL,
            kv_transfer_config=kv_config,
            enforce_eager=True,
            block_size=BLOCK_SIZE,
            attention_backend="TRITON_ATTN",
            enable_prefix_caching=False,
        )

        tok = llm.get_tokenizer()
        tokenizer_fn = lambda x: tok.convert_tokens_to_ids(tok.tokenize(x))

        return cls(llm, tokenizer_fn, samp_preload, samp_generate)

    def preload(self, docs: list[list[int]], prefix: list[int] | None = None) -> float:
        """Preload documents and optional prefix, returning elapsed time."""
        start = time.time()
        for doc in docs:
            self.llm.generate(
                wrap_prompt(doc), sampling_params=self.sampling_params_preload
            )
        if prefix:
            self.llm.generate(
                wrap_prompt(prefix), sampling_params=self.sampling_params_preload
            )
        return time.time() - start

    def generate(self, prompt: list[int]) -> tuple:
        """Generate response returning output and elapsed time."""
        start = time.time()
        response = self.llm.generate(
            wrap_prompt(prompt),
            sampling_params=self.sampling_params_generate, # self.sampling_params_preload
            use_tqdm=False,
        )
        elapsed = time.time() - start
        return response, elapsed

    def cleanup(self):
        """Cleanup the LLM instance."""
        del self.llm


# =============================================================================
# Document Builder
# =============================================================================


@dataclass
class DocumentSet:
    """Container for a set of segmented documents."""

    segments: list[list[int]]

    @classmethod
    def from_file(
        cls,
        filepath: str,
        tokenizer: Callable[[str], list[int]],
        num_segments: int = NUM_SEGMENTS,
    ) -> "DocumentSet":
        """Create document set from a text file."""
        segments = load_and_segment_text(filepath, num_segments)
        return cls(segments=[cls._format_doc(tokenizer(s)) for s in segments])

    @staticmethod
    def _format_doc(tokens: list[int]) -> list[int]:
        """Format a document with span token and padding."""
        return pad_tokens([SPAN_TOKEN_PLUS] + tokens, PAD_TOKEN)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> list[int]:
        return self.segments[idx]

    def __iter__(self):
        return iter(self.segments)


# =============================================================================
# Test Execution
# =============================================================================


@dataclass
class TestResult:
    """Result of a single test run."""

    name: str
    preload_time: float
    generation_time: float
    output_text: str
    model_name: str
    ttft: float = 0.0  # Time to first token
    tpot: float = 0.0  # Time per output token
    num_output_tokens: int = 0


class TestRunner:
    """Runs benchmarks and collects results."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results: list[TestResult] = []

    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{'=' * 80}")
        print(text)
        print("=" * 80)

    def print_summary(self):
        """Print final comparison summary."""
        self.print_header("SUMMARY COMPARISON")

        # Find specific test results for comparisons
        baseline_result = None
        segmented_result = None
        offloading_result = None
        
        for r in self.results:
            name_lower = r.name.lower()
            if "baseline" in name_lower:
                baseline_result = r
            elif "segmentedprefilloffload" in name_lower:
                segmented_result = r
            elif "offloadingconnector" in name_lower and "segmented" not in name_lower:
                offloading_result = r

        for result in self.results:
            total = result.preload_time + result.generation_time
            print(f"\n{result.name}:")
            print(f"  - Preload time: {result.preload_time:.4f} s")
            print(f"  - Generation time: {result.generation_time:.4f} s")
            print(f"  - Total time: {total:.4f} s")
            
            if result.ttft > 0:
                print(f"  - TTFT: {result.ttft:.4f} s ({result.ttft * 1000:.2f} ms)")
            if result.tpot > 0:
                print(f"  - TPOT: {result.tpot:.4f} s ({result.tpot * 1000:.2f} ms)")

            if baseline_result and result != baseline_result:
                diff = result.generation_time - baseline_result.generation_time
                pct = (
                    result.generation_time / baseline_result.generation_time - 1
                ) * 100
                print(f"  - vs Baseline: {diff:+.4f} s ({pct:+.2f}%)")
            
            # Add comparison between OffloadingConnector and SegmentedPrefillOffloadConnector
            if result == offloading_result and segmented_result:
                diff = result.generation_time - segmented_result.generation_time
                pct = (result.generation_time / segmented_result.generation_time - 1) * 100
                print(f"  - vs SegmentedPrefillOffloadConnector: {diff:+.4f} s ({pct:+.2f}%)")

        print("=" * 80)

    def save_results_to_file(self, filename: str = "test_results_summary.txt"):
        """Save all test results to a file."""
        with open(filename, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SPANS BENCHMARK RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {self.model_name}\n\n")

            # Find specific test results for comparisons
            baseline_result = None
            segmented_result = None
            offloading_result = None
            
            for r in self.results:
                name_lower = r.name.lower()
                if "baseline" in name_lower:
                    baseline_result = r
                elif "segmentedprefilloffload" in name_lower:
                    segmented_result = r
                elif "offloadingconnector" in name_lower and "segmented" not in name_lower:
                    offloading_result = r

            for i, result in enumerate(self.results, 1):
                f.write("=" * 80 + "\n")
                f.write(f"TEST {i}: {result.name}\n")
                f.write("=" * 80 + "\n\n")

                # Timing metrics
                f.write("TIMING METRICS:\n")
                f.write(f"  Preload time:     {result.preload_time:.4f} s\n")
                f.write(f"  Generation time:  {result.generation_time:.4f} s\n")
                f.write(f"  Total time:       {result.preload_time + result.generation_time:.4f} s\n")
                
                if result.ttft > 0:
                    f.write(f"  TTFT:             {result.ttft:.4f} s ({result.ttft * 1000:.2f} ms)\n")
                if result.tpot > 0:
                    f.write(f"  TPOT:             {result.tpot:.4f} s ({result.tpot * 1000:.2f} ms)\n")
                if result.num_output_tokens > 0:
                    f.write(f"  Output tokens:    {result.num_output_tokens}\n")

                # Comparison with baseline
                if baseline_result and result != baseline_result:
                    diff = result.generation_time - baseline_result.generation_time
                    pct = (result.generation_time / baseline_result.generation_time - 1) * 100
                    f.write(f"\n  vs Baseline:\n")
                    f.write(f"    Generation time diff: {diff:+.4f} s ({pct:+.2f}%)\n")
                    
                    if result.ttft > 0 and baseline_result.ttft > 0:
                        ttft_diff = result.ttft - baseline_result.ttft
                        ttft_pct = (result.ttft / baseline_result.ttft - 1) * 100
                        f.write(f"    TTFT diff:            {ttft_diff:+.4f} s ({ttft_pct:+.2f}%)\n")
                    
                    if result.tpot > 0 and baseline_result.tpot > 0:
                        tpot_diff = result.tpot - baseline_result.tpot
                        tpot_pct = (result.tpot / baseline_result.tpot - 1) * 100
                        f.write(f"    TPOT diff:            {tpot_diff:+.4f} s ({tpot_pct:+.2f}%)\n")

                # Comparison between SegmentedPrefillOffloadConnector and OffloadingConnector
                if result == offloading_result and segmented_result:
                    diff = result.generation_time - segmented_result.generation_time
                    pct = (result.generation_time / segmented_result.generation_time - 1) * 100
                    f.write(f"\n  vs SegmentedPrefillOffloadConnector:\n")
                    f.write(f"    Generation time diff: {diff:+.4f} s ({pct:+.2f}%)\n")
                    
                    if result.ttft > 0 and segmented_result.ttft > 0:
                        ttft_diff = result.ttft - segmented_result.ttft
                        ttft_pct = (result.ttft / segmented_result.ttft - 1) * 100
                        f.write(f"    TTFT diff:            {ttft_diff:+.4f} s ({ttft_pct:+.2f}%)\n")
                    
                    if result.tpot > 0 and segmented_result.tpot > 0:
                        tpot_diff = result.tpot - segmented_result.tpot
                        tpot_pct = (result.tpot / segmented_result.tpot - 1) * 100
                        f.write(f"    TPOT diff:            {tpot_diff:+.4f} s ({tpot_pct:+.2f}%)\n")

                # Output text
                f.write(f"\nOUTPUT TEXT:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{result.output_text}\n")
                f.write("-" * 80 + "\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ Results saved to: {filename}")


# =============================================================================
# Main
# =============================================================================


def main():
    setup_environment()
    model_name = MODEL_NAMES[SELECTED_MODEL_INDEX]

    # Create first LLM instance with SegmentedPrefillOffloadConnector
    runner = TestRunner(model_name)
    runner.print_header("Creating vLLM with SegmentedPrefillOffloadConnector")

    instance1 = LLMInstance.create(
        model_name, kv_config=KVConfigBuilder.segmented_prefill_offload()
    )
    tokenizer = instance1.tokenizer

    # Prompt template components
    prefix = pad_tokens(
        tokenizer(
            "<|system|>\nYou are an intelligent AI assistant. "
            "Please answer questions based on the user's instructions. "
            "Below are some reference documents that may help you in "
            "answering the user's question."
        ),
        PAD_TOKEN,
    )
    query_prefix = [SPAN_TOKEN_CROSS] + tokenizer(
        "<|user|>\nPlease write a high-quality answer for the "
        "given question using only the provided search documents "
        "(some of which might be irrelevant).\nQuestion: "
    )
    query_suffix = tokenizer("\n<|assistant|>\n")

    # Load documents
    docs_set1 = DocumentSet.from_file(TEXT_FILE_1, tokenizer, NUM_SEGMENTS)
    docs_set2 = None
    if Path(TEXT_FILE_2).exists():
        docs_set2 = DocumentSet.from_file(TEXT_FILE_2, tokenizer, NUM_SEGMENTS)
        print(f"Found {TEXT_FILE_2} - using 8 documents\n")
    else:
        print(f"{TEXT_FILE_2} not found - using 4 documents only\n")

    # Combine all documents
    all_docs = list(docs_set1)
    if docs_set2:
        all_docs.extend(docs_set2)

    # Print document lengths
    labels = ["doc_w", "doc_x", "doc_y", "doc_z", "doc_p", "doc_q", "doc_r", "doc_s"]
    for i, (label, doc) in enumerate(zip(labels[: len(all_docs)], all_docs)):
        print(f"  {label} length: {len(doc)}")

    # Preload
    runner.print_header(f"Preloading {len(all_docs)} documents + prefix")
    preload_time = instance1.preload(all_docs, prefix)
    print(f"Preload completed in {preload_time:.4f} s")
    time.sleep(2)

    # Build the full query
    question = "What is Northholm's value to Luminthia?"
    full_prompt = (
        prefix + sum(all_docs, []) + query_prefix + tokenizer(question) + query_suffix
    )
    print(f"\nFull prompt length: {len(full_prompt)} tokens")

    # =============================================================================
    # TEST 1: SegmentedPrefillOffloadConnector
    # =============================================================================

    test_name = f"SegmentedPrefillOffloadConnector ({len(all_docs)} docs)"
    runner.print_header(f"TEST 1: {test_name}")

    response, gen_time = instance1.generate(full_prompt)
    output_text = response[0].outputs[0].text
    
    # Extract metrics
    ttft = 0.0
    tpot = 0.0
    num_output_tokens = len(response[0].outputs[0].token_ids)
    
    metrics = response[0].metrics
    num_output_tokens = len(response[0].outputs[0].token_ids)
    # TTFT: time from arrival to first token
    if metrics and metrics.first_token_time and metrics.arrival_time:
        ttft = metrics.first_token_time - metrics.arrival_time
    else:
        ttft = 0.0

    # TPOT: time between tokens after the first
    if metrics and metrics.finished_time and metrics.first_token_time and num_output_tokens > 1:
        tpot = (metrics.finished_time - metrics.first_token_time) / (num_output_tokens - 1)
    else:
        tpot = 0.0
    print(f"Preload time: {preload_time:.4f} s")
    print(f"Generation time: {gen_time:.4f} s")
    # TTFT and TPOT measurements don't work yet
    # print(f"TTFT: {ttft:.4f} s ({ttft * 1000:.2f} ms)")
    # print(f"TPOT: {tpot:.4f} s ({tpot * 1000:.2f} ms)")
    print(f"Output tokens: {num_output_tokens}")
    print(f"Output: {output_text[:200]}...")

    dump_kv_cache_to_file(instance1.llm, "kv_cache_test1_segmented.txt", test_name)
    runner.results.append(
        TestResult(test_name, preload_time, gen_time, output_text, model_name, ttft, tpot, num_output_tokens)
    )

    # Cleanup instance 1
    runner.print_header("Destroying vLLM instance")
    instance1.cleanup()
    time.sleep(3)

    # =============================================================================
    # TEST 2: OffloadingConnector 
    # =============================================================================

    runner.print_header("Creating new vLLM with OffloadingConnector")
    instance2 = LLMInstance.create(model_name, kv_config=KVConfigBuilder.offloading())

    runner.print_header("TEST 2: OffloadingConnector")

    # Preload
    preload_time2 = instance2.preload(list(all_docs), prefix)
    print(f"Preload completed in {preload_time2:.4f} s")
    time.sleep(2)

    response2, gen_time2 = instance2.generate(full_prompt)
    output_text2 = response2[0].outputs[0].text
    
    # Extract metrics
    ttft2 = 0.0
    tpot2 = 0.0
    num_output_tokens2 = len(response2[0].outputs[0].token_ids)
    
    if response2[0].metrics and hasattr(response2[0].metrics, 'first_token_latency'):
        ttft2 = response2[0].metrics.first_token_latency
    
    if num_output_tokens2 > 1 and ttft2 > 0:
        tpot2 = (gen_time2 - ttft2) / (num_output_tokens2 - 1)

    print(f"Preload time: {preload_time2:.4f} s")
    print(f"Generation time: {gen_time2:.4f} s")
    # TTFT and TPOT measurements don't work yet
    # print(f"TTFT: {ttft:.4f} s ({ttft * 1000:.2f} ms)")
    # print(f"TPOT: {tpot:.4f} s ({tpot * 1000:.2f} ms)")
    print(f"Output tokens: {num_output_tokens2}")
    print(f"Output: {output_text2[:200]}...")

    test_name2 = "OffloadingConnector (with preload)"
    dump_kv_cache_to_file(instance2.llm, "kv_cache_test2_offloading.txt", test_name2)
    runner.results.append(
        TestResult(test_name2, preload_time2, gen_time2, output_text2, model_name, ttft2, tpot2, num_output_tokens2)
    )

    # Cleanup instance 2
    runner.print_header("Destroying vLLM instance")
    instance2.cleanup()
    time.sleep(3)

    # =============================================================================
    # TEST 3: Baseline (no spans, no KVTransferConfig)
    # =============================================================================

    disable_spans()
    runner.print_header(
        "TEST 3: Baseline (SPANS DISABLED, no preload, no KVTransferConfig)"
    )

    instance3 = LLMInstance.create(model_name, kv_config=None)

    print("Running generation without preload...")
    response3, gen_time3 = instance3.generate(full_prompt)
    output_text3 = response3[0].outputs[0].text
    
    # Extract metrics
    ttft3 = 0.0
    tpot3 = 0.0
    num_output_tokens3 = len(response3[0].outputs[0].token_ids)
    
    if response3[0].metrics and hasattr(response3[0].metrics, 'first_token_latency'):
        ttft3 = response3[0].metrics.first_token_latency
    
    if num_output_tokens3 > 1 and ttft3 > 0:
        tpot3 = (gen_time3 - ttft3) / (num_output_tokens3 - 1)

    print(f"Generation time: {gen_time3:.4f} s")
    # TTFT and TPOT measurements don't work yet
    # print(f"TTFT: {ttft:.4f} s ({ttft * 1000:.2f} ms)")
    # print(f"TPOT: {tpot:.4f} s ({tpot * 1000:.2f} ms)")
    print(f"Output tokens: {num_output_tokens3}")
    print(f"Output: {output_text3[:200]}...")

    test_name3 = "Baseline (no preload, spans disabled)"
    dump_kv_cache_to_file(instance3.llm, "kv_cache_test3_baseline.txt", test_name3)
    runner.results.append(
        TestResult(test_name3, 0.0, gen_time3, output_text3, model_name, ttft3, tpot3, num_output_tokens3)
    )

    instance3.cleanup()

    # =============================================================================
    # Summary
    # =============================================================================

    runner.print_summary()
    runner.save_results_to_file("test_results_summary.txt")


if __name__ == "__main__":
    main()
