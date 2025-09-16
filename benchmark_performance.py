#!/usr/bin/env python3
"""
VectorChat vLLM Performance Benchmark
Benchmarks the performance impact of encrypted tokenization
"""

import time
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add paths
current_dir = Path(__file__).resolve().parent
vllm_dir = current_dir
vectorchat_dir = current_dir.parent / 'AID-CORE-COMMERCIAL' / 'vectorchat' / 'daemon'

for path in [str(vllm_dir), str(vectorchat_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_tokenizer(tokenizer, test_texts: List[str], iterations: int = 10) -> Dict[str, Any]:
    """Benchmark tokenizer performance."""
    results = {
        'encode_times': [],
        'decode_times': [],
        'total_tokens': 0,
        'avg_tokens_per_text': 0,
        'throughput_encode': 0,
        'throughput_decode': 0,
    }

    # Warm up
    for text in test_texts[:3]:
        tokenizer.encode(text)
        tokens = tokenizer.encode(text)
        tokenizer.decode(tokens)

    # Benchmark
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations}")

        # Encode benchmark
        start_time = time.perf_counter()
        all_tokens = []
        for text in test_texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        encode_time = time.perf_counter() - start_time
        results['encode_times'].append(encode_time)

        # Decode benchmark
        start_time = time.perf_counter()
        for tokens in [all_tokens[i:i+50] for i in range(0, len(all_tokens), 50)]:  # Batch decode
            tokenizer.decode(tokens)
        decode_time = time.perf_counter() - start_time
        results['decode_times'].append(decode_time)

    # Calculate statistics
    total_chars = sum(len(text) for text in test_texts)
    results['total_tokens'] = len(all_tokens)
    results['avg_tokens_per_text'] = results['total_tokens'] / len(test_texts)

    # Throughput calculations
    avg_encode_time = sum(results['encode_times']) / len(results['encode_times'])
    avg_decode_time = sum(results['decode_times']) / len(results['decode_times'])

    results['throughput_encode'] = total_chars / avg_encode_time  # chars/second
    results['throughput_decode'] = results['total_tokens'] / avg_decode_time  # tokens/second

    return results

def run_performance_comparison():
    """Compare performance of standard vs encrypted tokenizers."""
    logger.info("🚀 VectorChat vLLM Performance Benchmark")
    logger.info("=" * 60)

    # Test data
    test_texts = [
        "Hello world!",
        "This is a simple test sentence.",
        "Machine learning models can now process encrypted data streams directly without decryption.",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
        "Large language models are becoming increasingly important for natural language processing tasks.",
        "Cryptography and machine learning intersect in fascinating ways, enabling secure AI inference.",
        "Tokenization is the process of converting text into tokens that can be processed by neural networks.",
        "VectorChat provides encrypted tokenization for secure AI communication and inference pipelines.",
        "Performance benchmarking helps us understand the trade-offs between security and computational efficiency.",
        "This benchmark tests both encoding and decoding performance with various text lengths and complexities.",
    ] * 10  # Repeat for more data

    logger.info(f"Test dataset: {len(test_texts)} texts, ~{sum(len(t) for t in test_texts)} total characters")

    results = {}

    try:
        # Test 1: Standard tokenizer (baseline)
        logger.info("\n📊 Test 1: Standard Tokenizer (Baseline)")
        from transformers import AutoTokenizer

        standard_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        standard_results = benchmark_tokenizer(standard_tokenizer, test_texts)
        results['standard'] = standard_results

        logger.info("Standard tokenizer results:")
        logger.info(".2f")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(".1f")

    except Exception as e:
        logger.error(f"Standard tokenizer benchmark failed: {e}")
        return None

    try:
        # Test 2: VectorChat encrypted tokenizer
        logger.info("\n🔐 Test 2: VectorChat Encrypted Tokenizer")
        from vectorchat_tokenizer import create_vectorchat_tokenizer

        crypto_config = {
            'emdm_seed_hex': 'benchmark_test_seed_2024',
            'emdm_anchor_indices': [0, 1, 2, 3],
            'emdm_window_len': 10,
            'pairing_sequence_length': 8,
            'session_id_length_bytes': 16,
            'checksum_length': 2,
        }

        encrypted_tokenizer = create_vectorchat_tokenizer("gpt2", crypto_config)
        encrypted_results = benchmark_tokenizer(encrypted_tokenizer, test_texts)
        results['encrypted'] = encrypted_results

        logger.info("Encrypted tokenizer results:")
        logger.info(".2f")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(".1f")

    except Exception as e:
        logger.error(f"Encrypted tokenizer benchmark failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    # Performance comparison
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE COMPARISON")
    logger.info("=" * 60)

    if 'standard' in results and 'encrypted' in results:
        std_enc = results['standard']['throughput_encode']
        enc_enc = results['encrypted']['throughput_encode']
        encode_overhead = ((std_enc - enc_enc) / std_enc) * 100

        std_dec = results['standard']['throughput_decode']
        enc_dec = results['encrypted']['throughput_decode']
        decode_overhead = ((std_dec - enc_dec) / std_dec) * 100

        logger.info("Encoding Performance:")
        logger.info(".1f")
        logger.info(".1f")
        logger.info(".1f")

        logger.info("Decoding Performance:")
        logger.info(".1f")
        logger.info(".1f")
        logger.info(".1f")

        # Save results to file
        results_file = current_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n💾 Results saved to: {results_file}")

        # Recommendations
        logger.info("\n" + "=" * 60)
        logger.info("🎯 RECOMMENDATIONS")
        logger.info("=" * 60)

        if abs(encode_overhead) < 50 and abs(decode_overhead) < 50:
            logger.info("✅ Performance impact is acceptable for most use cases")
            logger.info("   Consider using VectorChat encryption for sensitive applications")
        elif abs(encode_overhead) < 100 and abs(decode_overhead) < 100:
            logger.info("⚠️  Moderate performance impact")
            logger.info("   Evaluate use case requirements vs performance trade-offs")
        else:
            logger.info("❌ Significant performance impact")
            logger.info("   Consider optimization or alternative approaches")

        logger.info("\n🔧 Optimization Suggestions:")
        logger.info("   - Implement caching for key generation")
        logger.info("   - Use batch processing for crypto operations")
        logger.info("   - Consider GPU acceleration for encryption")
        logger.info("   - Profile and optimize crypto algorithms")

    return results

def main():
    """Main benchmark function."""
    try:
        results = run_performance_comparison()
        if results:
            logger.info("\n✅ Benchmark completed successfully!")
            return 0
        else:
            logger.error("\n❌ Benchmark failed!")
            return 1
    except Exception as e:
        logger.error(f"Benchmark failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
