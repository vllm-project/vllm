# Example usage:
# python3 TKNP/prompt_generator.py --model meta-llama/Llama-3.1-8B --batch-size 8 --seq-length 128


import random
import json
import hashlib
import os
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer
import argparse

# Cache directory
CACHE_DIR = Path("TKNP/prompt_cache")

def generate_benchmark_prompts(
    batch_size: int,
    seq_length: int,
    tokenizer=None,
    model_name: Optional[str] = None,
    vocab_style: str = "natural",
    add_summarization_prompt: bool = True,
    seed: Optional[int] = None,
    use_cache: bool = True
) -> List[str]:
    """
    Generate prompts for vLLM benchmarking with exact token lengths.
    
    Args:
        batch_size: Number of prompts to generate
        seq_length: Exact total sequence length in tokens (including instruction)
        tokenizer: Tokenizer instance (required for exact token lengths)
        model_name: Model name for tokenizer loading
        vocab_style: Style of text generation:
            - "natural": Natural language sentences
            - "random": Random words
            - "repetitive": Repetitive patterns (tests caching)
        add_summarization_prompt: If True, adds "Summarize the following text:\n\n"
        seed: Random seed for reproducibility
        use_cache: If True, use cached prompts when available
    
    Returns:
        List of prompts, each with exactly seq_length tokens total
    """
    if tokenizer is None and model_name is None:
        raise ValueError("tokenizer or model_name is required for exact token lengths")
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get model name for cache key
    if model_name is None:
        model_name = getattr(tokenizer, 'name_or_path', 'unknown_model')
    
    # Check cache first
    if use_cache:
        cached_prompts = _load_from_cache(
            batch_size, seq_length, model_name, 
            vocab_style, add_summarization_prompt, seed
        )
        if cached_prompts is not None:
            print(f"✓ Loaded {len(cached_prompts)} prompts from cache")
            return cached_prompts
    
    # Generate prompts (original logic)
    if seed is not None:
        random.seed(seed)
    
    # Calculate tokens needed for content after instruction
    instruction = "Summarize the following text:\n\n" if add_summarization_prompt else ""
    instruction_tokens = _get_token_count(instruction, tokenizer) if instruction else 0
    content_tokens = seq_length - instruction_tokens
    
    if content_tokens <= 0:
        raise ValueError(f"seq_length ({seq_length}) must be greater than instruction tokens ({instruction_tokens})")
    
    # Generate ONE base prompt with exact token count
    if vocab_style == "natural":
        base_content = _generate_natural_prompt(content_tokens, tokenizer)
    elif vocab_style == "random":
        base_content = _generate_random_words(content_tokens, tokenizer)
    elif vocab_style == "repetitive":
        base_content = _generate_repetitive_prompt(content_tokens, tokenizer)
    else:
        raise ValueError(f"Unknown vocab_style: {vocab_style}")
    
    # Create batch_size variations by shuffling sentences
    base_sentences = [s.strip() + "." for s in base_content.split(". ") if s.strip()]
    if base_sentences and base_sentences[-1].endswith(".."):
        base_sentences[-1] = base_sentences[-1][:-1]
    
    prompts = []
    
    for i in range(batch_size):
        if i == 0:
            content = base_content
        else:
            shuffled_sentences = base_sentences.copy()
            random.shuffle(shuffled_sentences)
            content = " ".join(shuffled_sentences)
        
        prompt = instruction + content
        prompts.append(prompt)
    
    # Save to cache
    if use_cache:
        _save_to_cache(
            prompts, batch_size, seq_length, model_name,
            vocab_style, add_summarization_prompt, seed
        )
    
    return prompts


def _get_cache_key(batch_size: int, seq_length: int, model_name: str,
                   vocab_style: str, add_summarization_prompt: bool, 
                   seed: Optional[int]) -> str:
    """Generate unique cache key from configuration."""
    config_str = f"{batch_size}_{seq_length}_{model_name}_{vocab_style}_{add_summarization_prompt}_{seed}"
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_cache_path(batch_size: int, seq_length: int, model_name: str,
                    vocab_style: str, add_summarization_prompt: bool,
                    seed: Optional[int]) -> Path:
    """Get cache file path for given configuration."""
    cache_key = _get_cache_key(batch_size, seq_length, model_name, 
                                vocab_style, add_summarization_prompt, seed)
    
    # Create descriptive filename
    model_safe = model_name.replace('/', '_').replace('\\', '_')
    filename = f"bs{batch_size}_seq{seq_length}_{model_safe}_{vocab_style}_{cache_key[:8]}.json"
    
    return CACHE_DIR / filename


def _save_to_cache(prompts: List[str], batch_size: int, seq_length: int,
                   model_name: str, vocab_style: str, 
                   add_summarization_prompt: bool, seed: Optional[int]) -> None:
    """Save prompts to cache file."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cache_path = _get_cache_path(batch_size, seq_length, model_name,
                                     vocab_style, add_summarization_prompt, seed)
        
        cache_data = {
            'config': {
                'batch_size': batch_size,
                'seq_length': seq_length,
                'model_name': model_name,
                'vocab_style': vocab_style,
                'add_summarization_prompt': add_summarization_prompt,
                'seed': seed
            },
            'prompts': prompts
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        
        print(f"✓ Saved {len(prompts)} prompts to cache: {cache_path.name}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def _load_from_cache(batch_size: int, seq_length: int, model_name: str,
                     vocab_style: str, add_summarization_prompt: bool,
                     seed: Optional[int]) -> Optional[List[str]]:
    """Load prompts from cache if available."""
    try:
        cache_path = _get_cache_path(batch_size, seq_length, model_name,
                                     vocab_style, add_summarization_prompt, seed)
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Verify configuration matches
        config = cache_data['config']
        if (config['batch_size'] == batch_size and
            config['seq_length'] == seq_length and
            config['model_name'] == model_name and
            config['vocab_style'] == vocab_style and
            config['add_summarization_prompt'] == add_summarization_prompt and
            config['seed'] == seed):
            return cache_data['prompts']
        
        return None
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return None


def clear_cache(confirm: bool = False) -> None:
    """Clear all cached prompts."""
    if not confirm:
        print("Warning: This will delete all cached prompts.")
        print("Call clear_cache(confirm=True) to proceed.")
        return
    
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"✓ Cleared cache directory: {CACHE_DIR}")
    else:
        print("Cache directory does not exist.")


def list_cache_files() -> None:
    """List all cached prompt files with their configurations."""
    if not CACHE_DIR.exists():
        print("No cache directory found.")
        return
    
    cache_files = list(CACHE_DIR.glob("*.json"))
    
    if not cache_files:
        print("No cached files found.")
        return
    
    print(f"Found {len(cache_files)} cached files:\n")
    
    for cache_file in sorted(cache_files):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                config = data['config']
                num_prompts = len(data['prompts'])
                
            print(f"📁 {cache_file.name}")
            print(f"   Batch size: {config['batch_size']}, Seq length: {config['seq_length']}")
            print(f"   Model: {config['model_name']}, Style: {config['vocab_style']}")
            print(f"   Prompts: {num_prompts}, Seed: {config['seed']}")
            print()
        except Exception as e:
            print(f"❌ {cache_file.name} (error reading: {e})\n")


def _get_token_count(text: str, tokenizer) -> int:
    """Get exact token count for a text."""
    if hasattr(tokenizer, 'encode'):
        return len(tokenizer.encode(text))
    elif hasattr(tokenizer, '__call__'):
        result = tokenizer(text)
        if hasattr(result, 'input_ids'):
            return len(result.input_ids)
        return len(result)
    else:
        raise ValueError("Tokenizer must have 'encode' method or be callable")


def _generate_natural_prompt(target_tokens: int, tokenizer) -> str:
    """Generate natural language sentences with exact token count."""
    templates = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require extensive training data.",
        "Natural language processing has advanced significantly.",
        "Deep neural networks power modern AI systems.",
        "Transformer architectures revolutionized language models.",
        "Attention mechanisms enable better context understanding.",
        "Large language models demonstrate emergent capabilities.",
        "Efficient inference requires careful optimization.",
        "Batch processing improves throughput performance.",
        "Token generation speed depends on model size.",
    ]
    
    sentences = []
    for _ in range(target_tokens // 5 + 20):
        sentences.append(random.choice(templates))
    
    text = " ".join(sentences)
    words = text.split()
    
    left, right = 1, len(words)
    
    while left <= right:
        mid = (left + right) // 2
        test_text = " ".join(words[:mid])
        token_count = _get_token_count(test_text, tokenizer)
        
        if token_count == target_tokens:
            return test_text
        elif token_count < target_tokens:
            left = mid + 1
        else:
            right = mid - 1
    
    return " ".join(words[:right])


def _generate_random_words(target_tokens: int, tokenizer) -> str:
    """Generate random dictionary words with exact token count."""
    word_list = [
        "algorithm", "benchmark", "compute", "dataset", "engine",
        "function", "generation", "hyperparameter", "inference", "kernel",
        "latency", "model", "network", "optimization", "performance",
        "query", "runtime", "system", "throughput", "utility",
        "vector", "workload", "execution", "memory", "processing"
    ]
    
    words = [random.choice(word_list) for _ in range(target_tokens * 2 + 100)]
    
    left, right = 1, len(words)
    
    while left <= right:
        mid = (left + right) // 2
        test_text = " ".join(words[:mid])
        token_count = _get_token_count(test_text, tokenizer)
        
        if token_count == target_tokens:
            return test_text
        elif token_count < target_tokens:
            left = mid + 1
        else:
            right = mid - 1
    
    return " ".join(words[:right])


def _generate_repetitive_prompt(target_tokens: int, tokenizer) -> str:
    """Generate repetitive patterns with exact token count."""
    pattern = "repeat this pattern again and again to test caching behavior"
    
    repetitions = (target_tokens // len(pattern.split())) * 2 + 10
    full_text = (pattern + " ") * repetitions
    words = full_text.split()
    
    left, right = 1, len(words)
    
    while left <= right:
        mid = (left + right) // 2
        test_text = " ".join(words[:mid])
        token_count = _get_token_count(test_text, tokenizer)
        
        if token_count == target_tokens:
            return test_text
        elif token_count < target_tokens:
            left = mid + 1
        else:
            right = mid - 1
    
    return " ".join(words[:right])


def parse_args():
    """Parse command line arguments for prompt generator."""
    parser = argparse.ArgumentParser(description="Prompt generator for vLLM benchmarking")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for prompts (default: 8)")
    parser.add_argument("--seq-length", type=int, default=32,
                        help="Sequence length for prompts (default: 1024)")
    parser.add_argument("--vocab-style", type=str, default="natural",
                        choices=["natural", "random", "repetitive"],
                        help="Vocabulary style for prompts (default: natural)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    
    args = parse_args()
    from transformers import AutoTokenizer
    import time
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print("="*60)
    print("First run: Generating and caching prompts")
    print("="*60)
    
    start = time.time()
    prompts = generate_benchmark_prompts(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        vocab_style=args.vocab_style,
        seed=args.seed
    )
    elapsed = time.time() - start
    
    print(f"Generated {len(prompts)} prompts in {elapsed:.2f}s\n")
    
    print("="*60)
    print("Second run: Loading from cache (should be instant)")
    print("="*60)
    
    start = time.time()
    prompts_cached = generate_benchmark_prompts(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        vocab_style=args.vocab_style,
        seed=args.seed
    )
    elapsed = time.time() - start
    
    print(f"Loaded {len(prompts_cached)} prompts in {elapsed:.2f}s\n")
    
    # Verify they're identical
    print(f"Prompts identical: {prompts == prompts_cached}\n")
    
    # # List cache files
    # print("="*60)
    # print("Cache contents:")
    # print("="*60)
    # list_cache_files()