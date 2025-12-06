"""
Motivation:
A frequent question from vLLM users is how to estimate the memory required for
the attention key/value (KV) cache when scaling up context length, batch size,
or model size. While the underlying formulas are simple, there was no clear,
standalone example in the repository that demonstrates how to compute an
approximate KV memory footprint directly from a model’s configuration.

What this example provides:
This script extracts the relevant architectural attributes (number of layers and
hidden size) from a Hugging Face model configuration and applies a simple KV
sizing rule to estimate memory usage for a given seq_len, batch_size, and dtype.
The goal is to give users a back-of-the-envelope understanding of how KV cache
memory scales — without requiring them to run inference or inspect GPU memory.

Why this is helpful:
- Helps plan for long-context inference workloads
- Allows users to reason about memory tradeoffs before running vLLM
- Clarifies how KV memory scales with model architecture
- Useful for educational purposes when learning about LLM inference internals

This estimator intentionally abstracts away fragmentation, paged layout
overhead, and other runtime details. It is meant as a planning aid, not a
precise profiler.
"""

import argparse
from dataclasses import dataclass

try:
    from transformers import AutoConfig
except ImportError as e:
    raise SystemExit(
        "This example requires `transformers`. Install it with:\n"
        "  pip install transformers\n"
    ) from e


DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "int8": 1,
}


@dataclass
class KVEstimate:
    model_name: str
    num_layers: int
    hidden_size: int
    seq_len: int
    batch_size: int
    dtype: str

    def total_elements(self) -> int:
        # KV per token per layer = 2 * hidden_size
        return self.batch_size * self.seq_len * self.num_layers * (2 * self.hidden_size)

    def total_bytes(self) -> int:
        return self.total_elements() * DTYPE_BYTES[self.dtype]

    def total_gb(self) -> float:
        return self.total_bytes() / (1024 ** 3)

    def pretty(self) -> str:
        return (
            f"Model:         {self.model_name}\n"
            f"Layers:        {self.num_layers}\n"
            f"Hidden size:   {self.hidden_size}\n"
            f"Batch size:    {self.batch_size}\n"
            f"Seq length:    {self.seq_len}\n"
            f"Dtype:         {self.dtype}\n"
            f"-------------------------------\n"
            f"Approx KV cache memory: {self.total_gb():.2f} GB\n"
        )


def load_model_config(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name)

    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))

    if num_layers is None or hidden_size is None:
        raise ValueError(
            f"Could not extract num_layers/hidden_size from config for {model_name}."
        )

    return num_layers, hidden_size


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate KV cache memory usage.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="fp16", choices=DTYPE_BYTES.keys())
    return parser.parse_args()


def main():
    args = parse_args()
    num_layers, hidden_size = load_model_config(args.model)

    est = KVEstimate(
        model_name=args.model,
        num_layers=num_layers,
        hidden_size=hidden_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )

    print(est.pretty())


if __name__ == "__main__":
    main()
