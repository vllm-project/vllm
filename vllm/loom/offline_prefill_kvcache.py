import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.benchmarks.datasets import gen_prompt_decode_to_target_len


def _generate_exact_length_tokens(
    *,
    tokenizer,
    target_length: int,
    rng: np.random.Generator,
) -> list[int]:
    vocab_size = int(tokenizer.vocab_size)
    tokens = rng.integers(0, vocab_size, size=target_length).tolist()
    _, adjusted_tokens, _ = gen_prompt_decode_to_target_len(
        tokenizer=tokenizer,
        token_sequence=tokens,
        target_token_len=target_length,
        add_special_tokens=False,
        rng=rng,
    )
    return adjusted_tokens


def _to_cpu_past_key_values(past_key_values):
    cpu_pkv = []
    for layer_past in past_key_values:
        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
            k, v = layer_past
            cpu_pkv.append((k.detach().to("cpu"), v.detach().to("cpu")))
        else:
            raise TypeError(
                "Unsupported past_key_values layer type: "
                + f"{type(layer_past)} with len={getattr(layer_past, '__len__', None)}"
            )
    return tuple(cpu_pkv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix-repetition-prefix-len",
        dest="prefix_len",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--prefix-repetition-suffix-len",
        dest="suffix_len",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--prefix-repetition-num-prefixes",
        dest="num_prefixes",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--kvache-output",
        dest="kvache_output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset-output",
        dest="dataset_output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=False)

    args = parser.parse_args()

    if args.prefix_len % 8 != 0:
        warnings.warn(
            "--prefix-repetition-prefix-len is not a multiple of 8: "
            f"prefix_len={args.prefix_len}. This may reduce alignment efficiency.",
            stacklevel=2,
        )
    if args.suffix_len % 8 != 0:
        warnings.warn(
            "--prefix-repetition-suffix-len is not a multiple of 8: "
            f"suffix_len={args.suffix_len}. This may reduce alignment efficiency.",
            stacklevel=2,
        )

    kvache_output_path = Path(args.kvache_output)
    kvache_output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_output_path = Path(args.dataset_output)
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    if args.dtype == "auto":
        torch_dtype = "auto"
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    prefixes = []
    for prefix_id in range(args.num_prefixes):
        prefix_token_ids = _generate_exact_length_tokens(
            tokenizer=tokenizer, target_length=args.prefix_len, rng=rng
        )

        input_ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=True)

        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("Model did not return past_key_values (use_cache=True)")

        prefixes.append(
            {
                "prefix_id": prefix_id,
                "prefix_token_ids": prefix_token_ids,
                "past_key_values": _to_cpu_past_key_values(past_key_values),
            }
        )

    kvache_obj = {
        "format": "kvcache_prefill_v1",
        "model": args.model,
        "seed": args.seed,
        "prefix_len": args.prefix_len,
        "num_prefixes": args.num_prefixes,
        "dtype": str(model.dtype),
        "prefixes": prefixes,
    }
    torch.save(kvache_obj, kvache_output_path)

    num_prompts = args.num_prompts if args.num_prompts is not None else args.num_prefixes
    prompts_per_prefix = num_prompts // args.num_prefixes
    if prompts_per_prefix <= 0:
        raise ValueError(
            f"num_prompts ({num_prompts}) must be >= num_prefixes ({args.num_prefixes})"
        )

    with dataset_output_path.open("w", encoding="utf-8") as f:
        prefix_token_ids_by_id = {
            int(p["prefix_id"]): p["prefix_token_ids"] for p in prefixes
        }

        req_id = 0
        for round_idx in range(prompts_per_prefix):
            for prefix_id in range(args.num_prefixes):
                prefix_token_ids = prefix_token_ids_by_id[prefix_id]
                suffix_token_ids = _generate_exact_length_tokens(
                    tokenizer=tokenizer, target_length=args.suffix_len, rng=rng
                )
                prompt_token_ids = prefix_token_ids + suffix_token_ids

                line_obj = {
                    "format": "prefix_repetition_sample_v1",
                    "model": args.model,
                    "seed": args.seed,
                    "prefix_len": args.prefix_len,
                    "suffix_len": args.suffix_len,
                    "num_prefixes": args.num_prefixes,
                    "request_id": str(req_id),
                    "prefix_id": prefix_id,
                    "prompt_token_ids": prompt_token_ids,
                    "suffix_token_ids": suffix_token_ids,
                }
                f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")
                req_id += 1


if __name__ == "__main__":
    main()
