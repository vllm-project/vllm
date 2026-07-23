"""Longer-running E2E bench for py-spy sampling.

Runs generate() in a loop so py-spy has enough samples.
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--max-in", type=int, default=512)
    ap.add_argument("--max-out", type=int, default=256)
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--loops", type=int, default=6)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    random.seed(0)
    prompts = [
        TokensPrompt(prompt_token_ids=[random.randint(0, 10000)
                                       for _ in range(random.randint(64, args.max_in))])
        for _ in range(args.num_seqs)
    ]
    sp = [SamplingParams(temperature=0.6, ignore_eos=True,
                         max_tokens=random.randint(64, args.max_out))
          for _ in range(args.num_seqs)]

    llm = LLM(model=args.model, max_model_len=1024, enforce_eager=True,
              gpu_memory_utilization=0.6)
    # warmup
    llm.generate([TokensPrompt(prompt_token_ids=[1, 2, 3])],
                 SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8))

    total_out = sum(s.max_tokens for s in sp)
    total_time = 0.0
    for i in range(args.loops):
        t0 = time.time()
        llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        t = time.time() - t0
        total_time += t
        print(f"[loop {i}] {total_out} tok in {t:.2f}s = {total_out/t:.1f} tok/s", flush=True)

    print(f"[bench_e2e_long] TOTAL  {total_out * args.loops} tok  {total_time:.2f}s  "
          f"tput={total_out*args.loops/total_time:.1f} tok/s")


if __name__ == "__main__":
    main()
