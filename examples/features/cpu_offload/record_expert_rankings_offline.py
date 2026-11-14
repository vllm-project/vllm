# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Records the routed-expert rankings file that `--cpu-offload-expert-rankings`
reads: `{"<layer index>": [global expert ids, hottest first], ...}`.

It runs prompts with routed-expert capture on and counts, per decoder layer,
how often each expert was routed to over the prompt and generated tokens. Use
prompts like the traffic the model will serve; the ranking is only as good as
that sample.

Example usage:

python record_expert_rankings_offline.py \
    --model Qwen/Qwen3-30B-A3B \
    --prompts prompts.txt \
    --output rankings.json

Then serve with `--cpu-offload-gb <GiB> --cpu-offload-expert-rankings
rankings.json`.
"""

import json

import numpy as np

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--prompts", required=True, type=str, help="text file, one prompt per line"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=str, help="rankings JSON to write"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="tokens to generate per prompt"
    )
    return parser.parse_args()


def main(args):
    with open(args.prompts) as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    engine_args = EngineArgs.from_cli_args(args)
    engine_args.enable_return_routed_experts = True
    llm = LLM.from_engine_args(engine_args)
    num_experts = llm.llm_engine.model_config.get_num_experts()

    counts: np.ndarray | None = None
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    for output in llm.generate(prompts, params):
        # [tokens, layers, top_k] global expert ids
        routed = output.outputs[0].routed_experts
        assert routed is not None
        if counts is None:
            counts = np.zeros((routed.shape[1], num_experts), dtype=np.int64)
        for layer in range(routed.shape[1]):
            counts[layer] += np.bincount(
                routed[:, layer].ravel(), minlength=num_experts
            )
    assert counts is not None, "no prompts"

    rankings = {
        str(layer): [int(e) for e in np.argsort(-row, kind="stable") if row[e]]
        for layer, row in enumerate(counts)
        if row.any()
    }
    with open(args.output, "w") as handle:
        json.dump(rankings, handle)
    print(f"Wrote rankings for {len(rankings)} layers to {args.output}")


if __name__ == "__main__":
    main(parse_args())
