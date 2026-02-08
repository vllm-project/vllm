# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.speculative import DynamicSpeculativeConfig
import json
from vllm.utils.argparse_utils import FlexibleArgumentParser


_DYNAMIC_STATS_TEST = DynamicSpeculativeConfig(
    max_num_speculative_tokens=7,
    acceptance_rate_per_pos=[0.68, 0.39, 0.20, 0.10, 0.06, 0.03, 0.02],  # E1
    batch_stats={
        1: {0: 6.87, 1: 7.97, 3: 9.41, 4: 9.91, 5: 10.8, 7: 12.29,},
        4: {0: 6.87, 1: 7.97, 3: 9.41, 4: 9.91, 5: 10.8, 7: 12.29,},
        16: {0: 7.3, 1: 8.39, 3: 9.95, 4: 10.8, 5: 11.59, 7: 13.11,},
        32: {0: 7.64, 1: 8.97, 3: 10.78, 4: 11.79, 5: 12.81, 7: 14.86,},
        64: {0: 8.53, 1: 10.44, 3: 13.16, 4: 15.7, 5: 17.54, 7: 120.57,},
        128: {0: 8.53, 1: 15.44, 3: 25.16, 4: 30.7, 5: 37.54, 7: 220.57,}, # fake
    },
)


class DynamicSpeculativeDecodingManager:
    """A mapping from batch size to optimal number of drafts to use for that
    batch size. This is used to dynamically adjust the number of drafts used
    based on the current batch size."""

    _optimal_num_speculative_tokens: dict[int, int]

    def __init__(
        self,
        dynamic_config: DynamicSpeculativeConfig | None,
        vllm_max_batch_size: int,
        vllm_num_speculative_tokens: int,
        warmup_steps: int = 10, # TODO: make this configurable
    ):
        self.dynamic_config = dynamic_config
        self.vllm_max_batch_size = vllm_max_batch_size
        self.vllm_num_speculative_tokens = vllm_num_speculative_tokens
        self.batch_stats = self.dynamic_config.batch_stats
        self.available_batch_sizes = sorted(self.dynamic_config.batch_stats.keys())
        self.steps = 0
        self.warmup_steps = warmup_steps

        # Sanity check
        assert (
            vllm_num_speculative_tokens
            <= self.dynamic_config.max_num_speculative_tokens
        ), "vllm_num_speculative_tokens must be <= max_num_speculative_tokens"

        # if self.dynamic_config.is_online:
        assert self.dynamic_config.max_num_speculative_tokens == len(
            self.dynamic_config.acceptance_rate_per_pos
        ), (
            "max_num_speculative_tokens must be equal to the length of acceptance_rate_per_pos"
        )
        assert self.dynamic_config.max_num_speculative_tokens > 0, (
            "max_num_speculative_tokens must be > 0"
        )
        assert all(0 < a < 1 for a in self.dynamic_config.acceptance_rate_per_pos), (
            "all acceptance_rate_per_pos values must be in (0, 1)"
        )
        assert 1 in self.dynamic_config.batch_stats, (
            f"batch size 1 must be available, found: {self.dynamic_config.batch_stats.keys()}"
        )
        assert vllm_max_batch_size in self.dynamic_config.batch_stats, (
            f"vllm max_num_seqs {vllm_max_batch_size} must be available, found: {self.dynamic_config.batch_stats.keys()}"
        )

        for bs in self.available_batch_sizes:
            assert bs > 0
            assert 0 in self.dynamic_config.batch_stats[bs], (
                f"batch size {bs} must have draft 0 stats"
            )
            assert 1 in self.dynamic_config.batch_stats[bs], (
                f"batch size {bs} must have draft 1 stats"
            )
            assert sorted(self.dynamic_config.batch_stats[bs].keys()) == list(
                self.dynamic_config.batch_stats[bs].keys()
            ), f"batch size {bs} draft keys must be sorted"

        self.update_optimal_num_speculative_tokens()

    def step(self, spec_decoding_stats_all, batch_size: int) -> int:
        self.steps += 1
        if self.should_update():
            acceptance_rate_per_pos = self.compute_acceptance_rate_per_pos(spec_decoding_stats_all)
            self.update_acceptance_rate_per_pos(
                acceptance_rate_per_pos
            )
            
        optimal_num_speculative_tokens = (
            self.get_optimal_num_speculative_tokens(
                batch_size
            )
        )

        return optimal_num_speculative_tokens
    
    def compute_acceptance_rate_per_pos(self, spec_decoding_stats_all) -> list[float]:
        acceptance_rate_per_pos = []
        for i in range(self.vllm_num_speculative_tokens):
            if spec_decoding_stats_all.num_draft_tokens_per_pos[i] == 0:
                acceptance_rate = 0.0
            else:
                acceptance_rate = (
                    spec_decoding_stats_all.num_accepted_tokens_per_pos[i]
                    / spec_decoding_stats_all.num_draft_tokens_per_pos[i]
                )

            acceptance_rate_per_pos.append(acceptance_rate)

        return acceptance_rate_per_pos

    def should_update(self) -> bool:
        # making this a separate function for easier overriding or extension
        return self.steps > self.warmup_steps


    def get_optimal_num_speculative_tokens(self, batch_size: int) -> int:
        assert batch_size > 0, "batch_size must be > 0"
        assert batch_size <= self.vllm_max_batch_size, (
            "batch_size must be <= vllm_max_batch_size"
        )
        return self._optimal_num_speculative_tokens[batch_size]

    def update_optimal_num_speculative_tokens(self):
        self._optimal_num_speculative_tokens = {
            bs: self._compute_optimal_num_speculative_tokens(bs)
            for bs in range(1, self.vllm_max_batch_size + 1)
        }

    def update_acceptance_rate_per_pos(
        self, acceptance_rate_per_pos: list[float]
    ):
        self.dynamic_config.acceptance_rate_per_pos = acceptance_rate_per_pos
        self.update_optimal_num_speculative_tokens()


    def _get_batch_stats(self, batch_size: int) -> dict:
        # import pdb; pdb.set_trace()
        if batch_size not in self.batch_stats:
            # find the nearest batch size smaller and bigger than the given batch size
            # and return the weighted avg of their stats
            # print(
            #     f"Finding batch stats for batch_size: {batch_size} in self.available_batch_sizes: {self.available_batch_sizes}"
            # )

            smaller_bs = [bs for bs in self.available_batch_sizes if bs < batch_size]
            smaller_bs = (
                max(smaller_bs) if len(smaller_bs) else self.available_batch_sizes[0]
            )
            larger_bs = [bs for bs in self.available_batch_sizes if bs > batch_size]
            larger_bs = (
                min(larger_bs) if len(larger_bs) else self.available_batch_sizes[-1]
            )

            # REMOVE
            # print(
            #     f"smaller_bs: {smaller_bs}, larger_bs: {larger_bs}, batch_size: {batch_size}"
            # )

            smaller_bs_stat = self.batch_stats[smaller_bs]
            larger_bs_stat = self.batch_stats[larger_bs]

            if larger_bs == smaller_bs:
                return self.batch_stats[smaller_bs]

            ratio = (batch_size - smaller_bs) / (larger_bs - smaller_bs)

            # REMOVE
            # print(f"ratio: {ratio}")

            avg_stat: dict[int, float] = {}
            for k in smaller_bs_stat:
                avg_stat[k] = smaller_bs_stat[k] + ratio * (
                    larger_bs_stat[k] - smaller_bs_stat[k]
                )

            return avg_stat
        else:
            return self.batch_stats[batch_size]

    def _get_itl(self, batch_stats, num_drafts: int) -> float:
        if num_drafts in batch_stats:
            return batch_stats[num_drafts]
        else:
            lower_num_draft = max(k for k in batch_stats.keys() if k < num_drafts)
            upper_num_draft = min(k for k in batch_stats.keys() if k > num_drafts)

            # REMOVE
            # print(f"lower_num_draft: {lower_num_draft}, upper_num_draft: {upper_num_draft}, num_drafts: {num_drafts}")

            ratio = (num_drafts - lower_num_draft) / (upper_num_draft - lower_num_draft)
            lower_itl = batch_stats[lower_num_draft]
            upper_itl = batch_stats[upper_num_draft]
            return lower_itl + ratio * (upper_itl - lower_itl)

    def _compute_optimal_num_speculative_tokens(self, batch_size: int) -> int:
        batch_stats = self._get_batch_stats(batch_size)

        max_goodput = -1
        for num_drafts in range(self.dynamic_config.max_num_speculative_tokens + 1):
            curr_al = 1 + sum(self.dynamic_config.acceptance_rate_per_pos[:num_drafts])
            curr_itl = self._get_itl(batch_stats, num_drafts)
            curr_goodput = curr_al / curr_itl
            if curr_goodput > max_goodput:
                max_goodput = curr_goodput
                chosen_num_drafts = num_drafts

            # REMOVE
            # print(
            #     f"num_drafts: {num_drafts}, al: {curr_al}, itl: {curr_itl}, goodput: {curr_goodput}"
            # )

        return chosen_num_drafts


# python3 vllm/v1/spec_decode/dynamic/manager.py --dynamic-config-path log/dynamic_sd_test_2/tp-1_temp-0.0_top_p-1.0_top_k--1/philschmid/mt-bench/dynamic_speculative_config.json
if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--dynamic-config-path",
        type=str,
        default=None,
        help="Path to the dynamic speculative decoding config json file.",
    )
    args = parser.parse_args()

    MAX_TEST_BS = 128
    if args.dynamic_config_path:
        with open(args.dynamic_config_path) as f:
            data = json.load(f)

        dynamic_config = DynamicSpeculativeConfig.model_validate(data)
        dynamic_sd = DynamicSpeculativeDecodingManager(
            dynamic_config=dynamic_config,
            vllm_max_batch_size=max(dynamic_config.batch_stats.keys()),
            vllm_num_speculative_tokens=dynamic_config.max_num_speculative_tokens,
        )
    else:
        dynamic_sd = DynamicSpeculativeDecodingManager(
            dynamic_config=_DYNAMIC_STATS_TEST,
            vllm_max_batch_size=MAX_TEST_BS,
            vllm_num_speculative_tokens=7,
        )
    for i in range(1, dynamic_sd.vllm_max_batch_size + 1, 4):
        print("\n====================================")
        print(
            f"bs: {i}, optimal num drafts: {dynamic_sd.get_optimal_num_speculative_tokens(i)}"
        )
