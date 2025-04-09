# SPDX-License-Identifier: Apache-2.0
import torch
import os
import csv
import time
from vllm.logger import init_logger
from typing import Optional
from vllm.distributed.utils import stateless_init_torch_distributed_process_group
import vllm.envs as envs
from typing import Tuple

logger = init_logger(__name__)


class MoEExpertLoad:
    """Load data for each expert on each layer.
    """

    def __init__(self, ep_rank: int, ep_size: int, global_expert_num: int):
        self.layer_index = []  # all layer ids, which using the moe feature
        self.layer_expert_map: torch.tensor = None  # expert id list on each layer
        self.global_expert_num = global_expert_num
        self.ep_rank_id = ep_rank
        self.ep_size = ep_size
        self.layer_expert_tokens_tensor_zero: torch.tensor = None
        self.layer_expert_tokens_tensor: torch.tensor = None
        self._current_layer_id: str = "unknown"
        self.ep_load_master_ip = envs.VLLM_EP_LOAD_COLLECT_MASTER_IP
        self.ep_load_port = envs.VLLM_EP_LOAD_COLLECT_MASTER_PORT
        self._collect_load_enabled = False

        assert self.ep_load_master_ip != "", "VLLM_EP_LOAD_COLLECT_MASTER_IP must be set"
        self.ep_load_group = stateless_init_torch_distributed_process_group(
            self.ep_load_master_ip,
            self.ep_load_port,
            ep_rank,
            ep_size,
            backend="gloo")
        logger.info("initialize ep load collecting processGroup with {}:{} for rank {} out of {} done"
                    .format(self.ep_load_master_ip, self.ep_load_port, ep_rank, ep_size))

        logger.info("initialized MoE Expert Load for rank {}".format(ep_rank))

    def add_layer(self, layer_id: str, expert_map: torch.tensor):
        """Add the layer to the expert load"""

        self.layer_index.append(layer_id)

        # each layer have the different expert map
        if self.layer_expert_map is None:
            self.layer_expert_map = expert_map.unsqueeze(0)
        else:
            self.layer_expert_map = torch.cat([self.layer_expert_map, expert_map.unsqueeze(0)], dim=0)

        if self.layer_expert_tokens_tensor is None:
            self.layer_expert_tokens_tensor = torch.zeros(1, self.global_expert_num,
                                                          device=f'cuda:{torch.cuda.current_device()}',
                                                          dtype=torch.int64)
        else:
            self.layer_expert_tokens_tensor = torch.cat([self.layer_expert_tokens_tensor,
                                                         torch.zeros(1, self.global_expert_num,
                                                                     device=f'cuda:{torch.cuda.current_device()}',
                                                                     dtype=torch.int64)], dim=0)

        logger.debug("[rank {}]add layer {} to MoE expert Load with expert list {}"
                     .format(self.ep_rank_id, layer_id, expert_map.tolist()))

    def set_current_layer_id(self, layer_id: str):
        self._current_layer_id = layer_id

    def add_token_experts(self, token_experts: torch.tensor):
        expert_tokens_list_index = self.layer_index.index(self._current_layer_id)
        flattened = token_experts.flatten().long()
        src = torch.ones_like(flattened, dtype=self.layer_expert_tokens_tensor.dtype)
        self.layer_expert_tokens_tensor[expert_tokens_list_index].scatter_add_(dim=0, index=flattened, src=src)

    def enable_collect_load(self):
        logger.info("enable moe collecting expert load")
        self._collect_load_enabled = True

    def disable_collect_load(self):
        logger.info("disable moe collecting expert load")
        self._collect_load_enabled = False

    def collect_load_enabled(self) -> bool:
        return self._collect_load_enabled

    def dump_load(self):
        ex_load, rank_load = self.calculate_load()
        torch.distributed.all_reduce(ex_load, group=self.ep_load_group)
        torch.distributed.all_reduce(rank_load, group=self.ep_load_group)

        # just generate the load file on the node with rank 0
        if self.ep_rank_id == 0:
            self.dump_load_to_file(ex_load, rank_load)

        return

    def calculate_load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate load for each expert and rank in each layer."""
        load_rank = torch.zeros(
                len(self.layer_index), self.ep_size,
                device="cpu", dtype=torch.int64
            )
        if self.layer_expert_tokens_tensor is not None:
            if self.layer_expert_tokens_tensor_zero is None:
                self.layer_expert_tokens_tensor_zero = torch.zeros_like(self.layer_expert_tokens_tensor)

            # filter the expert id on the rank
            mask = self.layer_expert_map != -1
            filter_tokens = torch.where(mask, self.layer_expert_tokens_tensor, self.layer_expert_tokens_tensor_zero[0])
            load_expert = filter_tokens.cpu()
            torch.cuda.synchronize()

            load_rank[:, self.ep_rank_id] = load_expert.sum(dim=1).squeeze()
        else:
            logger.error("[rank {}]No expert tokens available".format(self.ep_rank_id))
            load_expert = torch.zeros(
                len(self.layer_index), self.global_expert_num,
                device="cpu", dtype=torch.int64
            )

        # must clear the date to compute for next time
        self.reset()

        return load_expert, load_rank

    def reset(self):
        """Reset metrics.
        """
        if self.layer_expert_tokens_tensor is not None:
            self.layer_expert_tokens_tensor = torch.zeros_like(self.layer_expert_tokens_tensor_zero)

    def dump_load_to_file(self, expert_load: torch.Tensor, rank_load: torch.Tensor):
        timestamp = int(time.time())

        # statics data in expert level
        expert_load_csv_file = f"/tmp/moe_load_for_expert_on_layer_{timestamp}.csv"
        MoEExpertLoad.create_load_file(expert_load_csv_file, expert_load.shape[1])
        expert_load_statistics_csv_file = f"/tmp/moe_load_for_expert_statistics_on_layer_{timestamp}.csv"
        MoEExpertLoad.create_load_statistics_file(expert_load_statistics_csv_file)
        self.generate_load_file(expert_load_csv_file, expert_load_statistics_csv_file, expert_load)

        # statics data in rank level
        rank_load_csv_file = f"/tmp/moe_load_for_rank_on_layer_{timestamp}.csv"
        MoEExpertLoad.create_load_file(rank_load_csv_file, rank_load.shape[1])
        rank_load_statistics_csv_file = f"/tmp/moe_load_for_rank_statistics_on_layer_{timestamp}.csv"
        MoEExpertLoad.create_load_statistics_file(rank_load_statistics_csv_file)
        self.generate_load_file(rank_load_csv_file, rank_load_statistics_csv_file, rank_load)

    def generate_load_file(self, load_csv_file: str, load_statistics_csv_file: str, loads: torch.Tensor):
        for layer_id_index in range(len(self.layer_index)):
            layer_id = self.layer_index[layer_id_index]
            layer_load = loads[layer_id_index].tolist()
            with open(load_csv_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([layer_id] + layer_load)

            rank_load_stats_data = MoEExpertLoad.calculate_statistics_value(str(layer_id), layer_load)
            with open(load_statistics_csv_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    rank_load_stats_data["layer"],
                    rank_load_stats_data["total"],
                    rank_load_stats_data["mean"],
                    rank_load_stats_data["stddev"],
                    rank_load_stats_data["max"],
                    rank_load_stats_data["min"],
                    rank_load_stats_data["disparity"]
                ])

    @staticmethod
    def calculate_statistics_value(layer_id: str, layer_load: []):
        total_loads = sum(layer_load)

        # Calculate mean and variance
        mean = total_loads / len(layer_load) if len(layer_load) > 0 else 0
        variance = sum((x - mean) ** 2 for x in layer_load) / len(layer_load) if len(layer_load) > 0 else 0
        std_dev = variance ** 0.5

        sorted_loads = sorted(enumerate(layer_load), key=lambda x: x[1], reverse=True)

        max_value = sorted_loads[0][1]
        min_value = sorted_loads[-1][1]
        disparity = max_value / min_value if min_value != 0 else float('inf')

        return {
            "layer": layer_id,
            "total": total_loads,
            "mean": round(mean, 2),
            "stddev": round(std_dev, 2),
            "max": max_value,
            "min": min_value,
            "disparity": round(disparity, 2)
        }

    @staticmethod
    def create_load_statistics_file(csv_file: str):
        file_exists = os.path.exists(csv_file)
        if not file_exists:
            with open(csv_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["layer", "total", "mean", "stddev", "max", "min", "disparity"])

    @staticmethod
    def create_load_file(csv_file: str, num_columns: int):
        file_exists = os.path.exists(csv_file)
        if not file_exists:
            with open(csv_file, "w", newline='') as f:
                writer = csv.writer(f)
                if "expert" in csv_file:
                    header = ["layer"] + [f"exp_{i}" for i in range(num_columns)]
                elif "rank" in csv_file:
                    header = ["layer"] + [f"rank_{i}" for i in range(num_columns)]
                writer.writerow(header)


_EL: Optional[MoEExpertLoad] = None


def get_expert_load() -> MoEExpertLoad:
    return _EL


def new_expert_load(ep_rank: int, ep_size: int, global_expert_num: int) -> bool:
    if not envs.VLLM_EP_LOAD_COLLECT:
        logger.warning("expert load is disabled for VLLM_EP_LOAD_COLLECT env variable is not set")
        return False

    global _EL
    _EL = MoEExpertLoad(ep_rank=ep_rank, ep_size=ep_size, global_expert_num=global_expert_num)

    return True


def add_layer_to_expert_load(layer_id: str, expert_map: torch.tensor):
    _EL.add_layer(layer_id, expert_map)


def set_current_layer_id(layer_id: str):
    if _EL is not None:
        _EL.set_current_layer_id(layer_id)


def add_token_exper_list(token_experts: torch.tensor):
    if _EL is not None and _EL.collect_load_enabled():
        _EL.add_token_experts(token_experts=token_experts)


def enable_collect_load():
    if _EL is None:
        logger.warning("MoELoadBalancer is not initialized.")
        return None
    logger.info("Enable collecting expert load.")
    _EL.enable_collect_load()


def disable_collect_load():
    if _EL is None:
        logger.warning("MoELoadBalancer is not initialized.")
        return None
    logger.info("Disable collecting expert load.")
    _EL.disable_collect_load()


def dump_load():
    if _EL is None:
        logger.warning("MoELoadBalancer is not initialized.")
        return None
    logger.info("Dump expert load.")
    _EL.dump_load()
