import time
import threading
from typing import Optional

import torch
import prometheus_client

from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.models.interfaces import MixtureOfExperts

RECORDING_TIME = 10


class EplbStatLogger:
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter

    def __init__(self, model: MixtureOfExperts, device: torch.device, phy2log_map: Optional[torch.Tensor]):
        self.rank = get_ep_group().rank
        self.layers_num = model.num_moe_layers
        self.local_expert_num = model.num_local_physical_experts
        self.global_expert_num = model.num_physical_experts
        self.phy2log_map = phy2log_map
        if phy2log_map is None:
            self.phy2log_map = torch.arange(self.global_expert_num, device=device).repeat(self.layers_num, 1)

        labelnames_phy_load = ["rank", "layer", "phy_expert_id"]
        labelnames_phy2log = ["rank", "layer", "phy_expert_id", "log_expert_id"]

        self.phy_expert = self._counter_cls(
            name="vllm:phy_expert_heat",
            documentation="Heat of each physical expert per rank",
            labelnames=labelnames_phy_load)

        self.phy2log = self._gauge_cls(
            name="vllm:phy2log",
            documentation="physical expert to logical expert per rank",
            labelnames=labelnames_phy2log)

        self.do_record_loop = threading.Thread(target=self.record_loop)
        self.moe_load = None
        # only init in rank0
        if self.rank == 0:
            for layer_id in range(self.layers_num):
                for phy_expert_id in range(self.global_expert_num):
                    self.phy_expert.labels(rank=phy_expert_id // self.local_expert_num,
                                           layer=layer_id,
                                           phy_expert_id=phy_expert_id % self.local_expert_num)

            if self.phy2log_map is not None:
                cpu_phy2log = self.phy2log_map.cpu().tolist()
                for layer_id in range(len(cpu_phy2log)):
                    for phy_expert_id, log_expert_id in enumerate(cpu_phy2log[layer_id]):
                        self.phy2log.labels(rank=phy_expert_id // self.local_expert_num,
                                            layer=layer_id,
                                            phy_expert_id=phy_expert_id % self.local_expert_num,
                                            log_expert_id=log_expert_id).set(1)

            self.moe_load = [torch.zeros((self.layers_num, self.local_expert_num), dtype=torch.int32, device=device)
                             for _ in range(get_ep_group().world_size)]

        self.lock = threading.Lock()
        self.start_loop()

    def record(self, moe_load: Optional[torch.Tensor], phy2log_map: Optional[torch.Tensor]) -> None:
        if self.rank != 0:
            return
        with self.lock:
            if moe_load:
                self.moe_load = [old + new for old, new in zip(self.moe_load, moe_load)]
            if phy2log_map:
                self.phy2log_map = phy2log_map

    def record_loop(self):
        while True:
            with self.lock:
                moe_load = torch.stack(self.moe_load).cpu().tolist()
                for load in self.moe_load:
                    load.zero_()
            self.record_expert_load(moe_load)
            with self.lock:
                phy2log_map = self.phy2log_map.cpu().tolist()
            self.record_phy2log(phy2log_map)
            time.sleep(RECORDING_TIME)

    def start_loop(self):
        if self.rank == 0:
            self.do_record_loop.start()

    def record_phy2log(self, phy2log_map: list[list[int]]):
        if self.rank == 0:
            for layer_id in range(len(phy2log_map)):
                for phy_expert_id, log_expert_id in enumerate(phy2log_map[layer_id]):
                    self.phy2log.labels(
                        rank=phy_expert_id // self.local_expert_num,
                        layer=layer_id,
                        phy_expert_id=phy_expert_id % self.local_expert_num,
                        log_expert_id=self.phy2log_map[layer_id][phy_expert_id]
                    ).set(0)

                    self.phy2log.labels(
                        rank=phy_expert_id // self.local_expert_num,
                        layer=layer_id,
                        phy_expert_id=phy_expert_id % self.local_expert_num,
                        log_expert_id=log_expert_id
                    ).set(1)
                    self.phy2log_map[layer_id][phy_expert_id] = log_expert_id

    def record_expert_load(self, moe_load: list[list[int]]):
        if self.rank == 0:
            for layer_id in range(len(moe_load)):
                for phy_expert_id, load in enumerate(moe_load[layer_id]):
                    self.phy_expert.labels(
                        rank=phy_expert_id // self.local_expert_num,
                        layer=layer_id,
                        phy_expert_id=phy_expert_id
                    ).inc(load)

    def clear(self):
        for layer_id in range(self.layers_num):
            for phy_expert_id in range(self.global_expert_num):
                self.phy_expert.labels(rank=phy_expert_id // self.local_expert_num,
                                       layer=layer_id,
                                       phy_expert_id=phy_expert_id % self.local_expert_num).reset()
