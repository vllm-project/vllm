from collections import deque
from vllm.config import LoRAConfig, LoraPolicy
from typing import List

import logging

logger = logging.getLogger(__name__)


class LoRAScheduler:
    """Round robin scheduler for LoRAs."""

    def __init__(
        self,
        lora_config: LoRAConfig,
    ):
        logger.info(f"Initializing LoRA Scheduler with policy {lora_config.lora_policy} and {lora_config.num_iters_before_reschedule} iterations before rescheduling. Lora_config {lora_config}")

        self.lora_config = lora_config
        self.max_loras_each_iter = lora_config.max_loras

        self.active_loras = deque()
        self.all_loras = set()

        self.__counter = 0
        self.__prev_scheduled_loras = []
    
    def _register_lora(self, lora_id: int):
        if lora_id not in self.all_loras:
            self.all_loras.add(lora_id)
            self.active_loras.append(lora_id)

    def _unregister_lora(self, lora_id: int):
        if lora_id in self.all_loras:
            self.all_loras.remove(lora_id)
            if lora_id in self.active_loras:
                self.active_loras.remove(lora_id)

    def update_loras(self, all_lora_ids: List[int]):
        """Update the list of LoRAs that are available for scheduling."""
        logger.info(f"Updating LoRA available list: {all_lora_ids}")
        for lora_id in all_lora_ids:
            self._register_lora(lora_id)
        to_remove = [lora_id for lora_id in self.all_loras if lora_id not in all_lora_ids]
        for lora_id in to_remove:
            self._unregister_lora(lora_id)

    def schedule_loras(self):
        """Schedule which LoRAs requests can belong to for the next iteration."""
        if self.lora_config.lora_policy == LoraPolicy.NAIVE:
            return list(self.all_loras)
        
        assert self.lora_config.lora_policy == LoraPolicy.ROUND_ROBIN

        scheduled_loras = []
        logger.info(f"LoRA scheduler has {len(self.active_loras)} active loras")

        if self.__counter == 0:
            for _ in range(min(self.max_loras_each_iter, len(self.active_loras))):
                lora_id = self.active_loras.popleft()  # Get the next LoRA in round-robin order
                scheduled_loras.append(lora_id)
                self.active_loras.append(lora_id)  # Rotate it back to the end of the queue
        else:
            scheduled_loras = self.__prev_scheduled_loras
        
        self.__counter = (self.__counter + 1) % self.lora_config.num_iters_before_reschedule
        self.__prev_scheduled_loras = scheduled_loras
        return list(set([0] + scheduled_loras))
    
    # TODO: make lora schedulign dynamic (what if the final scheduled request list doesn't use all the LoRAs we allow?)