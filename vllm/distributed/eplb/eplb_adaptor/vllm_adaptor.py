import json
from typing import Any


class VllmEplbAdaptor():
    def __init__(self, model, **args):
        self.model = model
        self.expert_map_per_layer_cpu = dict()
        self.buffer_tensor_list: list[list[Any]] = [[]]
        self.expert_param_per_layer = dict()

    def do_update_expert_map(self, *args, **kwargs):
        pass

    def do_update_log2phy_map(self, *args, **kwargs):
        pass

    def do_update_expert_weight(self, *args, **kwargs):
        pass