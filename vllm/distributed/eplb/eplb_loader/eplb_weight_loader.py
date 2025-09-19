# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EplbWeightLoader:

    def __init__(self, *args):
        self.reqs = list()
        self.layer_id = -1
        self.recv_expert_list = list()
        self.comm_op_list = list()
        self.updated_expert_map = None
        self.updated_log2phy_map = None

    def set_log2phy_map(self, *args, **kargs):
        pass

    def generate_expert_d2d_transfer_task(self, *args, **kargs):
        pass

    def async_expert_weight_transfer(self):
        pass

    def update_expert_map_and_weight(self):
        pass

    def shuffle_layer(self, *args, **kargs):
        pass