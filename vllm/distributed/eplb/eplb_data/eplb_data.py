#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from abc import abstractmethod
from typing import Any


class BaseAdaptor:

    def __init__(self, **args):
        pass

    @abstractmethod
    def get_rank_expert_workload(self):
        raise NotImplementedError

    @abstractmethod
    def get_init_expert_map(self, num_moe_layers: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def do_update_expert_map(self, layer_id: Any,
                             updated_expert_map: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def do_update_expert_weight(self, layer_id: Any,
                                local_expert_to_replace: Any,
                                buffer_tensor_id: Any) -> Any:
        raise NotImplementedError
