# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import numpy as np
import torch

from vllm.platforms.interface import Platform


def test_seed_behavior():
    # Test with a specific seed
    Platform.seed_everything(42)
    random_value_1 = random.randint(0, 100)
    np_random_value_1 = np.random.randint(0, 100)
    torch_random_value_1 = torch.randint(0, 100, (1,)).item()

    Platform.seed_everything(42)
    random_value_2 = random.randint(0, 100)
    np_random_value_2 = np.random.randint(0, 100)
    torch_random_value_2 = torch.randint(0, 100, (1,)).item()

    assert random_value_1 == random_value_2
    assert np_random_value_1 == np_random_value_2
    assert torch_random_value_1 == torch_random_value_2
