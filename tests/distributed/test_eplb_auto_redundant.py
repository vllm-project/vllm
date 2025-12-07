# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test automatic calculation of num_redundant_experts for EPLB."""

import pytest
import torch

from vllm.config import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.platforms import current_platform


class MockMixtureOfExperts:
    """Mock MoE model for testing."""

    def __init__(
        self,
        num_logical_experts: int,
        num_routed_experts: int,
        num_moe_layers: int = 1,
        num_expert_groups: int = 1,
        num_redundant_experts: int = -1,
    ):
        self.num_logical_experts = num_logical_experts
        self.num_routed_experts = num_routed_experts
        self.num_moe_layers = num_moe_layers
        self.num_expert_groups = num_expert_groups
        self.num_redundant_experts = num_redundant_experts
        self.num_physical_experts = (
            num_routed_experts + num_redundant_experts
            if num_redundant_experts >= 0
            else num_routed_experts
        )
        self.num_local_physical_experts = 0
        self.num_shared_experts = 0
        self.expert_weights = [
            [torch.empty(1, 1) for _ in range(3)] for _ in range(num_moe_layers)
        ]

    def set_eplb_state(self, *args):
        """Mock method for setting EPLB state."""
        pass


class MockModelConfig:
    """Mock ModelConfig for testing."""

    def __init__(self):
        self.model = "test-model"

    def compute_hash(self):
        """Return a test hash."""
        return "test-hash"


class MockEPGroup:
    """Mock EP group for testing."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.device_group = self
        self.cpu_group = self

    def size(self):
        """Return world size."""
        return self.world_size

    def rank(self):
        """Return rank 0 for testing."""
        return 0


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
@pytest.mark.parametrize(
    "num_logical_experts,ep_size,expected_redundant",
    [
        (256, 8, 0),  # 256 % 8 == 0, no redundancy needed
        (256, 32, 0),  # 256 % 32 == 0, no redundancy needed
        (200, 8, 0),  # 200 % 8 == 0, no redundancy needed
        (256, 10, 4),  # 256 % 10 == 6, need 4 redundant (10 - 6 = 4)
        (100, 12, 8),  # 100 % 12 == 4, need 8 redundant (12 - 4 = 8)
        (7, 4, 1),  # 7 % 4 == 3, need 1 redundant (4 - 3 = 1)
        (13, 5, 2),  # 13 % 5 == 3, need 2 redundant (5 - 3 = 2)
        (64, 16, 0),  # 64 % 16 == 0, no redundancy needed
    ],
)
def test_auto_calculate_redundant_experts(
    num_logical_experts: int,
    ep_size: int,
    expected_redundant: int,
    monkeypatch,
):
    """Test automatic calculation of num_redundant_experts.

    Args:
        num_logical_experts: Number of logical experts in the model
        ep_size: Expert parallel world size
        expected_redundant: Expected calculated num_redundant_experts
        monkeypatch: Pytest monkeypatch fixture
    """
    # Mock EP group
    monkeypatch.setattr(
        "vllm.distributed.eplb.eplb_state.get_ep_group",
        lambda: MockEPGroup(ep_size),
    )

    # Create config with num_redundant_experts = -1 (auto-calculate)
    parallel_config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=ep_size,
        data_parallel_size=1,
        eplb_config=EPLBConfig(num_redundant_experts=-1),
    )

    # Create EPLB state
    eplb_state = EplbState(parallel_config, torch.device("cpu"))

    # Create mock model with -1 for num_redundant_experts (will be auto-calculated)
    mock_model = MockMixtureOfExperts(
        num_logical_experts=num_logical_experts,
        num_routed_experts=num_logical_experts,
        num_redundant_experts=-1,
    )

    # Add model (this should trigger auto-calculation)
    eplb_state.add_model(mock_model, MockModelConfig())

    # Verify the calculated value in config
    assert parallel_config.eplb_config.num_redundant_experts == expected_redundant

    # Verify the calculated value in model
    assert mock_model.num_redundant_experts == expected_redundant

    # Verify num_physical_experts is updated correctly
    assert mock_model.num_physical_experts == num_logical_experts + expected_redundant

    # Verify even distribution
    total_experts = num_logical_experts + expected_redundant
    assert total_experts % ep_size == 0, (
        f"Total experts {total_experts} should be evenly divisible by EP size {ep_size}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
def test_manual_redundant_experts_preserved(monkeypatch):
    """Test that manually specified num_redundant_experts is preserved."""
    ep_size = 8
    manual_value = 32

    # Mock EP group
    monkeypatch.setattr(
        "vllm.distributed.eplb.eplb_state.get_ep_group",
        lambda: MockEPGroup(ep_size),
    )

    # User explicitly sets num_redundant_experts = 32
    parallel_config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=ep_size,
        data_parallel_size=1,
        eplb_config=EPLBConfig(num_redundant_experts=manual_value),
    )

    # Create EPLB state
    eplb_state = EplbState(parallel_config, torch.device("cpu"))

    # Create mock model
    mock_model = MockMixtureOfExperts(
        num_logical_experts=256,
        num_routed_experts=256,
        num_redundant_experts=manual_value,
    )

    # Add model
    eplb_state.add_model(mock_model, MockModelConfig())

    # The manually set value should be preserved
    assert parallel_config.eplb_config.num_redundant_experts == manual_value
    assert mock_model.num_redundant_experts == manual_value


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
def test_zero_redundant_experts_preserved(monkeypatch):
    """Test that explicitly setting num_redundant_experts=0 is preserved."""
    ep_size = 8

    # Mock EP group
    monkeypatch.setattr(
        "vllm.distributed.eplb.eplb_state.get_ep_group",
        lambda: MockEPGroup(ep_size),
    )

    # User explicitly sets num_redundant_experts = 0
    parallel_config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=ep_size,
        data_parallel_size=1,
        eplb_config=EPLBConfig(num_redundant_experts=0),
    )

    # Create EPLB state
    eplb_state = EplbState(parallel_config, torch.device("cpu"))

    # Create mock model with experts that are already evenly distributed
    mock_model = MockMixtureOfExperts(
        num_logical_experts=256,  # 256 % 8 == 0
        num_routed_experts=256,
        num_redundant_experts=0,
    )

    # Add model
    eplb_state.add_model(mock_model, MockModelConfig())

    # Explicitly set 0 should be preserved (not auto-calculated)
    assert parallel_config.eplb_config.num_redundant_experts == 0
    assert mock_model.num_redundant_experts == 0


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
def test_validation_eplb_disabled_with_positive_redundant():
    """Test that setting num_redundant_experts > 0 with EPLB disabled raises error."""
    with pytest.raises(ValueError, match="num_redundant_experts is set to"):
        ParallelConfig(
            enable_eplb=False,
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
            eplb_config=EPLBConfig(num_redundant_experts=32),
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
def test_validation_eplb_disabled_with_auto_allowed():
    """Test that num_redundant_experts=-1 (auto) becomes 0 when EPLB disabled.

    When EPLB is disabled, auto-calculation cannot happen, so -1 is converted
    to 0 as a safe fallback to prevent MoE models from using an invalid
    num_physical_experts value (num_routed_experts - 1).
    """
    # This should not raise an error
    config = ParallelConfig(
        enable_eplb=False,
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        data_parallel_size=1,
        eplb_config=EPLBConfig(num_redundant_experts=-1),
    )
    # -1 should be converted to 0 when EPLB is disabled
    assert config.eplb_config.num_redundant_experts == 0


def test_validation_eplb_enabled_without_ep():
    """Test that enable_eplb=True without EP raises error."""
    with pytest.raises(ValueError, match="enable_expert_parallel must be True"):
        ParallelConfig(
            enable_eplb=True,
            enable_expert_parallel=False,
            tensor_parallel_size=8,
            data_parallel_size=1,
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
def test_validation_eplb_disabled_with_zero_allowed():
    """Test that num_redundant_experts=0 is allowed when EPLB disabled."""
    # This should not raise an error
    config = ParallelConfig(
        enable_eplb=False,
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        data_parallel_size=1,
        eplb_config=EPLBConfig(num_redundant_experts=0),
    )
    assert config.eplb_config.num_redundant_experts == 0


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
@pytest.mark.parametrize(
    "num_experts,ep_size",
    [
        (256, 8),
        (100, 12),
        (64, 7),
        (17, 3),
    ],
)
def test_auto_calculation_formula(num_experts: int, ep_size: int, monkeypatch):
    """Test that the auto-calculation formula works correctly.

    The formula should ensure: (num_experts + num_redundant) % ep_size == 0
    """
    # Mock EP group
    monkeypatch.setattr(
        "vllm.distributed.eplb.eplb_state.get_ep_group",
        lambda: MockEPGroup(ep_size),
    )

    parallel_config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=ep_size,
        data_parallel_size=1,
        eplb_config=EPLBConfig(num_redundant_experts=-1),
    )

    eplb_state = EplbState(parallel_config, torch.device("cpu"))

    mock_model = MockMixtureOfExperts(
        num_logical_experts=num_experts,
        num_routed_experts=num_experts,
        num_redundant_experts=-1,
    )

    eplb_state.add_model(mock_model, MockModelConfig())

    calculated_redundant = parallel_config.eplb_config.num_redundant_experts
    total = num_experts + calculated_redundant

    # Verify even distribution
    assert total % ep_size == 0, (
        f"Formula failed: ({num_experts} + {calculated_redundant}) % {ep_size} != 0"
    )

    # Verify it's the minimum (removing one would break even distribution)
    if calculated_redundant > 0:
        assert (num_experts + calculated_redundant - 1) % ep_size != 0, (
            f"Not minimum: {calculated_redundant - 1} would also work"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
def test_preserve_minus_one_when_expert_count_unknown():
    """Test that -1 is preserved when expert count cannot be determined.

    This addresses the scenario where get_num_experts() returns 0 because the
    model config doesn't specify expert counts. The -1 sentinel must be preserved
    to allow fallback auto-calculation in EplbState.add_model() after model
    construction, when the actual expert count becomes available.

    This prevents assertion failures in vllm/distributed/eplb/policy/default.py:146
    for models whose config omits expert counts but whose actual MoE expert count
    is not divisible by EP size.
    """
    # Test the early calculation in VllmConfig.__post_init__()
    config = EPLBConfig(num_redundant_experts=-1)

    # When num_logical_experts is 0 (unknown from config), should preserve -1
    config.auto_set_num_redundant_experts(
        num_logical_experts=0,  # get_num_experts() returned 0
        ep_size=8,
        enable_eplb=True,
    )

    # The value should remain -1 to allow fallback calculation later
    assert config.num_redundant_experts == -1, (
        f"Expected -1 to be preserved when expert count unknown, "
        f"got {config.num_redundant_experts}. "
        "This blocks fallback auto-calculation in EplbState.add_model()"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="EPLB is only supported on CUDA/ROCm devices",
)
@pytest.mark.parametrize(
    "num_logical_experts,tp_size,dp_size,pcp_size,expected_redundant",
    [
        # Test cases with PCP > 1
        (256, 2, 2, 2, 0),  # EP_SIZE = 2*2*2 = 8, 256 % 8 == 0
        (100, 2, 2, 3, 8),  # EP_SIZE = 2*2*3 = 12, 100 % 12 == 4, need 8
        (64, 1, 2, 4, 0),  # EP_SIZE = 1*2*4 = 8, 64 % 8 == 0
        (50, 1, 3, 2, 4),  # EP_SIZE = 1*3*2 = 6, 50 % 6 == 2, need 4
        (17, 2, 1, 3, 1),  # EP_SIZE = 2*1*3 = 6, 17 % 6 == 5, need 1
        (200, 1, 4, 5, 0),  # EP_SIZE = 1*4*5 = 20, 200 % 20 == 0
        (97, 3, 2, 2, 11),  # EP_SIZE = 3*2*2 = 12, 97 % 12 == 1, need 11
    ],
)
def test_auto_calculate_with_prefill_context_parallel(
    num_logical_experts: int,
    tp_size: int,
    dp_size: int,
    pcp_size: int,
    expected_redundant: int,
    monkeypatch,
):
    """Test automatic calculation of num_redundant_experts with PCP > 1.

    This test validates the fix for issue #30075 which ensures that the EP
    group size calculation includes prefill_context_parallel_size (PCP).

    Args:
        num_logical_experts: Number of logical experts in the model
        tp_size: Tensor parallel size
        dp_size: Data parallel size
        pcp_size: Prefill context parallel size
        expected_redundant: Expected calculated num_redundant_experts
        monkeypatch: Pytest monkeypatch fixture
    """
    # Calculate actual EP size: TP * DP * PCP
    ep_size = tp_size * dp_size * pcp_size

    # Mock EP group
    monkeypatch.setattr(
        "vllm.distributed.eplb.eplb_state.get_ep_group",
        lambda: MockEPGroup(ep_size),
    )

    # Create config with num_redundant_experts = -1 (auto-calculate)
    parallel_config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        prefill_context_parallel_size=pcp_size,
        eplb_config=EPLBConfig(num_redundant_experts=-1),
    )

    # Create EPLB state
    eplb_state = EplbState(parallel_config, torch.device("cpu"))

    # Create mock model with -1 for num_redundant_experts (will be auto-calculated)
    mock_model = MockMixtureOfExperts(
        num_logical_experts=num_logical_experts,
        num_routed_experts=num_logical_experts,
        num_redundant_experts=-1,
    )

    # Add model (this should trigger auto-calculation)
    eplb_state.add_model(mock_model, MockModelConfig())

    # Verify the calculated value in config
    assert parallel_config.eplb_config.num_redundant_experts == expected_redundant, (
        f"Expected num_redundant_experts={expected_redundant} for "
        f"experts={num_logical_experts}, TP={tp_size}, DP={dp_size}, "
        f"PCP={pcp_size} (EP_SIZE={ep_size}), but got "
        f"{parallel_config.eplb_config.num_redundant_experts}"
    )

    # Verify the calculated value in model
    assert mock_model.num_redundant_experts == expected_redundant

    # Verify num_physical_experts is updated correctly
    assert mock_model.num_physical_experts == num_logical_experts + expected_redundant

    # Verify even distribution
    total_experts = num_logical_experts + expected_redundant
    assert total_experts % ep_size == 0, (
        f"Total experts {total_experts} should be evenly divisible by "
        f"EP size {ep_size} (TP={tp_size} × DP={dp_size} × PCP={pcp_size})"
    )

    # Verify it's the minimum (removing one would break even distribution)
    if expected_redundant > 0:
        assert (num_logical_experts + expected_redundant - 1) % ep_size != 0, (
            f"Not minimum: {expected_redundant - 1} redundant experts would also work"
        )
