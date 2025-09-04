import pytest
import torch
from unittest.mock import Mock, patch

from vllm.distributed.eplb.eplb_state import EplbState


@pytest.fixture
def mock_model():
    """Create a mock MixtureOfExperts model"""
    model = Mock()
    model.num_routed_experts = 4
    model.num_redundant_experts = 2
    model.num_logical_experts = 4
    model.num_physical_experts = 6  # 4 + 2
    model.num_moe_layers = 2
    model.num_expert_groups = 1
    model.expert_weights = [Mock() for _ in range(2)]  # 2 MoE layers

    # Mock set_eplb_state method
    model.set_eplb_state = Mock()
    return model


@pytest.fixture
def mock_parallel_config():
    """Create a mock ParallelConfig"""
    config = Mock()
    config.eplb_config.window_size = 10
    config.eplb_config.step_interval = 100
    config.eplb_config.num_wait_worker_iterations = 5
    return config


@pytest.fixture
def mock_ep_group():
    """Create a mock EP group"""
    group = Mock()
    group.device_group = Mock()
    group.device_group.size.return_value = 2  # 2 ranks
    group.device_group.rank.return_value = 0
    group.cpu_group = Mock()
    return group


@pytest.fixture
def mock_distributed_env():
    """Mock distributed environment"""
    with patch(
            'vllm.distributed.eplb.eplb_state.get_ep_group') as mock_get_ep_group, \
            patch(
                'vllm.distributed.eplb.eplb_state.get_node_count') as mock_get_node_count, \
            patch(
                'vllm.distributed.eplb.eplb_state.all_reduce') as mock_all_reduce, \
            patch(
                'vllm.distributed.eplb.eplb_state.in_the_same_node_as') as mock_in_the_same_node_as:
        # Set mock return values
        mock_get_node_count.return_value = 1  # Mock 1 node
        mock_in_the_same_node_as.return_value = [True] * 2  # Mock 2 ranks on same node

        # Create mock EP group
        mock_ep_group = Mock()
        mock_ep_group.device_group = Mock()
        mock_ep_group.device_group.size.return_value = 2  # 2 ranks
        mock_ep_group.device_group.rank.return_value = 0
        mock_ep_group.cpu_group = Mock()

        mock_get_ep_group.return_value = mock_ep_group

        yield {
            'get_ep_group': mock_get_ep_group,
            'get_node_count': mock_get_node_count,
            'all_reduce': mock_all_reduce,
            'in_the_same_node_as': mock_in_the_same_node_as,
            'ep_group': mock_ep_group
        }


def test_eplb_state_initialization(mock_model, mock_parallel_config,
                                   mock_distributed_env):
    """Test EplbState initialization"""
    device = torch.device('cpu')

    # Create EplbState
    state = EplbState.build(
        model=mock_model,
        device=device,
        parallel_config=mock_parallel_config
    )

    # Verify basic attributes
    assert state.physical_to_logical_map.shape == (2,
                                                   6)  # 2 layers, 6 physical experts
    assert state.logical_to_physical_map.shape == (2, 4,
                                                   1024)  # 2 layers, 4 logical experts, max redundancy+1
    assert state.logical_replica_count.shape == (2,
                                                 4)  # 2 layers, 4 logical experts
    assert state.expert_load_pass.shape == (2,
                                            6)  # 2 layers, 6 physical experts
    assert state.expert_load_window.shape == (10, 2,
                                              6)  # window size 10, 2 layers, 6 physical experts
    assert state.expert_load_window_size == 10
    assert state.expert_rearrangement_step_interval == 100
    assert state.num_wait_worker_iterations == 5

    # Verify async processor is initialized
    assert state._async_processor is not None
    assert state._async_processor.target_func.__name__ == 'rebalance_experts'


def test_eplb_state_step_without_rearrangement(mock_model, mock_parallel_config,
                                               mock_distributed_env):
    """Test EplbState step method (without triggering rearrangement)"""
    device = torch.device('cpu')
    state = EplbState.build(
        model=mock_model,
        device=device,
        parallel_config=mock_parallel_config
    )

    # Set initial step to ensure no rearrangement is triggered
    state.expert_rearrangement_step = 50  # Less than interval 100

    # Record initial state
    initial_step = state.expert_rearrangement_step
    initial_window_step = state.expert_load_window_step

    # Execute step
    state.step(model=mock_model, is_dummy=False, log_stats=False)

    # Verify steps increased but no rearrangement triggered
    assert state.expert_rearrangement_step == initial_step + 1
    assert state.expert_load_window_step == initial_window_step + 1

    # Verify no rearrangement triggered (async processor should have no task)
    assert not state._async_processor.has_pending_task


def test_eplb_state_step_with_rearrangement(mock_model, mock_parallel_config,
                                            mock_distributed_env):
    """Test EplbState step method (triggering rearrangement)"""
    device = torch.device('cpu')
    state = EplbState.build(
        model=mock_model,
        device=device,
        parallel_config=mock_parallel_config
    )

    # Set initial step to ensure rearrangement is triggered
    state.expert_rearrangement_step = 99  # Next step will equal interval 100

    # Mock async processor's submit_task method
    original_submit_task = state._async_processor.submit_task
    state._async_processor.submit_task = Mock(return_value=True)

    # Execute step
    state.step(model=mock_model, is_dummy=False, log_stats=False)

    # Verify step counter reset
    assert state.expert_rearrangement_step == 0

    # Verify rearrangement task was submitted
    assert state._async_processor.submit_task.called

    # Restore original method
    state._async_processor.submit_task = original_submit_task


def test_eplb_state_rearrange_method(mock_model, mock_parallel_config,
                                     mock_distributed_env):
    """Test EplbState rearrange method"""
    device = torch.device('cpu')
    state = EplbState.build(
        model=mock_model,
        device=device,
        parallel_config=mock_parallel_config
    )

    # Mock async processor's submit_task method
    original_submit_task = state._async_processor.submit_task
    state._async_processor.submit_task = Mock(return_value=True)

    # Call rearrange method
    state.rearrange(model=mock_model)

    # Verify rearrangement task was submitted
    assert state._async_processor.submit_task.called

    # Get call arguments
    call_args = state._async_processor.submit_task.call_args
    args, kwargs = call_args

    # Verify parameters
    assert 'model' in kwargs['post_process_args']
    assert 'ep_group' in kwargs['post_process_args']

    # Restore original method
    state._async_processor.submit_task = original_submit_task


def test_eplb_state_process_async_result(mock_model, mock_parallel_config,
                                         mock_distributed_env):
    """Test EplbState async result processing"""
    # Mock rearrange_expert_weights_inplace function
    with patch(
            'vllm.distributed.eplb.eplb_state.rearrange_expert_weights_inplace') as mock_rearrange:
        device = torch.device('cpu')
        state = EplbState.build(
            model=mock_model,
            device=device,
            parallel_config=mock_parallel_config
        )

        # Create mock rearrangement results
        new_physical_to_logical_map = torch.tensor(
            [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]])
        new_logical_to_physical_map = torch.full((2, 4, 1024), -1)
        new_logical_replica_count = torch.tensor([[2, 2, 1, 1], [2, 2, 1, 1]])

        # Set mock results
        state._async_processor._result = (
            new_physical_to_logical_map,
            new_logical_to_physical_map,
            new_logical_replica_count
        )

        state._async_processor._post_process_args = {
            "model": mock_model,
            "ep_group": mock_distributed_env['ep_group'],
            "is_profile": False,
            "rank_mapping": None,
            "device": device
        }

        # Process async result
        state._process_async_result()

        # Verify expert weights were rearranged
        assert mock_rearrange.called

        # Verify state mappings were updated
        assert torch.equal(state.physical_to_logical_map,
                           new_physical_to_logical_map)
        assert torch.equal(state.logical_to_physical_map,
                           new_logical_to_physical_map)
        assert torch.equal(state.logical_replica_count,
                           new_logical_replica_count)


def test_eplb_state_cleanup(mock_model, mock_parallel_config,
                            mock_distributed_env):
    """Test EplbState cleanup functionality"""
    device = torch.device('cpu')
    state = EplbState.build(
        model=mock_model,
        device=device,
        parallel_config=mock_parallel_config
    )

    # Mock async processor's cleanup method
    original_cleanup = state._async_processor.cleanup
    state._async_processor.cleanup = Mock()

    # Call cleanup
    state.__del__()

    # Verify async processor was cleaned up
    assert state._async_processor.cleanup.called

    # Restore original method
    state._async_processor.cleanup = original_cleanup


if __name__ == "__main__":
    pytest.main([__file__])