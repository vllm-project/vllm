import time
from collections import deque
from typing import List
from unittest.mock import MagicMock

import pytest  # noqa

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.policy import PolicyFactory
from vllm.core.scheduler import Scheduler, SchedulingBudget
from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob, SequenceGroup

from .utils import create_dummy_prompt


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]


def test_scheduler_add_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1, cache_dtype="auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq group to scheduler.
    num_seq_group = 4
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        scheduler.add_seq_group(seq_group)
        assert scheduler.get_num_unfinished_seq_groups() == i + 1


def test_scheduler_abort_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 1)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add multiple seq groups to scheduler.
    num_seq_group = 4
    request_ids = set()
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        scheduler.add_seq_group(seq_group)
        request_ids.add(str(i))

    # Abort all added seq groups.
    assert scheduler.get_num_unfinished_seq_groups() == num_seq_group
    scheduler.abort_seq_group(request_ids)
    assert scheduler.get_num_unfinished_seq_groups() == 0


def test_scheduler_schedule_simple():
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, num_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    seq_group_meta, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group

    # Schedule seq groups generation.
    seq_group_meta, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_seq_group
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group


def test_scheduler_schedule_preempt_abort():
    block_size = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, 2, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 2
    cache_config.num_gpu_blocks = 2
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    seq_a, seq_group_a = create_dummy_prompt("1", block_size)
    seq_b, seq_group_b = create_dummy_prompt("2", block_size)
    scheduler.add_seq_group(seq_group_a)
    scheduler.add_seq_group(seq_group_b)

    # Schedule seq groups prompts.
    seq_group_meta, out = scheduler.schedule()
    assert get_sequence_groups(out) == [seq_group_a, seq_group_b]
    assert out.num_batched_tokens == block_size * 2  # seq_a and seq_b
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 2
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Append "generated" tokens, allowing the sequence to mark prompt tokens as
    # processed.
    token_id = 0
    seq_a.append_token_id(token_id, {token_id: Logprob(0.0)})
    seq_b.append_token_id(token_id, {token_id: Logprob(0.0)})

    # Schedule seq groups generation and preempt seq group b.
    seq_group_meta, out = scheduler.schedule()
    assert get_sequence_groups(out) == [seq_group_a]
    assert out.num_batched_tokens == 1
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Abort seq group a. Re-schedule seq group b prompt with recomputation.
    scheduler.abort_seq_group("1")
    seq_group_meta, out = scheduler.schedule()
    assert get_sequence_groups(out) == [seq_group_b]
    assert out.num_batched_tokens == 5  # 4 prompt + 1 generation.
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 1


def test_scheduler_max_seqs():
    block_size = 4
    num_seq_group = 4
    max_seq_group = 2
    max_model_len = 16
    scheduler_config = SchedulerConfig(64, max_seq_group, max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    all_seq_groups: List[SequenceGroup] = []
    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        all_seq_groups.append(seq_group)

    # Append 1 seq group
    scheduler.add_seq_group(all_seq_groups[0])

    # Schedule seq groups prompts.
    _, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])

    # Schedule seq groups generation.
    _, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])

    # Append 2 more seq group
    scheduler.add_seq_group(all_seq_groups[1])
    scheduler.add_seq_group(all_seq_groups[2])

    # Schedule seq groups prompts.
    # Only 1 seq group should be scheduled since max_seq_group is 2
    # and one is prompting.
    _, out = scheduler.schedule()
    assert set(get_sequence_groups(out)) == set([all_seq_groups[1]])


def test_scheduler_delay_factor():
    block_size = 4
    scheduler_config = SchedulerConfig(100, 64, 16, delay_factor=0.5)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # schedule first prompt
    _, seq_group = create_dummy_prompt("0", prompt_length=block_size)
    scheduler.add_seq_group(seq_group)
    seq_group_meta, out = scheduler.schedule()
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '0'

    # wait for a second before scheduling next prompt
    time.sleep(1)
    _, seq_group = create_dummy_prompt("1", prompt_length=block_size)
    scheduler.add_seq_group(seq_group)

    # second prompt should *not* be scheduled
    seq_group_meta, out = scheduler.schedule()
    assert out.num_prefill_groups == 0
    assert seq_group_meta[0].request_id == '0'

    # wait for more than 0.5 second and try again
    time.sleep(0.6)
    seq_group_meta, out = scheduler.schedule()
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '1'


def test_swapped_out_prioritized():
    scheduler = initialize_scheduler(max_num_seqs=6)
    # best_of=2 * 3 == 6 sequences.
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
        scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 3

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    _, out = scheduler.schedule()
    assert len(out.scheduled_seq_groups) == 2
    assert out.num_batched_tokens == 2
    assert out.blocks_to_swap_out != {}
    assert out.blocks_to_swap_in == {}

    # Add 1 more task. Swap should be prioritized over prefill.
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
    scheduler.add_seq_group(seq_group)
    _, out = scheduler.schedule()
    assert len(out.scheduled_seq_groups) == 3
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 3
    assert out.blocks_to_swap_in != {}
    assert out.blocks_to_swap_out == {}


def initialize_scheduler(*,
                         max_num_seqs=1000,
                         max_token_budget=1000,
                         max_model_len=1000,
                         lora_config=None):
    block_size = 4
    scheduler_config = SchedulerConfig(max_token_budget, max_num_seqs,
                                       max_model_len)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, lora_config)
    return scheduler


def create_token_budget(num_batched_tokens: int = 0,
                        num_curr_seqs: int = 0,
                        token_budget: int = 10000,
                        max_num_seqs: int = 10000) -> SchedulingBudget:
    return SchedulingBudget(
        num_batched_tokens=num_batched_tokens,
        num_curr_seqs=num_curr_seqs,
        token_budget=token_budget,
        max_num_seqs=max_num_seqs,
    )


def test_prefill_schedule_max_prompt_len():
    """
    Test prompt longer than max_prompt_len is aborted.
    """
    scheduler = initialize_scheduler(max_model_len=30)
    _, seq_group = create_dummy_prompt(0, prompt_length=60)
    waiting = deque([seq_group])
    budget = create_token_budget()
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 1
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 0


def test_prefill_schedule_token_budget():
    """
    Test token budget respected.
    """
    scheduler = initialize_scheduler()
    waiting = deque()
    budget = create_token_budget(token_budget=0)
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        waiting.append(seq_group)

    # 0 token budget == nothing is scheduled.
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 2

    # 60 token budget == 1 request scheduled.
    budget = create_token_budget(token_budget=60)
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 1
    assert budget.num_batched_tokens == 60
    assert budget.num_curr_seqs == 1
    assert len(remaining_waiting) == 1

    # Test when current_batched_tokens respected.
    scheduler = initialize_scheduler()
    waiting = deque()
    budget = create_token_budget(num_batched_tokens=30, token_budget=60)
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
    # Cannot schedule a prompt that doesn't fit the budget.
    waiting.append(seq_group)
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 30
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 1
    budget = create_token_budget(num_batched_tokens=30, token_budget=90)
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.seq_groups) == 1
    assert budget.num_batched_tokens == 90
    assert budget.num_curr_seqs == 1
    assert len(remaining_waiting) == 0


def test_prefill_schedule_max_seqs():
    """
    Test max seq respected.
    """
    scheduler = initialize_scheduler()
    waiting = deque()
    budget = create_token_budget(max_num_seqs=2)
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        waiting.append(seq_group)
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert budget.num_batched_tokens == 120
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 1

    # Verify curr_num_seqs respected.
    waiting = deque()
    budget = create_token_budget(num_curr_seqs=2, max_num_seqs=2)
    _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
    waiting.append(seq_group)
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 1


def test_prefill_schedule_max_lora():
    """
    Test max lora is respected and prioritized.
    """
    lora_config = LoRAConfig(max_lora_rank=8, max_loras=1)
    scheduler = initialize_scheduler(lora_config=lora_config)
    waiting = deque()
    budget = create_token_budget(token_budget=120)
    curr_loras = set()
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           lora_request=LoRARequest(
                                               lora_name=str(i),
                                               lora_int_id=i + 1,
                                               lora_local_path="abc"))
        waiting.append(seq_group)
    # Add two more requests to verify lora is prioritized.
    # 0: Lora, 1: Lora, 2: regular, 3: regular
    # In the first iteration, index 0, 2 is scheduled.
    # If a request is not scheduled because it hits max lora, it is
    # prioritized. Verify that.
    for i in range(2, 4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        waiting.append(seq_group)
    # Schedule 2 requests (0 and 2)
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, curr_loras)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert budget.num_batched_tokens == 120
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 2
    assert len(curr_loras) == 1
    # The second lora request is scheduled next as FCFS policy.
    # Reset curr_loras so that it can be scheduled.
    curr_loras = set()
    budget = create_token_budget(token_budget=60)
    remaining_waiting, output = scheduler._schedule_prefills(
        remaining_waiting, budget, curr_loras)
    assert len(output.seq_groups) == 1
    assert output.seq_groups[0].seq_group.request_id == "1"
    assert len(remaining_waiting) == 1
    assert len(curr_loras) == 1
    assert budget.num_batched_tokens == 60


def test_prefill_schedule_no_block_manager_capacity():
    """
    Test sequence cannot be scheduled due to block manager has no capacity.
    """
    scheduler = initialize_scheduler()
    waiting = deque()
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        waiting.append(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.LATER
    remainig_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remainig_waiting) == 3

    scheduler = initialize_scheduler()
    waiting = deque()
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        waiting.append(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.NEVER
    remaining_waiting, output = scheduler._schedule_prefills(
        waiting, budget, None)
    assert len(output.ignored_seq_groups) == 3
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 0


def test_decode_schedule_preempted():
    """
    Test decodes cannot be scheduled and preempted.
    """
    scheduler = initialize_scheduler()
    running = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler._allocate_and_set_running(seq_group)
        running.append(seq_group)
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "1"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    # 1 cannot be scheduled, and the lowest priority (request 2)
    # should be preempted. 1 will also be preempted.
    budget = create_token_budget(num_batched_tokens=3, num_curr_seqs=3)
    remainig_running, output = scheduler._schedule_decodes(
        running, budget, curr_loras, policy)
    assert len(remainig_running) == 0
    assert len(output.seq_groups) == 1
    assert output.seq_groups[0].seq_group.request_id == "0"
    assert len(output.preempted) == 2
    # Verify budgets are updated.
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 1
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == {}
    # Nothing is copied.
    assert output.blocks_to_copy == {}


def test_decode_swap_beam_search():
    """
    Test best_of > 1 swap out blocks
    """
    scheduler = initialize_scheduler()
    running = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        running.append(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)
    scheduler.block_manager.swap_out = MagicMock()
    expected_swap_mapping = {"5": "7"}
    scheduler.block_manager.swap_out.return_value = expected_swap_mapping

    budget = create_token_budget(num_batched_tokens=3, num_curr_seqs=3)
    remainig_running, output = scheduler._schedule_decodes(
        running, budget, curr_loras, policy)
    assert len(remainig_running) == 0
    assert len(output.seq_groups) == 2
    assert output.seq_groups[0].seq_group.request_id == "0"
    assert output.seq_groups[1].seq_group.request_id == "1"
    assert len(output.preempted) == 0
    assert len(output.swapped_out) == 1
    # Budget should refledct preempted requests.
    assert budget.num_batched_tokens == 2
    # since there are 2 sequences, 2 should be subtracted.
    assert budget.num_curr_seqs == 1
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == expected_swap_mapping
    # Nothing is copied.
    assert output.blocks_to_copy == {}


def test_schedule_decode_blocks_to_copy_update():
    """
    Verify blocks_to_copy is updated.
    """
    scheduler = initialize_scheduler()
    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    running = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    scheduler._allocate_and_set_running(seq_group)
    running.append(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.append_slots = MagicMock()
    scheduler.block_manager.append_slots.return_value = {2: [3]}

    budget = create_token_budget()
    remaining_running, output = scheduler._schedule_decodes(
        running, budget, curr_loras, policy)
    assert len(remaining_running) == 0
    assert len(output.seq_groups) == 1
    assert len(output.preempted) == 0
    assert len(output.swapped_out) == 0
    # Nothing is preempted.
    assert output.blocks_to_swap_out == {}
    # Since append_slot returns the source -> dist mapping, it should
    # applied.
    assert output.blocks_to_copy == {2: [3]}


def test_schedule_swapped_simple():
    scheduler = initialize_scheduler()
    swapped = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    blocks_to_swap_out = {}
    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    scheduler._allocate_and_set_running(seq_group)
    scheduler._swap_out(seq_group, blocks_to_swap_out)
    swapped.append(seq_group)

    budget = create_token_budget()
    remaining_swapped, output = scheduler._schedule_swapped(
        swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 0
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 2
    assert len(output.seq_groups) == 1
    # swap in is the reverse of swap out
    blocks_to_swap_in_reverse = {}
    for swapin, swapout in output.blocks_to_swap_in.items():
        blocks_to_swap_in_reverse[swapout] = swapin
    assert blocks_to_swap_out == blocks_to_swap_in_reverse


def test_schedule_swapped_max_token_budget():
    scheduler = initialize_scheduler()
    swapped = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    blocks_to_swap_out = {}
    for _ in range(2):
        _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        swapped.append(seq_group)

    budget = create_token_budget(token_budget=1)
    remaining_swapped, output = scheduler._schedule_swapped(
        swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 2
    assert len(output.seq_groups) == 1

    # Verify num_batched_tokens are respected.
    budget = create_token_budget(num_batched_tokens=1, token_budget=1)
    remaining_swapped, output = scheduler._schedule_swapped(
        remaining_swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 0
    assert len(output.seq_groups) == 0


def test_schedule_swapped_max_seqs():
    scheduler = initialize_scheduler()
    swapped = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    blocks_to_swap_out = {}
    for _ in range(2):
        _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        swapped.append(seq_group)

    budget = create_token_budget(max_num_seqs=2)
    remaining_swapped, output = scheduler._schedule_swapped(
        swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 2
    assert len(output.seq_groups) == 1

    # Verify num_curr_seqs are respected.
    budget = create_token_budget(num_curr_seqs=2, max_num_seqs=2)
    remaining_swapped, output = scheduler._schedule_swapped(
        remaining_swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 2
    assert len(output.seq_groups) == 0


def test_schedule_swapped_max_loras():
    lora_config = LoRAConfig(max_lora_rank=8, max_loras=1)
    scheduler = initialize_scheduler(lora_config=lora_config)
    swapped = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = set()
    blocks_to_swap_out = {}
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           lora_request=LoRARequest(
                                               lora_name=str(i),
                                               lora_int_id=i + 1,
                                               lora_local_path="abc"))
        scheduler._allocate_and_set_running(seq_group)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        swapped.append(seq_group)

    budget = create_token_budget()
    remaining_swapped, output = scheduler._schedule_swapped(
        swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 1
    assert len(output.seq_groups) == 1
    assert len(curr_loras) == 1


def test_schedule_swapped_cannot_swap_in():
    scheduler = initialize_scheduler()
    swapped = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    blocks_to_swap_out = {}
    for _ in range(2):
        _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        swapped.append(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.can_swap_in = MagicMock()
    scheduler.block_manager.can_swap_in.return_value = False
    # Since we cannot swap in, none of the requests are swapped in.
    budget = create_token_budget()
    remaining_swapped, output = scheduler._schedule_swapped(
        swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 2
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(output.seq_groups) == 0


def test_schedule_swapped_blocks_to_copy():
    scheduler = initialize_scheduler()
    swapped = deque()
    policy = PolicyFactory.get_policy(policy_name="fcfs")
    curr_loras = None
    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    scheduler._allocate_and_set_running(seq_group)
    blocks_to_swap_out = {}
    scheduler._swap_out(seq_group, blocks_to_swap_out)
    swapped.append(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.append_slots = MagicMock()
    scheduler.block_manager.append_slots.return_value = {2: [3]}

    budget = create_token_budget()
    remaining_swapped, output = scheduler._schedule_swapped(
        swapped, budget, curr_loras, policy)
    assert len(remaining_swapped) == 0
    assert len(output.seq_groups) == 1
    assert output.blocks_to_copy == {2: [3]}
