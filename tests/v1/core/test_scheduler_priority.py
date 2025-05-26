import heapq
import time
import unittest
from collections import deque
from unittest.mock import MagicMock, PropertyMock

from vllm.config import VllmConfig, SchedulerConfig, KVCacheConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm.v1.engine import EngineCoreEventType


# Helper function to create a mock Request
def create_mock_request(request_id: str,
                        priority: int,
                        arrival_time: float,
                        num_prompt_tokens: int = 10,
                        num_output_tokens: int = 1,
                        status: RequestStatus = RequestStatus.WAITING) -> Request:
    mock_req = MagicMock(spec=Request)
    mock_req.request_id = request_id
    mock_req.priority = priority
    mock_req.arrival_time = arrival_time
    mock_req.prompt_token_ids = list(range(num_prompt_tokens))
    mock_req.num_prompt_tokens = num_prompt_tokens
    mock_req._output_token_ids = [] # Internal list for output_token_ids
    mock_req.output_token_ids = [] # Public view
    mock_req._all_token_ids = list(range(num_prompt_tokens)) # Internal list for all_token_ids
    mock_req.all_token_ids = list(range(num_prompt_tokens)) # Public view

    # This property needs to be dynamic based on _output_token_ids and spec_token_ids
    type(mock_req).num_tokens_with_spec = PropertyMock(
        return_value=num_prompt_tokens + len(mock_req.output_token_ids) + len(getattr(mock_req, 'spec_token_ids', []))
    )
    type(mock_req).num_tokens = PropertyMock(
        return_value=num_prompt_tokens + len(mock_req.output_token_ids)
    )
    type(mock_req).num_output_tokens = PropertyMock(
        return_value=len(mock_req.output_token_ids)
    )

    mock_req.num_computed_tokens = 0
    mock_req.status = status
    mock_req.sampling_params = MagicMock()
    mock_req.sampling_params.max_tokens = num_output_tokens + 10 # Allow some generation
    mock_req.lora_request = None
    mock_req.structured_output_request = None
    mock_req.mm_positions = []
    mock_req.mm_inputs = []
    mock_req.mm_hashes = []
    mock_req.has_encoder_inputs = False
    mock_req.spec_token_ids = []
    mock_req.use_structured_output = False
    mock_req.num_cached_tokens = -1


    def record_event_side_effect(event_type, timestamp=None):
        if not hasattr(mock_req, 'events'):
            mock_req.events = []
        mock_req.events.append(MagicMock(type=event_type, timestamp=timestamp or time.monotonic()))

    mock_req.record_event = MagicMock(side_effect=record_event_side_effect)
    
    def append_output_token_ids_side_effect(token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        mock_req._output_token_ids.extend(token_ids)
        mock_req._all_token_ids.extend(token_ids)
        # Update property mocks if necessary, though for these tests, fixed values might be okay
        # For simplicity, we assume num_tokens_with_spec and num_tokens don't change drastically after creation
        # in a way that affects basic scheduling decisions for these tests.

    mock_req.append_output_token_ids = MagicMock(side_effect=append_output_token_ids_side_effect)

    def get_finished_reason_side_effect():
        if mock_req.status == RequestStatus.FINISHED_STOPPED:
            return "stop"
        return None
    mock_req.get_finished_reason = MagicMock(side_effect=get_finished_reason_side_effect)
    mock_req.stop_reason = None

    return mock_req


class TestSchedulerPriority(unittest.TestCase):

    def setUp(self):
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        
        # SchedulerConfig
        self.mock_scheduler_config = MagicMock(spec=SchedulerConfig)
        self.mock_scheduler_config.max_num_seqs = 2 # Max 2 running requests
        self.mock_scheduler_config.max_num_batched_tokens = 2048 
        self.mock_scheduler_config.max_model_len = 4096
        self.mock_scheduler_config.long_prefill_token_threshold = 0 # Disable for these tests
        self.mock_scheduler_config.disable_chunked_mm_input = False
        self.mock_vllm_config.scheduler_config = self.mock_scheduler_config

        # KVCacheConfig
        self.mock_kv_cache_config = MagicMock(spec=KVCacheConfig)
        self.mock_kv_cache_config.block_size = 16
        self.mock_kv_cache_config.num_gpu_blocks = 128  # Enough blocks for tests
        self.mock_kv_cache_config.num_cpu_blocks = 128
        self.mock_kv_cache_config.kv_cache_dtype = "auto"
        self.mock_kv_cache_config.kv_cache_groups = [MagicMock()] # Simplistic mock for kv_cache_groups
        self.mock_vllm_config.cache_config = self.mock_kv_cache_config # Assign to vllm_config

        # Other configs needed by Scheduler's KVCacheManager
        self.mock_vllm_config.model_config = MagicMock()
        self.mock_vllm_config.model_config.max_model_len = self.mock_scheduler_config.max_model_len
        self.mock_vllm_config.model_config.get_vocab_size = MagicMock(return_value=32000)
        self.mock_vllm_config.speculative_config = None
        self.mock_vllm_config.lora_config = None
        self.mock_vllm_config.kv_events_config = None
        self.mock_vllm_config.kv_transfer_config = None # Important for disabling KVConnector

        # KVCacheManager Mock
        self.mock_kv_cache_manager = MagicMock(spec_set=KVCacheManager)
        self.mock_kv_cache_manager.allocate_slots = MagicMock()
        self.mock_kv_cache_manager.free = MagicMock()
        self.mock_kv_cache_manager.get_block_ids = MagicMock(return_value=[[1,2,3]]) # Dummy block IDs
        self.mock_kv_cache_manager.get_computed_blocks = MagicMock(return_value=(MagicMock(), 0))
        self.mock_kv_cache_manager.take_events = MagicMock(return_value=[])


        # StructuredOutputManager Mock
        self.mock_structured_output_manager = MagicMock(spec=StructuredOutputManager)
        self.mock_structured_output_manager.grammar_bitmask = MagicMock(return_value={})
        
        # EncoderCacheManager Mock (needed by Scheduler)
        self.mock_encoder_cache_manager = MagicMock()
        self.mock_encoder_cache_manager.get_freed_ids = MagicMock(return_value=[])


    def _create_scheduler(self, policy: str) -> Scheduler:
        self.mock_scheduler_config.policy = policy
        
        # Re-create KVCacheManager instance for each scheduler if its behavior depends on config
        # For these tests, the shared mock_kv_cache_manager is okay as its methods are mocked.
        
        scheduler = Scheduler(
            vllm_config=self.mock_vllm_config,
            kv_cache_config=self.mock_kv_cache_config, # Pass the KVCacheConfig object
            structured_output_manager=self.mock_structured_output_manager,
            log_stats=False
        )
        # Replace the internally created managers with mocks
        scheduler.kv_cache_manager = self.mock_kv_cache_manager
        scheduler.encoder_cache_manager = self.mock_encoder_cache_manager
        return scheduler

    def test_basic_priority_scheduling(self):
        scheduler = self._create_scheduler(policy="priority")
        self.mock_kv_cache_manager.allocate_slots.return_value = MagicMock() # Always successful allocation

        req1 = create_mock_request("req1", priority=1, arrival_time=time.monotonic() + 0.01, num_prompt_tokens=10)
        req2 = create_mock_request("req2", priority=0, arrival_time=time.monotonic() + 0.02, num_prompt_tokens=10) # Higher priority
        req3 = create_mock_request("req3", priority=1, arrival_time=time.monotonic() + 0.03, num_prompt_tokens=10)
        req4 = create_mock_request("req4", priority=0, arrival_time=time.monotonic() + 0.04, num_prompt_tokens=10) # Higher priority, later arrival

        scheduler.add_request(req1)
        scheduler.add_request(req2)
        scheduler.add_request(req3)
        scheduler.add_request(req4)

        # Schedule first batch (max 2 running)
        output1 = scheduler.schedule()
        scheduled_ids1 = [r.req_id for r in output1.scheduled_new_reqs]
        self.assertEqual(len(scheduled_ids1), 2)
        self.assertEqual(scheduled_ids1, ["req2", "req4"]) # req2 and req4 have priority 0

        # Schedule second batch
        output2 = scheduler.schedule()
        scheduled_ids2 = [r.req_id for r in output2.scheduled_new_reqs]
        self.assertEqual(len(scheduled_ids2), 2)
        self.assertEqual(scheduled_ids2, ["req1", "req3"]) # req1 and req3 have priority 1

    def test_fcfs_for_same_priority(self):
        scheduler = self._create_scheduler(policy="priority")
        self.mock_kv_cache_manager.allocate_slots.return_value = MagicMock()

        # All priority 0, different arrival times
        req1_p0_t1 = create_mock_request("req1_p0_t1", priority=0, arrival_time=time.monotonic() + 0.01)
        req2_p0_t2 = create_mock_request("req2_p0_t2", priority=0, arrival_time=time.monotonic() + 0.02)
        req3_p1_t0 = create_mock_request("req3_p1_t0", priority=1, arrival_time=time.monotonic() + 0.00) # Lower priority, earliest arrival
        
        scheduler.add_request(req3_p1_t0) # Added first, but lower priority
        scheduler.add_request(req1_p0_t1) 
        scheduler.add_request(req2_p0_t2)

        output = scheduler.schedule()
        scheduled_ids = [r.req_id for r in output.scheduled_new_reqs]
        self.assertEqual(len(scheduled_ids), 2) # Max 2 running
        self.assertEqual(scheduled_ids, ["req1_p0_t1", "req2_p0_t2"]) # FCFS among priority 0

        output2 = scheduler.schedule()
        scheduled_ids2 = [r.req_id for r in output2.scheduled_new_reqs]
        self.assertEqual(len(scheduled_ids2), 1)
        self.assertEqual(scheduled_ids2, ["req3_p1_t0"])


    def test_preemption_lowest_priority(self):
        scheduler = self._create_scheduler(policy="priority")
        self.mock_scheduler_config.max_num_seqs = 2 # Allow 2 running requests

        # Initial running requests
        running_req1_p1_t1 = create_mock_request("running_req1_p1_t1", priority=1, arrival_time=1.0, status=RequestStatus.RUNNING)
        running_req2_p2_t2 = create_mock_request("running_req2_p2_t2", priority=2, arrival_time=2.0, status=RequestStatus.RUNNING) # Lowest priority among running
        
        scheduler.running = [running_req1_p1_t1, running_req2_p2_t2]
        scheduler.requests = {r.request_id: r for r in scheduler.running}


        # Simulate KVCacheManager being full for new requests initially, then becoming available after preemption
        # For the new high-priority request, allocate_slots will initially fail, triggering preemption.
        # Then, for the same request in the next schedule call (after it's added to waiting and picked again), it should succeed.
        
        allocation_attempts = 0
        def allocate_slots_side_effect(req, num_new_tokens, *args, **kwargs):
            nonlocal allocation_attempts
            if req.request_id == "waiting_req0_p0_t0": # The new high-priority request
                allocation_attempts +=1
                if allocation_attempts == 1: # First attempt for the new request
                     # Ensure this is called when trying to schedule the new high-prio request
                    self.assertEqual(len(scheduler.running), 2) # Max capacity before preemption
                    return None # No space, trigger preemption
                else: # Second attempt for this request (after preemption)
                    return MagicMock() # Space is now available
            return MagicMock() # Allocate for other requests if any (e.g. if running ones are scheduled again)

        self.mock_kv_cache_manager.allocate_slots.side_effect = allocate_slots_side_effect

        # New high-priority request
        waiting_req0_p0_t0 = create_mock_request("waiting_req0_p0_t0", priority=0, arrival_time=0.5)
        scheduler.add_request(waiting_req0_p0_t0)

        # First schedule call: should preempt running_req2_p2_t2
        output1 = scheduler.schedule()
        
        # Check running queue: running_req1 should still be there, waiting_req0 should have been scheduled
        self.assertEqual(len(scheduler.running), 2)
        running_ids_after_schedule1 = {r.request_id for r in scheduler.running}
        self.assertIn("running_req1_p1_t1", running_ids_after_schedule1)
        self.assertIn("waiting_req0_p0_t0", running_ids_after_schedule1)
        self.assertEqual(waiting_req0_p0_t0.status, RequestStatus.RUNNING)

        # Check waiting queue (heap): running_req2_p2_t2 should be there
        self.assertEqual(len(scheduler.waiting), 1)
        preempted_priority, _, preempted_req_obj = scheduler.waiting[0] # It's a heap
        self.assertEqual(preempted_req_obj.request_id, "running_req2_p2_t2")
        self.assertEqual(preempted_req_obj.status, RequestStatus.PREEMPTED)
        self.assertEqual(preempted_priority, 2)
        
        # Check that allocate_slots was called twice for the new request (once failed, once succeeded)
        # This check is a bit tricky due to how many times schedule might be called internally or how allocate_slots is used.
        # The important part is that preemption happened and the correct request was preempted.
        # A simpler check might be to ensure the preempted request is in the waiting queue.

    def test_fcfs_policy_scheduling(self):
        scheduler = self._create_scheduler(policy="fcfs")
        self.mock_kv_cache_manager.allocate_slots.return_value = MagicMock()

        req1_p1_t1 = create_mock_request("req1_p1_t1", priority=1, arrival_time=time.monotonic() + 0.01)
        req2_p0_t2 = create_mock_request("req2_p0_t2", priority=0, arrival_time=time.monotonic() + 0.02) # Priority ignored
        req3_p2_t0 = create_mock_request("req3_p2_t0", priority=2, arrival_time=time.monotonic() + 0.00) # Priority ignored, earliest arrival

        scheduler.add_request(req1_p1_t1) # Added first by this call, but t0 is earlier
        scheduler.add_request(req2_p0_t2)
        scheduler.add_request(req3_p2_t0) # Added last by this call, but t0 is earliest

        # Order of addition to scheduler's internal deque matters for FCFS if arrival_time were identical
        # But here, arrival_time should dominate if the scheduler sorts by it or processes in order of add_request
        # vLLM's FCFS is based on the order requests are added to the self.waiting deque.

        # Expected order if add_request appends and schedule processes from left of deque:
        # req1_p1_t1, req2_p0_t2, req3_p2_t0
        
        output1 = scheduler.schedule()
        scheduled_ids1 = [r.req_id for r in output1.scheduled_new_reqs]
        self.assertEqual(len(scheduled_ids1), 2)
        self.assertEqual(scheduled_ids1, ["req1_p1_t1", "req2_p0_t2"])

        output2 = scheduler.schedule()
        scheduled_ids2 = [r.req_id for r in output2.scheduled_new_reqs]
        self.assertEqual(len(scheduled_ids2), 1)
        self.assertEqual(scheduled_ids2, ["req3_p2_t0"])

    def test_fcfs_policy_preemption(self):
        scheduler = self._create_scheduler(policy="fcfs")
        self.mock_scheduler_config.max_num_seqs = 1 # Max 1 running request

        running_req1 = create_mock_request("running_req1", priority=0, arrival_time=1.0, status=RequestStatus.RUNNING)
        scheduler.running = [running_req1]
        scheduler.requests = {running_req1.request_id: running_req1}
        
        self.mock_kv_cache_manager.allocate_slots.return_value = None # No space, trigger preemption

        waiting_req2 = create_mock_request("waiting_req2", priority=1, arrival_time=2.0) # Priority ignored
        scheduler.add_request(waiting_req2)

        output = scheduler.schedule()

        # running_req1 should be preempted
        self.assertEqual(len(scheduler.running), 1) # waiting_req2 should now be running
        self.assertEqual(scheduler.running[0].request_id, "waiting_req2")
        self.assertEqual(waiting_req2.status, RequestStatus.RUNNING)
        
        self.assertEqual(len(scheduler.waiting), 1)
        self.assertEqual(scheduler.waiting[0].request_id, "running_req1")
        self.assertEqual(running_req1.status, RequestStatus.PREEMPTED)

if __name__ == "__main__":
    unittest.main()
