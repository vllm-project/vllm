# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import multiprocessing
import queue
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import partial
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Optional
from weakref import finalize

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.utils import sched_yield
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.utils import LazyLoader, get_mp_context
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm.v1.structured_output.request import FutureGrammar

GRAMMAR_BITMASK_SHM_NAME = "vllm_grammar_bitmask_shm"
GRAMMAR_READY_FLAG_SHM_NAME = "vllm_grammar_ready_flag"

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)
"""
## Main Classes

This module contains 3 main classes for structured output processing:

### 1. StructuredOutputManager
**Process**: Main  
**Purpose**: Queue-based manager that coordinates structured output operations
**Responsibilities**: 
- Submit tasks via multiprocessing queues
- Retrieve results from child process  
- Create and manage single child process

### 2. StructuredOutputGateway  
**Process**: Child  
**Purpose**: Background process that receives and executes tasks
**Responsibilities**:
- Receive tasks from task queue
- Execute tasks using StructuredOutputExecutor
- Send results back via result queues
- Isolate heavy computation from main process

### 3. StructuredOutputExecutor
**Process**: Child (inside gateway)  
**Purpose**: Performs actual structured output operations
**Responsibilities**:
- Grammar compilation
- Bitmask generation  
- Token acceptance
- All core structured output work


## Communication Architecture

The structured output system uses a combination of multiprocessing queues and
shared memory for communication between the main process 
(StructuredOutputManager) and the child process (StructuredOutputGateway):

### 1. task_queue
**Direction**: Main → Child  
**Purpose**: Sends task requests from manager to gateway

### 2. bitmask_shared_memory + grammar_ready_flag  
**Direction**: Child → Main  
**Purpose**: Returns grammar bitmask results for token generation via shared 
memory
**Usage**: Runs asynchronously with submit_grammar_bitmask() and return 
GrammarBitmaskPlaceholder. When GrammarBitmaskPlaceholder.result() is called,
polls shared memory flag until bitmask is ready
**Content**: Bitmask numpy array written to shared memory with ready flag 
signaling
**Implementation**: Uses `GRAMMAR_BITMASK_SHM_NAME` and 
`GRAMMAR_READY_FLAG_SHM_NAME`

### 3. batch_validate_result_queue
**Direction**: Child → Main  
**Purpose**: Validated token sequences for speculative decoding
**Usage**: Synchronous - manager blocks until validation completes  
**Content**: StructuredOutputResult with validated token dictionaries

### 4. grammar_init_notification_queue
**Direction**: Child → Main  
**Purpose**: Notifies main process when grammar initialization completes
**Usage**: Asynchronous - manager polls queue to update initialization 
status
**Content**: StructuredOutputResult with completed request_id

### Communication Flow
1. **Task Submission**: Manager creates StructuredOutputTask 
   and puts in task_queue
2. **Task Execution**: Gateway retrieves task, 
   executes via StructuredOutputExecutor
3. **Result Routing**: Gateway routes results to appropriate result queue 
   based on task type
4. **Result Retrieval**: Manager retrieves results (synchronously or polls 
   asynchronously)
"""


@dataclass
class GrammarInitData:
    """
    Lightweight data structure containing only the necessary fields 
    from Request for grammar initialization.
    """
    request_id: str
    guided_decoding_backend: str
    structured_output_key: tuple

    @classmethod
    def from_request(cls, request: Request) -> GrammarInitData:
        assert request.structured_output_request is not None
        assert request.sampling_params is not None
        return cls(request_id=request.request_id,
                   guided_decoding_backend=request.sampling_params.
                   structured_outputs._backend,
                   structured_output_key=request.structured_output_request.
                   structured_output_key)


class TaskType(Enum):
    GRAMMAR_INIT = 1
    GRAMMAR_DELETE = 2
    GRAMMAR_BITMASK = 3
    BATCH_ACCEPT_TOKENS = 4
    BATCH_VALIDATE_TOKENS = 5
    CLEAR_BACKEND = 6
    SHUTDOWN = 7


class StructuredOutputTask:

    def __init__(self, task_type: TaskType, args: tuple, kwargs: dict):
        self.task_type = task_type
        self.args = args
        self.kwargs = kwargs


class StructuredOutputResult:

    def __init__(self,
                 task_type: TaskType,
                 result: Any,
                 error: Optional[Exception] = None):
        self.task_type = task_type
        self.result = result
        self.error = error


class StructuredOutputGateway:
    """
    Runs on single CHILD process (created by StructuredOutputManager).
    Background process that receives tasks from queue, executes them using
    StructuredOutputExecutor, and sends results back via queues. Isolates heavy
    computation from the main process.
    """

    def __init__(self, task_queue, batch_validate_result_queue,
                 grammar_init_notification_queue, vllm_config: VllmConfig,
                 bitmask_shm_name: str, ready_flag_shm_name: str):
        self.task_queue = task_queue
        self.batch_validate_result_queue = batch_validate_result_queue
        self.grammar_init_notification_queue = grammar_init_notification_queue
        self.vllm_config = vllm_config
        self.bitmask_shm_name = bitmask_shm_name
        self.ready_flag_shm_name = ready_flag_shm_name
        self.structured_output_executor: Optional[
            StructuredOutputExecutor] = None

    @staticmethod
    def run_gateway(task_queue, batch_validate_result_queue,
                    grammar_init_notification_queue, vllm_config: VllmConfig,
                    bitmask_shm_name: str, ready_flag_shm_name: str):
        """Static method to run the gateway in a separate process."""
        gateway = StructuredOutputGateway(task_queue,
                                          batch_validate_result_queue,
                                          grammar_init_notification_queue,
                                          vllm_config,
                                          bitmask_shm_name,
                                          ready_flag_shm_name)
        gateway.run()

    def run(self):
        """Main processing loop for the child process."""
        logger.debug("StructuredOutputGateway starting - PID: %s",
                     multiprocessing.current_process().pid)
        self.structured_output_executor = StructuredOutputExecutor(
            self.vllm_config)

        # Attach to shared memory in child process
        self.bitmask_shm = shared_memory.SharedMemory(
            name=self.bitmask_shm_name)

        while True:
            try:
                task = self.task_queue.get()
                if task.task_type == TaskType.SHUTDOWN:
                    logger.debug("StructuredOutputGateway shutting down")
                    self.bitmask_shm.close()
                    break
                result = self._execute_task(task)
                # Only put result in queue if it's needed
                if task.task_type == TaskType.GRAMMAR_INIT:
                    # Notify main process that grammar init is complete
                    self.grammar_init_notification_queue.put(result)
                elif task.task_type == TaskType.GRAMMAR_BITMASK:
                    # Write bitmask to shared memory and signal flag
                    if result.result is not None and result.error is None:
                        self._write_bitmask_to_shared_memory(result.result)
                    else:
                        # Set the flag even on error so result() doesn't hang
                        flag_shm = shared_memory.SharedMemory(
                            name=self.ready_flag_shm_name)
                        flag_shm.buf[0] = 1
                        flag_shm.close()
                elif task.task_type == TaskType.BATCH_VALIDATE_TOKENS:
                    self.batch_validate_result_queue.put(result)
            except Exception as e:
                logger.debug("Error in StructuredOutputGateway: %s", e)
                if task.task_type == TaskType.GRAMMAR_INIT:
                    error_result = StructuredOutputResult(
                        task.task_type, None, e)
                    self.grammar_init_notification_queue.put(error_result)
                elif task.task_type == TaskType.GRAMMAR_BITMASK:
                    # Signal flag even on error so result() doesn't hang
                    flag_shm = shared_memory.SharedMemory(
                        name=self.ready_flag_shm_name)
                    flag_shm.buf[0] = 1
                    flag_shm.close()
                elif task.task_type == TaskType.BATCH_VALIDATE_TOKENS:
                    error_result = StructuredOutputResult(
                        task.task_type, None, e)
                    self.batch_validate_result_queue.put(error_result)

    def _execute_task(self,
                      task: StructuredOutputTask) -> StructuredOutputResult:
        assert self.structured_output_executor is not None
        try:
            if task.task_type == TaskType.GRAMMAR_INIT:
                self.structured_output_executor.grammar_init(
                    *task.args, **task.kwargs)
                # Return the request_id so Gateway can notify main process
                grammar_init_data = task.args[0]
                return StructuredOutputResult(task.task_type,
                                              grammar_init_data.request_id)
            elif task.task_type == TaskType.GRAMMAR_DELETE:
                self.structured_output_executor.grammar_delete(
                    *task.args, **task.kwargs)
                return StructuredOutputResult(task.task_type, None)
            elif task.task_type == TaskType.GRAMMAR_BITMASK:
                result = self.structured_output_executor.grammar_bitmask(
                    *task.args, **task.kwargs)
                return StructuredOutputResult(task.task_type, result)
            elif task.task_type == TaskType.BATCH_ACCEPT_TOKENS:
                self.structured_output_executor.batch_accept_tokens(
                    *task.args, **task.kwargs)
                return StructuredOutputResult(task.task_type, None)
            elif task.task_type == TaskType.BATCH_VALIDATE_TOKENS:
                result = self.structured_output_executor.batch_validate_tokens(
                    *task.args, **task.kwargs)
                return StructuredOutputResult(task.task_type, result)
            elif task.task_type == TaskType.CLEAR_BACKEND:
                self.structured_output_executor.clear_backend()
                return StructuredOutputResult(task.task_type, None)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
        except Exception as e:
            return StructuredOutputResult(task.task_type, None, e)

    def _write_bitmask_to_shared_memory(self, bitmask_array: np.ndarray):
        """Write bitmask numpy array to shared memory."""
        # Write shape info first (2 int32 values)
        shape_info = np.array(bitmask_array.shape, dtype=np.int32)
        self.bitmask_shm.buf[:8] = shape_info.tobytes()

        # Write actual data
        data_bytes = bitmask_array.tobytes()
        self.bitmask_shm.buf[8:8 + len(data_bytes)] = data_bytes

        # Ensure all writes are visible before setting the flag
        # Python's SharedMemory is backed by mmap, which provides coherent
        # memory access across processes. On most architectures, the memory
        # model ensures that when process B opens a SharedMemory object and
        # sees flag=1, all previous writes by process A will be visible.
        # This is because SharedMemory creation/attachment involves system
        # calls that act as memory barriers.

        # Set the flag to indicate bitmask is ready
        flag_shm = shared_memory.SharedMemory(name=self.ready_flag_shm_name)
        flag_shm.buf[0] = 1
        flag_shm.close()


class StructuredOutputManager:
    """
    Runs on MAIN process. Queue-based manager that coordinates structured
    output operations by submitting tasks via multiprocessing queues and
    retrieving results. Creates and manages a single child process. Methods
    with `submit_` prefix are run on child process by passing the task via
    queues. Other methods are executed on the current process.
    """

    def __init__(self, vllm_config: VllmConfig):

        self.vllm_config = vllm_config
        self.reasoner: Optional[ReasoningParser] = None
        
        # Create shared memory names with data parallel rank
        data_parallel_rank = str(self.vllm_config.parallel_config.data_parallel_rank)
        self.bitmask_shm_name = GRAMMAR_BITMASK_SHM_NAME + data_parallel_rank
        self.ready_flag_shm_name = GRAMMAR_READY_FLAG_SHM_NAME + data_parallel_rank

        if not self.vllm_config.model_config.skip_tokenizer_init:
            self.tokenizer = init_tokenizer_from_configs(
                model_config=vllm_config.model_config)
            reasoning_parser = \
                    self.vllm_config.structured_outputs_config.reasoning_parser
            if reasoning_parser:
                reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser)
                self.reasoner = reasoner_cls(tokenizer=self.tokenizer)

        # Set to track initialized grammars in main process
        self.initialized_grammars: set[str] = set()

        # Start the child process using vLLM's multiprocessing context
        mp_context = get_mp_context()
        self.task_queue = mp_context.Queue()
        self.batch_validate_result_queue = mp_context.Queue()
        self.grammar_init_notification_queue = mp_context.Queue()

        # Create shared memory for bitmask results
        # Calculate size based on max batch size and vocab size
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        max_spec_tokens = 0
        if vllm_config.speculative_config is not None:
            max_spec_tokens = \
                vllm_config.speculative_config.num_speculative_tokens
        vocab_size = vllm_config.model_config.get_vocab_size()
        # Size: max_num_seqs * ((vocab_size + 31) // 32) * 4 bytes
        #       + 8 bytes for shape
        max_num_seqs = max_batch_size * (1 + max_spec_tokens)
        shm_size = max_num_seqs * ((vocab_size + 31) // 32) * 4 + 8

        self._cleanup_existing_shared_memory()
        self.bitmask_shm = shared_memory.SharedMemory(
            create=True, size=shm_size, name=self.bitmask_shm_name)

        # Create shared memory flag for bitmask ready signaling
        self.flag_shm = shared_memory.SharedMemory(
            name=self.ready_flag_shm_name, create=True, size=1)
        self.flag_shm.buf[0] = 0  # Initialize flag to 0 (not ready)

        # Create a partial config with only the required fields
        partial_config = VllmConfig(
            scheduler_config=self.vllm_config.scheduler_config,
            model_config=self.vllm_config.model_config,
            speculative_config=self.vllm_config.speculative_config,
            structured_outputs_config=self.vllm_config.
            structured_outputs_config,
            lora_config=self.vllm_config.lora_config,
        )

        self.gateway_process = mp_context.Process(
            target=StructuredOutputGateway.run_gateway,
            name="StructuredOutputGateway",
            args=(self.task_queue, self.batch_validate_result_queue,
                  self.grammar_init_notification_queue, partial_config,
                  self.bitmask_shm_name, self.ready_flag_shm_name),
            daemon=True)
        self.gateway_process.start()

        logger.debug(
            "StructuredOutputManager started with child process PID: %s",
            self.gateway_process.pid)

    def submit_grammar_bitmask(self, requests: dict[str, Request],
                               structured_output_request_ids: dict[str, int],
                               scheduled_spec_decode_tokens: dict[str,
                                                                  list[int]]):
        """Submit grammar_bitmask task asynchronously."""
        if not structured_output_request_ids:
            return None

        # Clear the flag for new bitmask calculation
        self.flag_shm.buf[0] = 0

        self.update_reasoning_ended(requests, structured_output_request_ids)
        req_reasoning_ended = {}
        for request_id, _ in structured_output_request_ids.items():
            request = requests[request_id]
            assert request.structured_output_request is not None
            req_reasoning_ended[request_id] = (
                request.structured_output_request.reasoning_ended)

        task = StructuredOutputTask(
            TaskType.GRAMMAR_BITMASK,
            (structured_output_request_ids, req_reasoning_ended,
             scheduled_spec_decode_tokens), {})
        self.task_queue.put(task)

        # Return a placeholder that consumer can check
        return GrammarBitmaskPlaceholder(self.bitmask_shm_name, self.ready_flag_shm_name)

    def submit_batch_accept_tokens(
            self, request_id_to_new_token_ids: list[tuple[str, list[int]]]):
        """Submit batch_accept_tokens task (fire-and-forget)."""
        if len(request_id_to_new_token_ids) == 0:
            return
        task = StructuredOutputTask(TaskType.BATCH_ACCEPT_TOKENS,
                                    (request_id_to_new_token_ids, ), {})
        self.task_queue.put(task)

    def submit_batch_validate_tokens(
            self,
            request_id_to_token_ids: dict[str,
                                          list[int]]) -> dict[str, list[int]]:
        """Validate tokens for multiple requests and return validated tokens."""
        if len(request_id_to_token_ids) == 0:
            return {}

        task = StructuredOutputTask(TaskType.BATCH_VALIDATE_TOKENS,
                                    (request_id_to_token_ids, ), {})
        self.task_queue.put(task)
        result = self.batch_validate_result_queue.get()
        if result.error:
            raise Exception(f"Error in batch_validate_tokens: {result.error}")
        return result.result

    def submit_grammar_init(self, request):
        """Submit grammar_init task."""
        if request.structured_output_request is None:
            return

        # Extract only the necessary fields from request
        # to reduce data transfer overhead
        grammar_init_data = GrammarInitData.from_request(request)

        task = StructuredOutputTask(TaskType.GRAMMAR_INIT,
                                    (grammar_init_data, ), {})
        self.task_queue.put(task)
        # Set up automatic cleanup when structured_output_request
        # is garbage collected
        finalize(request.structured_output_request,
                 partial(self.submit_grammar_delete, request.request_id))
        # Set the compiled_grammar AFTER putting the task in the queue
        # to avoid pickling the callback
        request.structured_output_request.compiled_grammar = FutureGrammar(
            self._is_grammar_init_done, request.request_id)

    def _is_grammar_init_done(self, request_id: str) -> bool:
        # Read all available notifications from the queue
        # and add them to the set
        while not self.grammar_init_notification_queue.empty():
            try:
                result = self.grammar_init_notification_queue.get_nowait()
                if result.error:
                    # Log error but don't add to initialized set
                    logger.debug("Error in grammar initialization: %s",
                                 result.error)
                else:
                    completed_request_id = result.result
                    self.initialized_grammars.add(completed_request_id)
            except queue.Empty:
                # Queue is empty
                break

        # Check if this request_id is in our set of initialized grammars
        return request_id in self.initialized_grammars

    def submit_grammar_delete(self, request_id: str):
        """Submit grammar_delete task (fire-and-forget)."""
        task = StructuredOutputTask(TaskType.GRAMMAR_DELETE, (request_id, ),
                                    {})
        self.task_queue.put(task)

        # Remove from our set of initialized grammars
        self.initialized_grammars.discard(request_id)

    def _submit_clear_backend(self):
        """Submit clear_backend task (fire-and-forget)."""
        task = StructuredOutputTask(TaskType.CLEAR_BACKEND, (), {})
        self.task_queue.put(task)

    def update_reasoning_ended(self, requests: dict[str, Request],
                               structured_output_request_ids: dict[str, int]):
        """Update the reasoning_ended flag for the given requests."""
        if self.reasoner is not None:
            for request_id, _ in structured_output_request_ids.items():
                request = requests[request_id]
                structured_output = request.structured_output_request
                assert structured_output is not None
                if structured_output.reasoning_ended is None:
                    structured_output.reasoning_ended = \
                        self.reasoner.is_reasoning_end(request.prompt_token_ids)

    def should_advance(self, request: Request) -> bool:
        """Determine whether we can advance the FSM."""
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert (request.structured_output_request.compiled_grammar
                    is not None)
        # by default, we should always advance
        # for cases that doesn't uses thinking mode.
        if self.reasoner is not None:
            structured_req = request.structured_output_request

            if structured_req.reasoning_ended:
                return True

            # Check if reasoning ends in *this* step
            if self.reasoner.is_reasoning_end(request.all_token_ids):
                # Reasoning just ended, so we shouldn't advanced til
                # next pass
                structured_req.reasoning_ended = True

            return False
        else:
            return True

    def shutdown(self):
        """Shutdown the manager and child process."""
        self._submit_clear_backend()
        task = StructuredOutputTask(TaskType.SHUTDOWN, (), {})
        self.task_queue.put(task)
        self.gateway_process.join(timeout=5)
        if self.gateway_process.is_alive():
            logger.debug("Force terminating StructuredOutputGateway")
            self.gateway_process.terminate()
            self.gateway_process.join()

        # Clean up shared memory
        self.bitmask_shm.close()
        self.bitmask_shm.unlink()
        self.flag_shm.close()
        self.flag_shm.unlink()

    def _cleanup_existing_shared_memory(self):
        """Clean up any existing shared memory segments from previous runs.
        
        This handles cases where a previous process was killed abruptly
        and didn't properly clean up its shared memory segments.
        """
        # Try to unlink bitmask shared memory if it exists
        try:
            existing_bitmask_shm = shared_memory.SharedMemory(
                name=self.bitmask_shm_name)
            existing_bitmask_shm.close()
            existing_bitmask_shm.unlink()
            logger.debug("Cleaned up existing bitmask shared memory from "
                         "previous run")
        except FileNotFoundError:
            # No existing shared memory, which is fine
            pass
        except Exception as e:
            logger.warning("Error cleaning up bitmask shared memory: %s", e)

        # Try to unlink flag shared memory if it exists
        try:
            existing_flag_shm = shared_memory.SharedMemory(
                name=self.ready_flag_shm_name)
            existing_flag_shm.close()
            existing_flag_shm.unlink()
            logger.debug("Cleaned up existing flag shared memory from "
                         "previous run")
        except FileNotFoundError:
            # No existing shared memory, which is fine
            pass
        except Exception as e:
            logger.warning("Error cleaning up flag shared memory: %s", e)


class GrammarBitmaskPlaceholder:
    """
    Placeholder object that gpu_model_runner.py can check and get result from.
    Uses shared memory flag to wait for result 
    and shared memory to retrieve bitmask data.
    """

    def __init__(self, bitmask_shm_name: str, ready_flag_shm_name: str):
        self.bitmask_shm_name = bitmask_shm_name
        self.ready_flag_shm_name = ready_flag_shm_name

    def result(self) -> np.ndarray:
        import numpy as np

        # Poll the shared memory flag until it's set
        flag_shm = shared_memory.SharedMemory(name=self.ready_flag_shm_name)
        while True:
            flag_value = flag_shm.buf[0]

            if flag_value == 1:  # Flag is set, bitmask is ready
                break

            sched_yield()  # Short sleep to avoid busy waiting

        flag_shm.close()

        bitmask_shm = shared_memory.SharedMemory(name=self.bitmask_shm_name)
        # Read shape info first
        shape_bytes = bytes(bitmask_shm.buf[:8])  # Create a copy of the bytes
        shape = np.frombuffer(shape_bytes, dtype=np.int32).copy()
        # Read bitmask data
        data_size = shape[0] * shape[1] * 4  # int32 = 4 bytes
        # Create a complete copy of the data before closing shared memory
        data_bytes = bytes(bitmask_shm.buf[8:8 + data_size])
        bitmask_shm.close()
        # Now create the numpy array from the copied bytes
        bitmask = np.frombuffer(data_bytes,
                                dtype=np.int32).reshape(shape).copy()
        return bitmask


class StructuredOutputExecutor:
    """
    Runs on CHILD process (inside StructuredOutputGateway).
    Executor that performs the actual structured output work 
    including grammar compilation, bitmask generation, and token acceptance.
    All communication between processes happens via queues.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.backend: Optional[StructuredOutputBackend] = None
        self.vllm_config = vllm_config

        self._grammar_bitmask: Optional[torch.Tensor] = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        self.request_id_to_grammar: dict[str, StructuredOutputGrammar] = {}

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(
                max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of CPUs*5,
            # which is way too high since these tasks are CPU-bound,
            # not I/O bound.
            # We also know we would never dominate CPU usage with just grammar
            # compilation, so we set it to half the number of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = init_tokenizer_from_configs(
                model_config=self.vllm_config.model_config)

    def _get_grammar(self, request_id: str) -> StructuredOutputGrammar:
        return self.request_id_to_grammar[request_id]

    def grammar_init(self, grammar_init_data: GrammarInitData) -> None:
        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            backend = grammar_init_data.guided_decoding_backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import (
                    OutlinesBackend)

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend)
                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(
                    f"Unsupported structured output backend: {backend}")

        grammar = self.executor.submit(self._async_create_grammar,
                                       grammar_init_data).result()
        self.request_id_to_grammar[grammar_init_data.request_id] = grammar

    def _async_create_grammar(
        self,
        grammar_init_data: GrammarInitData,
    ) -> StructuredOutputGrammar:
        key = grammar_init_data.structured_output_key

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self,
        batch: list[tuple[StructuredOutputGrammar, int, bool]],
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self,
        batch: list[tuple[StructuredOutputGrammar, int, bool]],
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        structured_output_request_ids: dict[str, int],
        req_reasoning_ended: dict[str, Optional[bool]],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Optional[npt.NDArray[np.int32]]:
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = \
                self.vllm_config.speculative_config.num_speculative_tokens

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = \
                self.backend.allocate_token_bitmask(
                    max_batch_size * (1 + max_num_spec_tokens))

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0
        ordered_seq = sorted(structured_output_request_ids.items(),
                             key=lambda x: x[1])

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if len(ordered_seq) > self.fill_bitmask_parallel_threshold and \
                max_num_spec_tokens == 0:
            promises = []
            batch = []

            for req_id, _ in ordered_seq:
                grammar = self._get_grammar(req_id)
                apply_bitmask = self.should_fill_bitmask(
                    req_reasoning_ended, req_id)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id, _ in ordered_seq:
                grammar = self._get_grammar(req_id)
                apply_bitmask = self.should_fill_bitmask(
                    req_reasoning_ended, req_id)

                state_advancements = 0
                req_tokens = scheduled_spec_decode_tokens.get(req_id, [])
                for i, token in enumerate(req_tokens + [None]):
                    self._fill_bitmasks([(grammar, cumulative_index,
                                          apply_bitmask)])

                    if apply_bitmask and token is not None and \
                        not grammar.is_terminated():
                        assert grammar.accept_tokens(req_id, [token])
                        state_advancements += 1
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, req_reasoning_ended, req_id) -> bool:
        if req_reasoning_ended[req_id] is not None:
            return req_reasoning_ended[req_id]
        return True

    def _accept_tokens(self, request_id: str,
                       new_token_ids: list[int]) -> None:
        grammar = self._get_grammar(request_id)
        grammar.accept_tokens(request_id, new_token_ids)

    def batch_accept_tokens(
            self, request_id_to_new_token_ids: list[tuple[str,
                                                          list[int]]]) -> None:
        for req_id, new_token_ids in request_id_to_new_token_ids:
            self._accept_tokens(req_id, new_token_ids)

    def batch_validate_tokens(
            self,
            request_id_to_token_ids: dict[str,
                                          list[int]]) -> dict[str, list[int]]:
        """Validate tokens for multiple requests without advancing the FSM
        state."""
        result = {}
        for req_id, token_ids in request_id_to_token_ids.items():
            grammar = self._get_grammar(req_id)
            validated_tokens = grammar.validate_tokens(token_ids)
            result[req_id] = validated_tokens
        return result

    def grammar_delete(self, request_id: str) -> None:
        """Remove grammar for the given request_id to prevent memory leaks."""
        if request_id in self.request_id_to_grammar:
            grammar = self.request_id_to_grammar[request_id]
            # Reset the grammar state before deletion for clean cleanup
            grammar.reset()
            # Remove from dictionary to allow garbage collection
            del self.request_id_to_grammar[request_id]

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
