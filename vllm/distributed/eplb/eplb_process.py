# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
import random
from collections.abc import Callable
from contextlib import suppress
from multiprocessing import Queue
from queue import Empty

import torch
import torch.distributed as dist

from vllm.logger import init_logger

from .eplb_expert_mapper import BipartiteExpertUpdate, GreedyExpertUpdate
from .eplb_state import ExpertMapperArgs, RebalanceTaskArgs

logger = init_logger(__name__)


class EPLBProcess:
    """
    Encapsulates lifecycle management for asynchronous expert
    rearrangement processes
    """

    def __init__(self, target_func: Callable, num_wait_worker_iterations: int):
        """
        Initialize asynchronous process manager

        Args:
            target_func: Target function to execute in asynchronous process
                (e.g., rebalance_experts)
            num_wait_worker_iterations: Number of steps to wait before
                checking results
        """
        self.target_func = target_func
        self._num_wait_worker_iterations = num_wait_worker_iterations

        # Process management related
        self._process: mp.Process | None = None
        self._input_queue: Queue | None = None
        self._result_queue: Queue | None = None
        self._exception_queue: Queue | None = None
        self._step_counter = 0
        self._result: tuple | None = None
        self._is_running = False
        self._has_pending_task = False
        self._is_post_processing = False
        self.rank_id = dist.get_rank()

        # Initialize process and queues
        self._initialize_process()

    def _initialize_process(self) -> None:
        """Initialize the background process and queues"""
        try:
            # Initialize queues
            self._input_queue = Queue()
            self._result_queue = Queue()
            self._exception_queue = Queue()

            # Start the process
            self._process = mp.Process(
                target=self._worker_loop,
                name="EPLBProcess",
                args=(self._input_queue, self._result_queue, self._exception_queue),
            )
            self._process.start()
            self._is_running = True
            logger.debug("EPLB background process started")

        except Exception:
            self.cleanup()
            raise

    def pack_update_info(self, update_info_generator):
        """
        Pack a list of update info tuples for efficient IPC.
        """
        send_all = []
        recv_all = []
        maps = []
        log2phy_all = []
        layer_ids = []
        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:
            send_info_this_rank = send_info.get(self.rank_id, [])
            recv_info_this_rank = recv_info.get(self.rank_id, [])
            send_all.append(send_info_this_rank)
            recv_all.append(recv_info_this_rank)

            maps.append(new_expert_map[self.rank_id].numpy().tolist())

            log2phy_map = self.generate_log2phy_map(new_expert_map)
            log2phy_all.append(log2phy_map[self.rank_id].numpy().tolist())

            layer_ids.append(layer_id)

        return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))

    def generate_log2phy_map(self, expert_map):
        """
        Generates a logical-to-physical expert mapping for all ranks based on an
        initial expert distribution map. This map indicates which physical expert
        slot (on which rank) corresponds to a given logical expert. It handles cases
        where an expert might not be present on all ranks and fills in missing
        entries by replicating existing ones.

        Args:
            expert_map: A 2D tensor of shape [num_ranks, num_global_experts].
                        `expert_map[r, g]` contains the local physical ID of global
                        expert `g` on rank `r`, or -1 if global expert `g` is not
                        on rank `r`.

        Returns:
            A 2D tensor `log2phy_map` of shape [num_ranks, num_global_experts].
            `log2phy_map[r, g]` will contain the *global physical ID* of the expert
            that rank `r` should use for logical expert `g`.
            A global physical ID is
            `rank_id * num_local_experts + local_physical_expert_id`.
        """
        num_local_experts = expert_map.max() + 1
        log2phy_map = expert_map.clone()
        num_ranks, num_global_expert = log2phy_map.shape

        row_indices = (
            torch.arange(num_ranks).view(-1, 1).expand(num_ranks, num_global_expert)
            * num_local_experts
        )
        log2phy_map[log2phy_map != -1] += row_indices[log2phy_map != -1]

        for idx in range(num_global_expert):
            positive_rank_idx = torch.where(log2phy_map[:, idx] != -1)[0]
            negative_rank_idx = torch.where(log2phy_map[:, idx] == -1)[0]
            num_rank_holding_expert = positive_rank_idx.size(0)

            if num_rank_holding_expert == 1:
                log2phy_map[negative_rank_idx, idx] = torch.full(
                    (num_ranks - 1,),
                    log2phy_map[positive_rank_idx, idx].item(),
                    dtype=log2phy_map.dtype,
                )
            else:
                random_list = [
                    random.choice(log2phy_map[positive_rank_idx, idx])
                    for _ in range(num_ranks - num_rank_holding_expert)
                ]
                log2phy_map[negative_rank_idx, idx] = torch.tensor(
                    random_list, dtype=log2phy_map.dtype
                )
        return log2phy_map

    def _worker_loop(
        self, input_queue: Queue, output_queue: Queue, exception_queue: Queue
    ) -> None:
        """Subprocess worker loop that processes tasks continuously"""
        try:
            while True:
                # Get arguments from input queue
                try:
                    args, expert_mapper_args = input_queue.get(timeout=1.0)
                    # Sentinel value to stop the process
                    if args is None or expert_mapper_args is None:
                        break

                    # Execute target function
                    result = self.target_func(*args)

                    new_physical_to_logical_map = result[0]
                    policy_type = expert_mapper_args.policy_type
                    # Generate migration information
                    new_deployment = new_physical_to_logical_map.reshape(
                        expert_mapper_args.num_moe_layers,
                        args.num_gpus,
                        -1,
                    )
                    old_deployment = (
                        expert_mapper_args.phyhsical_to_logical_map.reshape(
                            expert_mapper_args.num_moe_layers,
                            args.num_gpus,
                            -1,
                        )
                    )
                    if policy_type == "bipartite":
                        update_info = BipartiteExpertUpdate(
                            new_deployment, old_deployment
                        ).generate()
                    elif policy_type == "greedy":
                        update_info = GreedyExpertUpdate(
                            new_deployment, old_deployment
                        ).generate()
                    output_queue.put(self.pack_update_info(update_info))

                except Empty:
                    # Timeout, check if we should continue
                    continue
                except Exception as e:
                    output_queue.put(None)
                    if hasattr(e, "add_note"):
                        import traceback

                        e.add_note(traceback.format_exc())
                    exception_queue.put(e)
                    logger.exception("Task execution failed in worker process")

        except Exception as e:
            exception_queue.put(e)
            logger.exception("Worker process encountered fatal error")
        finally:
            logger.debug("EPLB worker process exiting")

    def submit_task(
        self, args: RebalanceTaskArgs, expert_mapper_args: ExpertMapperArgs
    ) -> bool:
        """
        Submit a task to the asynchronous process

        Args:
            args: Tuple of arguments to pass to the target function
            expert_mapper_args: Tuple of arguments to pass to expert mapper strategy

        Returns:
            True if task submitted successfully, False otherwise
        """
        if not self._is_running:
            logger.error("Cannot submit task: process is not running")
            return False

        if self._has_pending_task:
            logger.warning("Cannot submit task: already has a pending task")
            return False

        if not self._input_queue:
            logger.error("Cannot submit task: input queue is not initialized")
            return False

        try:
            # Put arguments to the input queue
            combined_args = (args, expert_mapper_args)
            self._input_queue.put(combined_args)
            self._has_pending_task = True
            self._step_counter = 0
            self._result = None
            return True

        except Exception as e:
            logger.error("Failed to submit task to asynchronous process: %s", str(e))
            return False

    def step(self) -> bool:
        """
        Increment step counter and check if results need processing

        Returns:
            Whether results have been processed
        """
        if not self._is_running or not self._has_pending_task:
            return False

        self._step_counter += 1

        # Check for exceptions first
        if self._exception_queue and not self._exception_queue.empty():
            error_msg = self._exception_queue.get()
            self._has_pending_task = False
            raise RuntimeError(f"Asynchronous process failed: {error_msg}")

        # Check if processing conditions are met
        if self._should_process():
            self._fetch_result()
            self._has_pending_task = False
            return True

        return False

    def _should_process(self) -> bool:
        """Determine if results need processing"""
        if not self._process or not self._result_queue:
            return True

        return (
            self._step_counter >= self._num_wait_worker_iterations
            or not self._process.is_alive()
            or not self._result_queue.empty()
        )

    def _fetch_result(self) -> None:
        """Retrieve subprocess results"""
        if self._result_queue and not self._result_queue.empty():
            self._result = self._result_queue.get()
        else:
            self._result = None
            logger.warning("Asynchronous process completed but no result was returned")

    def cleanup(self) -> None:
        """Clean up process resources"""
        # Send sentinel value to stop the process
        if self._input_queue:
            with suppress(Exception):
                self._input_queue.put(None)

        if self._process:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

        for q in (self._input_queue, self._result_queue, self._exception_queue):
            if q:
                with suppress(Exception):
                    q.close()
                with suppress(Exception):
                    q.join_thread()

        self._input_queue = None
        self._result_queue = None
        self._exception_queue = None
        self._is_running = False
        self._has_pending_task = False

    @property
    def is_running(self) -> bool:
        """Return whether the process is running"""
        return self._is_running

    @property
    def has_pending_task(self) -> bool:
        """Return whether there is a pending task"""
        return self._has_pending_task

    @property
    def is_post_processing(self) -> bool:
        return self._is_post_processing

    @is_post_processing.setter
    def is_post_processing(self, value: bool):
        self._is_post_processing = value

    @property
    def result(self) -> tuple | None:
        """Return processing results"""
        return self._result

    def __del__(self):
        """Ensure resource cleanup when object is destroyed"""
        self.cleanup()
