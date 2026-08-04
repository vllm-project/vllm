# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controllers.full_sync_tracker import FullSyncTracker
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
    KVOpEvent,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    OpType,
    PinMsg,
    PinRetMsg,
)
from lmcache.v1.cache_controller.observability import PrometheusLogger
from lmcache.v1.cache_controller.utils import RegistryTree
from lmcache.v1.token_database import ChunkedTokenDatabase

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.controllers import RegistrationController

logger = init_logger(__name__)


"""
The kv controller use `(instance_id, worker_id)` -> [location -> set[chunk_hash]] 
as kv_pool. When the number of instances is small and stable, the time complexity 
of `lookup` in kv controller is O(n). If the number of instance is large or unknown, 
the time complexity will degrade to O(n^2), and the ReverseIndexKVController is a 
better choice.
"""


class KVController:
    def __init__(
        self,
        registry: RegistryTree,
        full_sync_completion_threshold: float = 0.8,
        full_sync_timeout_s: float = 300.0,
    ) -> None:
        # TODO(Jiayi): remove this hardcode
        self.token_database = ChunkedTokenDatabase()
        self.registry = registry
        self.cluster_executor: Any = None

        # Full sync tracker
        self.full_sync_tracker = FullSyncTracker(
            registry_tree=registry,
            completion_threshold=full_sync_completion_threshold,
            sync_timeout_s=full_sync_timeout_s,
        )

    def _setup_metrics(self) -> None:
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.kv_pool_keys_count.set_function(
                self.registry.get_total_kv_count
            )
            prometheus_logger.kv_op_seq_discontinuity_count.set_function(
                self.registry.get_seq_discontinuity_count
            )
            # Full sync metrics
            prometheus_logger.full_sync_workers_syncing.set_function(
                self.full_sync_tracker.get_syncing_count
            )
            prometheus_logger.full_sync_workers_completed.set_function(
                self.full_sync_tracker.get_completed_count
            )
            prometheus_logger.full_sync_global_progress.set_function(
                self.full_sync_tracker.get_global_progress
            )
            prometheus_logger.full_sync_missing_batches_total.set_function(
                self.full_sync_tracker.get_total_missing_batches_count
            )

    def post_init(
        self, reg_controller: "RegistrationController", cluster_executor: Any
    ) -> None:
        """
        Post initialization of the KV controller.
        """
        self.reg_controller = reg_controller
        self.cluster_executor = cluster_executor
        self._setup_metrics()

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("clear", msg)

    async def pin(self, msg: PinMsg) -> PinRetMsg:
        """
        Pin kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("pin", msg)

    async def compress(self, msg: CompressMsg) -> CompressRetMsg:
        """
        Compress kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("compress", msg)

    async def decompress(self, msg: DecompressMsg) -> DecompressRetMsg:
        """
        Decompress kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("decompress", msg)

    async def move(self, msg: MoveMsg) -> MoveRetMsg:
        """
        Move kv chunks of instance-worker(s).
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("move", msg)

    async def check_finish(self, msg: CheckFinishMsg) -> CheckFinishRetMsg:
        """
        Check if an event is finished.
        """
        assert self.cluster_executor is not None
        return await self.cluster_executor.execute("check_finish", msg)

    async def handle_batched_kv_operations(self, msg: BatchedKVOperationMsg) -> None:
        """Handle batched KV operations by forwarding to registry."""
        if not msg.operations:
            return

        # Check if worker is currently in full sync
        if self.full_sync_tracker.is_worker_syncing(msg.instance_id, msg.worker_id):
            # During full sync, incremental operations should be discarded
            logger.debug(
                "Discarding incremental KV operations during full sync: "
                "instance=%s, worker=%d, sync_id=%s, operation_count=%d",
                msg.instance_id,
                msg.worker_id,
                self.full_sync_tracker.get_sync_id(msg.instance_id, msg.worker_id),
                len(msg.operations),
            )
            return

        if not self.registry.handle_batched_kv_operations(msg):
            logger.warning(
                "Failed to handle batched KV operations, instance: %s, worker: %d",
                msg.instance_id,
                msg.worker_id,
            )

    # ============= Full Sync Message Handlers =============

    async def handle_full_sync_start(
        self, msg: FullSyncStartMsg
    ) -> FullSyncStartRetMsg:
        """
        Handle full sync start request from a worker.

        This is called when a worker wants to start full sync.
        The controller should:
        1. Clear existing keys for this worker
        2. Mark the worker as syncing (incremental events will be discarded)
        3. Return acceptance
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        sync_id = msg.sync_id
        report_id = (instance_id, worker_id)

        # Start sync tracking first (mark worker as SYNCING)
        success = self.full_sync_tracker.start_sync(
            instance_id=instance_id,
            worker_id=worker_id,
            sync_id=sync_id,
            total_keys=msg.total_keys,
            batch_count=msg.batch_count,
        )

        if not success:
            logger.warning(
                "Failed to start sync for worker %s: sync_id=%s", report_id, sync_id
            )
            return FullSyncStartRetMsg(
                sync_id=sync_id,
                accepted=False,
                error_msg="Failed to start sync: worker already syncing with "
                "different sync_id or worker not found",
            )

        # Now clear existing keys for this worker/location using efficient batch method
        # This prevents new incremental messages from being processed while we clear
        existing_keys = self.registry.get_worker_kv_keys(
            instance_id, worker_id, msg.location
        )
        if existing_keys:
            old_count = len(existing_keys)
            # Use efficient batch clear method
            cleared = self.registry.clear_worker_kv(
                instance_id, worker_id, msg.location
            )
            if cleared:
                logger.info(
                    "Cleared %d existing keys for worker %s location %s "
                    "before full sync",
                    old_count,
                    report_id,
                    msg.location,
                )
            else:
                logger.warning(
                    "Failed to clear keys for worker %s location %s",
                    report_id,
                    msg.location,
                )

        logger.info(
            "Accepted full sync start: worker=%s, sync_id=%s, "
            "total_keys=%d, batch_count=%d",
            report_id,
            sync_id,
            msg.total_keys,
            msg.batch_count,
        )
        return FullSyncStartRetMsg(sync_id=sync_id, accepted=True)

    async def handle_full_sync_batch(self, msg: FullSyncBatchMsg) -> None:
        """
        Handle full sync batch message from a worker.

        This adds the keys from the batch to the registry.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        location = msg.location
        sync_id = msg.sync_id
        batch_id = msg.batch_id
        keys = msg.keys
        report_id = (instance_id, worker_id)

        # Record batch receipt
        if not self.full_sync_tracker.receive_batch(
            instance_id=instance_id,
            worker_id=worker_id,
            sync_id=sync_id,
            batch_id=batch_id,
            keys_count=len(keys),
        ):
            logger.warning(
                "Failed to record batch %d for worker %s", batch_id, report_id
            )
            return

        # Add keys to registry using batched operations
        operations = []
        for seq_num, key in enumerate(keys):
            operations.append(
                KVOpEvent(
                    op_type=OpType.ADMIT,
                    key=key,
                    seq_num=seq_num,
                )
            )
        if operations:
            batch_msg = BatchedKVOperationMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                location=location,
                operations=operations,
            )
            self.registry.handle_batched_kv_operations(batch_msg, is_full_sync=True)

        current_keys = self.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        logger.debug(
            "Added %d keys from batch %d for worker %s, total now: %d",
            len(keys),
            batch_id,
            report_id,
            len(current_keys),
        )

    async def handle_full_sync_end(self, msg: FullSyncEndMsg) -> None:
        """
        Handle full sync end message from a worker.

        This marks the sync as end-received and records actual total keys.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        sync_id = msg.sync_id
        actual_total_keys = msg.actual_total_keys
        report_id = (instance_id, worker_id)

        success = self.full_sync_tracker.complete_sync(
            instance_id=instance_id,
            worker_id=worker_id,
            sync_id=sync_id,
            actual_total_keys=actual_total_keys,
        )

        if success:
            # Verify registry has the expected number of keys
            actual_keys_in_pool = len(
                self.registry.get_worker_kv_keys(instance_id, worker_id, msg.location)
            )
            logger.info(
                "Full sync completed for worker %s: sync_id=%s, "
                "reported_keys=%d, keys_in_pool=%d",
                report_id,
                sync_id,
                actual_total_keys,
                actual_keys_in_pool,
            )
        else:
            logger.warning(
                "Failed to complete full sync for worker %s: sync_id=%s",
                report_id,
                sync_id,
            )

    async def handle_full_sync_status(
        self, msg: FullSyncStatusMsg
    ) -> FullSyncStatusRetMsg:
        """
        Handle full sync status query from a worker.

        Returns the sync status including any missing batches that need resending.
        """
        is_complete, global_progress, can_exit_freeze, missing_batches = (
            self.full_sync_tracker.get_sync_status(
                instance_id=msg.instance_id,
                worker_id=msg.worker_id,
                sync_id=msg.sync_id,
            )
        )

        if missing_batches:
            logger.info(
                "Full sync status query: worker=(%s, %d), sync_id=%s, "
                "is_complete=%s, missing_batches=%s",
                msg.instance_id,
                msg.worker_id,
                msg.sync_id,
                is_complete,
                missing_batches,
            )

        return FullSyncStatusRetMsg(
            sync_id=msg.sync_id,
            is_complete=is_complete,
            global_progress=global_progress,
            can_exit_freeze=can_exit_freeze,
            missing_batches=missing_batches,
        )

    # TODO(Jiayi): The current implementation does not handle
    # the case where the prefix chunks are evicted while the
    # suffix chunk is still in the system. LMCache should guarantee
    # this does not happen.
    # TODO(Jiayi): The current implementation does not consider
    # the location of the kv chunks. It simply returns the
    # `instance_id` with longest prefix.
    # TODO(Jiayi): Need to get rid of the hash somehow
    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        tokens = msg.tokens
        layout_info = {}
        for start, end, key in self.token_database.process_tokens(
            tokens, make_key=False
        ):
            result = self.registry.find_kv(key)
            if result is None:
                break
            matched_instance = result.instance_id
            matched_location = result.location
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    # TODO: improve the matching logic, return multi results
    async def batched_p2p_lookup(
        self, msg: BatchedP2PLookupMsg
    ) -> BatchedP2PLookupRetMsg:
        """
        Perform batched P2P lookup for multiple keys.

        :param BatchedP2PLookupMsg msg: The batched P2P lookup message containing keys.

        :return: A BatchedP2PLookupRetMsg containing the lookup results.
        """
        hashes = msg.hashes
        if not hashes:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        # Single lookup to get all needed info (optimized path)
        result = self.registry.find_kv_with_worker_info(
            hashes[0], exclude_instance_id=msg.instance_id
        )
        if result is None:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        kv_info, peer_init_url, current_keys = result
        if peer_init_url is None:
            return BatchedP2PLookupRetMsg(layout_info=[("", "", 0, "")])

        # Count hits efficiently
        num_hit_chunks = 0
        for key in hashes:
            if key not in current_keys:
                break
            num_hit_chunks += 1

        return BatchedP2PLookupRetMsg(
            layout_info=[
                (kv_info.instance_id, kv_info.location, num_hit_chunks, peer_init_url),
            ]
        )
