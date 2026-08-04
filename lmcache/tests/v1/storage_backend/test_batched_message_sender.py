# SPDX-License-Identifier: Apache-2.0
# Standard
import gc
import threading
import time

# Third Party
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.batched_message_sender import (
    BatchedMessageSender,
)


class MockLookupServer:
    def __init__(self):
        self.removed_keys = []
        self.inserted_keys = []

    def batched_remove(self, keys):
        self.removed_keys.extend(keys)

    def batched_insert(self, keys):
        self.inserted_keys.extend(keys)


class MockLMCacheWorker:
    def __init__(self):
        self.messages = []
        self._lock = threading.Lock()

    def put_msg(self, msg):
        with self._lock:
            self.messages.append(msg)


def create_test_config(
    local_cpu: bool = True, use_layerwise: bool = False, enable_blending: bool = False
):
    """Create a test configuration for LocalCPUBackend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=local_cpu,
        use_layerwise=use_layerwise,
        enable_blending=enable_blending,
        lmcache_instance_id="test_instance",
    )
    return config


def create_test_key(key_id: str = "test_key") -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


class TestBatchedMessageSender:
    """Test cases for BatchedMessageSender."""

    def teardown_method(self, method):
        # Clean up any lingering BatchedMessageSender instances
        for obj in gc.get_objects():
            if isinstance(obj, BatchedMessageSender) and hasattr(obj, "running"):
                obj.close()

        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_basic_batching(self, lmcache_engine_metadata):
        """Test basic message batching functionality."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 3
        config.extra_config["kv_msg_batch_timeout"] = 0.1

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        # Add operations below batch size threshold
        sender.add_kv_op(OpType.ADMIT, 1)
        sender.add_kv_op(OpType.ADMIT, 2)

        # Should not have sent messages yet
        assert len(lmcache_worker.messages) == 0

        # Add one more to reach batch size
        sender.add_kv_op(OpType.ADMIT, 3)

        # Give some time for the message to be sent
        time.sleep(0.05)

        # Should have sent one batched message
        assert len(lmcache_worker.messages) == 1
        msg = lmcache_worker.messages[0]
        assert len(msg.operations) == 3

        sender.close()

    def test_timeout_based_flush(self, lmcache_engine_metadata):
        """Test that messages are flushed based on timeout."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 100
        config.extra_config["kv_msg_batch_timeout"] = 0.05

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        # Add a few operations (below batch size)
        sender.add_kv_op(OpType.ADMIT, 1)
        sender.add_kv_op(OpType.EVICT, 2)

        # Wait for timeout to trigger flush
        time.sleep(0.1)

        # Should have sent messages due to timeout
        assert len(lmcache_worker.messages) >= 1

        sender.close()

    def test_manual_flush(self, lmcache_engine_metadata):
        """Test manual flush functionality."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 100
        config.extra_config["kv_msg_batch_timeout"] = 10.0

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        # Add operations
        sender.add_kv_op(OpType.ADMIT, 1)
        sender.add_kv_op(OpType.EVICT, 2)

        # Manually flush
        sender.flush()

        # Should have sent messages
        assert len(lmcache_worker.messages) == 1
        msg = lmcache_worker.messages[0]
        assert len(msg.operations) == 2

        sender.close()

    def test_sequence_numbers(self, lmcache_engine_metadata):
        """Test that sequence numbers are monotonically increasing."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 10
        config.extra_config["kv_msg_batch_timeout"] = 0.1

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        # Add multiple operations
        for i in range(15):
            sender.add_kv_op(OpType.ADMIT, i)

        sender.flush()

        # Collect all sequence numbers
        seq_nums = []
        for msg in lmcache_worker.messages:
            for op in msg.operations:
                seq_nums.append(op.seq_num)

        # Verify sequence numbers are strictly consecutive
        # Since we have a single consumer thread, messages must arrive in order
        assert seq_nums == list(range(len(seq_nums))), (
            "Sequence numbers must be consecutive starting from 0"
        )

        sender.close()

    def test_concurrent_producers(self, lmcache_engine_metadata):
        """Stress test with multiple producer threads."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 50
        config.extra_config["kv_msg_batch_timeout"] = 0.01

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        num_threads = 10
        ops_per_thread = 100

        def producer_task(thread_id):
            for i in range(ops_per_thread):
                key = thread_id * ops_per_thread + i
                sender.add_kv_op(OpType.ADMIT, key)

        threads = [
            threading.Thread(target=producer_task, args=(i,))
            for i in range(num_threads)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Flush remaining messages
        sender.flush()

        # Collect all operations
        all_ops = []
        for msg in lmcache_worker.messages:
            all_ops.extend(msg.operations)

        # Verify total number of operations
        assert len(all_ops) == num_threads * ops_per_thread

        # Verify sequence numbers are strictly increasing within each batch
        # Since sequence numbers are assigned during drain (not add_kv_op),
        # the order in which operations are dequeued determines their
        # sequence numbers. This guarantees that within each batch,
        # sequence numbers are strictly increasing.
        seq_nums = [op.seq_num for op in all_ops]

        # Verify no duplicate sequence numbers
        assert len(seq_nums) == len(set(seq_nums)), "Sequence numbers must be unique"

        # Verify all sequence numbers are present (no messages lost)
        expected_seq_nums = set(range(num_threads * ops_per_thread))
        actual_seq_nums = set(seq_nums)
        assert expected_seq_nums == actual_seq_nums, (
            f"Missing sequence numbers: {expected_seq_nums - actual_seq_nums}"
        )

        # Verify sequence numbers are strictly increasing within each batch
        for msg in lmcache_worker.messages:
            batch_seq_nums = [op.seq_num for op in msg.operations]
            for i in range(1, len(batch_seq_nums)):
                assert batch_seq_nums[i] > batch_seq_nums[i - 1], (
                    f"Sequence numbers must be strictly increasing within batch: "
                    f"{batch_seq_nums[i - 1]} -> {batch_seq_nums[i]}"
                )

        sender.close()

    def test_mixed_operations_order(self, lmcache_engine_metadata):
        """Test that admit and evict operations maintain order."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 100
        config.extra_config["kv_msg_batch_timeout"] = 0.1

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        # Add operations in specific order
        operations = [
            (OpType.ADMIT, 1),
            (OpType.EVICT, 1),
            (OpType.ADMIT, 1),
            (OpType.ADMIT, 2),
            (OpType.EVICT, 2),
        ]

        for op_type, key in operations:
            sender.add_kv_op(op_type, key)

        sender.flush()

        # Collect all operations
        all_ops = []
        for msg in lmcache_worker.messages:
            all_ops.extend(msg.operations)

        # Verify operations are in the same order
        assert len(all_ops) == len(operations)
        for i, (expected_op_type, expected_key) in enumerate(operations):
            assert all_ops[i].op_type == expected_op_type
            assert all_ops[i].key == expected_key

        sender.close()

    def test_high_throughput_stress(self, lmcache_engine_metadata):
        """Stress test with high throughput operations."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 100
        config.extra_config["kv_msg_batch_timeout"] = 0.01

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        num_operations = 10000

        # Add many operations rapidly
        for i in range(num_operations):
            op_type = OpType.ADMIT if i % 2 == 0 else OpType.EVICT
            sender.add_kv_op(op_type, i)

        sender.flush()

        # Collect all operations
        all_ops = []
        for msg in lmcache_worker.messages:
            all_ops.extend(msg.operations)

        # Verify all operations were sent
        assert len(all_ops) == num_operations

        # Verify sequence numbers are strictly increasing
        # In a concurrent environment, we need to ensure that:
        # 1. Sequence numbers are strictly increasing within each batch
        # 2. No sequence numbers are duplicated
        # 3. All sequence numbers from 0 to num_operations-1 are present

        seq_nums = [op.seq_num for op in all_ops]

        # Verify no duplicate sequence numbers
        assert len(seq_nums) == len(set(seq_nums)), "Sequence numbers must be unique"

        # Verify all sequence numbers are present (no messages lost)
        expected_seq_nums = set(range(num_operations))
        actual_seq_nums = set(seq_nums)
        assert expected_seq_nums == actual_seq_nums, (
            f"Missing sequence numbers: {expected_seq_nums - actual_seq_nums}"
        )

        # Verify sequence numbers are strictly increasing within each batch
        # Since we have a single consumer thread, each batch should be processed
        # in the order it was created, and within each batch, messages should be
        # in the order they were added to the queue.
        for msg in lmcache_worker.messages:
            batch_seq_nums = [op.seq_num for op in msg.operations]
            for i in range(1, len(batch_seq_nums)):
                assert batch_seq_nums[i] > batch_seq_nums[i - 1], (
                    f"Sequence numbers must be strictly increasing within batch: "
                    f"{batch_seq_nums[i - 1]} -> {batch_seq_nums[i]}"
                )

        sender.close()

    def test_close_with_pending_messages(self, lmcache_engine_metadata):
        """Test that close() flushes pending messages."""

        config = create_test_config()
        config.extra_config = {}
        config.extra_config["kv_msg_batch_size"] = 100
        config.extra_config["kv_msg_batch_timeout"] = 10.0

        lmcache_worker = MockLMCacheWorker()
        sender = BatchedMessageSender(
            metadata=lmcache_engine_metadata,
            config=config,
            location="test_location",
            lmcache_worker=lmcache_worker,
        )

        # Add operations
        sender.add_kv_op(OpType.ADMIT, 1)
        sender.add_kv_op(OpType.EVICT, 2)

        # Close should flush pending messages
        sender.close()

        # Should have sent messages
        assert len(lmcache_worker.messages) >= 1

        # Collect all operations
        all_ops = []
        for msg in lmcache_worker.messages:
            all_ops.extend(msg.operations)

        assert len(all_ops) == 2
