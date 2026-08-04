# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for RegistryTree with fine-grained locking.

Tests verify that the object-based locking mechanism provides:
1. Thread-safe operations on instances, workers, and KV stores
2. Concurrent operations on different instances don't block each other
3. Data consistency under high concurrency
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# First Party
from lmcache.v1.cache_controller.message import BatchedKVOperationMsg, KVOpEvent, OpType
from lmcache.v1.cache_controller.utils import (
    InstanceNode,
    RegistryTree,
    WorkerNode,
)


class TestWorkerNodeLocking:
    """Test WorkerNode's internal locking for kv_store and seq_tracker."""

    def test_concurrent_admit_kv(self):
        """Test concurrent KV admission on the same worker."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        keys_per_thread = 100

        def admit_keys(thread_id):
            for i in range(keys_per_thread):
                key = thread_id * keys_per_thread + i
                msg = BatchedKVOperationMsg(
                    instance_id="test_instance",
                    worker_id=worker.worker_id,
                    location=location,
                    operations=[
                        KVOpEvent(
                            op_type=OpType.ADMIT,
                            key=key,
                            seq_num=thread_id * keys_per_thread + i,
                        )
                    ],
                )
                worker.handle_batched_kv_operations(msg)

        threads = [
            threading.Thread(target=admit_keys, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All keys should be admitted
        assert worker.get_kv_count() == num_threads * keys_per_thread

    def test_concurrent_admit_evict_kv(self):
        """Test concurrent admit and evict on the same worker."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 5
        operations_per_thread = 100
        errors = []

        def mixed_operations(thread_id):
            for i in range(operations_per_thread):
                key = i  # Same key range for all threads
                try:
                    if i % 2 == 0:
                        msg = BatchedKVOperationMsg(
                            instance_id="test_instance",
                            worker_id=worker.worker_id,
                            location=location,
                            operations=[
                                KVOpEvent(
                                    op_type=OpType.ADMIT,
                                    key=key,
                                    seq_num=thread_id * operations_per_thread + i,
                                )
                            ],
                        )
                    else:
                        msg = BatchedKVOperationMsg(
                            instance_id="test_instance",
                            worker_id=worker.worker_id,
                            location=location,
                            operations=[
                                KVOpEvent(
                                    op_type=OpType.EVICT,
                                    key=key,
                                    seq_num=thread_id * operations_per_thread + i,
                                )
                            ],
                        )
                    worker.handle_batched_kv_operations(msg)
                except Exception as e:
                    errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=mixed_operations, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_concurrent_seq_num_update(self):
        """Test concurrent sequence number updates."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        updates_per_thread = 100

        def update_seq(thread_id):
            for i in range(updates_per_thread):
                worker.update_seq_num(location, thread_id * updates_per_thread + i)

        threads = [
            threading.Thread(target=update_seq, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final seq_num should be set (exact value depends on thread order)
        final_seq = worker.get_seq_num(location)
        assert final_seq is not None


class TestInstanceNodeLocking:
    """Test InstanceNode's internal locking for workers dict."""

    def test_concurrent_add_workers(self):
        """Test concurrent worker additions to same instance."""
        instance = InstanceNode(instance_id="test_instance")
        num_threads = 10

        def add_worker(worker_id):
            worker = WorkerNode(
                worker_id=worker_id,
                ip="127.0.0.1",
                port=8000 + worker_id,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
                last_heartbeat_time=time.time(),
            )
            instance.add_worker(worker)

        threads = [
            threading.Thread(target=add_worker, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All workers should be added
        assert len(instance.get_worker_ids()) == num_threads

    def test_concurrent_add_remove_workers(self):
        """Test concurrent worker add/remove on same instance."""
        instance = InstanceNode(instance_id="test_instance")
        errors = []

        # Pre-add some workers
        for i in range(50):
            worker = WorkerNode(
                worker_id=i,
                ip="127.0.0.1",
                port=8000 + i,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
                last_heartbeat_time=time.time(),
            )
            instance.add_worker(worker)

        def add_workers(start_id):
            for i in range(10):
                try:
                    worker = WorkerNode(
                        worker_id=start_id + i,
                        ip="127.0.0.1",
                        port=9000 + start_id + i,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                        last_heartbeat_time=time.time(),
                    )
                    instance.add_worker(worker)
                except Exception as e:
                    errors.append("Add error: %s" % e)

        def remove_workers(start_id):
            for i in range(10):
                try:
                    instance.remove_worker(start_id + i)
                except Exception as e:
                    errors.append("Remove error: %s" % e)

        threads = []
        # Add workers 100-199
        for i in range(10):
            threads.append(threading.Thread(target=add_workers, args=(100 + i * 10,)))
        # Remove workers 0-49
        for i in range(5):
            threads.append(threading.Thread(target=remove_workers, args=(i * 10,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors


class TestRegistryTreeFineGrainedLocking:
    """Test RegistryTree's fine-grained object-based locking."""

    def test_concurrent_operations_different_instances(self):
        """
        Test that operations on different instances don't block each other.
        This is the key benefit of fine-grained locking.
        """
        registry = RegistryTree()
        num_instances = 5
        workers_per_instance = 20
        errors = []
        timing_results = []

        def operate_on_instance(instance_idx):
            nonlocal errors
            instance_id = "instance_%d" % instance_idx
            start_time = time.time()

            for worker_id in range(workers_per_instance):
                try:
                    # Register worker
                    registry.register_worker(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        ip="192.168.%d.%d" % (instance_idx, worker_id),
                        port=8000 + worker_id,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )

                    # KV operations
                    for kv_key in range(10):
                        msg = BatchedKVOperationMsg(
                            instance_id=instance_id,
                            worker_id=worker_id,
                            location="location_%d" % worker_id,
                            operations=[
                                KVOpEvent(
                                    op_type=OpType.ADMIT,
                                    key=kv_key,
                                    seq_num=kv_key,
                                )
                            ],
                        )
                        registry.handle_batched_kv_operations(msg)

                except Exception as e:
                    errors.append("Instance %d error: %s" % (instance_idx, e))

            elapsed = time.time() - start_time
            timing_results.append((instance_idx, elapsed))

        threads = [
            threading.Thread(target=operate_on_instance, args=(i,))
            for i in range(num_instances)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

        # Verify all data
        for i in range(num_instances):
            instance_id = "instance_%d" % i
            worker_ids = registry.get_worker_ids(instance_id)
            assert len(worker_ids) == workers_per_instance, (
                "Instance %d: expected %d workers, got %d"
                % (
                    i,
                    workers_per_instance,
                    len(worker_ids),
                )
            )

    def test_concurrent_register_deregister_same_instance(self):
        """Test concurrent register/deregister on the same instance."""
        registry = RegistryTree()
        instance_id = "test_instance"
        num_threads = 10
        operations_per_thread = 50
        errors = []

        def register_deregister(thread_id):
            for i in range(operations_per_thread):
                worker_id = thread_id * 1000 + i
                try:
                    # Register
                    registry.register_worker(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        ip="192.168.1.%d" % thread_id,
                        port=8000 + i,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )

                    # Verify registration
                    worker = registry.get_worker(instance_id, worker_id)
                    if worker is None:
                        errors.append(
                            "Thread %d: Worker %d not found after register"
                            % (thread_id, worker_id)
                        )
                        continue

                    # Deregister
                    result = registry.deregister_worker(instance_id, worker_id)
                    if result is None:
                        errors.append(
                            "Thread %d: Deregister failed for worker %d"
                            % (thread_id, worker_id)
                        )

                except Exception as e:
                    errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=register_deregister, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_concurrent_kv_operations_same_worker(self):
        """Test concurrent KV operations on the same worker."""
        registry = RegistryTree()
        instance_id = "test_instance"
        worker_id = 0

        registry.register_worker(
            instance_id=instance_id,
            worker_id=worker_id,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        keys_per_thread = 100
        errors = []

        def kv_operations(thread_id):
            nonlocal errors
            for i in range(keys_per_thread):
                key = thread_id * keys_per_thread + i
                try:
                    # Admit using batched operation message
                    msg = BatchedKVOperationMsg(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        location=location,
                        operations=[
                            KVOpEvent(
                                op_type=OpType.ADMIT,
                                key=key,
                                seq_num=thread_id * keys_per_thread + i,
                            )
                        ],
                    )
                    result = registry.handle_batched_kv_operations(msg)
                    if not result:
                        errors.append(
                            "Thread %d: handle_batched_kv_operations failed for key %d"
                            % (thread_id, key)
                        )
                except Exception as e:
                    errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=kv_operations, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

        # Verify total KV count
        total = registry.get_total_kv_count()
        assert total == num_threads * keys_per_thread

    def test_find_kv_concurrent_with_admit(self):
        """Test find_kv while concurrent admit operations are happening."""
        registry = RegistryTree()
        num_instances = 3
        workers_per_instance = 3

        # Setup: register workers
        for i in range(num_instances):
            for w in range(workers_per_instance):
                registry.register_worker(
                    instance_id="instance_%d" % i,
                    worker_id=w,
                    ip="192.168.%d.%d" % (i, w),
                    port=8000 + w,
                    peer_init_url=None,
                    socket=None,
                    registration_time=time.time(),
                )

        errors = []
        found_keys = []

        def admit_keys(instance_idx, worker_id):
            nonlocal errors
            for key in range(100):
                try:
                    msg = BatchedKVOperationMsg(
                        instance_id="instance_%d" % instance_idx,
                        worker_id=worker_id,
                        location="location_%d" % worker_id,
                        operations=[
                            KVOpEvent(
                                op_type=OpType.ADMIT,
                                key=key + instance_idx * 1000,
                                seq_num=key,
                            )
                        ],
                    )
                    registry.handle_batched_kv_operations(msg)
                except Exception as e:
                    errors.append("Admit error: %s" % e)

        def find_keys():
            for _ in range(50):
                for key in range(100):
                    try:
                        result = registry.find_kv(key)
                        if result:
                            found_keys.append(key)
                    except Exception as e:
                        errors.append("Find error: %s" % e)
                time.sleep(0.001)

        threads = []
        # Admit threads
        for i in range(num_instances):
            for w in range(workers_per_instance):
                threads.append(threading.Thread(target=admit_keys, args=(i, w)))
        # Find threads
        threads.append(threading.Thread(target=find_keys))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_heartbeat_update_concurrent(self):
        """Test concurrent heartbeat updates."""
        registry = RegistryTree()
        instance_id = "test_instance"
        num_workers = 10

        # Register workers
        for w in range(num_workers):
            registry.register_worker(
                instance_id=instance_id,
                worker_id=w,
                ip="127.0.0.1",
                port=8000 + w,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
            )

        errors = []
        num_threads = 20
        updates_per_thread = 100

        def update_heartbeats():
            for _ in range(updates_per_thread):
                for w in range(num_workers):
                    try:
                        result = registry.update_heartbeat(instance_id, w, time.time())
                        if not result:
                            errors.append("Heartbeat update failed for worker %d" % w)
                    except Exception as e:
                        errors.append("Heartbeat error: %s" % e)

        threads = [
            threading.Thread(target=update_heartbeats) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_get_all_worker_infos_concurrent(self):
        """Test get_all_worker_infos during concurrent modifications."""
        registry = RegistryTree()
        errors = []

        def register_workers():
            for i in range(100):
                try:
                    registry.register_worker(
                        instance_id="instance_%d" % (i % 5),
                        worker_id=i,
                        ip="192.168.1.%d" % i,
                        port=8000 + i,
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )
                except Exception as e:
                    errors.append("Register error: %s" % e)

        def get_infos():
            for _ in range(50):
                try:
                    infos = registry.get_all_worker_infos_cached()
                    # Just verify it returns a list without error
                    assert isinstance(infos, list)
                except Exception as e:
                    errors.append("Get infos error: %s" % e)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=register_workers),
            threading.Thread(target=get_infos),
            threading.Thread(target=get_infos),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_instance_cleanup_on_last_worker_deregister(self):
        """Test that empty instances are cleaned up correctly."""
        registry = RegistryTree()
        instance_id = "test_instance"
        num_workers = 10

        # Register workers
        for w in range(num_workers):
            registry.register_worker(
                instance_id=instance_id,
                worker_id=w,
                ip="127.0.0.1",
                port=8000 + w,
                peer_init_url=None,
                socket=None,
                registration_time=time.time(),
            )

        # Deregister all workers concurrently
        errors = []

        def deregister(worker_id):
            try:
                registry.deregister_worker(instance_id, worker_id)
            except Exception as e:
                errors.append("Deregister error: %s" % e)

        threads = [
            threading.Thread(target=deregister, args=(w,)) for w in range(num_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors

    def test_high_contention_stress(self):
        """Stress test with high contention on a single instance."""
        registry = RegistryTree()
        instance_id = "stress_instance"
        errors = []
        num_threads = 30
        operations_per_thread = 100

        def stress_operations(thread_id):
            nonlocal errors
            for i in range(operations_per_thread):
                worker_id = thread_id * operations_per_thread + i
                try:
                    # Register
                    registry.register_worker(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        ip="10.0.0.%d" % (thread_id % 256),
                        port=8000 + (i % 1000),
                        peer_init_url=None,
                        socket=None,
                        registration_time=time.time(),
                    )

                    # KV operations
                    msg = BatchedKVOperationMsg(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        location="loc1",
                        operations=[
                            KVOpEvent(
                                op_type=OpType.ADMIT,
                                key=i,
                                seq_num=i,
                            )
                        ],
                    )
                    registry.handle_batched_kv_operations(msg)

                    # Read operations
                    registry.get_worker(instance_id, worker_id)

                    # Deregister
                    registry.deregister_worker(instance_id, worker_id)

                except Exception as e:
                    errors.append(
                        "Thread %d iteration %d error: %s" % (thread_id, i, e)
                    )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(stress_operations, i) for i in range(num_threads)
            ]
            for f in futures:
                f.result()

        assert not errors, "Errors occurred (first 10): %s" % errors[:10]

    def test_data_consistency_after_concurrent_ops(self):
        """Verify data consistency after heavy concurrent operations."""
        registry = RegistryTree()
        num_instances = 5
        workers_per_instance = 10
        kv_keys = 50

        # Register all workers
        for inst in range(num_instances):
            for w in range(workers_per_instance):
                registry.register_worker(
                    instance_id="inst_%d" % inst,
                    worker_id=w,
                    ip="10.%d.%d.1" % (inst, w),
                    port=8000,
                    peer_init_url=None,
                    socket=None,
                    registration_time=time.time(),
                )

        errors = []

        def concurrent_operations(inst_idx):
            nonlocal errors
            instance_id = "inst_%d" % inst_idx
            for w in range(workers_per_instance):
                for key in range(kv_keys):
                    try:
                        msg = BatchedKVOperationMsg(
                            instance_id=instance_id,
                            worker_id=w,
                            location="loc",
                            operations=[
                                KVOpEvent(
                                    op_type=OpType.ADMIT,
                                    key=key,
                                    seq_num=key,
                                )
                            ],
                        )
                        registry.handle_batched_kv_operations(msg)
                    except Exception as e:
                        errors.append("Admit error: %s" % e)

        threads = [
            threading.Thread(target=concurrent_operations, args=(i,))
            for i in range(num_instances)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors during concurrent ops: %s" % errors

        # Verify final state
        total_kv = registry.get_total_kv_count()
        expected_kv = num_instances * workers_per_instance * kv_keys
        assert total_kv == expected_kv, "Expected %d KV entries, got %d" % (
            expected_kv,
            total_kv,
        )

        for inst in range(num_instances):
            worker_ids = registry.get_worker_ids("inst_%d" % inst)
            assert len(worker_ids) == workers_per_instance, (
                "Instance %d: expected %d workers, got %d"
                % (
                    inst,
                    workers_per_instance,
                    len(worker_ids),
                )
            )


class TestBatchOperations:
    """Test batch KV operations for performance optimization."""

    def test_batch_admit_kv_basic(self):
        """Test basic batch admit functionality using handle_batched_kv_operations."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        keys = list(range(100))

        # Create batch operation message
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=key) for key in keys
        ]
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=worker.worker_id,
            location=location,
            operations=operations,
        )
        worker.handle_batched_kv_operations(msg)

        assert worker.get_kv_count() == 100
        for key in keys:
            assert worker.has_kv(location, key)

    def test_batch_evict_kv_basic(self):
        """Test basic batch evict functionality using handle_batched_kv_operations."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        keys = list(range(100))

        # First admit all keys using batch operation - start with seq_num=0
        admit_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=i)
            for i, key in enumerate(keys)
        ]
        admit_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=worker.worker_id,
            location=location,
            operations=admit_operations,
        )
        worker.handle_batched_kv_operations(admit_msg)
        assert worker.get_kv_count() == 100

        # Evict half of them using batch operation - continue sequence
        evict_keys = list(range(50))
        evict_operations = [
            KVOpEvent(op_type=OpType.EVICT, key=key, seq_num=100 + i)
            for i, key in enumerate(evict_keys)
        ]
        evict_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=worker.worker_id,
            location=location,
            operations=evict_operations,
        )
        worker.handle_batched_kv_operations(evict_msg)

        assert worker.get_kv_count() == 50

        # Check remaining keys
        for key in range(50, 100):
            assert worker.has_kv(location, key)

    def test_batch_evict_nonexistent_keys(self):
        """Test batch evict with some non-existent keys
        using handle_batched_kv_operations."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"

        # Admit only keys 0-49 using batch operation - start with seq_num=0
        admit_keys = list(range(50))
        admit_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=i)
            for i, key in enumerate(admit_keys)
        ]
        admit_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=worker.worker_id,
            location=location,
            operations=admit_operations,
        )
        worker.handle_batched_kv_operations(admit_msg)
        assert worker.get_kv_count() == 50

        # Try to evict keys 0-99 (half don't exist)
        # using batch operation - continue sequence
        evict_keys = list(range(100))
        evict_operations = [
            KVOpEvent(op_type=OpType.EVICT, key=key, seq_num=50 + i)
            for i, key in enumerate(evict_keys)
        ]
        evict_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=worker.worker_id,
            location=location,
            operations=evict_operations,
        )
        worker.handle_batched_kv_operations(evict_msg)

        # All admitted keys should be evicted
        assert worker.get_kv_count() == 0

    def test_registry_batch_operations(self):
        """Test batch operations through RegistryTree
        using handle_batched_kv_operations."""
        registry = RegistryTree()

        registry.register_worker(
            instance_id="inst_0",
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"
        keys = list(range(1000))

        # Batch admit using handle_batched_kv_operations - start with seq_num=0
        admit_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=i)
            for i, key in enumerate(keys)
        ]
        admit_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location=location,
            operations=admit_operations,
        )
        result = registry.handle_batched_kv_operations(admit_msg)
        assert result is not False  # Should succeed
        assert registry.get_total_kv_count() == 1000

        # Batch evict using handle_batched_kv_operations - continue sequence
        evict_keys = keys[:500]
        evict_operations = [
            KVOpEvent(op_type=OpType.EVICT, key=key, seq_num=1000 + i)
            for i, key in enumerate(evict_keys)
        ]
        evict_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location=location,
            operations=evict_operations,
        )
        registry.handle_batched_kv_operations(evict_msg)
        assert registry.get_total_kv_count() == 500

    def test_batch_operations_nonexistent_worker(self):
        """Test batch operations on non-existent worker
        using handle_batched_kv_operations."""
        registry = RegistryTree()

        # Batch admit to non-existent worker using handle_batched_kv_operations
        admit_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=key) for key in [1, 2, 3]
        ]
        admit_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location="loc",
            operations=admit_operations,
        )
        result = registry.handle_batched_kv_operations(admit_msg)
        # Should return None or indicate failure when worker doesn't exist
        assert result is None or result is False

        # Batch evict from non-existent worker using handle_batched_kv_operations
        evict_operations = [
            KVOpEvent(op_type=OpType.EVICT, key=key, seq_num=key) for key in [1, 2, 3]
        ]
        evict_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location="loc",
            operations=evict_operations,
        )
        result = registry.handle_batched_kv_operations(evict_msg)
        # Should return None or indicate failure when worker doesn't exist
        assert result is None or result is False

    def test_concurrent_batch_operations(self):
        """Test concurrent batch operations on same worker
        using handle_batched_kv_operations."""
        worker = WorkerNode(
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        location = "test_location"
        num_threads = 10
        keys_per_batch = 100
        errors = []

        def batch_admit(thread_id):
            keys = [thread_id * keys_per_batch + i for i in range(keys_per_batch)]
            try:
                operations = [
                    KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=key)
                    for key in keys
                ]
                msg = BatchedKVOperationMsg(
                    instance_id="test_instance",
                    worker_id=worker.worker_id,
                    location=location,
                    operations=operations,
                )
                worker.handle_batched_kv_operations(msg)
            except Exception as e:
                errors.append("Thread %d error: %s" % (thread_id, e))

        threads = [
            threading.Thread(target=batch_admit, args=(i,)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Errors occurred: %s" % errors
        assert worker.get_kv_count() == num_threads * keys_per_batch

    def test_registry_batch_with_seq_check(self):
        """Test batch operations with sequence check through RegistryTree
        using handle_batched_kv_operations."""
        registry = RegistryTree()

        registry.register_worker(
            instance_id="inst_0",
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"
        discontinuity_detected = False

        def on_discontinuity():
            nonlocal discontinuity_detected
            discontinuity_detected = True

        # First batch admit with seq check (continuous)
        batch1_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=seq_num)
            for key, seq_num in [(1, 0), (2, 1), (3, 2)]
        ]
        batch1_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location=location,
            operations=batch1_operations,
        )
        result = registry.handle_batched_kv_operations(batch1_msg)
        # Check discontinuity count instead
        discontinuity_count = registry.get_seq_discontinuity_count()
        assert result is not False and discontinuity_count == 0, (
            "First batch should succeed and be continuous"
        )
        assert registry.get_total_kv_count() == 3

        # Second batch admit (continuous)
        batch2_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=seq_num)
            for key, seq_num in [(4, 3), (5, 4), (6, 5)]
        ]
        batch2_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location=location,
            operations=batch2_operations,
        )
        result = registry.handle_batched_kv_operations(batch2_msg)
        discontinuity_count = registry.get_seq_discontinuity_count()
        assert result is not False and discontinuity_count == 0, (
            "Second batch should succeed and be continuous"
        )
        assert registry.get_total_kv_count() == 6

        # Third batch with gap
        batch3_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=seq_num)
            for key, seq_num in [(10, 10), (11, 11)]
        ]
        batch3_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location=location,
            operations=batch3_operations,
        )
        result = registry.handle_batched_kv_operations(batch3_msg)
        discontinuity_count = registry.get_seq_discontinuity_count()
        assert result is not False, "Batch with gap should still succeed"
        assert discontinuity_count > 0, "Sequence discontinuity should be detected"
        assert registry.get_total_kv_count() == 8

    def test_registry_batch_evict_with_seq_check(self):
        """Test batch evict with sequence check through RegistryTree
        using handle_batched_kv_operations."""
        registry = RegistryTree()

        registry.register_worker(
            instance_id="inst_0",
            worker_id=0,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        location = "test_location"

        # First admit some keys (continuous sequence starting from 0)
        admit_operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=i)
            for i, key in enumerate(range(100))
        ]
        admit_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location=location,
            operations=admit_operations,
        )
        result = registry.handle_batched_kv_operations(admit_msg)
        discontinuity_count = registry.get_seq_discontinuity_count()
        assert result is not False and discontinuity_count == 0, (
            "Admit should succeed and be continuous"
        )
        assert registry.get_total_kv_count() == 100

        # Batch evict with seq check (continuous) - continue sequence from 100
        evict_operations = [
            KVOpEvent(op_type=OpType.EVICT, key=key, seq_num=100 + i)
            for i, key in enumerate(range(50))
        ]
        evict_msg = BatchedKVOperationMsg(
            instance_id="inst_0",
            worker_id=0,
            location=location,
            operations=evict_operations,
        )
        result = registry.handle_batched_kv_operations(evict_msg)
        discontinuity_count_after_evict = registry.get_seq_discontinuity_count()
        assert discontinuity_count_after_evict == 0, (
            "Evict should succeed and be continuous"
        )
        assert result is not False, "Evict should succeed"
        # Note: Evict operations don't check sequence continuity,
        # so count shouldn't increase
        assert registry.get_total_kv_count() == 50
