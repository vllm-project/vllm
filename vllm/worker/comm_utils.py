import cupy as cp
import os

try:
    from mscclpp.utils import KernelBuilder, pack
except ImportError:
    pass

MAX_SEMIDS = 10
FLUSH_COUNT = 128

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../csrc"

# Seq2SemMapper is a class that maps sequence ids to semaphore ids
# It is used to manage the semaphore ids for MSCCL++ proxy channels
class Seq2SemMapper:
    def __init__(self):
        self.available_semids = list(range(MAX_SEMIDS))
        self.seq_to_sem = {}

    def set_seq(self, seq_id):
        sem_id = self.available_semids.pop(0)
        self.seq_to_sem[seq_id] = sem_id

    def free_seq(self, seq_id):
        sem_id = self.seq_to_sem.pop(seq_id)
        self.available_semids.insert(0, sem_id)

    def get_sem_id(self, seq_id):
        return self.seq_to_sem[seq_id]

class SplitCommInfo():
    def __init__(self,
                 worker_type,
                 proxy_service,
                 device_handles,
                 flush_counter,
                 block_size,
                 memory_ids,
                 my_rank,
                 remote_rank,):
        self.worker_type = worker_type
        self.proxy_service = proxy_service
        self.device_handles = device_handles
        self.flush_counter = flush_counter
        self.block_size = block_size
        self.memory_ids = memory_ids
        self.my_rank = my_rank
        self.remote_rank = remote_rank

class SendKVKernel:
    def __init__(self):
        self._kernel = KernelBuilder(
            file="kv_comm_kernels.cu",
            kernel_name="nw_cache_out_kernel",
            file_dir=KERNEL_DIR
        ).get_compiled_kernel()
        self.nblocks = 1
        self.nthreads = 1

    def __call__(self, params):
        return self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, None)

class SignalKVKernel:
    def __init__(self):
        self._kernel = KernelBuilder(
            file="kv_comm_kernels.cu",
            kernel_name="nw_cache_out_signal_kernel",
            file_dir=KERNEL_DIR
        ).get_compiled_kernel()
        self.nblocks = 1
        self.nthreads = 1

    def __call__(self, params):
        return self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, None)

class WaitKVKernel:
    def __init__(self):
        self._kernel = KernelBuilder(
            file="kv_comm_kernels.cu",
            kernel_name="nw_cache_in_kernel",
            file_dir=KERNEL_DIR
        ).get_compiled_kernel()
        self.nblocks = 1
        self.nthreads = 1

    def __call__(self, params):
        return self._kernel.launch_kernel(params, self.nblocks, self.nthreads, 0, None)

class KVCacheCommunicator:
    def __init__(self, comm_info):
        self.comm_info = comm_info
        self.send_kernel = SendKVKernel()
        self.signal_kernel = SignalKVKernel()
        self.wait_kernel = WaitKVKernel()

    def wait(self, sem_id):
        dh = cp.asarray(memoryview(b"".join([self.comm_info.device_handles[sem_id]])), dtype=cp.uint8)
        params = pack(dh)
        self.wait_kernel(params)

    def signal_and_flush(self, sem_id):
        dh = cp.asarray(memoryview(b"".join([self.comm_info.device_handles[sem_id]])), dtype=cp.uint8)
        self.comm_info.flush_counter += 1
        flush = True if self.comm_info.flush_counter % FLUSH_COUNT == 0 else False
        params = pack(dh, flush)
        self.signal_kernel(params)

    def put(self, sem_id, layer_id, block_start, num_blocks):
        block_size = self.comm_info.block_size
        remote_rank = self.comm_info.remote_rank
        my_rank = self.comm_info.my_rank
        for k_or_v in [0, 1]:
            block_offset = block_start * block_size
            dh = cp.asarray(memoryview(b"".join([self.comm_info.device_handles[sem_id]])), dtype=cp.uint8)
            self.comm_info.flush_counter += 1
            flush = True if self.comm_info.flush_counter % FLUSH_COUNT == 0 else False
            params = b""
            params += pack(
                dh,
                self.comm_info.memory_ids[layer_id][k_or_v][remote_rank],
                self.comm_info.memory_ids[layer_id][k_or_v][my_rank],
                block_offset,
                block_size * num_blocks,
                flush
            )
            self.send_kernel(params)
