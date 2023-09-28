# TODO maybe wrap in a try except ImportError? idk

import os
import rpyc
from rpyc.utils.server import ThreadedServer

class RPyCWorkerService(rpyc.Service):
    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_print_debug_msg(self, msg):
        print(f"{os.getpid()}:{msg}")

    def exposed_init_torch_distributed(self, master_addr, master_port, gpu_ids, world_size, rank):
        # https://github.com/ray-project/ray/blob/7a3ae5ba5dbd6704f435bde8dba91a8a8d207ae4/python/ray/air/util/torch_dist.py#L95
        # for reference
        print(f"Running on {os.getpid()}")
        
        os.environ["MASTER_ADDR"] = str(master_addr)  # idk lmao search up torch distributed
        os.environ["MASTER_PORT"] = str(master_port)

        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # idk what this does
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id for gpu_id in gpu_ids))


        # running on one node, local_{rank|world_size} is same as {rank|world_size}
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)

    def exposed_init_worker(self, worker_init_fn):
        print(f"Running on {os.getpid()}")
        # TODO check that worker_init_fn runs on the worker process
        self.worker = worker_init_fn()

    def exposed_execute_method(self, method: str, *args, **kwargs):
        executor = getattr(self.worker, method)  # TODO f"exposed_{method}"?
        return executor(*args, **kwargs)
    
class RPyCWorkerClient:
    def __init__(self, conn):
        self.conn = conn

    def print_debug_msg(self, msg):
        self.conn.root.print_debug_msg(msg)
    
    def init_torch_distributed(self, master_addr, master_port, gpu_ids, world_size, rank):
        self.conn.root.init_torch_distributed(master_addr, master_port, gpu_ids, world_size, rank)

    def init_worker(self, worker_init_fn):
        self.conn.root.init_worker(worker_init_fn)

    def execute_method(self, method, *args, **kwargs):
        return self.conn.root.execute_method(method, *args, **kwargs)  # TODO is this right?



def init_rpyc_env(port):
    t = ThreadedServer(RPyCWorkerService(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return