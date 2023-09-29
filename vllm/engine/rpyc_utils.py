# TODO maybe wrap in a try except ImportError? idk

import os
import asyncio as aio
import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.utils.classic import obtain

from vllm.worker.worker import Worker

class RPyCWorkerService(rpyc.Service):
    def on_connect(self, conn):

        pass

    def on_disconnect(self, conn):
        pass

    def exposed_print_debug_msg(self, msg):
        print(f"in service {os.getpid()}:{msg}")

    def exposed_init_torch_distributed(self, master_addr, master_port, gpu_ids, world_size, rank):
        # https://github.com/ray-project/ray/blob/7a3ae5ba5dbd6704f435bde8dba91a8a8d207ae4/python/ray/air/util/torch_dist.py#L95
        # for reference
        print(f"Running on {os.getpid()}")
        print(f"{master_addr}:{master_port}, #gpus {gpu_ids}, ws {world_size}, rank {rank}")
        
        os.environ["MASTER_ADDR"] = str(master_addr)  # idk lmao search up torch distributed
        os.environ["MASTER_PORT"] = str(master_port)

        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # idk what this does
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id for gpu_id in gpu_ids))


        # running on one node, local_{rank|world_size} is same as {rank|world_size}
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)

    def exposed_init_worker_doesnt_work(self, worker_init_fn):
        print(f"init_worker running on {os.getpid()}")
        # TODO check that worker_init_fn runs on the worker process
        # can't pickle worker_init_fn shrug
        worker_init_fn = obtain(worker_init_fn)
        print(worker_init_fn, type(worker_init_fn))
        self.worker = worker_init_fn()
        print(type(self.worker))
        # dumb hack idk how to get the thing to initialize the worker here so will do it manually

    def exposed_init_worker(self, model_config, parallel_config, scheduler_config):
        print(f"in worker {os.getpid()}, {model_config}, {parallel_config}, {scheduler_config}")
        model_config, parallel_config, scheduler_config = obtain(model_config), obtain(parallel_config), obtain(scheduler_config)
        print(f"type after serializing/deserializing: {type(parallel_config)}")
        print(f"idk {parallel_config.tensor_parallel_size}")
        print(f"ws after serializing/deserializing: {parallel_config.world_size}")
        self.worker = Worker(
            model_config,
            parallel_config,
            scheduler_config,
            None,
            "env://", # f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",  # TODO ????????? figure this out, goes into torch.dist.init_process_group whatever that does 
        )

    def exposed_execute_method(self, method: str, *args, **kwargs):
        print(f"execute_method running on {os.getpid()}")
        print(type(self.worker))
        executor = getattr(self.worker, method)
        return executor(*args, **kwargs)
    
class RPyCWorkerClient:
    def __init__(self, conn):
        self.conn = conn
        def async_wrap(f):
            f = rpyc.async_(f)
            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                await aio.to_thread(ans.wait)
                # raise if exception
                return ans.value
            return _func
        self.async_wrap = async_wrap
        self._ainit_torch_distributed = self.async_wrap(self.conn.root.init_torch_distributed)
        self._ainit_worker = self.async_wrap(self.conn.root.init_worker)
        self._aexecute_method = self.async_wrap(self.conn.root.execute_method)
        self._init_torch_distributed = self.conn.root.init_torch_distributed
        self._init_worker = self.conn.root.init_worker
        self._execute_method = self.conn.root.execute_method
        


    def print_debug_msg(self, msg):
        self.conn.root.print_debug_msg(msg)
    
    async def aprint_debug_msg(self, msg):
        return await self.async_wrap(self.conn.root.print_debug_msg)(msg)

    # TODO will I end up needing the nonasync fns?
    
    def init_torch_distributed(self, master_addr, master_port, gpu_ids, world_size, rank):
        self._init_torch_distributed(master_addr, master_port, gpu_ids, world_size, rank)

    def init_worker_doesnt_work(self, worker_init_fn):
        self._init_worker(worker_init_fn)

    def init_worker(self, model_config, parallel_config, scheduler_config):
        # TODO something to mark as transferable? idk lightllm does this
        print(f"ws, type before serializing: {parallel_config.world_size}, {type(parallel_config)}")
        print(f"help: {parallel_config.world_size}")
        self._init_worker(model_config, parallel_config, scheduler_config)

    def execute_method(self, method, *args, **kwargs):
        print(f"executing method {method}")
        return self._execute_method(method, *args, **kwargs)  # TODO is this right?
    
    async def aexecute_method(self, method, *args, **kwargs):
        return await self._aexecute_method(method, *args, **kwargs)
    
    async def ainit_torch_distributed(self, master_addr, master_port, gpu_ids, world_size, rank):
        return await self._ainit_torch_distributed(master_addr, master_port, gpu_ids, world_size, rank)
    
    async def ainit_worker(self, worker_init_fn):
        return await self._ainit_worker(worker_init_fn)



def init_rpyc_env(port):
    print(f"init_rpyc_env for port {port}")
    t = ThreadedServer(RPyCWorkerService(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return