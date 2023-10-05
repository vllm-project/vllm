# TODO maybe wrap in a try except ImportError? idk

import os
import asyncio as aio
import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.utils.classic import obtain
from contextlib import closing
import socket
from datetime import timedelta
import time

# doesn't work
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(">>> importing rpycutils", os.getpid())


def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]


class RPyCWorkerService(rpyc.Service):
    def on_connect(self, conn):

        pass

    def on_disconnect(self, conn):
        pass

    def exposed_print_debug_msg(self, msg):
        print(f"in service {os.getpid()}:{msg}")

    

    def exposed_get_addr_and_port(self):
        # equivalent of
        # addr = ray.util.get_node_ip_address()
        # port = find_free_port()
        addr = "127.0.0.1"  # we should be local I think
        port = find_free_port()
        return addr, port


    def exposed_init_torch_distributed(self, master_addr, master_port, gpu_ids, world_size, rank):
        # https://github.com/ray-project/ray/blob/7a3ae5ba5dbd6704f435bde8dba91a8a8d207ae4/python/ray/air/util/torch_dist.py#L95
        # for reference
        print(f"Running on {os.getpid()}")
        print(f"{master_addr}:{master_port}, #gpus {gpu_ids}, ws {world_size}, rank {rank}")
        
        os.environ["MASTER_ADDR"] = str(master_addr)  # idk lmao search up torch distributed
        os.environ["MASTER_PORT"] = str(master_port)

        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # idk what this does
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id for gpu_id in gpu_ids))  # TODO wrong type?
        if "NCCL_SOCKET_IFNAME" not in os.environ:
            os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker,veth"

        # TODO debug stuff
        print(os.getpid(), "importing torch", time.time())
        # import importlib
        import torch  # this import is fast, which suggests we've already imported it
        import torch.distributed as dist
        # importlib.reload(torch)  # tried reloading torch/dist, get some error generic_type: cannot initialize type "GradBucket": an object with that name is already defined
        # importlib.reload(dist)
        print(os.getpid(), "done importing torch", time.time())
        print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")

        # ray makes a call to init process group here
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size, timeout=timedelta(seconds=1800))

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
        # import inside worker process since if not it'll break the engine process
        # probably same reason as why _init_workers_ray imports this so late?
        from vllm.worker.worker import Worker
        model_config, parallel_config, scheduler_config = obtain(model_config), obtain(parallel_config), obtain(scheduler_config)
        self.worker = Worker(
            model_config,
            parallel_config,
            scheduler_config,
            None,
            None,
        )

    def exposed_execute_method(self, method: str, *args, **kwargs):
        # print(f"execute_method running on {os.getpid()}")
        # raise ValueError("crashing to view call stack")
        print(os.getpid(), "starthead", time.time())
        # I believe this obtain() makes a call to the other process, which may be a bottleneck. 
        args, kwargs = obtain(args), obtain(kwargs)  # with prints, seems like this takes about 0.0025 seconds 4 workers n=1
        executor = getattr(self.worker, method)
        print(os.getpid(), "startexec", time.time())
        retval = executor(*args, **kwargs)
        print(os.getpid(), "stopexec", time.time())
        return retval
    
class RPyCWorkerClient:
    def __init__(self, conn):
        self.conn = conn
        # conn is type rpyc.core.protocol.Connection
        # import pdb; pdb.set_trace()
        # print("workerclient.conn type", type(self.conn))  # apparently this hangs? wtf
        def async_wrap(f):
            f = rpyc.async_(f)
            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                # ans.set_expiry(3600)  # absurdly long, wait for model to init
                # print(ans._ttl)
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
        self._get_addr_and_port = self.conn.root.get_addr_and_port
        


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
        # we run obtain() on the worker to send {model|parallel|scheduler}_config to the workers
        self._init_worker(model_config, parallel_config, scheduler_config)

    def execute_method(self, method, *args, **kwargs):
        print(f"executing method {method} at {time.time()}")  # with threadpoolexecutor these prints seem to execute at the same time
        ans = self._execute_method(method, *args, **kwargs)  # TODO is this right?
        print(f"finish executing method {method} at {time.time()}")
        new_ans = obtain(ans)
        return new_ans
    
    def get_addr_and_port(self):
        return self._get_addr_and_port()
    
    async def aexecute_method(self, method, *args, **kwargs):
        t1 = time.time()
        print(f'started at {t1}')
        ans = await self._aexecute_method(method, *args, **kwargs)
        # t2 = time.time()
        new_ans = obtain(ans)  # seems fast enough
        # t3 = time.time()
        # print(t2 - t1, t3 - t2)
        return new_ans  # do we need to check None? probably not?
    
    async def ainit_torch_distributed(self, master_addr, master_port, gpu_ids, world_size, rank):
        return await self._ainit_torch_distributed(master_addr, master_port, gpu_ids, world_size, rank)
    
    async def ainit_worker(self, model_config, parallel_config, scheduler_config):
        return await self._ainit_worker(model_config, parallel_config, scheduler_config)



def init_rpyc_env(port):
    # We need to import torch here, otherwise torch won't recognize CUDA devices as available.
    # Not sure why unfortunately, but I think it's related to some ordering of imports/environment set up
    import torch
    t = ThreadedServer(RPyCWorkerService(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return

def example(local_rank):
    # debug torch cuda
    import torch
    print("Example Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")