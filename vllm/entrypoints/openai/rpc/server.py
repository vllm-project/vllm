import asyncio
import pickle
import zmq
import zmq.asyncio

from vllm import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.openai.rpc import (VLLM_GENERATE_RPC_PATH,
                                         VLLM_GET_DATA_RPC_PATH,
                                         VLLM_IS_READY_RPC_PATH,
                                         GetDataRequest)

class RPCServer:
    def __init__(self, async_engine_args):        
        # Initialize engine first.
        self.engine = AsyncLLMEngine.from_engine_args(
            async_engine_args, UsageContext.OPENAI_API_SERVER)

        # Initialize context.
        self.context = zmq.asyncio.Context()
        
        # Init socket for readiness state.
        self.is_ready_socket = self.context.socket(zmq.REP)
        self.is_ready_socket.bind(VLLM_IS_READY_RPC_PATH)

        # Init socket for generation.
        self.generate_socket = self.context.socket(zmq.ROUTER)
        self.generate_socket.bind(VLLM_GENERATE_RPC_PATH)

        # TODO (robertgshaw2-neuralmagic): 
        # add socket for generation without streaming

        # Init socket for simple data requests.
        self.get_data_socket = self.context.socket(zmq.REP)
        self.get_data_socket.bind(VLLM_GET_DATA_RPC_PATH)

        # Setup polling so we can listen on both sockets.
        self.poller = zmq.asyncio.Poller()
        self.poller.register(self.generate_socket, zmq.POLLIN)
        self.poller.register(self.get_data_socket, zmq.POLLIN)


    async def get_data(self, message):
        request_type = pickle.loads(message)
        if request_type == GetDataRequest.MODEL_CONFIG:
            return await self.engine.get_model_config()
        else:
            raise ValueError(f"Unknown request type: {request_type}")


    async def generate(self, identity, message):
        request = pickle.loads(message)
        results_generator = self.engine.generate(
            request.inputs, 
            sampling_params=request.sampling_params,
            request_id=request.request_id)
        
        async for request_output in results_generator:
            self.socket.send_multipart([
                identity, 
                pickle.dumps(request_output, pickle.HIGHEST_PROTOCOL)
            ])
    
    async def run_loop(self):
        # Notify the RPC client that we are ready to recieve requests.
        await self.is_ready_socket.send_string("Ready!")
        self.is_ready_socket.close()

        # Avoid GC of running tasks.
        running_tasks = set()
        while True:
            try:
                socks = dict(await self.poller.poll())
            except KeyboardInterrupt:
                # TODO: should there be some other exception here?
                break
            
            task = None
            if self.generate_socket in socks:
                identity, message = await self.generate_socket.recv_multipart()
                task = asyncio.create_task(self.generate(identity, message)) 

            elif self.get_data_socket in socks:
                message = await self.get_data_socket.recv()
                task = asyncio.create_task(self.get_data(message))

            # We need to keep around a strong reference to the task, 
            # to avoid the task disappearing mid-execution as running tasks
            # can be GC'ed. Below is a common "fire-and-forget" tasks
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
            if task is not None:
                running_tasks.add(task)
                task.add_done_callback(running_tasks.discard)

        # TODO: Do I need to close the generate / get_data sockets?

def run_rpc_server(async_engine_args):
    server = RPCServer(async_engine_args=async_engine_args)
    asyncio.run(server.run_loop())
