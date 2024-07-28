from vllm import AsyncEngineArgs, AsyncLLMEngine
import asyncio
import pickle
import zmq, zlib
import zmq.asyncio

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


class RPCServer:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind('tcp://*:5570')

        self.running_tasks = set()
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=MODEL, enable_chunked_prefill=True))

    async def generate(self, identity, message):
        request = pickle.loads(zlib.decompress(message))
        results_generator = self.engine.generate(
            request.inputs, 
            sampling_params=request.sampling_params,
            request_id=request.request_id)
        
        async for request_output in results_generator:
            self.socket.send_multipart([
                identity, 
                zlib.compress(
                    pickle.dumps(request_output, pickle.HIGHEST_PROTOCOL))
            ])
        
    async def run_loop(self):
        while True:
            identity, message = await self.socket.recv_multipart()
            
            # Process the request in the background.
            task = asyncio.create_task(self.generate(identity=identity,
                                                     message=message))

            # We need to keep around a strong reference to the task, 
            # to avoid the task disappearing mid-execution as running tasks
            # can be GC'ed. Below is a common "fire-and-forget" tasks
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
            self.running_tasks.add(task)
            task.add_done_callback(self.running_tasks.discard)


if __name__ == "__main__":
    server = RPCServer()
    asyncio.run(server.run_loop())
