from vllm import AsyncEngineArgs, AsyncLLMEngine
import asyncio
import zmq
import zmq.asyncio
from vllm.grpc.pb import generate_pb2
from vllm import SamplingParams
from vllm.inputs.data import TextPrompt, TokensPrompt

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_TOKENS = 150
TEMPERATURE = 0

class RPCServer:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind('tcp://*:5570')

        self.running_tasks = set()
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=MODEL,
                            enable_chunked_prefill=True))

    async def generate(self, identity, message):
        # request = pickle.loads(message)
        request = generate_pb2.GenerateRequest()
        request.ParseFromString(message)
        
        if len(request.prompt_inputs.prompt_token_ids) > 0:
            inputs = TokensPrompt(prompt_token_ids=request.prompt_inputs.prompt_token_ids)
        else:
            inputs = TextPrompt(prompt=request.prompt_inputs.prompt)

        results_generator = self.engine.generate(
            inputs,
            sampling_params=SamplingParams(max_tokens=MAX_TOKENS, 
                                           temperature=TEMPERATURE),
            request_id=request.request_id)
        
        async for request_output in results_generator:
            outputs = [ 
                generate_pb2.CompletionOutput(
                    index=output.index,
                    token_ids=output.token_ids,
                    text=output.text,
                    finish_reason=output.finish_reason)
                for output in request_output.outputs
            ]
            proto = generate_pb2.GenerateResponse(outputs=outputs)
            self.socket.send_multipart([identity, proto.SerializeToString()])
        
    async def run_loop(self):
        while True:
            identity, message = await self.socket.recv_multipart()
            print("got message")
            
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
