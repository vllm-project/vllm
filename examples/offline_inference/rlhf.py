from vllm import LLM, SamplingParams
import cloudpickle
from vllm.worker.worker import Worker

class MyWorker(Worker):
    def echo_rank(self):
        from vllm.distributed.parallel_state import get_world_group
        return get_world_group().rank

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m", enforce_eager=True, worker_cls=cloudpickle.dumps(MyWorker), tensor_parallel_size=2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(llm.collective_rpc("echo_rank"))
