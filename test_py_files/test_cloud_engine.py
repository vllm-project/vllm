import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PretrainedConfig, AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel
from typing import Callable, List, Optional, Tuple, Union, Dict
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from vllm import LLM
from vllm import SamplingParams
from functools import partial
import os
import time
def register():
    from vllm import ModelRegistry
    from decoder import XCodeDecForCausalLM, XCodeDecConfig  # Import decoder classes

    AutoConfig.register("xcodedec", XCodeDecConfig)  # Register decoder config
    ModelRegistry.register_model("XCodeDecModelForCausalLM", XCodeDecForCausalLM)  # Register decoder model
    from middle_model import XCodeForCausalLM, XCodeMiddleConfig  # Changed to absolute import

    AutoConfig.register("xcodemiddle", XCodeMiddleConfig)
    ModelRegistry.register_model("XCodeMiddleModelForCausalLM", XCodeForCausalLM)

    from encoder import XCodeEncForCausalLM, XCodeEncConfig  # Import encoder classes

    AutoConfig.register("xcodeenc", XCodeEncConfig)  # Register encoder config
    ModelRegistry.register_model("XCodeEncModelForCausalLM", XCodeEncForCausalLM)  # Register encoder model

    from enc_dec import XCodeEncDecConfig, XCodeEncDecForCausalLM  # Import encoder classes

    AutoConfig.register("xcodeencdec", XCodeEncDecConfig)  # Register encoder config
    ModelRegistry.register_model("XCodeEncDecModelForCausalLM", XCodeEncDecForCausalLM)  # Register encoder model

register()



middle_model = LLM(
    model="/project/phan/kt477/OppyAI_backend/qwen7b_middle_clean_no_att_on_client_dec",
    # model="Qwen/Qwen2.5-Coder-32B-Instruct",
    tokenizer="Qwen/Qwen2.5-Coder-7B-Instruct",
    skip_tokenizer_init=True,
    # task="reward",
    enable_prompt_embeds=True,
    model_part="middle",  # Set to False for encoder
    gpu_memory_utilization=0.5,
    max_model_len=1024,
    tensor_parallel_size=1,
    enforce_eager=True
)




middle_engine = middle_model.llm_engine


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)





request_id = 0

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
# # input ids to list of integers
input_ids = model_inputs.input_ids[0].tolist()
tokens = []


def send_intermediate_states(_, __, output, prefix = "client"):
    hidden_states, residual = output
    # Right now, save the hidden states and residual to file
    if os.path.exists("test_py_files") is False:
        os.makedirs("test_py_files")
            


    torch.save(hidden_states, f"test_py_files/{prefix}_hidden_states_tensor.pt")
    torch.save(residual, f"test_py_files/{prefix}_residual_tensor.pt")



    # serialized_hidden_states = pickle.dumps(hidden_states.to("cpu"))
    # serialized_residual = pickle.dumps(residual.to("cpu"))
    # node.isend(serialized_hidden_states, tag=0, latency=None).wait()
    # node.isend(serialized_residual, tag=0, latency=None).wait()
    # logger.debug(f"Sent hidden_states: {hidden_states.shape} ({len(serialized_hidden_states)} bytes sent) and residual: {residual.shape} ({len(serialized_residual)} bytes sent)")


def recv_intermediate_states(_, input, prefix = "client"):

    positions, _, _ = input
    device = positions.device

    # Load the hidden states and residual from file
    if os.path.exists("test_py_files") is False:
        os.makedirs("test_py_files")

        # If the 2 files do not exist, wait until they are created
    if not os.path.exists(f"test_py_files/{prefix}_hidden_states_tensor.pt") or not os.path.exists(f"test_py_files/{prefix}_residual_tensor.pt"):

        while not (os.path.exists(f"test_py_files/{prefix}_hidden_states_tensor.pt") and os.path.exists(f"test_py_files/{prefix}_residual_tensor.pt")) and not os.path.exists("test_py_files/terminate.json"):
            pass
                # time.sleep(10)  # Wait for 10 seconds before checking again
    if os.path.exists("test_py_files/terminate.json"):

        raise Exception("Termination file detected")
    print(f"Loading hidden states and residual from {prefix} files...")
    i = 0
    # Retry loading until successful
    while i < 5:
        try:
            hidden_states = torch.load(f"test_py_files/{prefix}_hidden_states_tensor.pt").to(device)
            residual = torch.load(f"test_py_files/{prefix}_residual_tensor.pt").to(device)

            break
        except Exception as e:
            time.sleep(1)
            i += 1


    
    # Delete the files after loading
    os.remove(f"test_py_files/{prefix}_hidden_states_tensor.pt")
    os.remove(f"test_py_files/{prefix}_residual_tensor.pt")
    print(f"Removed files: {prefix}_hidden_states_tensor.pt and {prefix}_residual_tensor.pt")


    # serialized_hidden_states = node.irecv(tag=0).wait()
    # serialized_residual = node.irecv(tag=0).wait()
    # hidden_states = pickle.loads(serialized_hidden_states).to(device)
    # residual = pickle.loads(serialized_residual).to(device)
    # logger.debug(f"Got hidden_states: {hidden_states.shape} ({len(serialized_hidden_states)} bytes sent), residual: {residual.shape} ({len(serialized_residual)} bytes sent) and positions {positions.shape}")

    return positions, hidden_states, residual


middle_engine.model_executor.driver_worker.model_runner.model.middle.layers[-1].register_forward_hook(partial(send_intermediate_states, prefix="cloud"))
middle_engine.model_executor.driver_worker.model_runner.model.middle.layers[0].register_forward_pre_hook(partial(recv_intermediate_states, prefix="client"))


# middle_output = middle_model.generate(
#     {
#         "prompt_embeds": torch.zeros((35, 3584), device="cuda:0")  # Placeholder for middle model,
#     },
#     SamplingParams(max_tokens=2048)
# )

middle_engine.add_request(
    request_id="123",
    prompt={
        "prompt_embeds": torch.zeros((35, 3584), device="cuda:0")  # Placeholder for middle model,
    },
    params=SamplingParams(max_tokens=2048)
)
while middle_engine.has_unfinished_requests():
    try:
        if os.path.exists("test_py_files/terminate.json"):
            print("Termination file detected. Stopping...")
            middle_engine.abort_request("123")
            os.remove("test_py_files/terminate.json")
        else:
            middle_output = middle_engine.step()
    except Exception as e:
        print(f"Error occurred: {e}")
