import queue
import numpy as np
import tritonclient.grpc as grpcclient
from functools import partial
import uuid


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        print(result.as_numpy("response"))
        user_data._completed_requests.put(result)


if __name__ == "__main__":
    timeout = 10000
    model_name = "vicuna_13b_13"
    prompts = ["How to make vLLM great again"]
    max_tokens = [256]
    user_data = UserData()

    prompt_bytes = [prompt.encode() for prompt in prompts]
    line = np.array(prompt_bytes, dtype=np.object_)

    inputs = [
        grpcclient.InferInput("prompt", line.shape, "BYTES"),
        grpcclient.InferInput("max_tokens", line.shape, "INT32"),
    ]

    outputs = [grpcclient.InferRequestedOutput("response")]
    with grpcclient.InferenceServerClient(url="localhost:8001") as triton_client:
        triton_client.start_stream(callback=partial(callback, user_data))

        inputs[0].set_data_from_numpy(line)
        line = np.array(max_tokens, dtype=np.int32)
        inputs[1].set_data_from_numpy(line)

        request_id = str(uuid.uuid4())

        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            request_id=request_id,
            outputs=outputs,
        )

    print("finished")
