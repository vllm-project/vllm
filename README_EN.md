This is a fork of vLLM(Version: 0.2.2). The features we added is to support gptq quantization, you can directly load an A16W4 quantization model trained by AutoGPTQ.

## New features
The features we added is to support gptq quantization. We test on the Qwen-72B and the test performance is shown in the table.

| context length | generate length | tokens/s    | tokens/s   | tokens/s    | tokens/s   | tokens/s    | tokens/s   |   tokens/s  |  tokens/s  |
|----------------|-----------------|-------------|------------|-------------|------------|-------------|------------|:-----------:|:----------:|
|                |                 |     tp=8    |    tp=8    |     tp=4    |    tp=4    |     tp=2    |    tp=2    |     tp=1    |    tp=1    |
|                |                 | fp16 a16w16 | int4 a16w4 | fp16 a16w16 | int4 a16w4 | fp16 a16w16 | int4 a16w4 | fp16 a16w16 | int4 a16w4 |
|        1       |        2k       |    26.42    |    27.68   |    24.98    |    27.19   |    17.39    |    20.76   |      -      |    14.63   |
|       6k       |        2k       |    24.93    |    25.98   |    22.76    |    24.56   |      -      |    18.07   |      -      |      -     |
|       14k      |        2k       |    22.67    |    22.87   |    19.38    |    19.28   |      -      |    14.51   |      -      |      -     |
|       30k      |        2k       |    19.95    |    19.87   |    17.05    |    16.93   |      -      |      -     |      -      |      -     |


## Quickstart

### Installation
You can build and install vLLM from source.

```
   cd vllm 
   pip install -e .
```

### Usage Examples
We introduce how to run the gptq a16w4 quantization model here. If you want to learn more about vllm, please read [official documentation](https://github.com/vllm-project/vllm). The example codes of the QWen can be found in the directory 'tests/qwen/'.

#### Offline Batched Inference
Note: To run the following code, you need to enter the directory 'tests/qwen/' first.

```python
from vllm_wrapper import vLLMWrapper

if __name__ == '__main__':
    model = ""
    #gptq a16w4 model
    vllm_model = vLLMWrapper(model,
                             quantization = 'gptq',
                             dtype="float16",
                             tensor_parallel_size=2)

    response, history = vllm_model.chat(query="你好",
                                        history=None)
    print(response)
    response, history = vllm_model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。",
                                        history=history)
    print(response)
    response, history = vllm_model.chat(query="给这个故事起一个标题",
                                        history=history)
    print(response)

```

#### API Server

Note: In addition to installing the software required by vllm, you should install FastChat.

```bash
pip install fschat
pip install accelerate

```

##### Start the Server

step 1. Launch the controller
```
    cd FastChat/
    python -m fastchat.serve.controller
```

step 2. Launch the model worker 
```
    python -m fastchat.serve.vllm_worker --model-path $model_path --tensor-parallel-size 4 --trust-remote-code
```

step 3. Launch the openai api server
```
   python -m fastchat.serve.openai_api_server --host localhost --port 8000
```

##### Query the model by API

step 1. install openai-python
```bash
pip install --upgrade openai
```
step 2. Query codes
```python
import openai
# to get proper authentication, make sure to use a valid key that's listed in
# the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

model = "qwen"
call_args = {
    'temperature': 1.0,
    'top_p': 1.0,
    'top_k': -1,
    'max_tokens': 2048, # output-len
    'presence_penalty': 1.0,
    'frequency_penalty': 0.0,
}
# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}],
  **call_args
)
# print the completion
print(completion.choices[0].message.content)
```