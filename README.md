<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

本仓库是基于vLLM（版本0.2.2）进行修改的一个分支，主要为了支持[Qwen系列大语言模型](https://github.com/QwenLM/Qwen)的GPTQ量化推理。

This repo is a fork of vLLM(Version: 0.2.2), which supports the GPTQ model inference of [Qwen large language models](https://github.com/QwenLM/Qwen).

## 新增功能

该版本vLLM跟官方0.22版本的主要区别在于增加GPTQ int4量化模型支持。我们在Qwen-72B-Chat上测试了量化模型性能，结果如下表。

The features we added is to support GPTQ int4 quantization. We test on the Qwen-72B and the test performance is shown in the table.

| context length | generate length | tokens/s    | tokens/s   | tokens/s    | tokens/s   | tokens/s    | tokens/s   |   tokens/s  |  tokens/s  |
|----------------|-----------------|-------------|------------|-------------|------------|-------------|------------|:-----------:|:----------:|
|                |                 |     tp=8    |    tp=8    |     tp=4    |    tp=4    |     tp=2    |    tp=2    |     tp=1    |    tp=1    |
|                |                 | fp16 a16w16 | int4 a16w4 | fp16 a16w16 | int4 a16w4 | fp16 a16w16 | int4 a16w4 | fp16 a16w16 | int4 a16w4 |
|        1       |        2k       |    26.42    |    27.68   |    24.98    |    27.19   |    17.39    |    20.76   |      -      |    14.63   |
|       6k       |        2k       |    24.93    |    25.98   |    22.76    |    24.56   |      -      |    18.07   |      -      |      -     |
|       14k      |        2k       |    22.67    |    22.87   |    19.38    |    19.28   |      -      |    14.51   |      -      |      -     |
|       30k      |        2k       |    19.95    |    19.87   |    17.05    |    16.93   |      -      |      -     |      -      |      -     |


## 如何开始

### 安装

为了安装vLLM，你必须满足以下要求：

To install vLLM, you must meet the below requirements.

* torch >= 2.0
* cuda 11.8 or 12.1

目前，我们仅支持源码安装。

You can install vLLM from source.

如果你使用cuda 12.1和torch 2.1，你可以使用以下方法安装

If you use cuda 12.2 and torch 2.1, you can install vLLM by

```
   git clone https://github.com/QwenLM/vllm-gptq.git
   cd vllm-gptq
   pip install -e .
```

其他情况下，安装可能较为复杂。一个可能的方式是，安装对应版本的cuda和PyTorch后，**删除`requirements.txt`的torch依赖，并删除`pyproject.toml`**，再尝试执行`pip install -e .`。

In other cases, installation may be complicated. One possible way is to install the corresponding versions of CUDA and PyTorch, **delete the torch dependencies in `Requirements.txt`, delete `pyproject.toml`, and then try to execute `pip install -e.`

### 如何使用

我们在此仅介绍如何运行Qwen的量化模型。

We only introduce how to run Qwen's quantized model.

* 如果想了解更多关于Qwen系列模型的用法，请访问[Qwen官方仓库](https://github.com/QwenLM/Qwen)
* 如果想使用vLLM其他功能，请阅读 [官方文档](https://github.com/vllm-project/vllm)。

* If you want to know more about the Qwen series model, visit [Qwen's official repo] (https://github.com/qwenlm/qwen)
* If you want to use other functions of VLLM, read [Official Document] (https://github.com/vllm-project/vllm).

关于Qen量化模型的示例代码，代码目录在tests/qwen/。

Regarding the example code of Qwen quantized model, the code directory is in tests/qwen/.

注意：当前本仓库仅支持Int4量化模型。Int8量化模型将在后续支持。

Note: The current warehouse only supports Int4 quantized model. Int8 quantization will be supported in near future.

#### 批处理调用模型

注意：运行以下代码，需要先进入对应的目录：tests/qwen/。

Note: To run the following code, you need to enter the directory 'tests/qwen/' first.

```python
from vllm_wrapper import vLLMWrapper

if __name__ == '__main__':
    model = "Qwen/Qwen-72B-Chat-Int4"

    vllm_model = vLLMWrapper(model,
                             quantization = 'gptq',
                             dtype="float16",
                             tensor_parallel_size=1)

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

#### API方式调用模型

除去安装vLLM外，以API方式调用模型需要额外安装fastchat

In addition to installing vLLM, you should install FastChat.

```bash
pip install fschat
```

##### 启动Server

step 1. 启动控制器

step 1. Launch the controller

```
    python -m fastchat.serve.controller
```

step 2. 启动模型worker

step 2. Launch the model worker 

```
    python -m fastchat.serve.vllm_worker --model-path $model_path --tensor-parallel-size 1 --trust-remote-code
```

step 3. 启动服务器

step 3. Launch the openai api server

```
   python -m fastchat.serve.openai_api_server --host localhost --port 8000
```

##### API调用

step 1. 安装openai-python

step 1. install openai-python

```bash
pip install --upgrade openai
```

step 2. 调用接口

step 2. Query APIs

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
