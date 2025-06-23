---
title: GGUF
---
[](){ #gguf }

!!! warning
    Please note that GGUF support in vLLM is highly experimental and under-optimized at the moment, it might be incompatible with other features. Currently, you can use GGUF as a way to reduce memory footprint. If you encounter any issues, please report them to the vLLM team.

!!! warning
    Currently, vllm only supports loading single-file GGUF models. If you have a multi-files GGUF model, you can use [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) tool to merge them to a single-file model.

To run a GGUF model with vLLM, you can download and use the local GGUF model from [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) with the following command:

```bash
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

You can also add `--tensor-parallel-size 2` to enable tensor parallelism inference with 2 GPUs:

```bash
# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   --tensor-parallel-size 2
```

!!! warning
    We recommend using the tokenizer from base model instead of GGUF model. Because the tokenizer conversion from GGUF is time-consuming and unstable, especially for some models with large vocab size.

GGUF assumes that huggingface can convert the metadata to a config file. In case huggingface doesn't support your model you can manually create a config and pass it as hf-config-path

```bash
# If you model is not supported by huggingface you can manually provide a huggingface compatible config path
vllm serve ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
   --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   --hf-config-path Tinyllama/TInyLlama-1.1B-Chat-v1.0
```

You can also use the GGUF model directly through the LLM entrypoint:

??? Code

      ```python
      from vllm import LLM, SamplingParams

      # In this script, we demonstrate how to pass input to the chat method:
      conversation = [
         {
            "role": "system",
            "content": "You are a helpful assistant"
         },
         {
            "role": "user",
            "content": "Hello"
         },
         {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
         },
         {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
         },
      ]

      # Create a sampling params object.
      sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

      # Create an LLM.
      llm = LLM(model="./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
               tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
      # Generate texts from the prompts. The output is a list of RequestOutput objects
      # that contain the prompt, generated text, and other information.
      outputs = llm.chat(conversation, sampling_params)

      # Print the outputs.
      for output in outputs:
         prompt = output.prompt
         generated_text = output.outputs[0].text
         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
      ```
