# Prompt Embedding Inputs

This page teaches you how to pass prompt embedding inputs to vLLM.

## What are prompt embeddings?

The traditional flow of text data for a Large Language Model goes from text to token ids (via a tokenizer) then from token ids to prompt embeddings. For a traditional decoder-only model (such as meta-llama/Llama-3.1-8B-Instruct), this step of converting token ids to prompt embeddings happens via a look-up from a learned embedding matrix, but the model is not limited to processing only the embeddings corresponding to its token vocabulary.

:::{note}
Prompt embeddings are currently only supported in the v0 engine.
:::

## Offline Inference

To input multi-modal data, follow this schema in {class}`vllm.inputs.EmbedsPrompt`:

- `prompt_embeds`: A torch tensor representing a sequence of prompt/token embeddings. This has the shape (sequence_length, hidden_size), where sequence length is the number of tokens embeddings and hidden_size is the hidden size (embedding size) of the model.

### Hugging Face Transformers Inputs

You can pass prompt embeddings from Hugging Face Transformers models to the  `'prompt_embeds'` field of the prompt embedding dictionary, as shown in the following examples:

```python
from vllm import LLM
import transformers

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Transformers
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
transformers_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

llm = LLM(model=model_name, enable_prompt_embeds=True)

# Refer to the HuggingFace repo for the correct format to use
chat = [{"role": "user", "content": "Please tell me about the capital of France."}]
token_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt')

embedding_layer = transformers_model.get_input_embeddings()
prompt_embeds = embedding_layer(token_ids).squeeze(0)

# Single prompt inference
outputs = llm.generate({
    "prompt_embeds": prompt_embeds,
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

# Batch inference

chats = [
    [{"role": "user", "content": "Please tell me about the capital of France."}],
    [{"role": "user", "content": "When is the day longest during the year?"}],
    [{"role": "user", "content": "Where is bigger, the moon or the sun?"}]
]

token_ids_list = [
    tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt') for chat in chats
]
prompt_embeds_list = [embedding_layer(token_ids).squeeze(0) for token_ids in token_ids_list]

outputs = llm.generate(
    [
        {
            "prompt_embeds": prompt_embeds,
        } for prompt_embeds in prompt_embeds_list
    ]
)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
```

## Online Serving

Our OpenAI-compatible server accepts prompt embeddings inputs via the [Completions API](https://platform.openai.com/docs/api-reference/completions). Prompt embeddings inputs are added via a new `'prompt_embeds'` key in the JSON package.

When a mixture of `'prompt_embeds'` and `'prompt'` inputs are provided in a single request, the prompt embeds are always returned first.

Prompt embeddings are passed in as base64 encoded torch tensors.

### Transformers Inputs via OpenAI Client

First, launch the OpenAI-compatible server:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --task generate \
  --max-model-len 4096 --enable-prompt-embeds
```

Then, you can use the OpenAI client as follows:

```python
from openai import OpenAI
import transformers
import torch

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Transformers
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
transformers_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)


# Refer to the HuggingFace repo for the correct format to use
chat = [{"role": "user", "content": "Please tell me about the capital of France."}]
token_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors='pt')

embedding_layer = transformers_model.get_input_embeddings()
prompt_embeds = embedding_layer(token_ids).squeeze(0)

# Prompt embeddings
buffer = io.BytesIO()
torch.save(prompt_embeds, buffer)
buffer.seek(0)
binary_data = buffer.read()
encoded_embeds = base64.b64encode(binary_data).decode('utf-8')


completion = client_with_prompt_embeds.completions.create(
    model=model_name,
    # NOTE: The OpenAI client does not allow `None` as an input to 
    # `prompt`. Use an empty string if you have no text prompts.
    prompt="",  
    max_tokens=5,
    temperature=0.0,
    # NOTE: The OpenAI client allows passing in extra JSON body via the
    # `extra_body` argument.
    extra_body={"prompt_embeds": encoded_embeds}
)

print(completion.choices[0].text)
```
