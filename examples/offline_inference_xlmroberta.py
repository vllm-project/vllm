from typing import List, Tuple, Union

from transformers import AutoTokenizer

from vllm import LLM

model = "BAAI/bge-reranker-base"
llm = LLM(model=model, tensor_parallel_size=1)

prompt = "this is a useless prompt."
sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]] = \
        [("hello world", "nice to meet you"), ("head north", "head south")]
tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=None)

inputs = tokenizer(
    sentence_pairs,
    padding=True,
    truncation=True,
    return_tensors='pt',
    max_length=512,
).to("cuda")
outputs = llm.process([{
    "prompt": prompt,
    "multi_modal_data": {
        "xlmroberta": inputs,
    }
}],
                      use_tqdm=False)

for output in outputs:
    print(output.outputs.result)
