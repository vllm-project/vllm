# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time

# to ensure deterministic behaviour
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# standard imports
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.config import KVTransferConfig


BLOCK_SIZE=64
CPU_CACHE_BYTES=8000000000
# helper functions
def pad(toklist, padtok):
    return toklist[:-1] + [padtok] * ((BLOCK_SIZE - len(toklist)) % BLOCK_SIZE) + toklist[-1:]


def avg(list_of_numbers):
    return sum(list_of_numbers) / max(len(list_of_numbers), 1)


def wrap(prompt):
    if isinstance(prompt[0], list):
        return [TokensPrompt(prompt_token_ids=p) for p in prompt]
    return TokensPrompt(prompt_token_ids=prompt)


def load_and_segment_text(filename, num_segments=4):
    """
    Load a text file and segment it into multiple documents.
    
    Args:
        filename: Path to the text file
        num_segments: Number of segments to create (default: 4)
    
    Returns:
        List of segmented documents
    """
    # Read the text file
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split text into roughly equal segments
    # Remove extra whitespace and split into sentences/chunks
    text = ' '.join(text.split())  # Normalize whitespace

    # Formatting
    text = text.replace('"', '\\"')

    # Calculate approximate segment length
    segment_length = len(text) // num_segments
    
    segments = []
    start = 0
    
    for i in range(num_segments):
        if i == num_segments - 1:
            # Last segment gets remaining text
            segment = text[start:]
        else:
            # Find a good break point (space) near the segment boundary
            end = start + segment_length
            # Look for the next space to avoid breaking words
            while end < len(text) and text[end] != ' ':
                end += 1
            segment = text[start:end]
            start = end + 1  # Skip the space
        
        segments.append(segment.strip())
    
    return segments

def initialize_vllm(
    model, temp=0.6, logprobs=None, max_toks=32768, max_generated_toks=1
):
    # boot up vLLM
    samp_params_preload = SamplingParams(temperature=temp, max_tokens=1)
    samp_params_generate = SamplingParams(
        temperature=temp, max_tokens=max_generated_toks, logprobs=logprobs
    )
    ktc_example = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            # kv_connector_extra_config={
            #     "shared_storage_path": "local_storage",
            # },
        )
    ktc_offload = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": "local_storage",
                "cpu_bytes_to_use": CPU_CACHE_BYTES
            },
        )
    ktc_segment_simple = KVTransferConfig(
            kv_connector="SegmentedPrefillExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": "local_storage",
            },
            kv_connector_module_path="segmented_prefill_example_connector_2",
        )
    ktc_segment_offload = KVTransferConfig(
            kv_connector="SegmentedPrefillOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": "local_storage",
                "cpu_bytes_to_use": CPU_CACHE_BYTES
            },
            kv_connector_module_path="segmented_prefill_example_connector",
        )
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.9,
        kv_transfer_config=ktc_segment_offload,
        enforce_eager=True,  # <- so it boots faster
        block_size=BLOCK_SIZE,
        attention_backend="TRITON_ATTN",
        enable_prefix_caching=False,
        
    )
    tok = llm.get_tokenizer()
    tok_fun = lambda x: tok.convert_tokens_to_ids(tok.tokenize(x))
    return samp_params_preload, samp_params_generate, tok_fun, llm


def main():
    model_names = [
        "ldsjmdy/Tulu3-Block-FT",  # <- finetuned to handle block-attention
        "ldsjmdy/Tulu3-RAG",  #      <- baseline
        "ibm-granite/granite-3.1-8b-instruct",  #      <- Qwenchuk
    ]
    model_name = model_names[2]

    # tokens that need to be set to perform block-attention
    PAD_TOK = 27  # <-  "<"
    SPAN_TOK_PLUS = 10  # <- "+"
    SPAN_TOK_CROSS = 31  # <- "@"

    # vLLM-specific env vars

    # enables block attention
    # -> when this line is not commented, we expect a speedup
    #    in the execution of the last two .generate calls
    os.environ["VLLM_V1_SPANS_ENABLED"] = "True"

    # the token that tells vLLM "this is the beginning of a span"
    os.environ["VLLM_V1_SPANS_TOKEN_PLUS"] = str(SPAN_TOK_PLUS)

    # token that tells vLLM:
    # "from here on, recompute KV vectors if any previous tokens differ"
    os.environ["VLLM_V1_SPANS_TOKEN_CROSS"] = str(SPAN_TOK_CROSS)

    # will print every step of the span process if set to true
    os.environ["VLLM_V1_SPANS_DEBUG"] = "True"

    # will disable the adjustment of positional encodings when a KV cache
    # block is loaded to a different position than it was stored
    # -> when this line is not commented,
    #    spans overlap in their positional encodings
    os.environ["VLLM_V1_SPANS_DISABLE_REPOSITION"] = "True"

    # general env vars

    # now we instantiate the model
    samp_params_preload, samp_params_generate, tok, llm = initialize_vllm(
        model_name, max_generated_toks=128, max_toks=10_000, temp=0.0
    )

    # components of the prompt template
    prefix = pad(
        tok(
            "<|system|>\nYou are an intelligent AI assistant. "
            "Please answer questions based on the user's instructions. "
            "Below are some reference documents that may help you in "
            "answering the user's question."
        ),
        PAD_TOK,
    )
    midfx = [SPAN_TOK_CROSS] + tok(
        "<|user|>\nPlease write a high-quality answer for the "
        "given question using only the provided search documents "
        "(some of which might be irrelevant).\nQuestion: "
    )
    postfx = tok("""\n<|assistant|>\n""")

    print("---->", postfx)

    # task-specific documents
    doc_a = pad(
        [SPAN_TOK_PLUS]
        + tok(
            "[0] The Template-Assisted "
            "Selective Epitaxy (TASE) method, developed at "
            "IBM Research Europe – Zurich, permits to "
            "create a homogeneous integration route for "
            "various semiconductor materials which is "
            "compatible with the CMOS process."
        ),
        PAD_TOK,
    )

    doc_b = pad(
        [SPAN_TOK_PLUS]
        + tok(
            "[1] The dominant sequence transduction "
            "models are based on complex recurrent or "
            "convolutional neural networks in an encoder-decoder "
            "configuration. "
        ),
        PAD_TOK,
    )
    
    

    # # alt-docs (purely to check performance on longer documents)
    """
    a_toks = tok("Sequence Transduction Models")
    b_toks = tok("Template-Assisted Selective Epitaxy")
    doc_a = pad(
        [SPAN_TOK_PLUS]
        + [a_toks[idx % len(a_toks)] for idx in range(10_000)],
        PAD_TOK,
    )
    doc_b = pad(
        [SPAN_TOK_PLUS]
        + [b_toks[idx % len(a_toks)] for idx in range(10_000)],
        PAD_TOK,
    )
    """

    # user query
    query = (
        midfx
        + tok(
            "Tell me which one concerns deep learning. "
            "Indicate your answer with a number in brackets."
        )
        + postfx
    )
    query_2 = (
        midfx
        + tok(
            "What is Northholm's value to Luminthia?"
        )
        + postfx
    )

    segments = load_and_segment_text("example_text.txt", num_segments=4)
    # for i, seg in enumerate(segments):
    #     print(f"Segment nubmer {i}:")
    #     print(seg)
        
    # Create the four documents using the same pattern as the example
    doc_w = pad(
        [SPAN_TOK_PLUS] + tok(segments[0]),
        PAD_TOK,
    )
    
    doc_x = pad(
        [SPAN_TOK_PLUS] + tok(segments[1]),
        PAD_TOK,
    )
    
    doc_y = pad(
        [SPAN_TOK_PLUS] + tok(segments[2]),
        PAD_TOK,
    )
    
    doc_z = pad(
        [SPAN_TOK_PLUS] + tok(segments[3]),
        PAD_TOK,
    )
    
    print(f"doc_w length: {len(doc_w)}")
    print(f"doc_x length: {len(doc_x)}")
    print(f"doc_y length: {len(doc_y)}")
    print(f"doc_z length: {len(doc_z)}")

    # preload documents
    ts_pre = time.time()
    llm.generate(
        wrap(doc_a), sampling_params=samp_params_preload
    )
    llm.generate(
        wrap(doc_b), sampling_params=samp_params_preload
    )
    
    for doc in [doc_w, doc_x, doc_y, doc_z]:
        llm.generate(
        wrap(doc), sampling_params=samp_params_preload
    )
        
    llm.generate(
        wrap(prefix), sampling_params=samp_params_preload
    )
    te_pre = time.time() - ts_pre
    time.sleep(2)
    ts_gen = time.time()
    
    # this now will load prefix, doc_a, doc_b,
    # from the KV cache regardless of the order
    # print("=============Generate 1=================")
    # model_response_1 = llm.generate(
    #     wrap(prefix + doc_a + doc_b + query),
    #     sampling_params=samp_params_generate,
    #     use_tqdm=False,
    # )
    # print("=============Generate 2=================")

    # # this should also run faster:
    # model_response_2 = llm.generate(
    #     wrap(prefix + doc_b + doc_a + query),
    #     sampling_params=samp_params_generate,
    #     use_tqdm=False,
    # )
    
    model_response_3 = llm.generate(
        wrap(prefix + doc_w + doc_x + doc_y + doc_z + query_2),
        sampling_params=samp_params_generate,
        use_tqdm=False,
    )
    
    model_response_4 = llm.generate(
        wrap(prefix + doc_z + doc_x + doc_w + doc_y + query_2),
        sampling_params=samp_params_generate,
        use_tqdm=False,
    )

    te_gen = time.time() - ts_gen

    print(f"doc preload time / TTFT : {te_pre:.4f} / {te_gen:.4f} (s)")
    print("model output 1 was:", model_response_3[0].outputs[0].text)
    print("model output 2 was:", model_response_4[0].outputs[0].text)


if __name__ == "__main__":
    main()