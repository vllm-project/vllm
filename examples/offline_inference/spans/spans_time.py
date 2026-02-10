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
    ]
    model_name = model_names[0]

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
            "Tell me what is the name of the robot."
        )
        + postfx
    )

    segments = load_and_segment_text("example_text.txt", num_segments=4)
    
    # Check if example_text_2.txt exists
    use_text_2 = os.path.exists("example_text_2.txt")
    if use_text_2:
        segments_2 = load_and_segment_text("example_text_2.txt", num_segments=4)
        print("Found example_text_2.txt - will use 8 documents")
    else:
        print("example_text_2.txt not found - will use 4 documents only")
    
    # for i, seg in enumerate(segments):
    #     print(f"Segment nubmer {i}:")
    #     print(seg)
        
    # Create the four documents from example_text.txt
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
    
    # Create the four documents from example_text_2.txt if it exists
    if use_text_2:
        doc_p = pad(
            [SPAN_TOK_PLUS] + tok(segments_2[0]),
            PAD_TOK,
        )
        
        doc_q = pad(
            [SPAN_TOK_PLUS] + tok(segments_2[1]),
            PAD_TOK,
        )
        
        doc_r = pad(
            [SPAN_TOK_PLUS] + tok(segments_2[2]),
            PAD_TOK,
        )
        
        doc_s = pad(
            [SPAN_TOK_PLUS] + tok(segments_2[3]),
            PAD_TOK,
        )
    
    print(f"doc_w length: {len(doc_w)}")
    print(f"doc_x length: {len(doc_x)}")
    print(f"doc_y length: {len(doc_y)}")
    print(f"doc_z length: {len(doc_z)}")
    if use_text_2:
        print(f"doc_p length: {len(doc_p)}")
        print(f"doc_q length: {len(doc_q)}")
        print(f"doc_r length: {len(doc_r)}")
        print(f"doc_s length: {len(doc_s)}")

    # preload documents
    ts_pre = time.time()
    llm.generate(
        wrap(doc_a), sampling_params=samp_params_preload
    )
    llm.generate(
        wrap(doc_b), sampling_params=samp_params_preload
    )
    
    # Preload documents based on availability
    docs_to_preload = [doc_w, doc_x, doc_y, doc_z]
    if use_text_2:
        docs_to_preload.extend([doc_p, doc_q, doc_r, doc_s])
    
    for doc in docs_to_preload:
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
    
    # Measure time for model_response_3 with SegmentedPrefillOffloadConnector
    print("\n" + "="*80)
    if use_text_2:
        print("TEST 1: SegmentedPrefillOffloadConnector with all 8 docs")
    else:
        print("TEST 1: SegmentedPrefillOffloadConnector with 4 docs")
    print("="*80)
    ts_model3 = time.time()
    
    # Build prompt based on available documents
    prompt_docs = doc_w + doc_x + doc_y + doc_z
    if use_text_2:
        prompt_docs = prompt_docs + doc_p + doc_q + doc_r + doc_s
    
    model_response_3 = llm.generate(
        wrap(prefix + prompt_docs + query_2),
        sampling_params=samp_params_generate,
        use_tqdm=False,
    )
    te_model3 = time.time() - ts_model3

    te_gen = time.time() - ts_gen

    print(f"doc preload time / TTFT : {te_pre:.4f} / {te_gen:.4f} (s)")
    print(f"model_response_3 generation time: {te_model3:.4f} (s)")
    print("model output 1 was:", model_response_3[0].outputs[0].text)
    
    # Destroy the vllm instance
    print("\n" + "="*80)
    print("Destroying vLLM instance...")
    print("="*80)
    del llm
    time.sleep(3)  # Give time for cleanup
    
    # Instantiate new vllm with OffloadingConnector
    print("\n" + "="*80)
    print("Creating new vLLM instance with OffloadingConnector...")
    print("="*80)
    ktc_offload = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": "local_storage",
            "cpu_bytes_to_use": CPU_CACHE_BYTES
        },
    )
    llm_offload = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        kv_transfer_config=ktc_offload,
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        attention_backend="TRITON_ATTN",
        enable_prefix_caching=False,
    )
    
    # Permute documents to different ordering: reverse order for both sets
    print("\n" + "="*80)
    if use_text_2:
        print("TEST 2: OffloadingConnector with permuted docs (doc_z+y+x+w + doc_s+r+q+p)")
    else:
        print("TEST 2: OffloadingConnector with permuted docs (doc_z+y+x+w)")
    print("="*80)
    
    # Preload documents in new order
    ts_pre2 = time.time()
    llm_offload.generate(
        wrap(doc_a), sampling_params=samp_params_preload
    )
    llm_offload.generate(
        wrap(doc_b), sampling_params=samp_params_preload
    )
    
    # Preload in permuted order (reverse order for both document sets)
    docs_to_preload_2 = [doc_z, doc_y, doc_x, doc_w]
    if use_text_2:
        docs_to_preload_2.extend([doc_s, doc_r, doc_q, doc_p])
    
    for doc in docs_to_preload_2:
        llm_offload.generate(
            wrap(doc), sampling_params=samp_params_preload
        )
        
    llm_offload.generate(
        wrap(prefix), sampling_params=samp_params_preload
    )
    te_pre2 = time.time() - ts_pre2
    time.sleep(2)
    
    # Generate with permuted document order
    ts_model4 = time.time()
    
    # Build prompt based on available documents (in reverse order)
    prompt_docs_2 = doc_z + doc_y + doc_x + doc_w
    if use_text_2:
        prompt_docs_2 = prompt_docs_2 + doc_s + doc_r + doc_q + doc_p
    
    model_response_4 = llm_offload.generate(
        wrap(prefix + prompt_docs_2 + query_2),
        sampling_params=samp_params_generate,
        use_tqdm=False,
    )
    te_model4 = time.time() - ts_model4
    
    print(f"doc preload time (permuted): {te_pre2:.4f} (s)")
    print(f"model_response_4 generation time: {te_model4:.4f} (s)")
    print("model output 2 was:", model_response_4[0].outputs[0].text)
    
    # Destroy the second vllm instance
    print("\n" + "="*80)
    print("Destroying second vLLM instance...")
    print("="*80)
    del llm_offload
    time.sleep(3)  # Give time for cleanup
    
    # TEST 3: Disable spans and run without KVTransferConfig
    print("\n" + "="*80)
    print("TEST 3: Baseline test with SPANS DISABLED (no KVTransferConfig, no preload)")
    print("="*80)
    
    # Disable spans
    os.environ["VLLM_V1_SPANS_ENABLED"] = "False"
    
    # Create new LLM instance without KVTransferConfig
    llm_baseline = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        attention_backend="TRITON_ATTN",
        enable_prefix_caching=False,
    )
    
    # Run generation directly without any preload step
    print("Running generation without preload...")
    ts_model5 = time.time()
    
    # Use same document order as TEST 1 for fair comparison
    model_response_5 = llm_baseline.generate(
        wrap(prefix + prompt_docs + query_2),
        sampling_params=samp_params_generate,
        use_tqdm=False,
    )
    te_model5 = time.time() - ts_model5
    
    print(f"model_response_5 generation time (no preload, spans disabled): {te_model5:.4f} (s)")
    print("model output 3 was:", model_response_5[0].outputs[0].text)
    
    # Cleanup
    del llm_baseline
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"Test 1 (SegmentedPrefillOffloadConnector, with preload):")
    print(f"  - Preload time: {te_pre:.4f} s")
    print(f"  - Generation time: {te_model3:.4f} s")
    print(f"  - Total time: {te_pre + te_model3:.4f} s")
    print(f"\nTest 2 (OffloadingConnector, with preload, permuted):")
    print(f"  - Preload time: {te_pre2:.4f} s")
    print(f"  - Generation time: {te_model4:.4f} s")
    print(f"  - Total time: {te_pre2 + te_model4:.4f} s")
    print(f"\nTest 3 (Baseline, no preload, spans disabled):")
    print(f"  - Preload time: 0.0000 s")
    print(f"  - Generation time: {te_model5:.4f} s")
    print(f"  - Total time: {te_model5:.4f} s")
    print(f"\nComparison:")
    print(f"  - Test 1 vs Test 3: {te_model3 - te_model5:.4f} s ({((te_model3/te_model5 - 1) * 100):.2f}% change)")
    print(f"  - Test 2 vs Test 3: {te_model4 - te_model5:.4f} s ({((te_model4/te_model5 - 1) * 100):.2f}% change)")
    print("="*80)


if __name__ == "__main__":
    main()