from vllm import LLM
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract first LlamaAttention layer")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", 
                        help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", 
                        help="Whether to trust remote code")
    args = parser.parse_args()
    
    # Instantiate a normal vLLM engine
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
    )
    
    prompts = ["Hello, how are you?"]
    # Get the first LlamaAttention layer
    first_attention_layer = llm.llm_engine.model.model.layers[0].self_attn
    
    # Print some info about it
    print(f"Successfully retrieved first LlamaAttention layer")
    print(f"Number of heads: {first_attention_layer.num_heads}")
    print(f"Number of KV heads: {first_attention_layer.num_kv_heads}")
    print(f"Head dimension: {first_attention_layer.head_dim}")
    
    return first_attention_layer

if __name__ == "__main__":
    main()
