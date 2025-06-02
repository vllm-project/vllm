from typing import Optional
from vllm import LLM
import argparse
import torch

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
        enforce_eager=True,
    )
    
    prompts = ["Hello, how are you?"]
    
    def check_model(self):
            print("self.model_runner.model.model.__class__", self.model_runner.model.model.__class__)
            print("self.model_runner.model.model.layers", self.model_runner.model.model.layers)
            layer = self.model_runner.model.model.layers[0]
            print("layer:", layer)
            print("layer.self_attn:", layer.self_attn)
            print("layer.self_attn.attn:", layer.self_attn.attn)
            return layer.self_attn.attn

    ret = llm.collective_rpc(check_model)

    print("ret:", ret)


if __name__ == "__main__":
    main()
