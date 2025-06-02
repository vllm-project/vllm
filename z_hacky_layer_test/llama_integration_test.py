from typing import Optional
from vllm import LLM
import argparse
import torch
import os

from vllm.sampling_params import SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Extract first LlamaAttention layer")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", 
                        help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", 
                        help="Whether to trust remote code")
    parser.add_argument("--backend", type=str, 
                        choices=["FLASH_ATTN_VLLM_V1", "FLASHINFER_VLLM_V1", "BOK_V1"],
                        default="FLASHINFER_VLLM_V1",
                        help="vLLM attention backend to use")
    args = parser.parse_args()
    
    # Set the environment variable for the backend
    os.environ["VLLM_ATTENTION_BACKEND"] = args.backend
    
    # Create the output directory name based on the backend
    output_dir = f"{args.backend}_attn_captures"
    
    # Instantiate a normal vLLM engine
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
    )
    
    prompts = ["Hello, how are you?"]
    
    def attach_hook(self):
        print("self.model_runner.model.model.__class__", self.model_runner.model.model.__class__)
        print("self.model_runner.model.model.layers", self.model_runner.model.model.layers)
        layer = self.model_runner.model.model.layers[0]
        print("layer:", layer)
        print("layer.self_attn:", layer.self_attn)
        print("layer.self_attn.attn:", layer.self_attn.attn)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Counter to keep track of forward passes
        hook_counter = [0]
        
        def pre_forward_hook(module, args):
            query, key, value = args[0], args[1], args[2]
            
            # Get the forward context and kv_cache
            from vllm.forward_context import get_forward_context
            forward_context = get_forward_context()
            kv_cache = module.kv_cache[forward_context.virtual_engine]
            
            # Increment counter
            hook_counter[0] += 1
            count = hook_counter[0]
            print(f"Saving tensors from forward pass {count}", flush=True)
            
            # Create subdirectory for this forward pass
            pass_dir = f"{output_dir}/pass_{count}"
            os.makedirs(pass_dir, exist_ok=True)
            
            # Save tensors to files using torch.save
            if query is not None:
                torch.save(query.detach().cpu(), f"{pass_dir}/query.pt")
            
            if key is not None:
                torch.save(key.detach().cpu(), f"{pass_dir}/key.pt")
            else:
                torch.save(None, f"{pass_dir}/key.pt")
                
            if value is not None:
                torch.save(value.detach().cpu(), f"{pass_dir}/value.pt")
            else:
                torch.save(None, f"{pass_dir}/value.pt")
            
            # Save the KV cache
            if isinstance(kv_cache, list):
                # If it's a list, save each tensor in the list
                torch.save(len(kv_cache), f"{pass_dir}/kv_cache_list_len.pt")
                kv_cache_cpu = []
                for tensor in kv_cache:
                    if hasattr(tensor, 'detach'):
                        kv_cache_cpu.append(tensor.detach().cpu())
                    else:
                        kv_cache_cpu.append(tensor)
                torch.save(kv_cache_cpu, f"{pass_dir}/kv_cache.pt")
            else:
                # If it's a tensor or something else
                if hasattr(kv_cache, 'detach'):
                    torch.save(kv_cache.detach().cpu(), f"{pass_dir}/kv_cache.pt")
                else:
                    torch.save(kv_cache, f"{pass_dir}/kv_cache.pt")
            
            # Also save the forward context's virtual engine
            torch.save(forward_context.virtual_engine, f"{pass_dir}/virtual_engine.pt")
            
            return args
        
        # Register the pre-forward hook
        attn_layer = layer.self_attn.attn
        hook = attn_layer.register_forward_pre_hook(pre_forward_hook)
        
        print(f"Attached pre-forward hook to {attn_layer}", flush=True)
        # return attn_layer

    llm.collective_rpc(attach_hook)
    
    # Generate output with the model to trigger the hook
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1,
    )
    output = llm.generate(prompts, sampling_params)
    print("Output:", output, flush=True)
    
    # print("ret:", ret)


if __name__ == "__main__":
    main()
