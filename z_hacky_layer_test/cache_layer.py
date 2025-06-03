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
                        choices=["FLASH_ATTN", "FLASH_ATTN_VLLM_V1", "FLASHINFER", "BOK", "GARBAGE", "GARBAGE2"],
                        default="FLASHINFER",
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
        # print("self.model_runner.model.model.__class__", self.model_runner.model.model.__class__)
        # print("self.model_runner.model.model.layers", self.model_runner.model.model.layers)
        layer = self.model_runner.model.model.layers[0]
        # print("layer:", layer)
        # print("layer.self_attn:", layer.self_attn)
        # print("layer.self_attn.attn:", layer.self_attn.attn)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Counter to keep track of forward passes
        hook_counter = [0]
        
        def pre_forward_hook(module, args):
            hook_counter[0] += 1
            current_pass_count = hook_counter[0]
            print(f"Pre-forward hook for pass {current_pass_count}, saving pre-execution tensors.", flush=True)
            
            query, key, value = args[0], args[1], args[2]
            
            # Get the forward context and kv_cache
            from vllm.forward_context import get_forward_context
            forward_context = get_forward_context()
            kv_cache_pre = module.kv_cache[forward_context.virtual_engine]
            
            # Create subdirectory for this forward pass
            pass_dir = f"{output_dir}/pass_{current_pass_count}"
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
            
            # Save the KV cache *before* forward
            if isinstance(kv_cache_pre, (list, tuple)):
                torch.save(len(kv_cache_pre), f"{pass_dir}/kv_cache_pre_list_len.pt")
                kv_cache_pre_cpu = []
                for tensor_item in kv_cache_pre:
                    if hasattr(tensor_item, 'detach'):
                        kv_cache_pre_cpu.append(tensor_item.detach().cpu())
                    else:
                        kv_cache_pre_cpu.append(tensor_item)
                torch.save(kv_cache_pre_cpu, f"{pass_dir}/kv_cache_pre.pt")
            else:
                if hasattr(kv_cache_pre, 'detach'):
                    torch.save(kv_cache_pre.detach().cpu(), f"{pass_dir}/kv_cache_pre.pt")
                else:
                    torch.save(kv_cache_pre, f"{pass_dir}/kv_cache_pre.pt")
            
            # Also save the forward context's virtual engine
            torch.save(forward_context.virtual_engine, f"{pass_dir}/virtual_engine.pt")
            
            return args

        def post_forward_hook(module, args_input, output_tensor):
            # hook_counter[0] was already incremented by the pre_forward_hook for this pass
            current_pass_count = hook_counter[0]
            print(f"Post-forward hook for pass {current_pass_count}, saving post-execution tensors.", flush=True)

            pass_dir = f"{output_dir}/pass_{current_pass_count}"
            # Directory should have been created by pre_forward_hook

            # Save the output tensor
            if output_tensor is not None:
                # Assuming output_tensor is a single tensor as per Attention layer's forward method
                if hasattr(output_tensor, 'detach'):
                    torch.save(output_tensor.detach().cpu(), f"{pass_dir}/output.pt")
                else:
                    torch.save(output_tensor, f"{pass_dir}/output.pt")
            else:
                torch.save(None, f"{pass_dir}/output.pt")

            # Save the KV cache *after* forward pass
            from vllm.forward_context import get_forward_context
            forward_context = get_forward_context() 
            kv_cache_post = module.kv_cache[forward_context.virtual_engine]

            if isinstance(kv_cache_post, (list, tuple)):
                torch.save(len(kv_cache_post), f"{pass_dir}/kv_cache_post_list_len.pt")
                kv_cache_post_cpu = []
                for tensor_item in kv_cache_post:
                    if hasattr(tensor_item, 'detach'):
                        kv_cache_post_cpu.append(tensor_item.detach().cpu())
                    else:
                        kv_cache_post_cpu.append(tensor_item)
                torch.save(kv_cache_post_cpu, f"{pass_dir}/kv_cache_post.pt")
            else:
                if hasattr(kv_cache_post, 'detach'):
                    torch.save(kv_cache_post.detach().cpu(), f"{pass_dir}/kv_cache_post.pt")
                else:
                    torch.save(kv_cache_post, f"{pass_dir}/kv_cache_post.pt")
        
        # Register the pre-forward and post-forward hooks
        attn_layer : torch.nn.Module = layer.self_attn.attn
        pre_hook = attn_layer.register_forward_pre_hook(pre_forward_hook)
        post_hook = attn_layer.register_forward_hook(post_forward_hook)
        
        print(f"Attached pre-forward and post-forward hooks to {attn_layer}", flush=True)
        # To store hooks if you plan to remove them later:
        # if not hasattr(self, 'registered_hooks'):
        #     self.registered_hooks = []
        # self.registered_hooks.extend([pre_hook, post_hook])

    llm.collective_rpc(attach_hook)
    
    # Generate output with the model to trigger the hook
    sampling_params = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        top_k=1,
    )
    output = llm.generate(prompts, sampling_params)
    print("Output:", output, flush=True)
    
    # print("ret:", ret)


if __name__ == "__main__":
    main()
