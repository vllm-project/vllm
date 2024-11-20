from vllm import LLM
from vllm.lora.request import LoRARequest
import os

def extract_layer_names(llm):
    engine = getattr(llm, "llm_engine")
    model_executor = getattr(engine, "model_executor")
    driver_worker = getattr(model_executor, "driver_worker")
    model_runner = getattr(driver_worker, "model_runner")
    return [name for name, _ in model_runner.model.named_modules()]

def load_base_model(base_model_path):
    print(f"Loading base model from {base_model_path}...")
    llm = LLM(model=base_model_path, enable_lora=True)
    print("Base model loaded.")
    return llm

def load_lora_adapter(llm, lora_path):
    print(f"Loading LoRA adapter from {lora_path}...")
    lora_request = LoRARequest("lora_adapter", 1, lora_path)
    print("LoRA adapter loaded.")
    print("Sending a dummy request.")
    prompt = "Hi!"
    output = llm.generate(prompt, lora_request=lora_request)
    print("The request is sent.")
    return llm

def compare_layers(base_layers, lora_layers):
    print("Comparing layers...")
    base_set = set(base_layers)
    lora_set = set(lora_layers)

    added_layers = lora_set - base_set
    removed_layers = base_set - lora_set

    #print("Base model layers:")
    #for layer in base_set:
    #    print(f"    {layer}")

    if added_layers or removed_layers:
        print("Layer differences detected:")
        if added_layers:
            print("  Layers added by LoRA:")
            for layer in added_layers:
                print(f"    {layer}")
        if removed_layers:
            print("  Layers removed after LoRA:")
            for layer in removed_layers:
                print(f"    {layer}")
        return True
    else:
        print("No differences in layers detected.")
        return False

def main():
    base_model_path = "/data/llama-3/llama-3-8b"
    lora_adapter_path = "/home/oleg/lora_test/Meta-Llama-3-8B-oasst-Adapter"

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    if not os.path.exists(lora_adapter_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {lora_adapter_path}")

    base_model = load_base_model(base_model_path)
    base_layers = extract_layer_names(base_model)

    model_with_lora = load_lora_adapter(base_model, lora_adapter_path)
    lora_layers = extract_layer_names(model_with_lora)

    compare_layers(base_layers, lora_layers)

if __name__ == "__main__":
    main()

