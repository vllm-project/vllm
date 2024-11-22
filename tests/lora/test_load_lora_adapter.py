from vllm import LLM
from vllm.lora.request import LoRARequest
import os

def extract_layer_names(llm):
    engine = getattr(llm, "llm_engine")
    model_executor = getattr(engine, "model_executor")
    driver_worker = getattr(model_executor, "driver_worker")
    model_runner = getattr(driver_worker, "model_runner")
    list_adapters = list(model_runner.model.lora_manager.list_adapters().values())
    list_layers = []
    for adapter in list_adapters:
        loras = adapter.loras
        adapter_layers = []
        for k in loras:
            adapter_layers.append(loras[k].module_name)
        list_layers.append(adapter_layers)
    return list_layers

def load_base_model(base_model_path):
    print(f"Loading base model from {base_model_path}...")
    llm = LLM(model=base_model_path, enable_lora=True)
    print("Base model loaded.")
    return llm

def load_lora_adapter(llm, lora_path):
    print(f"Loading LoRA adapter from {lora_path}...")
    lora_request = LoRARequest("lora_adapter", 1, lora_path)
    print("LoRA adapter loaded.")
    return llm, lora_request

def send_request(llm, lora_request):
    print("Sending a dummy request.")
    prompt = "Hi!"
    output = llm.generate(prompt, lora_request=lora_request)
    print("The request is sent.")
    return llm

def compare_layers(base_layers, lora_layers):
    print("Comparing layers...")
    print(f"There are {len(base_layers)} LoRA layers in the base model.")
    print(f"There are {len(lora_layers)} LoRA layers in the LoRA adapter.")

    base_set = set(name for adapter in base_layers for name in adapter)
    lora_set = set(name for adapter in lora_layers for name in adapter)

    added_layers = lora_set - base_set
    removed_layers = base_set - lora_set

    if added_layers or removed_layers:
        print("Layer differences detected:")
        if added_layers:
            print(f"  Added {len(added_layers)} LoRA layers.")
        if removed_layers:
            print(f"  Removed {len(removed_layers)} LoRA layers.")
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

    model_with_lora, lora_request = load_lora_adapter(base_model, lora_adapter_path)
    lora_layers_before_request = extract_layer_names(model_with_lora)

    model_with_lora_after_request = send_request(model_with_lora, lora_request)
    lora_layers_after_request = extract_layer_names(model_with_lora_after_request)

    print("Compare the base model and the model with a loaded LoRA adapter...")
    compare_layers(base_layers, lora_layers_before_request)

    print("Compare the model with a loaded LoRA adapter before and after sending a request...")
    compare_layers(lora_layers_before_request, lora_layers_after_request)

if __name__ == "__main__":
    main()

