import torch
import torch.compile
from torch.compile import Mode
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_weight_loading():
    # Load model and tokenizer
    model_name = "vllm-base"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a baseline configuration
    baseline_config = torch.compile(config=Mode.default)

    # Create a fusion configuration
    fusion_config = torch.compile(config=Mode.fusion)

    # Run a few layers of the model and compare the outputs
    input_ids = torch.randint(0, 100, (1, 10))
    attention_mask = torch.ones((1, 10))

    # Run the baseline configuration
    baseline_output = model(input_ids, attention_mask=attention_mask)

    # Run the fusion configuration
    fusion_output = model(input_ids, attention_mask=attention_mask)

    # Compare the outputs
    assert torch.allclose(baseline_output.logits, fusion_output.logits)

    # Test weight loading with hf overrides
    model.hf_overrides = {"num_hidden_layers": 6}
    fusion_output = model(input_ids, attention_mask=attention_mask)
    assert torch.allclose(baseline_output.logits, fusion_output.logits)