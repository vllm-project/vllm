import pytest

@pytest.fixture
def model():
    model_name = "vllm-base"
    return AutoModelForCausalLM.from_pretrained(model_name)

@pytest.fixture
def tokenizer():
    model_name = "vllm-base"
    return AutoTokenizer.from_pretrained(model_name)