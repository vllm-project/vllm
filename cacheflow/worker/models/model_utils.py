import torch.nn as nn

from cacheflow.worker.models.opt import OPTForCausalLM

MODEL_CLASSES = {
    'opt': OPTForCausalLM,
}


def get_model(model_name: str) -> nn.Module:
    for model_class, model in MODEL_CLASSES.items():
        if model_class in model_name:
            return model.from_pretrained(model_name)
    raise ValueError(f'Invalid model name: {model_name}')
