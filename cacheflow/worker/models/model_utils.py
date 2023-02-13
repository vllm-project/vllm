import torch.nn as nn

from cacheflow.worker.models.opt import OPTForCausalLM

MODEL_CLASSES = {
    'opt': OPTForCausalLM,
}


def get_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f'Invalid model name: {model_name}')
    return MODEL_CLASSES[model_name].from_pretrained(model_name)
