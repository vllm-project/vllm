from vllm.control_vectors.control import ControlVector
from vllm.control_vectors.layers import MLPWithControlVector, BaseLayerWithControlVector
from vllm.config import ControlVectorConfig
from vllm.control_vectors.request import ControlVectorRequest
from vllm.lora.utils import replace_submodule
from typing import Optional, Dict, Set, Type
import torch
import re
import torch.nn as nn


_all_cv_classes = {
    "mlp": MLPWithControlVector
}

class ControlVectorModel:
    """A control vector model."""

    def __init__(
        self,
        model: nn.Module,
        control_vector_config: ControlVectorConfig,
    ):
        self.model = model
        self.control_vector_config = control_vector_config
        self._registered_control_vectors = {}
        self._active_control_vectors = []
        self.modules = {}
        self._create_cv_modules()


    def _create_cv_modules(self):
        for module_name, module in self.model.named_modules():
            for key in _all_cv_classes:
                if not module_name.endswith(key):
                    continue
                if isinstance(module, _all_cv_classes[key]):
                    continue
                new_module = replace_submodule(self.model, module_name, _all_cv_classes[key](module))
                self.register_module(module_name, new_module)
        
        print("HERE")

    
    def set_control_vector(self):
        for module_name, module in self.model.named_modules():
            for key in _all_cv_classes:
                if key not in module_name:
                    continue
                if isinstance(module, _all_cv_classes[key]):
                    cv_module = module
                else:
                    cv_module = _all_cv_classes[key](module)
                    new_module = replace_submodule(self.model, module_name, cv_module)
                    self.register_module(module_name, new_module)

                module_number = re.findall(r'\d+', module_name)

                if self._active_control_vectors:
                    cv_vector = self._active_control_vectors[0].directions.get(int(module_number[0]), None)
                else:
                    cv_vector = None

                cv_module.set_cv_vector(cv_vector)
            

    def register_module(self, module_name: str, module: "BaseLayerWithControlVector"):
        assert isinstance(module, BaseLayerWithControlVector)
        self.modules[module_name] = module
    
    def add_control_vector_request(self, request):
        assert isinstance(request, ControlVectorRequest), "Request must be an instance of ControlVectorRequest"
        self._registered_control_vectors[request.name] = request.get_control_vector()
    
    def remove_control_vector_request(self, request_name: str):
        if request_name in self._registered_control_vectors:
            del self._registered_control_vectors[request_name]

    def get_control_vector_request(self, request_name):
        return self._registered_control_vectors.get(request_name, None)
    
    def set_active_control_vector_request(self, request):
        if request.name in self._registered_control_vectors:
            self._active_control_vectors.append(self._registered_control_vectors[request.name])
            self.set_control_vector()
        else:
            raise ValueError(f"Control vector request{request.name} not found in registered control vectors.")
    
    def reset_control_vector(self):
        for _, module in self.model.named_modules():
            if type(module, MLPWithControlVector):
                module.reset_cv_vector()


    def _load_cv(self, cv_request: ControlVectorRequest):
        try:
            self.add_control_vector_adapter(cv_request)
            self.set_active_control_vector(cv_request)
            self.set_control_vector()
            return self
        except Exception as e:
            print("EXCEPTION: _load_cv", e)
        
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)




def create_cv_model(
        model: nn.Module,
        control_vector_config: ControlVectorConfig,
        **kwargs) -> ControlVectorModel:
    """Create a ControlVectorManager for a given model."""
    cv_model = ControlVectorModel(
        model=model,
        control_vector_config=control_vector_config,
        **kwargs)
    return cv_model