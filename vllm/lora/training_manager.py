# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import List, Optional, Set

import torch
import torch.nn as nn

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers.column_parallel_linear import MergedQKVParallelLinearWithLoRA
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.lora.request import LoRARequest
from vllm.lora.training_state import TrainingState

from transformers import get_scheduler


logger = init_logger(__name__)


class TrainingManager:
    """
    Training manager for managing LoRA adapters during training.
    
    This class implements the singleton pattern to ensure only one instance
    exists throughout the application lifecycle.

    TODO(girfan): Add support for multiple LoRA adapters.
    """
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TrainingManager, cls).__new__(cls)
        return cls._instance


    def __init__(
        self,
        model_runner: "GPUModelRunner",
        lora_manager: WorkerLoRAManager,
        lora_config: LoRAConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model_runner = model_runner
        self.lora_manager = lora_manager
        self.lora_config = lora_config
        self.device = device
        self.dtype = dtype

        self.rank = lora_config.max_lora_rank
        self.alpha = lora_config.lora_alpha
        self.target_modules = lora_config.lora_training_target_modules
        self.scheduler_type = lora_config.lora_scheduler_type

        self.trainable_lora_ids: Set[int] = set()
        self.in_prog_lora_ids: Set[int] = set()
        self.trainable_lora_params: List[nn.Parameter] = []
        self.trainable_lora_param_names: List[str] = []
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.max_grad_norm: float = 1.0
        self.grad_norm: float = 0.0

        self.training_state = TrainingState()

        # Initially set to None, will be set when the first training request is added.
        # Must be the same for all training requests later.
        self._num_training_steps: Optional[int] = None
        self._num_warmup_steps: Optional[int] = None

        # Logging
        self.log_interval: int = 50
        self._last_logged_step: int = 0

        # Captured tensors for debugging/comparison
        self.captured_input_ids: Optional[torch.Tensor] = None
        self.captured_labels: Optional[torch.Tensor] = None


    @classmethod
    def is_enabled(cls) -> bool:
        return cls._instance is not None


    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of TrainingManager.
        
        Returns:
            TrainingManager: The singleton instance
            
        Raises:
            RuntimeError: If the instance has not been initialized yet
        """
        if cls._instance is None:
            raise RuntimeError("TrainingManager has not been initialized yet.")
        return cls._instance


    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance. 
        
        This is primarily useful for testing or when you need to create
        a fresh instance. Use with caution as this will lose all state.
        """
        if cls._instance is not None:
            cls._instance = None
            cls._initialized = False


    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if the singleton instance has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return cls._instance is not None and cls._initialized


    @property
    def model(self):
        """Get the actual model, unwrapping if needed."""
        if hasattr(self.model_runner, 'model'):
            model = self.model_runner.model
            # Unwrap if it's a UBatchWrapper or similar
            if hasattr(model, 'unwrap'):
                return model.unwrap()
            return model
        return self.model_runner


    @property
    def grad_accumulation_steps(self) -> int:
        return self.training_state.grad_accumulation_steps


    @grad_accumulation_steps.setter
    def grad_accumulation_steps(self, value: int):
        self.training_state.grad_accumulation_steps = value


    @property
    def num_training_steps(self) -> int:
        return self._num_training_steps


    @num_training_steps.setter
    def num_training_steps(self, value: int):
        if self._num_training_steps is not None and value != self._num_training_steps:
            raise ValueError("num_training_steps can only be set once")
        self._num_training_steps = value


    @property
    def num_warmup_steps(self) -> int:
        return self._num_warmup_steps


    @num_warmup_steps.setter
    def num_warmup_steps(self, value: int):
        if self._num_warmup_steps is not None and value != self._num_warmup_steps:
            raise ValueError("num_warmup_steps can only be set once")
        self._num_warmup_steps = value


    @property
    def loss(self) -> Optional[torch.Tensor]:
        return self.training_state.loss


    def add_loss(self, loss: torch.Tensor):
        self.training_state.add_loss(loss)


    def step(self):
        self.training_state.step()


    def reset_steps(self):
        self.training_state.reset_steps()


    def reset_loss(self):
        self.training_state.reset_loss()


    def model_zero_grad(self):
        """
        Zero gradients for all trainable parameters.

        Manually zero the gradients of all LoRA stacked tensors since they
        are not registered as nn.Parameters.
        """
        self.model.zero_grad()
        for param_tensor in self.trainable_lora_params:
            if param_tensor.grad is not None:
                param_tensor.grad.zero_()
        self.optimizer.zero_grad()


    def should_log(self) -> bool:
        return self.training_state.total_steps % self.log_interval == 0


    def log(self, learning_rate: Optional[float] = None):
        steps_since_last_log: int = self.training_state.total_steps - self._last_logged_step
        loss_value: float = self.training_state.loss / steps_since_last_log
        learning_rate: float = self.get_learning_rate() if learning_rate is None else learning_rate
        self._last_logged_step = self.training_state.total_steps
        logger.info(f"loss = {loss_value:.6f}, learning rate = {learning_rate:.6f}, grad norm = {self.grad_norm:.6f}")


    def should_run_optimizer_step(self) -> bool:
        return self.training_state.steps % self.grad_accumulation_steps == 0 and self.training_state.steps > 0


    # TODO(girfan): Cache this result?
    def get_qkv_indices_for_training(self) -> List[int]:
        """
        Determine which Q/K/V indices to train based on target_patterns.
        Returns (a_indices, b_indices) where indices map to: 0=Q, 1=K, 2=V.

        This makes training configurable - only specified projections are trained.
        """
        # Map projection names to indices
        projection_map = {"q_proj": 0, "k_proj": 1, "v_proj": 2}

        # Determine which indices to enable based on target_patterns
        enabled_indices = []
        for pattern in self.target_modules:
            for proj_name, idx in projection_map.items():
                if proj_name in pattern:
                    enabled_indices.append(idx)
                    break

        # Remove duplicates and sort
        enabled_indices = sorted(list(set(enabled_indices)))

        if not enabled_indices:
            raise ValueError(f"No specific projections found for target_patterns={self.target_patterns}")

        return enabled_indices


    def _freeze_base_model(self):
        """Freeze all base model parameters (non-LoRA)."""
        for name, param in self.model.named_parameters():
            # Check if this is a LoRA parameter (including stacked tensors)
            is_lora_param = any([
                'lora_a' in name.lower(), 'lora_b' in name.lower(),
                'lora' in name.lower() and 'stacked' in name.lower(),
                'lora' in name.lower() and ('weight' in name.lower() or 'bias' in name.lower())
            ])

            if is_lora_param:
                raise ValueError(f"Base model parameter {name} is a LoRA parameter")

            param.requires_grad = False


    def is_registered_by_id(self, lora_id: int) -> bool:
        return lora_id in self.trainable_lora_ids or lora_id in self.in_prog_lora_ids


    def is_registered_by_index(self, index: int) -> bool:
        mapping = self.lora_manager._adapter_manager.lora_index_to_id
        if index < 0 or index >= len(mapping):
            raise ValueError(f"LoRA index {index} out of range")
        lora_id = mapping[index]
        if lora_id is None:
            raise ValueError(f"LoRA index {index} not found in LoRA manager")
        return self.is_registered_by_id(lora_id)


    def lora_eval(self, lora_request: LoRARequest):
        """Evaluate the LoRA adapter."""
        self.model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()
        # Set requires_grad to False for all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Set requires_grad to False for all trainable parameters
        for param in self.trainable_lora_params:
            param.requires_grad = False
        self.model.zero_grad()
        self.optimizer.zero_grad()


    def lora_train(self, lora_request: LoRARequest):
        """Make the LoRA adapter trainable."""
        self._try_initialize_lora_for_training(lora_request)
        self._setup_optimizer_and_scheduler()
        self.model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        # Set requires_grad to True for all trainable parameters
        for param in self.trainable_lora_params:
            param.requires_grad = True


    # TODO(girfan): Take the params from earlier in the code.
    def _try_initialize_lora_for_training(
        self,
        lora_request: LoRARequest,
    ):
        """Convert a loaded LoRA adapter to trainable parameters and setup optimizer."""
        lora_id = lora_request.lora_int_id

        # Check if the LoRA adapter is already trainable
        if lora_id in self.trainable_lora_ids:
            logger.debug(f"LoRA adapter {lora_id} is already trainable")
            return

        # Mark the LoRA adapter setup as in progress
        self.in_prog_lora_ids.add(lora_id)

        # Get QKV indices for training
        indices = self.get_qkv_indices_for_training()

        # Deactivate the LoRA adapter before adding it as a trainable adapter
        self.lora_manager.remove_adapter(lora_id)

        # Add the LoRA adapter to the LoRAManager
        self.lora_manager.add_adapter(lora_request, is_trainable=True, trainable_slices=indices)

        # Freeze the base model
        self._freeze_base_model()

        # Get LoRA model and stacked tensor index
        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            raise ValueError(f"LoRA adapter {lora_id} not found in LoRA manager")

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            raise ValueError(f"LoRA adapter {lora_id} could not be retrieved")

        # Clear existing trainable parameters and make stacked tensors trainable
        self.trainable_lora_params.clear()
        self.trainable_lora_param_names.clear()

        # Make stacked tensors trainable
        for module_name, module in self.model.named_modules():
            # Check if this module has loaded LoRA weights
            if module_name not in lora_model.loras:
                continue  # Skip modules that don't have LoRA loaded

            # Determine slice indices to train per module
            if isinstance(module, MergedQKVParallelLinearWithLoRA):
                # Only use indices that exist in the stacked tensors
                a_indices = [i for i in indices if i < len(module.lora_a_stacked)]
                b_indices = [i for i in indices if i < len(module.lora_b_stacked)]
            else:
                # Single-slice (includes QKVParallelLinearWithLoRA which packs qkv into one slice)
                a_indices = [0] if len(module.lora_a_stacked) > 0 else []
                b_indices = [0] if len(module.lora_b_stacked) > 0 else []

            if not a_indices or not b_indices:
                raise ValueError(f"No valid indices found for module {module_name}")

            # Both a_indices and b_indices should be the same
            assert a_indices == b_indices
            a_b_indices = a_indices

            # Process lora_a_stacked and lora_b_stacked
            for idx in a_b_indices:
                # lora_a
                stacked_tensor = module.lora_a_stacked[idx]
                param_name = f"{module_name}.lora_a_stacked[{idx}]"
                self.trainable_lora_params.append(stacked_tensor)
                self.trainable_lora_param_names.append(param_name)

                # lora_b
                stacked_tensor = module.lora_b_stacked[idx]
                param_name = f"{module_name}.lora_b_stacked[{idx}]"
                self.trainable_lora_params.append(stacked_tensor)
                self.trainable_lora_param_names.append(param_name)

        # Mark the LoRA adapter as trainable
        self.trainable_lora_ids.add(lora_id)
        self.in_prog_lora_ids.remove(lora_id)


    def _setup_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int,
    ):
        """Setup learning rate scheduler."""
        scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler


    def _setup_optimizer(self, weight_decay: float, learning_rate: float):
        assert len(self.trainable_lora_params) > 0
        optimizer_grouped_parameters = [
            {
                "params": list(self.trainable_lora_params),
                "weight_decay": weight_decay,
            },
            {
                "params": [],
                "weight_decay": 0.0,
            }
        ]
        optimizer_kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "fused": True,
        }
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            **optimizer_kwargs,
        )
        return optimizer


    def _setup_optimizer_and_scheduler(self, weight_decay: float = 0.0, learning_rate: float = 1e-4):
        if self.optimizer is not None and self.scheduler is not None:
            return

        self.optimizer = self._setup_optimizer(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
        )
        self.scheduler = self._setup_scheduler(
            optimizer=self.optimizer,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.num_warmup_steps,
        )


    def _clip_gradients(self):
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            trainable_params = [p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=self.max_grad_norm,
            )
            self.grad_norm = grad_norm
        else:
            # Calculate gradient norm without clipping
            total_norm = 0.0
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            self.grad_norm = total_norm


    def optimizer_step(self) -> float:
        """Perform optimizer step with gradient clipping."""
        self._clip_gradients()
        self.optimizer.step()
        # Save learning rate before update
        learning_rate = self.get_learning_rate()
        self.scheduler.step()
        return learning_rate


    def get_learning_rate(self) -> float:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.scheduler.get_last_lr()[0]

        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()

        return last_lr
