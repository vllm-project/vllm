"""
Custom worker extension for testing reload_weights.
This allows us to inspect actual model weights via RPC.
"""
import torch


class WeightInspectorExtension:
    """Worker extension to inspect model weights."""

    def get_weight_stats(self, param_name: str = None) -> dict:
        """
        Get statistics for model weights.

        Args:
            param_name: Optional specific parameter name to inspect.
                       If None, returns stats for the first parameter.

        Returns:
            Dictionary with weight statistics (mean, std, first 5 values)
        """
        # Get the model from the worker
        model = self.model_runner.model

        # Get all parameters
        params = dict(model.named_parameters())

        if param_name is None:
            # Get first parameter
            param_name = list(params.keys())[0]

        if param_name not in params:
            return {
                'error': f'Parameter {param_name} not found',
                'available_params': list(params.keys())[:10]
            }

        param = params[param_name]

        # Get statistics
        mean = param.data.mean().item()
        std = param.data.std().item()
        first_5 = param.data.flatten()[:5].cpu().tolist()

        return {
            'param_name': param_name,
            'mean': mean,
            'std': std,
            'first_5': first_5,
            'shape': list(param.shape),
            'device': str(param.device),
        }
