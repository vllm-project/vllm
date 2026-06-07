import torch
from torch.compile import register_custom_pass

# Define the custom pass for fused MoE Finalize + ResidualAdd + AllReduce + RMSNorm
@register_custom_pass
def fused_moe_finalize_residual_add_allreduce_rmsnorm(module):
    """
    Fused MoE Finalize + ResidualAdd + AllReduce + RMSNorm custom pass.
    """
    # Check if the module is a wrapped fused MoE op
    if isinstance(module, torch.nn.Module) and hasattr(module, 'fused_moe'):
        # Pull out the MoE finalize op
        moe_finalize = module.fused_moe.moe_finalize
        # Create a new module for the fused op
        fused_op = torch.nn.Module()
        fused_op.moe_finalize = moe_finalize
        fused_op.residual_add = module.fused_moe.residual_add
        fused_op.allreduce = module.fused_moe.allreduce
        fused_op.rms_norm = module.fused_moe.rms_norm
        # Replace the original module with the fused op
        return fused_op
    return module

# Define the kMoEFinalizeARResidualRMSNorm function
def kMoEFinalizeARResidualRMSNorm(input, weight, bias, residual, allreduce, rms_norm):
    """
    Fused MoE Finalize + ResidualAdd + AllReduce + RMSNorm function.
    """
    # Perform MoE finalize
    output = input + weight
    # Perform residual add
    output = output + residual
    # Perform allreduce
    output = allreduce(output)
    # Perform RMS norm
    output = rms_norm(output)
    return output