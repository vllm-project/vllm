from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import (divide, get_tp_group,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


class PackFactor(object):

    def __init__(self, pack_bits: int, num_bits: int) -> None:
        if num_bits not in [2, 3, 4]:
            raise ValueError
        self.pack_bits = pack_bits
        self.num_bits = num_bits

    def __rfloordiv__(self, other: int) -> int:
        if not isinstance(other, int):
            raise TypeError
        # the sole purpose of this class is to change the ordering
        # of the operands, so that it works with 3-bits. That is,
        # instead of using `other // (pack_bits // num_bits)`,
        # we use `(other // pack_bits) * num_bits`. The former
        # does not work with 3-bits, while the latter does.
        return divide(other, self.pack_bits) * self.num_bits


class FluteConfig(QuantizationConfig):
    """Config class for FLUTE Quantization."""

    def __init__(
        self,
        num_bits: int,
        group_size: int,
        num_sms_packed: int,
    ) -> None:
        if num_bits not in [2, 3, 4]:
            raise ValueError

        self.num_bits = num_bits
        self.group_size = group_size
        self.pack_factor = PackFactor(pack_bits=16, num_bits=num_bits)
        self.num_sms_packed = num_sms_packed

    def __repr__(self) -> str:
        return (f"FluteConfig("
                f"num_bits={self.num_bits}, "
                f"group_size={self.group_size}, "
                f"num_sms_packed={self.num_sms_packed})")

    @classmethod
    def get_name(cls) -> str:
        return "flute"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["flute_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FluteConfig":
        num_bits = cls.get_from_keys(config, ["num_bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        num_sms_packed = cls.get_from_keys(config, ["num_sms"])

        return cls(num_bits=num_bits,
                   group_size=group_size,
                   num_sms_packed=num_sms_packed)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["FluteLinearMethod"]:
        if isinstance(layer, LinearBase):
            return FluteLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class FluteLinearMethod(LinearMethodBase):
    """Linear method for Flute.

    Args:
        quant_config: The Flute quantization config.
    """

    def __init__(self, quant_config: FluteConfig) -> None:

        try:
            import flute
            if flute.__version__ < "0.0.6":
                raise ImportError("flute version is wrong. Please "
                                  "install flute>=0.0.6.")
        except ImportError as err:
            raise ImportError("Please install flute>=0.0.6 via "
                              "`pip install flute-kernel>=0.0.6` to use "
                              "flute quantizer.") from err

        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:

        import flute
        import flute.utils

        if params_dtype not in [torch.float16, torch.bfloat16]:
            raise TypeError

        K = input_size_per_partition
        N = sum(output_partition_sizes)
        P = int(N / 16 * self.quant_config.num_bits)
        G = int(K / self.quant_config.group_size)
        device = "cuda"

        weight = Parameter(
            torch.empty(
                (P, K),
                dtype=torch.int16,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        scales = Parameter(
            torch.empty(
                (N, G),
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
                # it's unclear if we need to specify `packed_dim`, but looks
                # like this is only useful if `pack_factor == output_dim`
                # "packed_dim": 1,
            },
        )

        tables = Parameter(
            torch.arange(
                2**self.quant_config.num_bits,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            tables,
            {
                **extra_weight_attrs,
                "input_dim": None,
                "output_dim": None,
                "ignore_warning": True,
            },
        )

        tables2 = Parameter(
            flute.utils.make_qmap2_from_qmap(tables),
            requires_grad=False,
        )
        set_weight_attrs(
            tables2,
            {
                **extra_weight_attrs,
                "input_dim": None,
                "output_dim": None,
                "ignore_warning": True,
            },
        )

        layer.num_bits = self.quant_config.num_bits
        layer.group_size = self.quant_config.group_size
        layer.num_sms_packed = self.quant_config.num_sms_packed
        layer.workspace = flute.utils.get_workspace_streamk(weight.device)

        layer.register_parameter("weight", weight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("tables", tables)
        layer.register_parameter("tables2", tables2)

        layer.needs_repacking = True
        layer.flute_input_size = input_size
        layer.flute_output_size = output_size
        layer.flute_output_partition_sizes = output_partition_sizes
        layer.flute_input_size_per_partition = input_size_per_partition
        layer.flute_is_K_partitioned = (input_size_per_partition != input_size)
        layer.flute_is_N_partitioned = (sum(output_partition_sizes) !=
                                        output_size)

    def _maybe_tensor_all_gather(
        self,
        tensor: torch.Tensor,
        shard_dim: Optional[int],
    ) -> torch.Tensor:

        if shard_dim is None:
            return tensor

        # NCCL does not support int16
        if tensor.dtype == torch.int16:
            tensor_dtype = tensor.dtype
            tensor_casted = tensor.to(dtype=torch.int32)
            if not (tensor_casted == tensor).all():
                raise ValueError
        else:
            tensor_dtype = None
            tensor_casted = tensor

        tensor_gathered = tensor_model_parallel_all_gather(tensor_casted,
                                                           dim=shard_dim)

        if tensor_dtype is not None:
            tensor_gathered_casted = tensor_gathered.to(dtype=tensor_dtype)
            if not (tensor_gathered_casted == tensor_gathered).all():
                raise ValueError
            return tensor_gathered_casted
        else:
            return tensor_gathered

    def _maybe_tensor_shard(
        self,
        tensor: torch.Tensor,
        shard_dim: Optional[int],
    ) -> torch.Tensor:

        if shard_dim is None:
            return tensor

        tp_group = get_tp_group()
        shard_dim_size = divide(tensor.shape[shard_dim], tp_group.world_size)
        tensor_shards = torch.split(tensor, shard_dim_size, dim=shard_dim)
        # NOTE: torch.split does not create contiguous tensors by default.
        tensor_shard = tensor_shards[tp_group.rank].contiguous()
        return tensor_shard

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        import flute
        import flute.utils

        # the weights packing are possibly shape-specialized, but as vLLM
        # potentially fuses parameters and/or partitions the weights (TP),
        # we need to potentially re-pack the weights.
        if ((not hasattr(layer, "needs_repacking")
             or not layer.needs_repacking)):
            return

        if ((layer.flute_is_K_partitioned is False)
                and (layer.flute_is_N_partitioned is False)):
            shard_dim = None
        if ((layer.flute_is_K_partitioned is True)
                and (layer.flute_is_N_partitioned is False)):
            shard_dim = 1
        if ((layer.flute_is_K_partitioned is False)
                and (layer.flute_is_N_partitioned is True)):
            shard_dim = 0
        if ((layer.flute_is_K_partitioned is True)
                and (layer.flute_is_N_partitioned is True)):
            raise NotImplementedError

        # split the combined tensors into individual tensors
        # weight: [P, K]
        # scales: [N, G]
        Ns = layer.flute_output_partition_sizes
        Ps = [int(N / 16 * layer.num_bits) for N in Ns]
        Qs = torch.split(layer.weight, Ps, dim=0)
        Ss = torch.split(layer.scales, Ns, dim=0)

        Qs_unpacked = []
        for Q, S in zip(Qs, Ss):
            # when the tensors are sharded, gather them before unpacking
            Q_gathered = self._maybe_tensor_all_gather(Q, shard_dim=shard_dim)
            S_gathered = self._maybe_tensor_all_gather(S, shard_dim=shard_dim)

            # unpack
            Q_gathered_unpacked = flute.utils.unpack(
                weight=Q_gathered,
                scales=S_gathered,
                workspace=layer.workspace,
                num_bits=layer.num_bits,
                group_size=layer.group_size,
                num_sms_packed=layer.num_sms_packed)

            # re-shard
            Qs_unpacked.append(
                self._maybe_tensor_shard(Q_gathered_unpacked,
                                         shard_dim=shard_dim))

        # reconstruct the unpacked tensor
        Q_unpacked = torch.cat(Qs_unpacked, dim=0)

        # re-pack the tensors
        Q_repacked = flute.utils.pack(
            Q_unpacked.T.contiguous().to(device="cpu"),
            num_bits=layer.num_bits,
            group_size=layer.group_size).to(device=layer.weight.device)

        if not all([
                Q_repacked.shape == layer.weight.shape, Q_repacked.dtype
                == layer.weight.dtype, Q_repacked.device == layer.weight.device
        ]):
            raise ValueError
        layer.weight = Parameter(Q_repacked, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        import flute

        output = flute.qgemm_simple(
            x,
            layer.weight,
            layer.scales,
            layer.tables,
            layer.tables2,
            layer.workspace,
            layer.num_bits,
            layer.group_size,
        )

        if bias is not None:
            output.add_(bias)  # In-place add

        return output
