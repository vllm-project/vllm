from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from vllm_simulator.utils import get_logger

_all_accs_: Dict[str, "AcceleratorInfo"] = {}
_acc_alias: Dict[str, str] = {}

logger = get_logger()


@dataclass
class AcceleratorInfo:
    # Must match an AISimulate system name.
    name: str
    hbm_capacity_gb: Optional[float] = None
    device_alias: list = field(default_factory=list)
    ref: str = ""

    @classmethod
    def from_dict(cls, config: Dict, save_to_registry: bool = False):
        acc = cls(**config)
        if save_to_registry:
            if acc.name in _acc_alias:
                logger.error(f"{acc.name} is already in registry")
            _all_accs_[acc.name.upper()] = acc
            for alias in acc.device_alias:
                if alias in _acc_alias:
                    logger.warning(f"Device alias [{alias}] is already in registry.")
                else:
                    _acc_alias[alias] = acc.name.upper()
        return acc

    @staticmethod
    def find_by_hw_name(hw_name: str) -> Union[None, "AcceleratorInfo"]:
        if hw_name in _acc_alias:
            hw = _all_accs_.get(_acc_alias[hw_name], None)
            if hw is not None:
                hw = deepcopy(hw)
                hw.name = hw_name
            return hw
        else:
            return _all_accs_.get(hw_name.upper(), None)

    @staticmethod
    def list_all_hws() -> Dict[str, "AcceleratorInfo"]:
        return _all_accs_


class NVIDIA:
    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H20",
            "device_alias": ["H20", "h20_sxm"],
            "hbm_capacity_gb": 96,
            "ref": "https://viperatech.com/product/nvidia-hgx-h20",
        },
        save_to_registry=True,
    )


__all__ = ["AcceleratorInfo", "NVIDIA"]
