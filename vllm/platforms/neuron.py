from .interface import Platform, PlatformEnum


class NeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON
    device_type: str = "neuron"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"
