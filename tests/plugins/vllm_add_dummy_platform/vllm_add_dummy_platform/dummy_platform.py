from vllm.platforms import Platform, PlatformEnum


class DummyPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_name = "DummyDevice"
    device_type = "DummyType"
    dispatch_key = "DUMMY"
    supported_quantization = ["dummy_quantization"]
