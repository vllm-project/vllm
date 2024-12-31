from vllm.platforms.cuda import CudaPlatform


class DummyPlatform(CudaPlatform):
    device_name = "DummyDevice"
