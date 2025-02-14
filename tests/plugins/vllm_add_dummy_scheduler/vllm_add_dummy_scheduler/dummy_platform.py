# SPDX-License-Identifier: Apache-2.0

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.platforms.cuda import CudaPlatform


class DummyPlatform(CudaPlatform):
    device_name = "DummyDevice"

    # function copied from vllm-project/vllm/platforms/cuda.py
    # last line added to change scheduler_cls to platform specific scheduler
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Multi-step scheduling is not supported (and not "
                        "needed) on VLLM V1. Please launch without "
                        "--num-scheduler-steps.")
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.multi_step_worker.MultiStepWorker"
            elif vllm_config.speculative_config:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Speculative decoding is not yet supported on VLLM V1."
                    )
                else:
                    parallel_config.worker_cls = \
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "vllm.worker.worker.Worker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # additional line to load plugin specific scheduler instead of default
        parallel_config.scheduler_cls = \
            "vllm_add_dummy_scheduler.dummy_scheduler.DummyScheduler"
