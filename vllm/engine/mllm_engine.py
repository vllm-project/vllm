#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
import time
from io import BytesIO
import requests
from PIL import Image

from vllm import LLMEngine, SamplingParams
from typing import List, Optional
from vllm.logger import init_logger
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler
from vllm.engine.ray_utils import DeviceID, ray
from vllm.sequence import Sequence, SequenceGroup
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import Counter
from vllm.worker.worker import MWorker

logger = init_logger(__name__)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class MLLMEngine(LLMEngine):
    def __init__(
            self,
            model_config: ModelConfig,
            cache_config: CacheConfig,
            parallel_config: ParallelConfig,
            scheduler_config: SchedulerConfig,
            distributed_init_method: str,
            stage_devices: List[List[DeviceID]],
            log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"use_dummy_weights={model_config.use_dummy_weights}, "
            f"download_dir={model_config.download_dir!r}, "
            f"use_np_weights={model_config.use_np_weights}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code)
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        self.workers: List[MWorker] = []
        assert len(stage_devices) == 1, "Only support one stage for now."
        for rank, node_resource, _ in stage_devices[0]:
            worker_cls = MWorker
            if self.parallel_config.worker_use_ray:
                worker_cls = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    resources={node_resource: 1e-3},
                )(worker_cls).remote

            worker = worker_cls(
                model_config,
                parallel_config,
                scheduler_config,
                rank,
                distributed_init_method,
            )
            self.workers.append(worker)
        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)

    def add_request(
            self,
            request_id: str,
            prompt: Optional[str],
            image: Optional[dict] = None,
            sampling_params: SamplingParams = None,
            prompt_token_ids: Optional[List[int]] = None,
            arrival_time: Optional[float] = None
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """

        mm_use_im_start_end = self.workers[0].model.model.vision_tower[0].config.use_im_start_end
        image_size = self.workers[0].model.model.vision_tower[0].config.image_size
        patch_size = self.workers[0].model.model.vision_tower[0].config.patch_size
        image_token_len = int((image_size / patch_size) ** 2)

        if arrival_time is None:
            arrival_time = time.time()
        if image:
            if mm_use_im_start_end:
                image_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
            else:
                image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            prompt += image_tokens
            image_data = self._load_image(image)
        else:
            image_data = None
        prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, image_data=image_data)
            seqs.append(seq)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def _load_image(self, image_srcs):
        images = []
        image_srcs = image_srcs if isinstance(image_srcs, list) else [image_srcs]
        for image_src_i in image_srcs:
            image_file = image_src_i.get("image_src")
            src_type = image_src_i.get("src_type")

            if src_type == "url":
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            elif src_type == "local":
                image = Image.open(image_file).convert('RGB')
            elif src_type == "base64":
                image = Image.open(BytesIO(base64.b64decode(image_file))).convert('RGB')
            else:
                assert 0, "src_type is not true"
            image_tensor = self.workers[0].model.model.image_processor(image, return_tensors='pt')['pixel_values'][0]
            images.append(image_tensor.half().cuda())
        return images

    def initialize_vision_tokenizer(self):
        self._run_workers(
            "initialize_vision_tokenizer",
            tokenizer=self.tokenizer
        )
