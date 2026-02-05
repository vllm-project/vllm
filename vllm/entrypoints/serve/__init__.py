# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_vllm_serve_api_routers(app: FastAPI):
    if envs.VLLM_SERVER_DEV_MODE:
        logger.warning(
            "SECURITY WARNING: Development endpoints are enabled! "
            "This should NOT be used in production!"
        )

    from vllm.entrypoints.serve.lora.api_router import (
        attach_router as attach_lora_router,
    )

    attach_lora_router(app)

    from vllm.entrypoints.serve.elastic_ep.api_router import (
        attach_router as attach_elastic_ep_router,
    )

    attach_elastic_ep_router(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.sleep.api_router import (
        attach_router as attach_sleep_router,
    )

    attach_sleep_router(app)

    from vllm.entrypoints.serve.rpc.api_router import (
        attach_router as attach_rpc_router,
    )

    attach_rpc_router(app)

    from vllm.entrypoints.serve.cache.api_router import (
        attach_router as attach_cache_router,
    )

    attach_cache_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)

    from vllm.entrypoints.serve.disagg.api_router import (
        attach_router as attach_disagg_router,
    )

    attach_disagg_router(app)

    from vllm.entrypoints.serve.rlhf.api_router import (
        attach_router as attach_rlhf_router,
    )

    attach_rlhf_router(app)

    from vllm.entrypoints.serve.pause.api_router import (
        attach_router as attach_pause_router,
    )

    attach_pause_router(app)

    from vllm.entrypoints.serve.instrumentator.metrics import (
        attach_router as attach_metrics_router,
    )

    attach_metrics_router(app)

    from vllm.entrypoints.serve.instrumentator.health import (
        attach_router as attach_health_router,
    )

    attach_health_router(app)

    from vllm.entrypoints.serve.instrumentator.offline_docs import (
        attach_router as attach_offline_docs_router,
    )

    attach_offline_docs_router(app)
    from vllm.entrypoints.serve.instrumentator.server_info import (
        attach_router as attach_server_info_router,
    )

    attach_server_info_router(app)
