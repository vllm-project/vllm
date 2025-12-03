# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import FastAPI


def register_vllm_serve_api_routers(app: FastAPI):
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

    from vllm.entrypoints.serve.instrumentator.metrics import (
        attach_router as attach_metrics_router,
    )

    attach_metrics_router(app)

    from vllm.entrypoints.serve.instrumentator.health import (
        attach_router as attach_health_router,
    )

    attach_health_router(app)
