# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Supply an isolated vLLM 0.26.0 / NIXL 1.4.1 parent with the desired UCX build.
# This layer neither builds UCX nor changes the parent's native libraries.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY peer_lab.py peer_lab_api.py patch_vllm.py /opt/peer-retirement/
ENV PYTHONPATH="/opt/peer-retirement:${PYTHONPATH}"
RUN python -c "import importlib.metadata as m; assert m.version('vllm') == '0.26.0'; assert m.version('nixl') == '1.4.1'" \
    && python /opt/peer-retirement/patch_vllm.py \
    && python -c "import peer_lab, peer_lab_api"
