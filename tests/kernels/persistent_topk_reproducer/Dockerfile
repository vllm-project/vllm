FROM vllm/vllm-openai:v0.27.0

ENV PYTHONUNBUFFERED=1
WORKDIR /repro

COPY tests/kernels/persistent_topk_reproducer/ /repro/
COPY csrc/libtorch_stable/persistent_topk.cuh /repro/
COPY csrc/libtorch_stable/topk_histogram_4096.cuh /repro/

RUN chmod 0755 /repro/run.sh

ENTRYPOINT ["/bin/bash", "-lc"]
