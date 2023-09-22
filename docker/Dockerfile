FROM continuumio/miniconda3:23.5.2-0

WORKDIR /app/vllm

RUN set -eux; \
    apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /app/vllm

RUN apt-get update \
    && apt-get install -y build-essential \
    && conda env create \
    && apt-get purge -y --auto-remove build-essential \
    && pip install --upgrade ray pyarrow pandas \
    && rm -fr /var/lib/apt/lists/*

COPY entrypoint.sh /app/vllm

ENV PATH /opt/conda/envs/vllm/bin:$PATH

ENTRYPOINT [ "/app/vllm/entrypoint.sh" ]