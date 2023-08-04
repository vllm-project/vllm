FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN conda create -n vllm python=3.10.11 -y

# Ensure that Python prints output immediately
ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app/

RUN pip uninstall torch -y
COPY . .
RUN conda run -n vllm pip install .

COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
