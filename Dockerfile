FROM nvcr.io/nvidia/pytorch:22.12-py3

# Ensure that Python prints output immediately
ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app/

# Install base requirements
RUN pip uninstall torch -y
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install vllm
COPY . .
RUN pip install .

COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
