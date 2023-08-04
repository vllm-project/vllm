FROM nvcr.io/nvidia/pytorch:22.12-py3

# Ensure that Python prints output immediately
ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8-venv \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install base requirements
RUN pip uninstall torch -y
COPY requirements-pinned.txt .
RUN pip install -r requirements-pinned.txt

# Install vllm
COPY . .
RUN pip install .

COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
