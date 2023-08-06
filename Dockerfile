FROM nvcr.io/nvidia/pytorch:22.12-py3

# Ensure that Python prints output immediately
ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Setup virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install base requirements
# COPY requirements.docker.txt .
# RUN pip install -r requirements.docker.txt

# # Install vllm
# COPY . .
# RUN pip install .
RUN pip install vllm==0.1.3 ray==2.6.2
RUN pip install ray[air]==2.6.2

COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
