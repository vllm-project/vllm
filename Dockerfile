FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2

RUN conda create -n vllm python=3.10.11 -y

# Ensure that Python prints output immediately
ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app/

COPY . .
RUN conda run -n vllm pip install .

COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
