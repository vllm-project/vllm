FROM vllm/vllm-openai:latest

# Set the working directory (optional)
WORKDIR /app

# Copy startup-time-logs, assuming it is running from startup-time-logs folder
COPY . .

# uninstall vllm
RUN pip uninstall -y vllm

# install vllm developer mode, I think that is where the vllm source is inside the image
RUN git clone https://github.com/vllm-project/vllm.git
RUN VLLM_USE_PRECOMPILED=1 pip install -e vllm

# install startup-time-logs
RUN pip install .

CMD ["/bin/bash", "-c", "sleep infinity"]
