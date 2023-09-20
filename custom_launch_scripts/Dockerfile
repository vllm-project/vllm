FROM nvcr.io/nvidia/pytorch:22.12-py3
RUN pip uninstall torch -y && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip install --upgrade pip
RUN pip install accelerate==0.21.0 bitsandbytes==0.40.2 gradio==3.37.0 protobuf==3.20.3 typing-inspect==0.8.0 typing_extensions==4.5.0 ray==2.5.1  transformers==4.33.1 scipy sentencepiece torch
COPY ./ /home/focusgpt/
WORKDIR /home/focusgpt/
RUN cd /home/focusgpt/vllm_src/vllm/ && pip install -e .
RUN cd /home/focusgpt/ && chmod +x /home/focusgpt/new_vllm_launch.sh
ENTRYPOINT ["./vllm_launch.sh"]
EXPOSE 8080
