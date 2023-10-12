FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel as build

# build requirements
RUN pip install -v ninja
RUN pip install -v packaging
RUN pip install -v setuptools
RUN pip install -v wheel
RUN pip install -v torch>=2.0.0

# copy things we need to build the c++ extensions
COPY csrc csrc
COPY setup.py setup.py
COPY README.md README.md
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY vllm/__init__.py vllm/__init__.py

ARG max_jobs=4

ENV MAX_JOBS=$max_jobs

# build just the c++ extensions
# TODO: we seem to be building each extension in order. Ideally we will build them in parallel
RUN python setup.py build_ext --inplace


FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime as runtime
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY README.md README.md
RUN pip install -r requirements.txt
RUN pip install accelerate fschat
COPY vllm vllm
COPY --from=build /workspace/vllm/*.so /workspace/vllm/

FROM runtime as vllm_api_server
ENTRYPOINT ["python", "-u", "-m", "vllm.entrypoints.api_server"]

FROM runtime as vllm_openai_api_server
ENTRYPOINT ["python", "-u", "-m", "vllm.entrypoints.openai.api_server"]
