FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && \
    apt-get install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

COPY startup.sh /startup.sh

RUN chmod +x /startup.sh

COPY . .

RUN /startup.sh

CMD ["/bin/bash"]
