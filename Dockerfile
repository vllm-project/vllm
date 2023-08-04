FROM nvcr.io/nvidia/pytorch:22.12-py3

# Ensure that Python prints output immediately
ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app/

RUN pip uninstall pytorch -y
COPY . .
RUN pip install .

COPY docker-entrypoint.sh .

ENTRYPOINT ["./docker-entrypoint.sh"]
