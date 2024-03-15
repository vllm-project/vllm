set -e
set -x

docker build -t rocm -f Dockerfile.rocm .

remove_docker_container() { docker rm -f rocm || true; }
trap remove_docker_container EXIT

remove_docker_container

docker run -d --p 8000:8000 -n rocm rocm
while [ "$(curl -s -o /dev/null -w ''%{http_code}'' localhost:8000)" != "200" ]; do sleep 1; done
python examples/openai_completion_client.py
