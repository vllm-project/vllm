set -e
set -x

rocminfo

docker build -t rocm -f Dockerfile.rocm .

remove_docker_container() { docker rm -f rocm || true; }
trap remove_docker_container EXIT

remove_docker_container

docker run --device /dev/kfd --device /dev/dri --network host --name rocm rocm python3 -m vllm.entrypoints.api_server &

timeout=300
counter=0

while [ "$(curl -s -o /dev/null -w ''%{http_code}'' localhost:8000)" != "200" ]; do
    sleep 1
    counter=$((counter+1))
    if [ $counter -ge $timeout ]; then
        echo "Timeout after $timeout seconds"
        break
    fi
done

python examples/api_client.py
