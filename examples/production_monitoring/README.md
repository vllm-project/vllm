# vLLM + Prometheus/Grafana 

This is a simple example that shows you how to connect vLLM metric logging to the Prometheus/Grafana stack.

> For this example, we launch Prometheus and Grafana via Docker. Make sure you have [`docker`](https://docs.docker.com/engine/install/) and [`docker compose`](https://docs.docker.com/compose/install/linux/#install-using-the-repository) installed.

### Launch

Prometheus metric logging is enabled by default in the OpenAI-compatible server. Launch vLLM as usual:
```bash
python3 ../../vllm/entrypoints/openai/api_server.py --model mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

Launch Prometheus and Grafana servers with `docker compose`:
```bash
docker compose up
```

Submit some sample requests to the server to generate some data:
```bash
python3 simulate_clients.py
```

> Note: you may need to run `pip install openai datasets` to run the above.

Navigating to [`http://localhost:8000/metrics`](http://localhost:8000/metrics) will show the raw Prometheus metrics being exposed by vLLM.

### Prometheus

#### Confirm Server Working
Visiting [`http://localhost:9090/targets`](http://localhost:9090/targets) should show that `http://host.docker.internal:8000/metrics` is in state `UP`


#### Query with PromQL
Navigate to [`http://localhost:9090/graph`](http://localhost:9090/graph) and execute the following expression to graph average time to first token over time:

```bash
rate(vllm_time_to_first_token_seconds_sum[30s])
/
rate(vllm_time_to_first_token_seconds_count[30s])
```

For more details on working with PromQL, see the [official docs](https://prometheus.io/docs/prometheus/latest/querying/basics/).

### Grafana

#### Login
Navigate to [`http://localhost:3000`](http://localhost:3000). Log in with the default username (`admin`) and password (`admin`).

#### Add Prometheus Data Source

Navigate to [`http://localhost:3000/connections/datasources/new`](http://localhost:3000/connections/datasources/new) and select Prometheus. 

On Prometheus configuration page, we need to add the `Prometheus Server URL` in `Connection`. Since Grafana and Prometheus are running in separate containers, put we need to put the IP address of the Prometheus container. Run the following to lookup the name of your Prometheus container:

```bash
docker container ls
CONTAINER ID   IMAGE                    COMMAND                  CREATED          STATUS          PORTS                                       NAMES
6b2eb9d7aa99   grafana/grafana:latest   "/run.sh"                45 minutes ago   Up 45 minutes   0.0.0.0:3000->3000/tcp, :::3000->3000/tcp   production_monitoring-grafana-1
d9b32bc6a02b   prom/prometheus:latest   "/bin/prometheus --c…"   45 minutes ago   Up 45 minutes   0.0.0.0:9090->9090/tcp, :::9090->9090/tcp   production_monitoring-prometheus-1
```

Run the following to lookup the IP address (replace `production_monitoring-prometheus-1` with your container name):
```bash 
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' production_monitoring-prometheus-1
>> 172.18.0.2
```

So, in our case, the `Prometheus Server URL` should be: `http://172.18.0.2:9090`.

Click `Save & Test`. We should get a green check saying "Successfully queried the Prometheus API.".

#### Import Dashboard 

Navigate to [`http://localhost:3000/dashboard/import`](http://localhost:3000/dashboard/import), upload `grafana.json`, and select the `prometheus` datasource.

You should see a screen that looks like the following:

![Grafana Dashboard Image](images/grafana-dashboard.png)
