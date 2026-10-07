# Prometheus and Grafana

This is a simple example that shows you how to connect vLLM metric logging to the Prometheus/Grafana stack. For this example, we launch Prometheus and Grafana via Docker. You can checkout other methods through [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/) websites.

Install:

- [`docker`](https://docs.docker.com/engine/install/)
- [`docker compose`](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

## Launch

Prometheus metric logging is enabled by default in the OpenAI-compatible server. Launch via the entrypoint:

```bash
vllm serve mistralai/Mistral-7B-v0.1 \
    --max-model-len 2048
```

Launch Prometheus and Grafana servers with `docker compose` from this example's directory:

```bash
cd examples/observability/prometheus_grafana
docker compose up
```

Prometheus scrapes the vLLM server on port 8000 of the host every 5 seconds (see `prometheus.yaml`). Grafana is provisioned with a Prometheus data source and the [vLLM dashboards](../dashboards/grafana/README.md).

Submit some sample requests to the server:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
    --model mistralai/Mistral-7B-v0.1 \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 3.0
```

Navigating to [`http://localhost:8000/metrics`](http://localhost:8000/metrics) will show the raw Prometheus metrics being exposed by vLLM.

## Grafana Dashboards

Navigate to [`http://localhost:3000`](http://localhost:3000) and log in with the default username (`admin`) and password (`admin`).

The dashboards are in the **vLLM** folder under [Dashboards](http://localhost:3000/dashboards). Start with **vLLM / Overview** and follow the *vLLM dashboards* links in the top-right corner to the other dashboards:

- **vLLM / Overview**: traffic, errors, latency SLOs, throughput and saturation
- **vLLM / Instances**: per-instance and per-engine drill-down
- **vLLM / KV Cache**: KV cache pressure, prefix caching, KV transfer and offloading
- **vLLM / Speculative Decoding**: draft acceptance and token flow

The files in `grafana/provisioning` configure the data source and load the dashboards from [`../dashboards/grafana`](../dashboards/grafana/README.md). To monitor a different deployment, edit the scrape targets in `prometheus.yaml`; to use the dashboards with your own Grafana instance, import them as described in [their README](../dashboards/grafana/README.md).
