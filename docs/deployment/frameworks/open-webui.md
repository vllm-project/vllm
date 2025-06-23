---
title: Open WebUI
---
[](){ #deployment-open-webui }

1. Install the [Docker](https://docs.docker.com/engine/install/)

2. Start the vLLM server with the supported chat completion model, e.g.

```bash
vllm serve qwen/Qwen1.5-0.5B-Chat
```

1. Start the [Open WebUI](https://github.com/open-webui/open-webui) docker container (replace the vllm serve host and vllm serve port):

```bash
docker run -d -p 3000:8080 \
--name open-webui \
-v open-webui:/app/backend/data \
-e OPENAI_API_BASE_URL=http://<vllm serve host>:<vllm serve port>/v1 \
--restart always \
ghcr.io/open-webui/open-webui:main
```

1. Open it in the browser: <http://open-webui-host:3000/>

On the top of the web page, you can see the model `qwen/Qwen1.5-0.5B-Chat`.

![](../../assets/deployment/open_webui.png)
