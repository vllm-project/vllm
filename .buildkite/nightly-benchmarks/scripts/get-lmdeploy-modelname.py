from lmdeploy.serve.openai.api_client import APIClient

api_client = APIClient("http://localhost:8000")
model_name = api_client.available_models[0]

print(model_name)
