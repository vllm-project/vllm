import os

from ...utils import RemoteOpenAIServer


def test_oot_registration_for_api_server():
    cli_args = "--gpu-memory-utilization 0.10 --dtype float32".split()
    path = os.path.dirname(os.path.dirname(__file__))  # tests/entrypoints/
    root_path = os.path.dirname(os.path.dirname(path))  # root of the repo
    env_dict = {
        "VLLM_PLUGINS": "vllm_add_dummy_model",
        "PYTHONPATH": f"{root_path}:{path}"
    }
    with RemoteOpenAIServer("facebook/opt-125m", cli_args,
                            env_dict=env_dict) as server:
        client = server.get_client()
        completion = client.chat.completions.create(
            model="facebook/opt-125m",
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": "Hello!"
            }],
            temperature=0,
        )
        generated_text = completion.choices[0].message.content
        # make sure only the first token is generated
        rest = generated_text.replace("<s>", "")
        assert rest == ""
