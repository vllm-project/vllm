.. _gguf:

GGUF
==================

.. warning::

   Please note that GGUF support in vLLM is highly experimental. Currently, you can use GGUF as a way to reduce memory footprint. If you encounter any issues, please report them to the vLLM team.

To run a GGUF model with vLLM, you can download and use the local GGUF model from `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF <https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ>`_ with the following command:

.. code-block:: console
   $ wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   $ # We recommend using the tokenizer from source model to avoid long-time tokenizer conversion from GGUF.
   $ vllm serve --model ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0

.. warning::

   We recommend using the tokenizer from the source model instead of GGUF model. This is because the tokenizer conversion from GGUF is time-consuming and unstable, especially some models with large vocab size.

You can also use the GGUF model directly through the LLM entrypoint:
.. code-block:: python
   from huggingface_hub import hf_hub_download

   from vllm import LLM, SamplingParams


   def run_gguf_inference(model_path):
      prompts = [
         "How many helicopters can a human eat in one sitting?",
         "What's the future of AI?",
      ]
      prompts = [
         PROMPT_TEMPLATE.format(system_message=system_message, prompt=prompt)
         for prompt in prompts
      ]
      # Create a sampling params object.
      sampling_params = SamplingParams(temperature=0, max_tokens=128)

      # Create an LLM.
      llm = LLM(model=model_path,
               tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

      outputs = llm.generate(prompts, sampling_params)
      # Print the outputs.
      for output in outputs:
         prompt = output.prompt
         generated_text = output.outputs[0].text
         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


   if __name__ == "__main__":
      repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
      filename = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
      model = hf_hub_download(repo_id, filename=filename)
      run_gguf_inference(model)



