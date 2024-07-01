Frequently Asked Questions
===========================

    Q: How can I serve multiple models on a single port using the OpenAI API?

A: Assuming that you're referring to using OpenAI compatible server to serve multiple models at once, that is not currently supported, you can run multiple instances of the server (each serving a different model) at the same time, and have another layer to route the incoming request to the correct server accordingly.

----------------------------------------

    Q: Which model to use for offline inference embedding?

A: If you want to use an embedding model, try: https://huggingface.co/intfloat/e5-mistral-7b-instruct. Instead models, such as Llama-3-8b, Mistral-7B-Instruct-v0.3, are generation models rather than an embedding model
