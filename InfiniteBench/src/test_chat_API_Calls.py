# test_chat_API_Calls.py
# Test file for testing the integration of the LLM API calls with the Chat APIs.
#
# Usage:
# python -m unittest test_chat_API_Calls.py

import unittest

from LLM_API_Calls import (
    chat_with_openai,
    chat_with_anthropic,
    chat_with_cohere,
    chat_with_groq,
    chat_with_openrouter,
    chat_with_huggingface,
    chat_with_deepseek,
    chat_with_mistral
)
from eval_utils import load_and_log_configs


class TestLLMAPICallsIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_and_log_configs()
        if cls.config is None:
            raise ValueError("Failed to load configuration")

    def test_chat_with_openai(self):
        api_key = self.config['api_keys'].get('openai')
        model = self.config['services'].get('openai')
        if not api_key:
            self.skipTest("OpenAI API key not available")
        response = chat_with_openai(api_key, "Hello, how are you?", "Respond briefly", temp=0.7, system_message="You are a helpful assistant.")
        print("OpenAI Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_anthropic(self):
        api_key = self.config['api_keys'].get('anthropic')
        model = self.config['services'].get('anthropic')
        if not api_key:
            self.skipTest("Anthropic API key not available")
        response = chat_with_anthropic(api_key, "Hello, how are you?", model, "Respond briefly")
        print("Anthropic Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_cohere(self):
        api_key = self.config['api_keys'].get('cohere')
        model = self.config['services'].get('cohere')
        if not api_key:
            self.skipTest("Cohere API key not available")
        response = chat_with_cohere(api_key, "Hello, how are you?", model, "Respond briefly")
        print("Cohere Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_groq(self):
        api_key = self.config['api_keys'].get('groq')
        if not api_key:
            self.skipTest("Groq API key not available")
        response = chat_with_groq(api_key, "Hello, how are you?", "Respond briefly")
        print("Groq Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_openrouter(self):
        api_key = self.config['api_keys'].get('openrouter')
        if not api_key:
            self.skipTest("OpenRouter API key not available")
        response = chat_with_openrouter(api_key, "Hello, how are you?", "Respond briefly")
        print("OpenRouter Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_huggingface(self):
        api_key = self.config['api_keys'].get('huggingface')
        if not api_key:
            self.skipTest("HuggingFace API key not available")
        response = chat_with_huggingface(api_key, "Hello, how are you?", "Respond briefly")
        print("Huggingface Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_deepseek(self):
        api_key = self.config['api_keys'].get('deepseek')
        if not api_key:
            self.skipTest("DeepSeek API key not available")
        response = chat_with_deepseek(api_key, "Hello, how are you?", "Respond briefly")
        print("DeepSeek Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_mistral(self):
        api_key = self.config['api_keys'].get('mistral')
        if not api_key:
            self.skipTest("Mistral API key not available")
        response = chat_with_mistral(api_key, "Hello, how are you?", "Respond briefly")
        print("Mistral Response: " + response + "\n")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()