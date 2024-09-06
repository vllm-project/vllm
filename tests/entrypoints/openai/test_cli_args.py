import unittest
import argparse
import json
from typing import Optional, Union, Sequence, List
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


class TestLoraParserAction(unittest.TestCase):
    def setUp(self):
        # Setting up argparse parser for tests
        parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
        self.parser = make_arg_parser(parser)

    def test_valid_key_value_format(self):
        # Test old format: name=path
        args = self.parser.parse_args([
            '--lora-modules',
            'module1=/path/to/module1',
        ])
        expected = [LoRAModulePath(name='module1', path='/path/to/module1')]
        self.assertEqual(args.lora_modules, expected)

    def test_valid_json_format(self):
        # Test valid JSON format input
        args = self.parser.parse_args([
            '--lora-modules',
            '{"name": "module2", "path": "/path/to/module2", "base_model_name": "llama"}'
        ])
        expected = [LoRAModulePath(name='module2', path='/path/to/module2', base_model_name='llama')]
        self.assertEqual(args.lora_modules, expected)

    def test_invalid_json_format(self):
        # Test invalid JSON format, should be list
        with self.assertRaises(ValueError):
            invalid_json_input = '{"name": "module3", "path": "/path/to/module3"}'
            self.action(self.parser, self.namespace, [invalid_json_input])

    def test_invalid_type_error(self):
        # Test type error when values is not a list
        with self.assertRaises(ValueError):
            self.action(self.parser, self.namespace, 'this_should_be_a_list')

    def test_invalid_json_field(self):
        # Test valid JSON format but missing required fields
        with self.assertRaises(ValueError):
            invalid_field_input = '{"name": "module4"}'
            self.action(self.parser, self.namespace, [invalid_field_input])

    def test_empty_values(self):
        # Test when values are None
        with self.assertRaises(TypeError):
            self.action(self.parser, self.namespace, None)
            self.assertEqual(self.namespace.lora_modules, [])


if __name__ == '__main__':
    unittest.main()