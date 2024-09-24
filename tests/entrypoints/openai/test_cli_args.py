import json
import unittest

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser

LORA_MODULE = {
    "name": "module2",
    "path": "/path/to/module2",
    "base_model_name": "llama"
}


class TestLoraParserAction(unittest.TestCase):

    def setUp(self):
        # Setting up argparse parser for tests
        parser = FlexibleArgumentParser(
            description="vLLM's remote OpenAI server.")
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
            json.dumps(LORA_MODULE),
        ])
        expected = [
            LoRAModulePath(name='module2',
                           path='/path/to/module2',
                           base_model_name='llama')
        ]
        self.assertEqual(args.lora_modules, expected)

    def test_invalid_json_format(self):
        # Test invalid JSON format input, missing closing brace
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                '--lora-modules',
                '{"name": "module3", "path": "/path/to/module3"'
            ])

    def test_invalid_type_error(self):
        # Test type error when values are not JSON or key=value
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                '--lora-modules',
                'invalid_format'  # This is not JSON or key=value format
            ])

    def test_invalid_json_field(self):
        # Test valid JSON format but missing required fields
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                '--lora-modules',
                '{"name": "module4"}'  # Missing required 'path' field
            ])

    def test_empty_values(self):
        # Test when no LoRA modules are provided
        args = self.parser.parse_args(['--lora-modules', ''])
        self.assertEqual(args.lora_modules, [])

    def test_multiple_valid_inputs(self):
        # Test multiple valid inputs (both old and JSON format)
        args = self.parser.parse_args([
            '--lora-modules',
            'module1=/path/to/module1',
            json.dumps(LORA_MODULE),
        ])
        expected = [
            LoRAModulePath(name='module1', path='/path/to/module1'),
            LoRAModulePath(name='module2',
                           path='/path/to/module2',
                           base_model_name='llama')
        ]
        self.assertEqual(args.lora_modules, expected)


if __name__ == '__main__':
    unittest.main()
