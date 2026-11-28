import importlib.util
import os
import sys
import unittest

# Load module directly to avoid torch/cuda hardware dependencies during unit test
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../../vllm/entrypoints/openai/action_gate.py",
)
spec = importlib.util.spec_from_file_location("vllm_action_gate", file_path)
action_gate_mod = importlib.util.module_from_spec(spec)
sys.modules["vllm_action_gate"] = action_gate_mod
spec.loader.exec_module(action_gate_mod)

ActionGateInferenceGuardrail = action_gate_mod.ActionGateInferenceGuardrail
GENESIS_HASH = action_gate_mod.GENESIS_HASH


class TestActionGateInferenceGuardrail(unittest.TestCase):
    def setUp(self):
        self.guardrail = ActionGateInferenceGuardrail(
            never_equate_intent_to_approval=True,
            enforce_action_boundary=True,
            max_tokens_per_request=4096,
        )

    def test_verify_request_allowed(self):
        res = self.guardrail.verify_request(
            model="meta-llama/Llama-3-70b-Instruct",
            max_tokens=1024,
            tools=[{"name": "sql_query"}],
            user_id="usr_001",
        )
        self.assertTrue(res["allowed"])
        self.assertIn("hash", res)
        entries = self.guardrail.get_ledger_entries()
        self.assertEqual(len(entries), 1)

    def test_token_budget_exceeded_raises_error(self):
        with self.assertRaises(ValueError):
            self.guardrail.verify_request(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                max_tokens=8192,  # Exceeds max_tokens_per_request=4096
                user_id="usr_002",
            )

    def test_record_completion_and_hash_chain_integrity(self):
        # 1. Request
        self.guardrail.verify_request(
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=2048,
            user_id="usr_003",
        )
        # 2. Completion
        self.guardrail.record_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            completion_tokens=512,
            tool_calls=[{"id": "call_1", "name": "vector_search"}],
            user_id="usr_003",
        )

        entries = self.guardrail.get_ledger_entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertTrue(self.guardrail.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
