# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.exceptions import LoRAAdapterNotFoundError, VLLMValidationError


class TestVLLMValidationError:
    def test_message_only(self):
        err = VLLMValidationError("invalid request")
        assert str(err) == "invalid request"

    def test_with_parameter(self):
        err = VLLMValidationError("invalid request", parameter="temperature")
        assert str(err) == "invalid request (parameter=temperature)"

    def test_with_value(self):
        err = VLLMValidationError("invalid request", value=999)
        assert str(err) == "invalid request (value=999)"

    def test_with_parameter_and_value(self):
        err = VLLMValidationError("invalid request", parameter="temperature", value=999)
        assert str(err) == "invalid request (parameter=temperature, value=999)"

    def test_with_none_parameter(self):
        err = VLLMValidationError("invalid request", parameter=None)
        assert str(err) == "invalid request"

    def test_with_none_value(self):
        err = VLLMValidationError("invalid request", value=None)
        assert str(err) == "invalid request"

    def test_is_valueerror_subclass(self):
        err = VLLMValidationError("test")
        assert isinstance(err, ValueError)


class TestLoRAAdapterNotFoundError:
    def test_message_format(self):
        err = LoRAAdapterNotFoundError("my-adapter", "/path/to/adapter")
        assert "my-adapter" in str(err)
        assert "/path/to/adapter" in str(err)

    def test_message_attribute(self):
        err = LoRAAdapterNotFoundError("my-adapter", "/path/to/adapter")
        assert err.message == str(err)
