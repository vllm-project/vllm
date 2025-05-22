# SPDX-License-Identifier: Apache-2.0
import platform
import unittest

import torch

from vllm.platforms.cpu import CpuPlatform


class TestAppleSiliconBF16(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.is_mac = platform.system() == "Darwin"
        cls.is_arm = platform.processor().startswith("arm")
        cls.is_apple_silicon = cls.is_mac and cls.is_arm

    def test_platform_detection(self):
        """Test if we correctly detect Apple Silicon platform"""
        self.assertEqual(self.is_mac, platform.system() == "Darwin")
        self.assertEqual(self.is_arm, platform.processor().startswith("arm"))
        self.assertEqual(self.is_apple_silicon, self.is_mac and self.is_arm)

    def test_bf16_support_detection(self):
        """Test if bf16 support is correctly detected"""
        if not self.is_apple_silicon:
            self.skipTest("Test only runs on Apple Silicon")

        has_support = CpuPlatform.bf16_support_mac()
        print(f"BF16 support detected: {has_support}")

        # Test PyTorch MPS backend
        self.assertTrue(torch.backends.mps.is_available(),
                        "MPS backend should be available on Apple Silicon")

        # Test bf16 tensor creation if support is detected
        if has_support:
            try:
                x = torch.zeros(1, dtype=torch.bfloat16, device="mps")
                self.assertEqual(x.dtype, torch.bfloat16)
                print("Successfully created bf16 tensor on MPS device")
            except Exception as e:
                self.fail(f"Failed to create bf16 tensor: {str(e)}")

    def test_supported_dtypes(self):
        """Test if supported dtypes are correctly reported"""
        if not self.is_apple_silicon:
            self.skipTest("Test only runs on Apple Silicon")

        platform = CpuPlatform()
        dtypes = platform.supported_dtypes

        # Should always support float16 and float32
        self.assertIn(torch.float16, dtypes)
        self.assertIn(torch.float32, dtypes)

        # Check bf16 support
        has_bf16 = CpuPlatform.bf16_support_mac()
        if has_bf16:
            self.assertIn(torch.bfloat16, dtypes)
        else:
            self.assertNotIn(torch.bfloat16, dtypes)


if __name__ == "__main__":
    unittest.main()
