# SPDX-License-Identifier: Apache-2.0
import platform
import unittest
from typing import ClassVar

import torch

from vllm.platforms.cpu import CpuPlatform


class TestAppleSiliconBF16(unittest.TestCase):
    is_mac: ClassVar[bool]
    is_arm: ClassVar[bool]
    is_apple_silicon: ClassVar[bool]

    @classmethod
    def setUpClass(cls) -> None:
        cls.is_mac = platform.system() == "Darwin"
        cls.is_arm = platform.machine() == "arm64"
        cls.is_apple_silicon = cls.is_mac and cls.is_arm

        if not cls.is_apple_silicon:
            print("Skipping tests: Not running on Apple Silicon")
            return

        if not torch.backends.mps.is_available():
            print("Skipping tests: MPS backend not available")
            return

    def test_platform_detection(self):
        """Test if we correctly detect Apple Silicon platform"""
        self.assertEqual(self.is_mac, platform.system() == "Darwin")
        self.assertEqual(self.is_arm, platform.machine() == "arm64")
        self.assertEqual(self.is_apple_silicon, self.is_mac and self.is_arm)

    def test_bf16_support_detection(self) -> None:
        """Test if bf16 support is correctly detected"""
        if not self.is_apple_silicon:
            self.skipTest("Test only runs on Apple Silicon")

        has_support = CpuPlatform._bf16_support_mac()
        print(f"BF16 support detected: {has_support}")

        # Test PyTorch MPS backend
        self.assertTrue(torch.backends.mps.is_available(),
                        "MPS backend should be available on Apple Silicon")

        # Test bf16 tensor creation if support is detected
        if has_support:
            try:
                tensor = torch.zeros(1, dtype=torch.bfloat16, device="mps")
                self.assertEqual(tensor.dtype, torch.bfloat16)
                self.assertEqual(tensor.device.type, "mps")
                print("Successfully created bf16 tensor on MPS device")
            except Exception as e:
                self.fail(f"Failed to create bf16 tensor: {str(e)}")

    def test_supported_dtypes(self) -> None:
        """Test if supported dtypes are correctly reported"""
        if not self.is_apple_silicon:
            self.skipTest("Test only runs on Apple Silicon")

        platform = CpuPlatform()
        dtypes = platform.supported_dtypes

        # Should always support float16 and float32
        self.assertIn(torch.float16, dtypes)
        self.assertIn(torch.float32, dtypes)

        # Check bf16 support
        has_bf16 = CpuPlatform._bf16_support_mac()
        if has_bf16:
            self.assertIn(torch.bfloat16, dtypes)
        else:
            self.assertNotIn(torch.bfloat16, dtypes)


if __name__ == "__main__":
    unittest.main()
