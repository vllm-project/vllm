# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for documentation encoding fixes.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from docs.mkdocs.hooks.generate_examples import Example


class TestDocumentationEncoding:
    """Test documentation generation with Unicode content."""
    
    def test_example_with_unicode_content(self):
        """Test that Example can handle files with Unicode characters."""
        # Create a temporary markdown file with Unicode content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            # Include the problematic fullwidth vertical bar character
            unicode_content = "# Test Title with Unicode ｜ Characters\n\nThis is a test file with Unicode content."
            f.write(unicode_content)
            temp_path = Path(f.name)
        
        try:
            # Create an Example instance
            example = Example(temp_path, "test_category")
            
            # Test that title determination works with Unicode
            assert "Test Title with Unicode ｜ Characters" in example.title
            
            # Test that content generation works
            generated_content = example.generate()
            assert "Test Title with Unicode ｜ Characters" in generated_content
            assert "｜" in generated_content  # Ensure Unicode character is preserved
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_example_with_code_file_unicode(self):
        """Test that Example can handle code files with Unicode in comments."""
        # Create a temporary Python file with Unicode content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            unicode_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script with Unicode characters: ｜ ← → ↑ ↓
This tests encoding handling in documentation generation.
"""

def main():
    print("Hello, World! ｜ Unicode test")
    # Comment with Unicode: ｜ ← → ↑ ↓

if __name__ == "__main__":
    main()
'''
            f.write(unicode_content)
            temp_path = Path(f.name)
        
        try:
            # Create an Example instance
            example = Example(temp_path, "test_category")
            
            # Test that content generation works with Unicode in code files
            generated_content = example.generate()
            assert "｜" in generated_content or "Unicode test" in generated_content
            
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_file_writing_with_unicode(self):
        """Test that file writing preserves Unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test markdown file with Unicode
            test_file = temp_path / "test.md"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("# Test ｜ Unicode\n\nContent with Unicode: ← → ↑ ↓")
            
            # Create an Example and generate content
            example = Example(test_file, "test")
            generated_content = example.generate()
            
            # Write the generated content to a new file (simulating the docs generation)
            output_file = temp_path / "output.md"
            with open(output_file, 'w+', encoding='utf-8') as f:
                f.write(generated_content)
            
            # Read back and verify Unicode is preserved
            with open(output_file, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            assert "｜" in read_content
            assert "← → ↑ ↓" in read_content
    
    def test_determine_title_with_unicode(self):
        """Test title determination with Unicode characters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            # Test various Unicode scenarios
            test_cases = [
                "# Simple Title",
                "# Title with ｜ fullwidth bar",
                "# Title with arrows ← → ↑ ↓",
                "# Title with emoji 🚀 and symbols ★",
                "# 中文标题 Chinese Title",
                "# العنوان العربي Arabic Title",
            ]
            
            for title_line in test_cases:
                f.seek(0)
                f.truncate()
                f.write(f"{title_line}\n\nContent here.")
                f.flush()
                
                temp_path = Path(f.name)
                example = Example(temp_path, "test")
                
                # Extract expected title (remove the "# " prefix)
                expected_title = title_line[2:]
                assert example.title == expected_title
        
        # Clean up
        temp_path.unlink()
    
    def test_windows_specific_encoding_issue(self):
        """Test the specific Windows encoding issue mentioned in the bug report."""
        # Simulate the problematic character from the error message
        problematic_char = '\uff5c'  # Fullwidth vertical bar
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            content = f"# Title with problematic char {problematic_char}\n\nContent here."
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            example = Example(temp_path, "test")
            
            # This should not raise UnicodeEncodeError
            title = example.title
            assert problematic_char in title
            
            generated_content = example.generate()
            assert problematic_char in generated_content
            
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
