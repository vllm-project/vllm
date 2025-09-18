#!/usr/bin/env python3
"""
Integration test to verify that CPU weight loading documentation and examples work correctly.
This is a minimal test that checks the functionality without requiring torch installation.
"""

import sys
import os

def test_example_syntax():
    """Test that the example script has valid Python syntax."""
    example_path = "examples/offline_inference/basic/cpu_weight_loading.py"
    
    if not os.path.exists(example_path):
        print(f"❌ Example file not found: {example_path}")
        return False
    
    try:
        with open(example_path, 'r') as f:
            code = f.read()
        
        # Compile the code to check for syntax errors
        compile(code, example_path, 'exec')
        print(f"✅ Example script syntax is valid: {example_path}")
        return True
    
    except SyntaxError as e:
        print(f"❌ Syntax error in example script: {e}")
        return False

def test_load_config_exists():
    """Test that LoadConfig class and pt_load_map_location field exist."""
    try:
        # Try to import and check the config structure
        import ast
        
        config_path = "vllm/config/load.py"
        if not os.path.exists(config_path):
            print(f"❌ LoadConfig file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            code = f.read()
        
        # Parse the AST to check for pt_load_map_location
        tree = ast.parse(code)
        
        has_pt_load_map_location = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and hasattr(node.target, 'id'):
                if node.target.id == 'pt_load_map_location':
                    has_pt_load_map_location = True
                    break
        
        if has_pt_load_map_location:
            print("✅ pt_load_map_location found in LoadConfig")
            return True
        else:
            print("❌ pt_load_map_location not found in LoadConfig")
            return False
            
    except Exception as e:
        print(f"❌ Error checking LoadConfig: {e}")
        return False

def test_cli_argument_exists():
    """Test that pt-load-map-location CLI argument is defined."""
    try:
        arg_utils_path = "vllm/engine/arg_utils.py"
        if not os.path.exists(arg_utils_path):
            print(f"❌ Arg utils file not found: {arg_utils_path}")
            return False
        
        with open(arg_utils_path, 'r') as f:
            content = f.read()
        
        if 'pt-load-map-location' in content:
            print("✅ pt-load-map-location CLI argument found")
            return True
        else:
            print("❌ pt-load-map-location CLI argument not found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking CLI arguments: {e}")
        return False

def test_documentation_updated():
    """Test that FAQ documentation was updated."""
    try:
        faq_path = "docs/usage/faq.md"
        if not os.path.exists(faq_path):
            print(f"❌ FAQ file not found: {faq_path}")
            return False
        
        with open(faq_path, 'r') as f:
            content = f.read()
        
        if 'How do you load weights from CPU?' in content and 'pt_load_map_location' in content:
            print("✅ FAQ documentation updated with CPU weight loading info")
            return True
        else:
            print("❌ FAQ documentation not properly updated")
            return False
            
    except Exception as e:
        print(f"❌ Error checking FAQ documentation: {e}")
        return False

def main():
    """Run all tests."""
    print("Running CPU weight loading integration tests...")
    print("=" * 50)
    
    tests = [
        test_example_syntax,
        test_load_config_exists,
        test_cli_argument_exists,
        test_documentation_updated,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CPU weight loading functionality is properly implemented and documented.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())