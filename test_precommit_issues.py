# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This file is intentionally created to test pre-commit checks
# It contains various issues that should trigger pre-commit hooks

import os,sys
import json
from typing import *

def bad_function(  x,y  ):
    """This function has bad formatting and typos"""
    # Bad spacing and formatting
    result=  x+y
    if result>10:
        print("The resutl is greater than ten")
    return result

def another_bad_function():
    # Missing type hints
    data = {"key": "value", "number": 42}
    for key, value in data.items():
        print(f"{key}: {value}")
    
    # Bad import style
    import re
    pattern = re.compile(r'\d+')
    return pattern

class BadClass:
    def __init__(self):
        self.value = None
    
    def method_with_issues(self):
        # More formatting issues
        x=1
        y=2
        z=x+y
        return z

# Bad spacing and formatting at module level
x=1
y=2
z=x+y

# This will cause issues with various linters
def unused_function():
    pass

# Bad string formatting
message = "Hello" + " " + "World"

# This will trigger the regex import check
import re
def regex_function():
    pattern = re.compile(r'\d+')
    return pattern

# This will trigger the triton import check
import triton

# This will trigger the pickle import check
import pickle
import cloudpickle
