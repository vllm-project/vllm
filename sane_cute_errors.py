#!/usr/bin/env python

# Usage:
#  ```
#  python setup.py build_ext --inplace | tee compile_log.txt
#  cat compile_log.txt | python sane_cute_errors.py
#  ```

import sys
import regex
from colorama import Fore

def _loop_replace(replace_fn, input_str, *args, **kwargs):
    new_string, count = replace_fn(input_str, *args, **kwargs)
    while count > 0:
        new_string, count = replace_fn(new_string, *args, **kwargs)
    return new_string

def replace_delimited_substring(input_str, start_delim, end_delim, replace_fn, prefix=""):    
    start_delim = regex.escape(start_delim)
    end_delim = regex.escape(end_delim)
    rx = f'{prefix}({start_delim}((?:(?!{start_delim}|{end_delim}).|(?1))*){end_delim})'
    return regex.subn(rx, lambda x: replace_fn(x.group(2)), input_str)

def replace_all_delimited_substrings(input_str, start_delim, end_delim, replace_fn, prefix=""):
    return _loop_replace(replace_delimited_substring, input_str, start_delim, end_delim, replace_fn, prefix=prefix)

def replace_delimiters(input_str, start_delim, end_delim, new_start, new_end, prefix=""):
    start_delim = regex.escape(start_delim)
    end_delim = regex.escape(end_delim)
    rx = f'{prefix}({start_delim}((?:(?!{start_delim}|{end_delim}).|(?1))*){end_delim})'
    return regex.subn(rx, f"{new_start}\\2{new_end}", input_str)

def replace_all_delimiters(input_str, start_delim, end_delim, new_start, new_end, prefix=""):
    return _loop_replace(replace_delimiters, input_str, start_delim, end_delim, new_start, new_end, prefix=prefix)

def replace(input_str, old, new):
    return regex.subn(old, new, input_str)

def replace_all(input_str, old, new):
    return _loop_replace(replace, input_str, old, new)
    
def sepreate_at_line_of(input_str):
    return regex.sub(r"at line (\d+) of ([^\n\r]*)", f"\n\t\tat {Fore.GREEN}\\2:\\1{Fore.RESET}", input_str)

def break_apart_instantiation_of(input_str):
    def replace_fn(x):
        def replace_fn_inner(x):
            x = regex.sub(r"([^\s=]+=)", r"\n\t\t  \1", x)
            return x
        x = regex.sub(r"(at line)", r"\n\t\t\1", x)
        y, _ = replace_delimited_substring(x, "[", "]", replace_fn_inner)
        return "instantiation of " + regex.sub(r"([^(]*)", f"{Fore.MAGENTA}\\1{Fore.RESET}", y, count=1)
    
    return replace_all_delimited_substrings(input_str, "\"", "\"", replace_fn, prefix=r"instantiation of ")

def template_replace_commas_at_depth_0(x, new_char):
    brace_stack = []
    brace_pairs = { "(": ")", "[": "]", "{": "}", "<": ">" }
    replaced_comma = False
    
    for idx in range(len(x)):
        if x[idx] in brace_pairs:
            brace_stack.append(x[idx])
        elif len(brace_stack) > 0 and x[idx] == brace_pairs[brace_stack[-1]]:
            brace_stack.pop()
        if len(brace_stack) == 0 and x[idx] == ",":
            x = x[:idx] + new_char + x[idx+1:]
            replaced_comma = True
    return x, replaced_comma


def replace_layout_commas(x):
    def replace_commas_inner(x):
        x, replaced = template_replace_commas_at_depth_0(x, new_char=" :")
        if not replaced:
            x, _ = replace_delimiters(x, "<", ">", "", "", prefix="cute::tuple")
            x, replaced = template_replace_commas_at_depth_0(x, new_char=" :")
        assert replaced == True
        return f"{Fore.BLUE}{x}{Fore.RESET}"
    
    x, _ = replace_delimited_substring(x, "<", ">", replace_commas_inner, prefix="cute::Layout")
    return x

def replace_composed_layout_commas(x):
    def replace_commas_inner(x):
        x, replaced = template_replace_commas_at_depth_0(x, new_char=" o")
        assert replaced == True
        return x
    
    x, _ = replace_delimited_substring(x, "<", ">", replace_commas_inner, prefix="cute::ComposedLayout")
    return x

def clean_up_log(log):
    new_str = sepreate_at_line_of(log)
    new_str = break_apart_instantiation_of(new_str)
    new_str = replace_layout_commas(new_str)
    new_str = replace_composed_layout_commas(new_str)
    new_str = replace_all_delimiters(new_str, "<", ">", "(", ")", prefix="cute::tuple")
    new_str = replace_all_delimiters(new_str, "<", ">", "S<", ">", prefix="cute::Swizzle")
    new_str = replace_all(new_str, r"cute::C<(\d+)>", r"_\1")
    new_str = replace_all(new_str, r"cute::_(\d+)", r"_\1")
    new_str = replace_all(new_str, r"cute::Underscore", r"_")

    template_type_abbreviations = (
        ("cute::ScaledBasis", "SB"),
        ("cute::Tensor", "T"),
        ("cute::ArithmeticTuple", "AT"),
        ("cute::ArithmeticTupleIterator", "ATI"),
        ("cute::ViewEngine", "VE")
    )

    for template_type, abrv in template_type_abbreviations:
        new_str = replace_all(new_str, template_type + "<", abrv + "<")
    print(new_str)


data = sys.stdin.read()
clean_up_log(data)