# SPDX-License-Identifier: Apache-2.0

import os

zero_overhead = os.environ.get('VLLM_ZERO_OVERHEAD') == '1'
disable_auto_finish_thread = os.environ.get(
    'VLLM_ZERO_DISABLE_AUTO_THREAD') == '1'
zero_no_thread = os.environ.get('VLLM_ZERO_NO_THREAD') == '1'


def is_zero_overhead():
    return zero_overhead


def is_zero_auto_thread():
    return (not disable_auto_finish_thread) and zero_overhead and (
        not zero_no_thread)


def is_zero_no_thread():
    return zero_no_thread and zero_overhead
