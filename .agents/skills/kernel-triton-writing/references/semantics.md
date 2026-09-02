<!--
SPDX-License-Identifier: Apache-2.0
The initial version was copied from NVIDIA TensorRT-LLM at commit
395985c025c8d1cf5aa842bc752b337ba88721b6 and substantially rewritten for vLLM.
-->

# Triton Semantics That Affect Correctness

Use the installed Triton API and the
[official language semantics](https://triton-lang.org/main/python-api/triton-semantics.html)
as the authority. These reminders highlight common porting hazards.

## Programs, shapes, and memory

A Triton program operates on blocks of values. The programmer still controls
the launch grid, block shapes, pointer arithmetic, masks, and access pattern;
the compiler does not make an arbitrary layout coalesced or race-free.

Broadcasting follows documented tensor-shape rules. Make dimensions explicit
with operations such as `[:, None]` and `[None, :]`, and ensure masks broadcast
to the corresponding pointer block. Do not assume Triton tensors are limited
to two dimensions.

Masked loads require an `other` value when masked lanes can participate in
later computation. Choose a value that is neutral for the operation. Masked
stores are still required at output boundaries.

`tl.where` evaluates both branches. It selects values; it does not guard an
otherwise-invalid load or store.

## Numeric behavior

Use the documented semantics of each operation rather than one global
promotion rule. In particular, reductions, dot products, transcendental
functions, and stores can have different conversion or precision behavior.
Specify accumulator or input precision when the operator contract requires it,
and make the reference use a comparable precision mode.

Python scalars and Triton tensors do not always promote like PyTorch tensors.
If promotion affects range or precision, cast deliberately and cover the case
with a focused test.

## Signed integer division

For integer division and remainder involving Triton tensor operands, Triton
uses C-style truncation toward zero. Python's `//` instead rounds toward
negative infinity. The difference matters only when an operand can be negative.
Keep offsets non-negative when possible, or implement and test the intended
floor-division or modulo operation explicitly.

For example, with a positive divisor, normalizing a remainder can be expressed
as `(value % divisor + divisor) % divisor`. Verify signed edge cases against the
intended public semantics rather than assuming translated Python expressions
behave identically.
