# mypy fixes — vllm/reasoning

## step3_reasoning_parser.py

### Line 100 — Incompatible types in assignment (`None` to `str`)

`content` was inferred as `str` from the slice, then reassigned to `None`.

**Fix:** Annotate as `str | None` and use the `or None` idiom in a single expression.

```python
# before
content = model_output[end_index + len(self.think_end_token) :]
if len(content) == 0:
    content = None

# after
content: str | None = (
    model_output[end_index + len(self.think_end_token) :] or None
)
```

### Lines 111 & 114 — `int | None` used with `Iterable[int]` / `islice[int]`

`vocab.get()` returns `int | None`, so `self.think_end_token_id` was inferred as
`int | None` even though the constructor raises when it is `None`.

**Fix:** Narrow via a local variable before the guard, then annotate the attribute as `int`.

```python
# before
self.think_end_token_id = self.vocab.get(self.think_end_token)
if self.think_end_token_id is None:
    raise RuntimeError(...)

# after
think_end_token_id = self.vocab.get(self.think_end_token)
if think_end_token_id is None:
    raise RuntimeError(...)
self.think_end_token_id: int = think_end_token_id
```

---

## olmo3_reasoning_parser.py

### Line 246 — `Sequence[int]` passed to `decode` expecting `list[int] | int`

`TokenizerLike.decode` accepts `list[int] | int`. The method signature must stay
`Sequence[int]` to honour the base class contract.

**Fix:** Convert at the call site.

```python
# before
text = self.model_tokenizer.decode(input_ids)

# after
text = self.model_tokenizer.decode(list(input_ids))
```

---

## hunyuan_a13b_reasoning_parser.py

### Lines 68–69 — Empty list literals need type annotations

mypy cannot infer the element type of `[]` without an annotation.

**Fix:** Add explicit annotations.

```python
# before
self.buffered_text = []
self.buffered_ids = []

# after
self.buffered_text: list[str] = []
self.buffered_ids: list[int] = []
```

### Line 79 — Empty list literal needs type annotation

Same issue for `token_buffer`.

**Fix:**

```python
# before
self.token_buffer = []

# after
self.token_buffer: list[int] = []
```

---

## gptoss_reasoning_parser.py

### Line 81 — `TokenizerLike` has no attribute `vocab`

`.vocab` is not part of the `TokenizerLike` protocol. The base class `ReasoningParser`
already exposes a `vocab` property that wraps `get_vocab()` for exactly this reason.

**Fix:** Use the base class property instead of accessing `.vocab` directly on the tokenizer.

```python
# before
self.eom_token_id = self.model_tokenizer.vocab["<|end|>"]

# after
self.eom_token_id = self.vocab["<|end|>"]
```

---

## basic_parsers.py

### Line 84 — `int | None` used with `Iterable[int]`

Same pattern as `step3_reasoning_parser.py` — `vocab.get()` returns `int | None`,
and the `None` guard does not narrow the attribute type.

**Fix:** Narrow via local variables, then annotate both attributes as `int`.

```python
# before
self.start_token_id = self.vocab.get(self.start_token)
self.end_token_id = self.vocab.get(self.end_token)
if self.start_token_id is None or self.end_token_id is None:
    raise RuntimeError(...)

# after
start_token_id = self.vocab.get(self.start_token)
end_token_id = self.vocab.get(self.end_token)
if start_token_id is None or end_token_id is None:
    raise RuntimeError(...)
self.start_token_id: int = start_token_id
self.end_token_id: int = end_token_id
```

---

## step3p5_reasoning_parser.py

### Line 53 — `Iterable[int]` passed to `_is_reasoning_end_from_ids` expecting `Sequence[int]`

`is_reasoning_end_streaming` receives `delta_ids: Iterable[int]` but
`_is_reasoning_end_from_ids` requires `Sequence[int]` (uses `len()` and index access).
The signature must stay `Iterable[int]` to honour the base class contract.

**Fix:** Convert at the call site.

```python
# before
return self._is_reasoning_end_from_ids(delta_ids)

# after
return self._is_reasoning_end_from_ids(list(delta_ids))
```

---

## kimi_k2_reasoning_parser.py

### Line 45 — `None` assigned to `IdentityReasoningParser`

mypy inferred `_identity_parser` as `IdentityReasoningParser` from the `if` branch,
then rejected `None` in the `else` branch.

**Fix:** Add an explicit `IdentityReasoningParser | None` annotation before the `if`.

```python
# before
if not thinking:
    self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
else:
    self._identity_parser = None

# after
self._identity_parser: IdentityReasoningParser | None
if not thinking:
    self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
else:
    self._identity_parser = None
```

### Line 79 — `IdentityReasoningParser | None` has no attribute (5 call sites)

mypy cannot narrow `self._identity_parser` through a helper method call
(`_is_identity_mode()`), only through direct `is not None` checks.

**Fix:** Replace all `if self._is_identity_mode():` with `if self._identity_parser is not None:`
at all 5 call sites. mypy then narrows the type to `IdentityReasoningParser` inside
each block. The `_is_identity_mode()` helper becomes unused and can be removed.

```python
# before
if self._is_identity_mode():
    return self._identity_parser.is_reasoning_end(input_ids)

# after
if self._identity_parser is not None:
    return self._identity_parser.is_reasoning_end(input_ids)
```
