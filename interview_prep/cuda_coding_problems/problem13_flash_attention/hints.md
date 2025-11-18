# Hints: Flash Attention

## Hint 1: Memory Problem
Standard: O(seq_len²) memory for attention matrix
Flash: O(seq_len × d) by computing in blocks

## Hint 2: Online Softmax
Update max and sum incrementally:
```
new_max = max(old_max, new_score)
rescale = exp(old_max - new_max)
new_sum = old_sum * rescale + exp(new_score - new_max)
new_output = old_output * rescale + exp(new_score - new_max) * V
```

## Hint 3: Tiling Strategy
- Outer loop: Q blocks
- Inner loop: K, V blocks
- Compute partial attention, update running stats

## Hint 4: Key Insight
Don't store full attention matrix!
Compute and use immediately in tiles.
