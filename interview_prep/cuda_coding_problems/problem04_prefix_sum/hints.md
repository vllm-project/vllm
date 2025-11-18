# Hints: Prefix Sum

## Hint 1: Blelloch Algorithm Overview
Two phases:
1. **Up-sweep:** Build reduction tree (like parallel reduction)
2. **Down-sweep:** Propagate partial sums back down

## Hint 2: Up-sweep Pattern
```cuda
for (int d = n/2; d > 0; d >>= 1) {
    if (tid < d) {
        int ai = offset * (2*tid + 1) - 1;
        int bi = offset * (2*tid + 2) - 1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
}
```

## Hint 3: Down-sweep Pattern
Set temp[n-1] = 0, then:
```cuda
for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    if (tid < d) {
        int ai = offset * (2*tid + 1) - 1;
        int bi = offset * (2*tid + 2) - 1;
        float t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
}
```

## Hint 4: Multi-block Strategy
1. Scan each block independently
2. Collect block sums
3. Scan block sums
4. Add scanned block sums to each block

## Hint 5: Inclusive vs Exclusive
Exclusive: Use output of algorithm directly
Inclusive: Add original input to exclusive scan result
