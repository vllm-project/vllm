# [Problem ID]: [Problem Title]

<!--
INSTRUCTIONS FOR USING THIS TEMPLATE:
1. Replace all [bracketed] sections with actual content
2. Ensure problem statement is clear and unambiguous
3. Provide diverse test cases including edge cases
4. Include multiple solution approaches when applicable
5. Add complexity analysis for each solution
6. Test all solutions thoroughly
-->

<!--
METADATA:
Difficulty: [Easy/Medium/Hard]
Category: [Array/String/Tree/Graph/DP/etc.]
Tags: [tag1, tag2, tag3]
Companies: [Company1, Company2] (if interview question)
Similar Problems: [Links to similar problems]
Estimated Time: [X] minutes
-->

## Problem Statement

[Clear, concise problem statement. Be precise and unambiguous.]

[Additional context or background if needed.]

---

## Constraints

- [Constraint 1: e.g., 1 ≤ n ≤ 10^5]
- [Constraint 2: e.g., -10^9 ≤ arr[i] ≤ 10^9]
- [Constraint 3: e.g., All elements are unique]
- [Constraint 4: e.g., Input array is sorted]

---

## Examples

### Example 1

**Input:**
```
[input format]
```

**Output:**
```
[output format]
```

**Explanation:**
[Step-by-step explanation of how to get from input to output]

---

### Example 2

**Input:**
```
[input format]
```

**Output:**
```
[output format]
```

**Explanation:**
[Step-by-step explanation]

---

### Example 3 (Edge Case)

**Input:**
```
[edge case input]
```

**Output:**
```
[output]
```

**Explanation:**
[Why this edge case is important]

---

## Test Cases

### Basic Test Cases

```python
# Test Case 1: [Description]
input_1 = [test input]
expected_output_1 = [expected output]

# Test Case 2: [Description]
input_2 = [test input]
expected_output_2 = [expected output]

# Test Case 3: [Description]
input_3 = [test input]
expected_output_3 = [expected output]
```

### Edge Cases

```python
# Edge Case 1: [Description - e.g., Empty input]
input_edge_1 = [test input]
expected_output_edge_1 = [expected output]

# Edge Case 2: [Description - e.g., Single element]
input_edge_2 = [test input]
expected_output_edge_2 = [expected output]

# Edge Case 3: [Description - e.g., Maximum size]
input_edge_3 = [test input]
expected_output_edge_3 = [expected output]

# Edge Case 4: [Description - e.g., All same elements]
input_edge_4 = [test input]
expected_output_edge_4 = [expected output]
```

### Large Test Cases

```python
# Large Test Case 1: [Description - e.g., Maximum constraints]
# input_large_1 = [description of how to generate]
# expected_output_large_1 = [expected output]
```

---

## Approach Overview

Before diving into solutions, consider these questions:

1. **What data structure would be most appropriate?**
   - [Consideration 1]
   - [Consideration 2]

2. **What are the key observations?**
   - [Observation 1]
   - [Observation 2]

3. **What are the tradeoffs between different approaches?**
   - [Tradeoff 1]
   - [Tradeoff 2]

---

## Solution 1: [Brute Force / Naive Approach]

### Intuition

[Explain the most straightforward approach, even if not optimal]

### Algorithm

1. [Step 1]
2. [Step 2]
3. [Step 3]
4. [Step 4]

### Implementation

```python
def solution_brute_force(input_params):
    """
    Brute force solution.

    Args:
        [param1] ([type]): [description]

    Returns:
        [type]: [description]

    Time Complexity: O([complexity])
    Space Complexity: O([complexity])
    """
    # Implementation with detailed comments

    # Step 1: [Description]
    [code for step 1]

    # Step 2: [Description]
    [code for step 2]

    # Step 3: [Description]
    [code for step 3]

    return result


# Test the solution
if __name__ == "__main__":
    # Test Case 1
    result = solution_brute_force([test_input])
    assert result == [expected_output], f"Expected {[expected_output]}, got {result}"
    print(f"✓ Test 1 passed")

    # Test Case 2
    result = solution_brute_force([test_input])
    assert result == [expected_output], f"Expected {[expected_output]}, got {result}"
    print(f"✓ Test 2 passed")
```

### Complexity Analysis

**Time Complexity:** O([complexity])
- [Explanation of why this complexity]
- [Break down by operation]

**Space Complexity:** O([complexity])
- [Explanation of space usage]
- [Auxiliary space breakdown]

### Pros and Cons

**Pros:**
- ✓ [Pro 1]
- ✓ [Pro 2]

**Cons:**
- ✗ [Con 1]
- ✗ [Con 2]

---

## Solution 2: [Optimized Approach]

### Intuition

[Explain the key insight that leads to optimization]

### Key Observation

[Critical observation that enables better solution]

### Algorithm

1. [Step 1]
2. [Step 2]
3. [Step 3]
4. [Step 4]

### Visualization

```
[Visual representation of the algorithm]
Step 1:
[diagram]

Step 2:
[diagram]

Step 3:
[diagram]
```

### Implementation

```python
def solution_optimized(input_params):
    """
    Optimized solution using [technique/data structure].

    Args:
        [param1] ([type]): [description]

    Returns:
        [type]: [description]

    Time Complexity: O([complexity])
    Space Complexity: O([complexity])
    """
    # Implementation with detailed comments

    # Initialize data structures
    [initialization code]

    # Main algorithm
    [main logic with explanatory comments]

    return result


# Test the solution
if __name__ == "__main__":
    # Test all cases
    test_cases = [
        ([input1], [expected1]),
        ([input2], [expected2]),
        ([input3], [expected3]),
    ]

    for i, (input_data, expected) in enumerate(test_cases, 1):
        result = solution_optimized(input_data)
        assert result == expected, f"Test {i} failed: expected {expected}, got {result}"
        print(f"✓ Test {i} passed")
```

### Complexity Analysis

**Time Complexity:** O([complexity])
- [Detailed explanation]
- [Best case]
- [Average case]
- [Worst case]

**Space Complexity:** O([complexity])
- [Detailed explanation]

### Pros and Cons

**Pros:**
- ✓ [Pro 1]
- ✓ [Pro 2]

**Cons:**
- ✗ [Con 1]
- ✗ [Con 2]

---

## Solution 3: [Most Optimal Approach] (If applicable)

### Intuition

[Explain the ultimate optimization]

### Algorithm

[Detailed algorithm steps]

### Implementation

```python
def solution_optimal(input_params):
    """
    Most optimal solution.

    Args:
        [param1] ([type]): [description]

    Returns:
        [type]: [description]

    Time Complexity: O([complexity])
    Space Complexity: O([complexity])
    """
    # Highly optimized implementation

    return result
```

### Complexity Analysis

**Time Complexity:** O([complexity])
**Space Complexity:** O([complexity])

---

## Follow-Up Questions

### Follow-Up 1: [Variation of the problem]

**Question:** [Modified problem statement]

**Approach:** [How to adapt the solution]

<details>
<summary>Solution</summary>

```python
def follow_up_1(input_params):
    """Solution to follow-up question 1."""
    # Implementation
    pass
```

</details>

---

### Follow-Up 2: [Another variation]

**Question:** [Modified problem statement]

**Approach:** [How to adapt the solution]

<details>
<summary>Solution</summary>

```python
def follow_up_2(input_params):
    """Solution to follow-up question 2."""
    # Implementation
    pass
```

</details>

---

## Interview Tips

### What Interviewers Look For

1. **Problem Understanding:**
   - [ ] Clarify ambiguities before coding
   - [ ] Ask about edge cases
   - [ ] Confirm input/output format

2. **Communication:**
   - [ ] Explain your thought process
   - [ ] Discuss tradeoffs
   - [ ] Walk through examples

3. **Coding:**
   - [ ] Write clean, readable code
   - [ ] Handle edge cases
   - [ ] Use meaningful variable names
   - [ ] Add comments for complex logic

4. **Testing:**
   - [ ] Test with provided examples
   - [ ] Consider edge cases
   - [ ] Verify time/space complexity

### Common Mistakes to Avoid

1. **[Mistake 1]:** [Description and how to avoid]
2. **[Mistake 2]:** [Description and how to avoid]
3. **[Mistake 3]:** [Description and how to avoid]

### Questions to Ask

Before coding, consider asking:

1. [Clarifying question 1]?
2. [Clarifying question 2]?
3. [Clarifying question 3]?

---

## Practice Plan

### Step 1: Understand (5-10 minutes)

- [ ] Read problem carefully
- [ ] Identify inputs and outputs
- [ ] Note all constraints
- [ ] Work through examples manually

### Step 2: Plan (10-15 minutes)

- [ ] Identify possible approaches
- [ ] Analyze time/space complexity
- [ ] Choose best approach
- [ ] Outline algorithm steps

### Step 3: Implement (20-30 minutes)

- [ ] Write code with comments
- [ ] Handle edge cases
- [ ] Keep code clean and organized

### Step 4: Test (10-15 minutes)

- [ ] Run provided examples
- [ ] Test edge cases
- [ ] Verify complexity matches analysis

### Step 5: Optimize (If time permits)

- [ ] Review for optimizations
- [ ] Consider alternative approaches
- [ ] Refactor if needed

---

## Related Problems

### Similar Difficulty
- [Problem 1](link): [Brief description]
- [Problem 2](link): [Brief description]

### Prerequisite Problems
- [Problem 1](link): [What concept it teaches]
- [Problem 2](link): [What concept it teaches]

### Advanced Problems
- [Problem 1](link): [How it extends this problem]
- [Problem 2](link): [How it extends this problem]

---

## Key Concepts

This problem tests understanding of:

- **[Concept 1]:** [Brief explanation]
- **[Concept 2]:** [Brief explanation]
- **[Concept 3]:** [Brief explanation]
- **[Data Structure]:** [When and why to use it]
- **[Algorithm]:** [When and why to use it]

---

## Complete Solution Template

```python
"""
Problem: [Problem Title]
Difficulty: [Level]
Category: [Category]
"""

from typing import [Type hints as needed]


class Solution:
    """Solution class for [Problem Title]."""

    def solve(self, input_params):
        """
        [Choose the best approach for interview]

        Args:
            [param1] ([type]): [description]

        Returns:
            [type]: [description]

        Time Complexity: O([complexity])
        Space Complexity: O([complexity])
        """
        # Your implementation here
        pass


def test_solution():
    """Test function with multiple test cases."""
    sol = Solution()

    # Test Case 1
    assert sol.solve([input]) == [expected], "Test 1 failed"
    print("✓ Test 1 passed")

    # Test Case 2
    assert sol.solve([input]) == [expected], "Test 2 failed"
    print("✓ Test 2 passed")

    # Edge Case 1
    assert sol.solve([input]) == [expected], "Edge case 1 failed"
    print("✓ Edge case 1 passed")

    # Edge Case 2
    assert sol.solve([input]) == [expected], "Edge case 2 failed"
    print("✓ Edge case 2 passed")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_solution()
```

---

## Hints

<details>
<summary>Hint 1: General direction</summary>
[High-level hint about approach]
</details>

<details>
<summary>Hint 2: Data structure</summary>
[Hint about what data structure to use]
</details>

<details>
<summary>Hint 3: Key insight</summary>
[More specific hint about the key observation]
</details>

<details>
<summary>Hint 4: Algorithm</summary>
[Hint about specific algorithm or technique]
</details>

---

## Additional Resources

### Tutorials
- [Tutorial 1](link): [What it covers]
- [Tutorial 2](link): [What it covers]

### Video Explanations
- [Video 1](link): [Creator, duration]
- [Video 2](link): [Creator, duration]

### Related Articles
- [Article 1](link): [Topic]
- [Article 2](link): [Topic]

---

## Notes

[Any additional notes, insights, or observations about this problem]

```markdown
## Personal Notes

### Key Insights
- [Insight 1]
- [Insight 2]

### Mistakes I Made
- [Mistake 1]: [How I fixed it]
- [Mistake 2]: [How I fixed it]

### Review Schedule
- [ ] First review: [Date]
- [ ] Second review: [Date]
- [ ] Third review: [Date]
```

---

**Problem Version:** 1.0
**Created:** [YYYY-MM-DD]
**Last Updated:** [YYYY-MM-DD]
**Source:** [Platform or book]
**Author:** [Author if applicable]
