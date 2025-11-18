# Lab [X]: [Lab Title]

<!--
INSTRUCTIONS FOR USING THIS TEMPLATE:
1. Replace all [bracketed] sections with actual content
2. Ensure problem is clearly defined and unambiguous
3. Provide complete setup instructions
4. Include all necessary files and data
5. Define clear success criteria
6. Test all instructions before publishing
-->

<!--
METADATA:
Lab Number: [X]
Topic: [Main topic covered]
Difficulty: [Beginner/Intermediate/Advanced]
Estimated Time: [X hours Y minutes]
Prerequisites: [Required knowledge/completed labs]
Related Tutorials: [Links]
-->

## Overview

**Topic:** [Main topic or concept this lab covers]

**Scenario:**
[2-3 paragraph realistic scenario that provides context for the lab. Make it practical and motivating.]

**What You'll Build:**
[Specific description of the deliverable(s)]

---

## Learning Objectives

By completing this lab, you will:

- [ ] [Specific skill or knowledge outcome 1]
- [ ] [Specific skill or knowledge outcome 2]
- [ ] [Specific skill or knowledge outcome 3]
- [ ] [Specific skill or knowledge outcome 4]

**Key Technologies/Concepts:**
- [Technology/Concept 1]
- [Technology/Concept 2]
- [Technology/Concept 3]

---

## Prerequisites

### Required Knowledge

- **[Topic 1]:** [Level of understanding needed]
- **[Topic 2]:** [Level of understanding needed]
- **[Topic 3]:** [Level of understanding needed]

**Recommended Preparation:**
- Complete [Tutorial X](link)
- Review [Concept Y](link)
- Understand [Topic Z](link)

### Required Software

```bash
# Python version
python --version  # Should be >= [X.Y]

# Required packages
pip install [package1]==[version]
pip install [package2]==[version]
pip install [package3]==[version]
```

### Required Files

Download and place in your lab directory:
- [`[filename.ext]`](link): [Description]
- [`[filename.ext]`](link): [Description]
- [`[data.csv]`](link): [Description]

---

## Setup Instructions

### Step 1: Create Lab Directory

```bash
# Create and navigate to lab directory
mkdir -p ~/vllm-learn/labs/lab[X]_[topic-name]
cd ~/vllm-learn/labs/lab[X]_[topic-name]

# Verify location
pwd
# Expected: /home/user/vllm-learn/labs/lab[X]_[topic-name]
```

### Step 2: Download Starter Code

```bash
# Download starter files
wget [URL to starter code zip]
unzip starter-code.zip

# Or clone from repository
git clone [repository URL] .
```

### Step 3: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import [key_package]; print('Setup successful!')"
```

### Step 4: Verify Setup

Run the verification script:

```bash
python verify_setup.py
```

**Expected Output:**

```
✓ Python version: [X.Y.Z]
✓ All required packages installed
✓ Data files present
✓ Directory structure correct
✓ Setup complete! You're ready to start.
```

If you see any ✗ marks, review the error messages and fix the issues.

---

## Problem Statement

### Background

[Detailed background information about the problem domain]

### The Challenge

[Clear, specific problem statement. Be precise about what needs to be solved.]

### Requirements

#### Functional Requirements

The system must:

1. **[Requirement Category 1]:**
   - [ ] [Specific requirement 1.1]
   - [ ] [Specific requirement 1.2]
   - [ ] [Specific requirement 1.3]

2. **[Requirement Category 2]:**
   - [ ] [Specific requirement 2.1]
   - [ ] [Specific requirement 2.2]
   - [ ] [Specific requirement 2.3]

3. **[Requirement Category 3]:**
   - [ ] [Specific requirement 3.1]
   - [ ] [Specific requirement 3.2]

#### Non-Functional Requirements

- **Performance:** [Specific performance criteria]
- **Scalability:** [Scalability requirements]
- **Error Handling:** [Error handling requirements]
- **Code Quality:** [Quality standards]
- **Documentation:** [Documentation requirements]

### Constraints

- [Constraint 1: e.g., Must use specific library/approach]
- [Constraint 2: e.g., Memory/time limitations]
- [Constraint 3: e.g., Compatibility requirements]

### Input/Output Specifications

**Input Format:**

```
[Detailed specification of input format]
Example:
[Concrete example of input]
```

**Output Format:**

```
[Detailed specification of output format]
Example:
[Concrete example of output]
```

---

## Architecture Overview

[Diagram or description of the system architecture]

```
┌─────────────────────────────────────────────┐
│          [System Component Name]            │
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │  Component A │─────▶│  Component B │   │
│  └──────────────┘      └──────────────┘   │
│         │                       │          │
│         │                       │          │
│         ▼                       ▼          │
│  ┌──────────────┐      ┌──────────────┐   │
│  │  Component C │      │  Component D │   │
│  └──────────────┘      └──────────────┘   │
└─────────────────────────────────────────────┘
```

**Component Responsibilities:**

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| [Component A] | [What it does] | [Input type] | [Output type] |
| [Component B] | [What it does] | [Input type] | [Output type] |
| [Component C] | [What it does] | [Input type] | [Output type] |

---

## Implementation Guide

### Phase 1: [First Implementation Phase] (Estimated: [X]m)

**Goal:** [What this phase accomplishes]

**Files to Create/Modify:**
- `[filename].py`: [Purpose]
- `[filename].py`: [Purpose]

**Tasks:**

#### Task 1.1: [Specific Task]

**What to implement:**
[Detailed description]

**Code Template:**

```python
# [filename].py

def [function_name]([parameters]):
    """
    [Function description]

    Args:
        [param1] ([type]): [description]
        [param2] ([type]): [description]

    Returns:
        [type]: [description]

    Raises:
        [ExceptionType]: [when it's raised]

    Example:
        >>> [example usage]
        [expected output]
    """
    # TODO: Implement this function
    pass
```

**Success Criteria:**
- [ ] [Specific criterion 1]
- [ ] [Specific criterion 2]
- [ ] [Passes unit test: `test_[function_name]()`]

**Testing:**

```bash
python -m pytest tests/test_phase1.py::test_[function_name] -v
```

---

#### Task 1.2: [Specific Task]

**What to implement:**
[Detailed description]

**Code Template:**

```python
# [filename].py

class [ClassName]:
    """
    [Class description]

    Attributes:
        [attr1] ([type]): [description]
        [attr2] ([type]): [description]

    Example:
        >>> obj = [ClassName]([args])
        >>> result = obj.[method]([args])
        >>> print(result)
        [expected output]
    """

    def __init__(self, [parameters]):
        """Initialize [ClassName]."""
        # TODO: Implement initialization
        pass

    def [method_name](self, [parameters]):
        """[Method description]."""
        # TODO: Implement method
        pass
```

**Success Criteria:**
- [ ] [Specific criterion 1]
- [ ] [Specific criterion 2]
- [ ] [Passes unit test: `test_[class_name]()`]

---

### Phase 2: [Second Implementation Phase] (Estimated: [X]m)

**Goal:** [What this phase accomplishes]

**Files to Create/Modify:**
- `[filename].py`: [Purpose]

**Tasks:**

#### Task 2.1: [Specific Task]

[Similar structure to Phase 1 tasks]

---

### Phase 3: [Third Implementation Phase] (Estimated: [X]m)

**Goal:** [What this phase accomplishes]

**Tasks:**

#### Task 3.1: [Integration Task]

[Integration and final assembly instructions]

---

## Testing Your Implementation

### Unit Tests

Run individual component tests:

```bash
# Test Phase 1 components
python -m pytest tests/test_phase1.py -v

# Test Phase 2 components
python -m pytest tests/test_phase2.py -v

# Test Phase 3 components
python -m pytest tests/test_phase3.py -v
```

**Expected Output:**

```
tests/test_phase1.py::test_function1 PASSED    [ 33%]
tests/test_phase1.py::test_function2 PASSED    [ 66%]
tests/test_phase1.py::test_function3 PASSED    [100%]

================== 3 passed in 0.45s ==================
```

### Integration Tests

Run full system tests:

```bash
python -m pytest tests/test_integration.py -v
```

### Manual Testing

Test with provided examples:

```bash
# Test case 1
python main.py --input data/test_case_1.txt

# Expected output:
[Expected output for test case 1]

# Test case 2
python main.py --input data/test_case_2.txt

# Expected output:
[Expected output for test case 2]
```

### Performance Testing

Benchmark your implementation:

```bash
python benchmark.py
```

**Performance Targets:**
- [ ] Completes test_case_1 in < [X]ms
- [ ] Handles [Y] items in < [Z]ms
- [ ] Memory usage < [N]MB

---

## Submission Requirements

### Code Files

Submit the following files:

1. **Source Code:**
   - `[file1].py`: [Description]
   - `[file2].py`: [Description]
   - `[file3].py`: [Description]

2. **Tests:**
   - `tests/test_[component].py`: [Your additional tests]

3. **Documentation:**
   - `README.md`: [Brief overview of your implementation]
   - `DESIGN.md`: [Design decisions and architecture]

4. **Results:**
   - `results.txt`: [Output from test cases]
   - `performance.txt`: [Benchmark results]

### Documentation Requirements

Your `README.md` should include:

```markdown
# Lab [X] Implementation: [Your Name]

## Overview
[Brief description of your approach]

## How to Run
[Exact commands to run your code]

## Design Decisions
[Key design decisions and justifications]

## Challenges Faced
[What was difficult and how you solved it]

## Results
[Summary of test results and performance]

## Additional Features
[Any extra features you implemented]
```

### Submission Checklist

Before submitting, ensure:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Code follows style guide (PEP 8 for Python)
- [ ] All functions have docstrings
- [ ] Code is commented appropriately
- [ ] No hardcoded values (use constants/config)
- [ ] Error handling implemented
- [ ] Edge cases handled
- [ ] Documentation complete
- [ ] Files organized properly

---

## Grading Rubric

| Category | Points | Criteria |
|----------|--------|----------|
| **Correctness** | 40 | • All requirements met (20 pts)<br>• Tests pass (20 pts) |
| **Code Quality** | 25 | • Clean, readable code (10 pts)<br>• Proper structure (10 pts)<br>• Following best practices (5 pts) |
| **Documentation** | 15 | • Code comments (5 pts)<br>• Docstrings (5 pts)<br>• README complete (5 pts) |
| **Performance** | 10 | • Meets performance targets (10 pts) |
| **Testing** | 10 | • Additional test cases (5 pts)<br>• Edge case coverage (5 pts) |
| **Total** | **100** | |

**Bonus Points (up to +10):**
- [ ] (+5) Exceptional code quality and design
- [ ] (+5) Additional features beyond requirements
- [ ] (+3) Outstanding documentation
- [ ] (+2) Comprehensive test coverage

**Grade Scale:**
- A: 90-100
- B: 80-89
- C: 70-79
- D: 60-69
- F: < 60

---

## Hints and Tips

### Getting Started

<details>
<summary>Where do I begin?</summary>

1. Read the entire problem statement carefully
2. Understand the input/output format
3. Start with Phase 1, Task 1.1
4. Test each component before moving on
5. Build incrementally

</details>

### Common Issues

<details>
<summary>Issue: [Common Problem 1]</summary>

**Symptoms:** [What you might see]

**Solution:** [How to fix it]

```python
# Example fix
[code snippet]
```

</details>

<details>
<summary>Issue: [Common Problem 2]</summary>

**Symptoms:** [What you might see]

**Solution:** [How to fix it]

</details>

### Debugging Tips

1. **Use print debugging:**
   ```python
   print(f"Debug: variable = {variable}")
   ```

2. **Use the debugger:**
   ```bash
   python -m pdb main.py
   ```

3. **Check your assumptions:**
   - Add assertions
   - Validate input/output at each step

4. **Test incrementally:**
   - Don't write everything then test
   - Test each function as you write it

---

## Additional Resources

### Relevant Documentation
- [Resource 1](link): [What it covers]
- [Resource 2](link): [What it covers]

### Related Tutorials
- [Tutorial 1](link): [Relevant sections]
- [Tutorial 2](link): [Relevant sections]

### Example Implementations
- [Example 1](link): [Similar problem]
- [Example 2](link): [Related technique]

### Research Papers (if applicable)
- [Paper 1](link): [Key insights]

---

## Extensions and Challenges

Once you complete the basic lab, try these extensions:

### Extension 1: [Enhanced Feature] (Medium)

**Objective:** [What this adds]

**Requirements:**
- [ ] [Requirement 1]
- [ ] [Requirement 2]

**Hints:** [Brief guidance]

---

### Extension 2: [Advanced Feature] (Hard)

**Objective:** [What this adds]

**Requirements:**
- [ ] [Requirement 1]
- [ ] [Requirement 2]

**Hints:** [Brief guidance]

---

### Extension 3: [Performance Optimization] (Advanced)

**Objective:** [Optimization goal]

**Requirements:**
- [ ] Achieve [X]x speedup
- [ ] Reduce memory by [Y]%

**Hints:** [Brief guidance]

---

## FAQ

**Q: [Common question]?**

A: [Clear answer]

---

**Q: [Common question]?**

A: [Clear answer]

---

**Q: [Common question]?**

A: [Clear answer]

---

## Support

**Getting Help:**

1. **Review Materials:** Re-read relevant tutorial sections
2. **Check Documentation:** Look up specific APIs
3. **Debug Systematically:** Follow the debugging tips above
4. **Office Hours:** [Time and location/link]
5. **Discussion Forum:** [Link to forum]
6. **Email:** [Contact email]

**When asking for help, include:**
- What you're trying to do
- What you've tried
- The specific error or unexpected behavior
- Relevant code snippet
- Your test environment details

---

## Acknowledgments

[Credits for datasets, libraries, or resources used]

---

**Lab Version:** 1.0
**Created:** [YYYY-MM-DD]
**Last Updated:** [YYYY-MM-DD]
**Author:** [Author Name]
**Estimated Completion Time:** [X hours Y minutes]
