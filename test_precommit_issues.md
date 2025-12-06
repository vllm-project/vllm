# Test Pre-commit Issues

This is a test markdown file with intentional issues to trigger pre-commit checks.

## Bad formatting

This paragraph has no blank line after the heading.

This paragraph has multiple spaces    and bad formatting.

- Bad list item
-Another bad list item
-   Bad indentation

## More Issues

This line is way too long and should trigger the line length rule because it exceeds the maximum allowed characters per line and continues on and on and on.

### Bad heading with trailing spaces    

This heading has trailing spaces.

## Code block without language

```
def bad_code():
    return "no language specified"
```

## Bad link formatting

[Bad link](https://example.com "title with spaces")

## Bad emphasis

This text has **bad** *emphasis* formatting.

## Missing blank lines

This paragraph is missing a blank line.
This is another paragraph without proper spacing.
