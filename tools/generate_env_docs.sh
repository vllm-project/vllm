#!/bin/bash

# Set the path to envs.py
ENVS_FILE="vllm/envs.py"

# Set input and temp files
INPUT_FILE="docs/source/serving/env_vars.md"
TEMP_FILE="/tmp/env_vars_temp.md"

# Check if envs.py exists
if [ ! -f "$ENVS_FILE" ]; then
    echo "Error: $ENVS_FILE file not found"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE file not found"
    exit 1
fi

# Create the table
cat << 'EOF' > "$TEMP_FILE"

| Name | Type | Default | Description |
|------|------|---------|-------------|
EOF

# Function to extract environment variables from envs.py
extract_env_vars() {
    # Extract variables that look like environment variables (all caps with underscores)
    grep -oP '(?<=")[A-Z][A-Z0-9_]*(?=")' "$ENVS_FILE" | grep -v "^__" | sort -u > /tmp/env_vars.txt
}

# Function to get description for a variable
get_description() {
    local var_name=$1
    # Look for comments above the variable definition
    awk -v var="$var_name" '
        /'"$var_name"'/ {
            # Print previous line if it starts with #
            if (prev ~ /^[[:space:]]*#/) {
                gsub(/^[[:space:]]*#[[:space:]]*/, "", prev)
                print prev
            }
        }
        { prev = $0 }
    ' "$ENVS_FILE" | head -n 1
}

# Function to get variable type
get_type() {
    local var_name=$1
    # Look for type hints or common type patterns in the variable definition
    local type_info=$(awk -v var="$var_name" '
        $0 ~ var {
            if ($0 ~ /int\(/) print "int"
            else if ($0 ~ /float\(/) print "float"
            else if ($0 ~ /bool\(/) print "bool"
            else if ($0 ~ /(True|False)/) print "bool"
            else if ($0 ~ /[0-9]+\.[0-9]+/) print "float"
            else if ($0 ~ /[0-9]+[^.]/) print "int"
            else print "str"
        }
    ' "$ENVS_FILE" | head -n 1)

    # Default to str if no type is found
    echo "${type_info:-str}"
}

# Function to add a table row
add_table_row() {
    local var_name=$1
    local var_type=$(get_type "$var_name")
    local default_value=${!var_name:-"not set"}
    local description=$(get_description "$var_name")

    # Escape any pipe characters in the values
    var_name=${var_name//|/\\|}
    var_type=${var_type//|/\\|}
    default_value=${default_value//|/\\|}
    description=${description//|/\\|}

    # If no description found, use "-"
    if [ -z "$description" ]; then
        description="-"
    fi

    echo "| \`$var_name\` | \`$var_type\` | \`$default_value\` | $description |" >> "$TEMP_FILE"
}

# Extract environment variables
echo "Extracting environment variables from $ENVS_FILE..."
extract_env_vars

# Process each found variable
while IFS= read -r var; do
    # Skip empty lines
    [ -z "$var" ] && continue
    add_table_row "$var"
done < /tmp/env_vars.txt

# Append the generated table to the existing file
cat "$TEMP_FILE" >> "$INPUT_FILE"

# Cleanup
rm /tmp/env_vars.txt "$TEMP_FILE"

echo "Documentation appended to $INPUT_FILE"