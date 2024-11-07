#!/bin/bash

set -eu

# ensure 1 argument is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pr_number>"
    exit 1
fi

PR_NUMBER=$1
OLD=/tmp/orig_pr_body.txt
NEW=/tmp/new_pr_body.txt

gh pr view --json body --template "{{.body}}" "${PR_NUMBER}" > "${OLD}"
cp "${OLD}" "${NEW}"

# Remove all lines after and including "**BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE**"
sed -i '/\*\*BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE\*\*/,$d' "${NEW}"

# Remove "FIX #xxxx (*link existing issues this PR will resolve*)"
sed -i '/FIX #xxxx.*$/d' "${NEW}"

# Remove "FILL IN THE PR DESCRIPTION HERE"
sed -i '/FILL IN THE PR DESCRIPTION HERE/d' "${NEW}"

# Run this only if ${NEW} is different than ${OLD}
if ! cmp -s "${OLD}" "${NEW}"; then
    echo "Updating PR body"
    gh pr edit --body-file "${NEW}" "${PR_NUMBER}"
else
    echo "No changes needed"
fi
