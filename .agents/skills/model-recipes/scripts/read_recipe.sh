#!/bin/bash
# Usage: read_recipe.sh <recipe.json> [guide-section...]
# Without sections, print a compact summary of a recipes.vllm.ai recipe JSON.
# With sections, print each guide section whose heading starts with one of
# them (case-insensitive), including its subsections.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <recipe.json> [guide-section...]" >&2
    exit 1
fi

# Print "<level>\t<heading text>\t<line>" for guide lines, marking headings
# outside code fences with their level and other lines with 0.
guide_lines() {
    jq -r '.guide // ""' "$1" | awk '
        /^[[:space:]]*(```|~~~)/ { fence = !fence }
        !fence && match($0, /^#+ /) {
            print RLENGTH - 1 "\t" substr($0, RLENGTH + 1) "\t" $0
            next
        }
        { print "0\t\t" $0 }'
}

if [ $# -gt 1 ]; then
    recipe=$1
    shift
    status=0
    for section in "$@"; do
        guide_lines "$recipe" | awk -F '\t' -v h="$section" '
            BEGIN { h = tolower(h); sub(/^#+[[:space:]]*/, "", h) }
            $1 > 0 && lvl && $1 <= lvl { lvl = 0 }
            $1 > 1 && !lvl && index(tolower($2), h) == 1 { lvl = $1; found = 1 }
            lvl { sub(/^[^\t]*\t[^\t]*\t/, ""); print }
            END {
                if (!found) {
                    print "No guide section starts with: " h > "/dev/stderr"
                    exit 1
                }
            }' || status=1
    done
    exit $status
fi

sections=$(guide_lines "$1" | awk -F '\t' '$1 == 2 || $1 == 3 { print $3 }' |
    jq -R . | jq -s .)

jq --argjson sections "$sections" '{
    hf_id,
    title: .meta.title,
    tasks: .meta.tasks,
    recipe_verified_hardware: .meta.hardware,
    omni_tasks: (.omni.tasks // null |
        if . then map(objects |= {id, label, endpoint, model_id, description})
        else null end),
    variants: (.variants // {} | map_values(
        {model_id, precision, vram_minimum_gb, supported_hardware} |
        with_entries(select(.value != null)))),
    features: (.features // {} | keys),
    opt_in_features,
    default: (.recommended_command // null |
        if . then {hardware, strategy, variant} else null end),
    hardware: .recommended_command.by_hardware,
    guide_sections: $sections
} | with_entries(select(.value != null))' "$1"
