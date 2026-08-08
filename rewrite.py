# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
with open("vllm/model_executor/models/ernie45_vl_moe.py") as f:
    lines = f.readlines()

# 1. Import AutoWeightsLoader
import_idx = -1
for i, line in enumerate(lines):
    if "from .utils import (" in line:
        import_idx = i
        break

if import_idx != -1:
    lines.insert(import_idx + 1, "    AutoWeightsLoader,\n")

# 2. Extract load_weights from Ernie4_5_VLMoeForCausalLM
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.startswith("class Ernie4_5_VLMoeForCausalLM("):
        for j in range(i, len(lines)):
            if lines[j].startswith("    def load_weights(self"):
                start_idx = j
                break
        break

for i in range(start_idx, len(lines)):
    if lines[i].startswith("        return loaded_params"):
        end_idx = i
        break

if start_idx == -1 or end_idx == -1:
    print("Could not find load_weights!")
    exit(1)

# Extract original load_weights lines
orig_lw_lines = lines[start_idx : end_idx + 1]

# Build new load_weights for Model
new_model_lw = []
skip = False
for line in orig_lw_lines:
    if (
        'if self.config.tie_word_embeddings and name.endswith("lm_head.weight"):'
        in line
    ):
        skip = True
    if (
        'if "mtp" in name or "vision_model" in name or "resampler_model" in name:'
        in line
    ):
        skip = True

    if not skip:
        new_model_lw.append(line)

    if skip and line.strip() == "continue":
        skip = False

# 3. Find end of Ernie4_5_VLMoeModel
model_end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("# only used as text backbone for ernie4.5-vl"):
        # The line before should be blank or return hidden_states
        model_end_idx = i - 1
        break

# We need to insert model_lw at model_end_idx (before the blank lines)
# Actually, insert it before the comment.
while lines[model_end_idx].strip() == "":
    model_end_idx -= 1
model_end_idx += 1

# 4. Replace original load_weights with AutoWeightsLoader
new_forcausal_lw = """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
            ignore_unexpected_prefixes=["mtp", "vision_model", "resampler_model"],
        )
        return loader.load_weights(weights)
"""

# Now apply all changes in reverse order to avoid index shifting!
# Or just build a new list.

# First, replace the ForCausalLM load_weights
new_lines = lines[:start_idx] + [new_forcausal_lw] + lines[end_idx + 1 :]

# Next, insert Model load_weights at model_end_idx
# Note: since model_end_idx < start_idx, the indices won't be messed up by the previous replacement!
new_lines = (
    new_lines[:model_end_idx]
    + ["\n"]
    + new_model_lw
    + ["\n"]
    + new_lines[model_end_idx:]
)

with open("vllm/model_executor/models/ernie45_vl_moe.py", "w") as f:
    f.writelines(new_lines)
print("Done")
