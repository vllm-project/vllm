# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSON rendering of a Selection."""

from __future__ import annotations

import json

from .select import Selection


def render_json(sel: Selection) -> str:
    return json.dumps(
        {
            "docs_only": sel.docs_only,
            "docs_affected": sel.docs_affected,
            "docs_reasons": sel.docs_reasons,
            "run_all": sel.run_all,
            "selected": sel.selected,
            "selected_rules": sel.selected_rules,
            "manual_hits": sel.manual_hits,
            "manual_rules": sel.manual_rules,
            "notes": sel.notes,
        },
        indent=1,
        sort_keys=True,
    )
