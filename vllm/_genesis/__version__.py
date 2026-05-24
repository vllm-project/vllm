# SPDX-License-Identifier: Apache-2.0
"""Genesis version — single source of truth.

Update this on each release tag. All other modules MUST import from
here rather than hardcoding their own version string.

Format: lowercase 'v' + dotted version + optional '.x' for in-dev.
This isn't strict PEP 440 because we use 'x' as a catch-all for
"any patch released under this minor". Tagged releases get a real
PEP 440 form (e.g. v7.63.0).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

# Bump on each release tag. The 'x' suffix means "any patch under this
# minor". On a real tagged release this becomes e.g. 'v7.63.0'.
VERSION = "v7.63.x"
__version__ = VERSION
