# SPDX-License-Identifier: Apache-2.0

# SinglePrefillPDJob is deprecated, use MultiplePrefillsPDJob instead
from vllm.entrypoints.cli.pd.multiple_prefills import MultiplePrefillsPDJob

# Alias for backward compatibility
SinglePrefillPDJob = MultiplePrefillsPDJob
