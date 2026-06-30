#include "core/registration.h"

// QuTLASS registers torch.ops._qutlass_C via TORCH_LIBRARY in bindings.cpp.
// This stub lets Python import vllm._qutlass_C to trigger op registration.
REGISTER_EXTENSION(_qutlass_C)
