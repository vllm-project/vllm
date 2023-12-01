
#include "flash.h"
#include "static_switch.h"

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream)
{
    FP16_SWITCH(!params.is_bf16, [&] {
        FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<elem_type, kHeadDim>(params, stream); });
    });
}
