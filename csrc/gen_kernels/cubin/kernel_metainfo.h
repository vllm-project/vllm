#pragma once

#include "../cuda_utils.h"
#ifndef EXCLUDE_SM_100
#endif
extern unsigned long long
    GemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin
        [];
extern unsigned long long
    GemmKernel_Fp16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin
        [];
extern unsigned long long
    GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin
        [];
struct TllmGenGemmKernelMetaInfo {
  Data_type mdtypeC;
  Data_type mdtypeElt;
  Data_type mdtypeAcc;
  int mTileM;
  int mTileN;
  int mTileK;
  int mEpilogueTileM;
  int mEpilogueTileN;
  int mMmaM;
  int mMmaN;
  int mMmaK;
  int mClusterX;
  int mClusterY;
  int mClusterZ;
  int mSharedMemBytes;
  int mSM;
  const unsigned long long* mCubin;
  unsigned int mCubinSize;
  const char* mFuncName;
};
extern unsigned int
    GemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_len;
extern unsigned int
    GemmKernel_Fp16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_len;
extern unsigned int
    GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_len;
#ifndef EXCLUDE_SM_100
static const TllmGenGemmKernelMetaInfo sTllmGenGemmKernelMetaInfos[] = {
    {DATA_TYPE_FP32, DATA_TYPE_E4M3, DATA_TYPE_FP32, 128, 128, 128, 128, 64,
     128, 64, 32, 1, 1, 1, 137216, kSM_100,
     GemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin,
     GemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_len,
     "gemmKernel_Fp32_E4m3_Fp32_tile128x128x128_epilogueTile128x64_"
     "mma128x64x32_cluster1x1x1_dsFp8_sm100a"},
    {DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP32, 128, 128, 128, 128, 64,
     128, 64, 32, 1, 1, 1, 104448, kSM_100,
     GemmKernel_Fp16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin,
     GemmKernel_Fp16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_len,
     "gemmKernel_Fp16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_"
     "mma128x64x32_cluster1x1x1_dsFp8_sm100a"},
    {DATA_TYPE_BF16, DATA_TYPE_E4M3, DATA_TYPE_FP32, 128, 128, 128, 128, 64,
     128, 64, 32, 1, 1, 1, 104448, kSM_100,
     GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin,
     GemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_mma128x64x32_cluster1x1x1_dsFp8_sm100a_cubin_len,
     "gemmKernel_Bfloat16_E4m3_Fp32_tile128x128x128_epilogueTile128x64_"
     "mma128x64x32_cluster1x1x1_dsFp8_sm100a"},

};
#endif

