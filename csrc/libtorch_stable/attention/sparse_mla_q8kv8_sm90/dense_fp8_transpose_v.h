/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

/**
 * ref to Fa3's SmemTranspose64x64:
 * https://github.com/Dao-AILab/flash-attention/blob/0823cf7b5d96499c1c79a4f64b1e256a035ba4b4/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L26
 */

#pragma once

template <int kBlockN, int kHeadDim>
struct SmemTransposeFp8_64x64 {
  static_assert((kBlockN % 64 == 0) && (kHeadDim % 64 == 0));

  using Element = cutlass::float_e4m3_t;
  using TransposeShapeAtomV = Shape<_64, _64>;
  using SmemLayoutAtomV = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

  // for fp8 in-kernel transpose -- src layout
  using SmemLayoutDivideV =
      decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
  using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
  using FactoringShapeV =
      decltype(make_shape(SmemShapeLDSM{}, shape<1>(SmemLayoutDivideV{}),
                          shape<2>(SmemLayoutDivideV{})));
  using SmemLayoutTransposeV = decltype(composition(
      SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

  // For fp8, this is the memory transpose.
  using SmemLayoutAtomVt = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

  // for fp8 in-kernel transpose -- dst layout
  using SmemLayoutVtTrans = decltype(composition(
      SmemLayoutVt{},
      make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1>{})));
  using SmemLayoutDivideVt =
      decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
  using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
  using FactoringShapeVt =
      decltype(make_shape(SmemShapeSTSM{}, shape<1>(SmemLayoutDivideVt{}),
                          shape<2>(SmemLayoutDivideVt{})));
  using SmemLayoutTransposeVt = decltype(composition(
      SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));

  using ldsm_thread_shape = Shape<_4, _1, _8, _4>;
  using ldsm_value_shape = Shape<_2, _8, _2, _1>;
  using ldsm_value_stride = Stride<_2, _4, _1, _0>;
  using TiledCopyLDSM = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<ldsm_thread_shape>{},
      Layout<ldsm_value_shape, ldsm_value_stride>{}));
  TiledCopyLDSM tiled_copy_ldsm;

  using stsm_thread_shape = Shape<_4, _1, _8, _4>;
  using stsm_value_shape = Shape<_4, _4, _2, _1>;
  using stsm_value_stride = Stride<_1, _8, _4, _0>;
  using TiledCopySTSM = decltype(make_tiled_copy(
      Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<stsm_thread_shape>{},
      Layout<stsm_value_shape, stsm_value_stride>{}));
  TiledCopySTSM tiled_copy_stsm;

  template <class SmemTensor, class SmemTensorOut>
  CUTLASS_DEVICE void transpose_pair(SmemTensor&& s_in0, SmemTensorOut&& s_out0,
                                     SmemTensor&& s_in1,
                                     SmemTensorOut&& s_out1) {
    using namespace cute;

    auto tid = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
    auto thr_copy_ldsm = tiled_copy_ldsm.get_thread_slice(tid);
    auto thr_copy_stsm = tiled_copy_stsm.get_thread_slice(tid);

    auto tXsX0 = thr_copy_ldsm.partition_S(s_in0);
    auto tXrX0 = make_tensor<Element>(shape(tXsX0));
    auto tXsX_out0 = thr_copy_stsm.partition_D(s_out0);

    auto tXsX1 = thr_copy_ldsm.partition_S(s_in1);
    auto tXrX1 = make_tensor<Element>(shape(tXsX1));
    auto tXsX_out1 = thr_copy_stsm.partition_D(s_out1);

    auto data0 = tXrX0.data();
    auto data1 = tXrX1.data();

    cute::copy(tiled_copy_ldsm, tXsX0, tXrX0);
    cute::copy(tiled_copy_ldsm, tXsX1, tXrX1);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size(tXrX0); n += 8) {
      uint32_t* d0 = reinterpret_cast<uint32_t*>(&data0[n]);
      uint32_t* d1 = reinterpret_cast<uint32_t*>(&data1[n]);
      auto upper0 = d0[0];
      auto lower0 = d0[1];
      auto upper1 = d1[0];
      auto lower1 = d1[1];
      d0[0] = __byte_perm(upper0, lower0, 0x6420);
      d0[1] = __byte_perm(upper0, lower0, 0x7531);
      d1[0] = __byte_perm(upper1, lower1, 0x6420);
      d1[1] = __byte_perm(upper1, lower1, 0x7531);
    }

    cute::copy(tiled_copy_stsm, tXrX0, tXsX_out0);
    cute::copy(tiled_copy_stsm, tXrX1, tXsX_out1);
  }
};
