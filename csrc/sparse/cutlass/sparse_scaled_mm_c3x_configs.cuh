namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_0 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_1 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_2 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_3 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_4 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_5 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_6 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_7 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_8 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_9 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_10 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_11 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_12 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_13 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_14 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _128, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_15 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_16 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _128, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_17 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_64, _128, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_18 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_19 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _256, _128>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_20 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_21 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_22 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _256, _128>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_23 {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _256, _128>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_0 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_1 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_2 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_3 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_4 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_5 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_6 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_7 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_8 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_9 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_10 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_11 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_12 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_13 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_14 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_15 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_16 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_17 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_18 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_19 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_20 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _128, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_21 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_64, _128, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_22 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_23 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_2, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_24 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_25 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_26 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _256, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_27 {
    static_assert(std::is_same<InType, int8_t>());
    using TileShape = Shape<_128, _256, _128>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = int32_t;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_0 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_1 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_2 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_3 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_4 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_5 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_6 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_7 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_8 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_9 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_10 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_11 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_12 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_13 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_14 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_15 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_16 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_17 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_18 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_19 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_20 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_21 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_22 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_23 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_2, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_24 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_25 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_26 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_4, _4, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_27 {
    static_assert(std::is_same<InType, cutlass::half_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_4, _4, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_0 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_1 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_2 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_3 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_4 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_5 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_6 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_7 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_8 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_9 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_4, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_10 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_11 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_12 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_13 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_14 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_15 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_16 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_17 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _64, _256>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_18 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_64, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_19 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_8, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_20 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_21 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _64, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_22 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_23 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_2, _2, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_24 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_25 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_26 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_4, _4, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_27 {
    static_assert(std::is_same<InType, cutlass::bfloat16_t>());
    using TileShape = Shape<_256, _128, _128>;
    using ClusterShape = Shape<_4, _4, _1>;
    using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedule = typename cutlass::gemm::PersistentScheduler;
    using AccType = float;
    static constexpr cutlass::gemm::GemmUniversalMode Mode = cutlass::gemm::GemmUniversalMode::kGemm;
    using Cutlass3xGemm =
        cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
            KernelSchedule, EpilogueSchedule, AccType, TileSchedule, Mode>;
};

}  // namespace