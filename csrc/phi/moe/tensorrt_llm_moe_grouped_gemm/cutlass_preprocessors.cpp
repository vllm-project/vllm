/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cutlass_preprocessors.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/stringUtils.h"

#include "cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

struct LayoutDetails
{
    enum class Layout
    {
        UNKNOWN,
        ROW_MAJOR,
        COLUMN_MAJOR
    };

    Layout layoutB = Layout::UNKNOWN;
    int rows_per_column_tile = 1;
    int columns_interleaved = 1;

    bool uses_imma_ldsm = false;
};

template <typename Layout>
struct getLayoutDetails
{
};

template <>
struct getLayoutDetails<cutlass::layout::RowMajor>
{
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::ROW_MAJOR;
        return layout_details;
    }
};

template <>
struct getLayoutDetails<cutlass::layout::ColumnMajor>
{
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
        return layout_details;
    }
};

template <int RowsPerTile, int ColumnsInterleaved>
struct getLayoutDetails<cutlass::layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>>
{
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
        layout_details.rows_per_column_tile = RowsPerTile;
        layout_details.columns_interleaved = ColumnsInterleaved;
        return layout_details;
    }
};

template <typename cutlassArch, typename TypeA, typename TypeB>
LayoutDetails getLayoutDetailsForArchAndQuantType()
{

    using CompileTraits = cutlass::gemm::kernel::LayoutDetailsB<TypeA, TypeB, cutlassArch>;
    using LayoutB = typename CompileTraits::Layout;
    using MmaOperator = typename CompileTraits::Operator;
    LayoutDetails details = getLayoutDetails<LayoutB>()();
    details.uses_imma_ldsm = std::is_same<MmaOperator, cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA>::value;
    return details;
}

template <typename cutlassArch>
LayoutDetails getLayoutDetailsForArch(QuantType quant_type)
{
    int const bits_per_weight_element = get_weight_quant_bits(quant_type);
    LayoutDetails details;
    switch (quant_type)
    {
    case QuantType::W8_A16:
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::half_t, uint8_t>();
        break;
    case QuantType::W4_A16:
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::half_t, cutlass::uint4b_t>();
        break;
    case QuantType::W4_AFP8:
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::float_e4m3_t, cutlass::uint4b_t>();
        break;
    default: TLLM_THROW("Unsupported quantization type");
    }
    return details;
}

LayoutDetails getLayoutDetailsForTransform(QuantType quant_type, int arch)
{
    if (arch >= 80 && arch < 90)
    {
        return getLayoutDetailsForArch<cutlass::arch::Sm80>(quant_type);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported Arch");
        return LayoutDetails();
    }
}

// Permutes the rows of B in a way that is compatible with Turing+ architectures.
//
// Throws an error for other architectures.
// The data is permuted such that:
// For W8_A16, each group of 16 rows is permuted using the map below:
//  0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
// For W4_A16, each group of 32 rows is permuted using the map below:
//  0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
// For W4_A8, see the map in the code. The idea is similar to above.
// The goal of this permutation is to ensure data ends up in the correct threads after
// we execute LDSM. It counteracts the effect of the data being of different widths.
// For more information about the expected layouts, see the MMA section in the PTX docs.
std::vector<int> get_permutation_map(QuantType quant_type)
{

    if (quant_type == QuantType::W8_A16)
    {
        return {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    }
    else
    {
        TLLM_THROW("Invalid quantization type for LDSM permutation");
    }
}

void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type, int64_t const arch_version)
{

    // We only want to run this step for weight only quant.
    std::vector<int> row_permutation = get_permutation_map(quant_type);

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    int const BITS_PER_ELT = get_weight_quant_bits(quant_type);
    int const K = 16 / BITS_PER_ELT;
    int const ELTS_PER_BYTE = 8 / BITS_PER_ELT;
    int const ELTS_PER_REG = 32 / BITS_PER_ELT;

    uint32_t const* input_byte_ptr = reinterpret_cast<uint32_t const*>(quantized_tensor);
    uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

    int MMA_SHAPE_N = 8;
    int B_ROWS_PER_MMA = 8 * K;
    int const elts_in_int32 = 32 / BITS_PER_ELT;

    int const num_vec_cols = num_cols / elts_in_int32;

    TLLM_CHECK_WITH_INFO(
        arch_version >= 75, "Unsupported Arch. Pre-volta not supported. Column interleave not needed on Volta.");

    TLLM_CHECK_WITH_INFO(num_rows % B_ROWS_PER_MMA == 0,
        fmtstr("Invalid shape for quantized tensor. Number of rows of quantized matrix must be a multiple of %d",
            B_ROWS_PER_MMA));
    TLLM_CHECK_WITH_INFO(num_cols % MMA_SHAPE_N == 0,
        fmtstr("Invalid shape for quantized tensor. On turing/Ampere, the number of cols must be a multiple of %d.",
            MMA_SHAPE_N));

    TLLM_CHECK_WITH_INFO(size_t(B_ROWS_PER_MMA) == row_permutation.size(), "Unexpected number of LDSM rows permuted.");

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int64_t matrix_offset = expert * int64_t(num_rows) * int64_t(num_vec_cols);
        for (int base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA)
        {
            for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row)
            {

                for (int write_col = 0; write_col < num_vec_cols; ++write_col)
                {
                    int const write_row = base_row + tile_row;
                    int const tile_read_row = row_permutation[tile_row];
                    int const read_row = base_row + tile_read_row;
                    int const read_col = write_col;

                    const int64_t read_offset = matrix_offset + int64_t(read_row) * num_vec_cols + read_col;
                    const int64_t write_offset = matrix_offset + int64_t(write_row) * num_vec_cols + write_col;

                    output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
                }
            }
        }
    }
}

// We need to use this transpose to correctly handle packed int4 and int8 data
// The reason this code is relatively complex is that the "trivial" loops took a substantial
// amount of time to transpose leading to long preprocessing times. This seemed to be a big
// issue for relatively large models.
template <QuantType quant_type>
void subbyte_transpose_impl(
    int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor, std::vector<size_t> const& shape)
{
    constexpr int bits_per_elt = get_weight_quant_bits(quant_type);

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    const size_t col_bytes = num_cols * bits_per_elt / 8;
    const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
    const size_t num_bytes = size_t(num_experts) * num_rows * col_bytes;

    uint8_t const* input_byte_ptr = reinterpret_cast<uint8_t const*>(quantized_tensor);
    uint8_t* output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

    static constexpr int ELTS_PER_BYTE = 8 / bits_per_elt;

    static constexpr int M_TILE_L1 = 64;
    static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
    uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

    static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

    // We assume the dims are a multiple of vector width. Our kernels only handle dims which are multiples
    // of 64 for weight-only quantization. As a result, this seemed like a reasonable tradeoff because it
    // allows GCC to emit vector instructions.
    TLLM_CHECK_WITH_INFO(!(col_bytes_trans % VECTOR_WIDTH) && !(col_bytes % VECTOR_WIDTH),
        fmtstr("Number of bytes for rows and cols must be a multiple of %d. However, num_rows_bytes = %ld and "
               "num_col_bytes = %ld.",
            VECTOR_WIDTH, col_bytes_trans, col_bytes));

    int const num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
    int const num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        const size_t matrix_offset = expert * num_rows * col_bytes;
        for (size_t row_tile_start = 0; row_tile_start < num_rows; row_tile_start += M_TILE_L1)
        {
            for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes; col_tile_start_byte += N_TILE_L1)
            {

                int const row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
                int const col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

                for (int ii = 0; ii < M_TILE_L1; ++ii)
                {
                    int const row = row_tile_start + ii;

                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                    {
                        int const col = col_tile_start_byte + jj;

                        const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

                        if (row < row_limit && col < col_limit)
                        {
                            for (int v = 0; v < VECTOR_WIDTH; ++v)
                            {
                                cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
                            }
                        }
                    }
                }

                if constexpr (bits_per_elt == 8)
                {
                    for (int ii = 0; ii < M_TILE_L1; ++ii)
                    {
                        for (int jj = ii + 1; jj < N_TILE_L1; ++jj)
                        {
                            std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
                        }
                    }
                }
                else if constexpr (bits_per_elt == 4)
                {

                    for (int ii = 0; ii < M_TILE_L1; ++ii)
                    {
                        // Using M_TILE_L1 here is deliberate since we assume that the cache tile
                        // is square in the number of elements (not necessarily the number of bytes).
                        for (int jj = ii + 1; jj < M_TILE_L1; ++jj)
                        {
                            int const ii_byte = ii / ELTS_PER_BYTE;
                            int const ii_bit_offset = ii % ELTS_PER_BYTE;

                            int const jj_byte = jj / ELTS_PER_BYTE;
                            int const jj_bit_offset = jj % ELTS_PER_BYTE;

                            uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
                            uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

                            cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
                            cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

                            cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
                            cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
                        }
                    }
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(false, "Unsupported quantization type.");
                }

                const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
                const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

                int const row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);
                int const col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

                for (int ii = 0; ii < M_TILE_L1; ++ii)
                {
                    int const row = row_tile_start_trans + ii;
                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                    {
                        int const col = col_tile_start_byte_trans + jj;

                        const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

                        if (row < row_limit_trans && col < col_limit_trans)
                        {
                            for (int v = 0; v < VECTOR_WIDTH; ++v)
                            {
                                output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
                            }
                        }
                    }
                }
            }
        }
    }
}

void subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type)
{

    if (quant_type == QuantType::W8_A16)
    {
        subbyte_transpose_impl<QuantType::W8_A16>(transposed_quantized_tensor, quantized_tensor, shape);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Invalid quant_type");
    }
}

void add_bias_and_interleave_int8s_inplace(int8_t* int8_tensor, const size_t num_elts)
{
    TLLM_CHECK_WITH_INFO(num_elts % 4 == 0, "Dimensions of int8 tensor must be a multiple of 4 for register relayout");
    for (size_t base = 0; base < num_elts; base += 4)
    {
        std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
    }
}

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type)
{
    if (quant_type == QuantType::W8_A16)
    {
        add_bias_and_interleave_int8s_inplace(tensor, num_elts);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Invalid quantization type for interleaving.");
    }
}

void interleave_column_major_tensor(int8_t* interleaved_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type, LayoutDetails details)
{

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    int const BITS_PER_ELT = get_weight_quant_bits(quant_type);
    int const elts_in_int32 = 32 / BITS_PER_ELT;

    int const rows_per_tile = details.rows_per_column_tile;

    TLLM_CHECK_WITH_INFO(!(num_rows % elts_in_int32),
        fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", elts_in_int32, num_rows));

    uint32_t const* input_byte_ptr = reinterpret_cast<uint32_t const*>(quantized_tensor);
    uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

    TLLM_CHECK_WITH_INFO(!(num_rows % rows_per_tile),
        fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", rows_per_tile, num_rows));

    int const num_vec_rows = num_rows / elts_in_int32;
    int const vec_rows_per_tile = rows_per_tile / elts_in_int32;
    int const interleave = details.columns_interleaved;

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int64_t matrix_offset = expert * int64_t(num_vec_rows) * int64_t(num_cols);
        for (int read_col = 0; read_col < num_cols; ++read_col)
        {
            const int64_t write_col = read_col / interleave;
            for (int base_vec_row = 0; base_vec_row < num_vec_rows; base_vec_row += vec_rows_per_tile)
            {
                for (int vec_read_row = base_vec_row;
                     vec_read_row < std::min(num_vec_rows, base_vec_row + vec_rows_per_tile); ++vec_read_row)
                {
                    const int64_t vec_write_row = interleave * base_vec_row
                        + vec_rows_per_tile * (read_col % interleave) + vec_read_row % vec_rows_per_tile;

                    const int64_t read_offset = matrix_offset + int64_t(read_col) * num_vec_rows + vec_read_row;
                    const int64_t write_offset
                        = matrix_offset + int64_t(write_col) * num_vec_rows * interleave + vec_write_row;
                    output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
                }
            }
        }
    }
}

void preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t const* row_major_quantized_weight,
    std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave)
{
    int arch = getSMVersion();
    if (force_interleave && arch == 90)
    {
        // Workaround for MOE which doesn't have specialised Hopper kernels yet
        arch = 80;
    }
    LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");

    size_t num_elts = 1;
    for (auto const& dim : shape)
    {
        num_elts *= dim;
    }

    const size_t num_bytes = num_elts * get_weight_quant_bits(quant_type) / 8;

    std::vector<int8_t> src_buf(num_bytes);
    std::vector<int8_t> dst_buf(num_bytes);
    std::copy(row_major_quantized_weight, row_major_quantized_weight + num_bytes, src_buf.begin());

    // Works on row major data, so issue this permutation first.
    if (details.uses_imma_ldsm)
    {
        permute_B_rows_for_mixed_gemm(dst_buf.data(), src_buf.data(), shape, quant_type, arch);
        src_buf.swap(dst_buf);
    }

    if (details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR)
    {
        subbyte_transpose(dst_buf.data(), src_buf.data(), shape, quant_type);
        src_buf.swap(dst_buf);
    }

    if (details.columns_interleaved > 1)
    {
        interleave_column_major_tensor(dst_buf.data(), src_buf.data(), shape, quant_type, details);
        src_buf.swap(dst_buf);
    }

    std::copy(src_buf.begin(), src_buf.end(), preprocessed_quantized_weight);
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
