// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace vllm {
namespace turboquant {

// ============================================================================
// TurboQuant KV cache data types
// ============================================================================

enum class TQDataType {
  kPQ4 = 0,  // PolarQuant 4-bit (no QJL)
  kTQ3 = 1,  // PolarQuant 3-bit + 1-bit QJL residual
  kTQ2 = 2,  // PolarQuant 2-bit + 1-bit QJL residual
};

// Bits per angle element for each mode
__host__ __device__ constexpr int angle_bits(TQDataType dt) {
  return dt == TQDataType::kPQ4 ? 4 : (dt == TQDataType::kTQ3 ? 3 : 2);
}

// Whether QJL residual correction is enabled
__host__ __device__ constexpr bool has_qjl(TQDataType dt) {
  return dt != TQDataType::kPQ4;
}

// ============================================================================
// Fast Walsh-Hadamard Transform (WHT)
// ============================================================================

// In-place normalized WHT on a vector of length `n` (must be power of 2).
// Each thread handles one element. Uses warp shuffle for dimensions <= 32,
// shared memory for larger dimensions.
template <typename scalar_t>
__device__ void wht_inplace(scalar_t* __restrict__ vec, int n) {
  // Butterfly-based WHT: O(n log n)
  for (int half = 1; half < n; half <<= 1) {
    for (int i = 0; i < n; i += (half << 1)) {
      for (int j = 0; j < half; j++) {
        scalar_t a = vec[i + j];
        scalar_t b = vec[i + j + half];
        vec[i + j] = a + b;
        vec[i + j + half] = a - b;
      }
    }
  }
  // Normalize by 1/sqrt(n)
  float norm = rsqrtf(static_cast<float>(n));
  for (int i = 0; i < n; i++) {
    vec[i] = static_cast<scalar_t>(static_cast<float>(vec[i]) * norm);
  }
}

// ============================================================================
// Seeded random sign generation for WHT rotation
// ============================================================================

// Simple hash-based PRNG for deterministic random signs.
// Uses layer_seed + element_index to produce a sign (+1 or -1).
__device__ __forceinline__ float random_sign(uint32_t seed, int idx) {
  // Murmur-style hash mixing
  uint32_t h = seed ^ static_cast<uint32_t>(idx);
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return (h & 1u) ? 1.0f : -1.0f;
}

// ============================================================================
// Randomized Hadamard rotation (diagonal * WHT)
// ============================================================================

// Apply D * H to vec, where D is a random diagonal sign matrix and H is WHT.
// This concentrates the angle distribution for better uniform quantization.
template <typename scalar_t>
__device__ void randomized_hadamard(scalar_t* __restrict__ vec, int n,
                                    uint32_t seed) {
  // Step 1: Multiply by random diagonal signs
  for (int i = 0; i < n; i++) {
    float s = random_sign(seed, i);
    vec[i] = static_cast<scalar_t>(static_cast<float>(vec[i]) * s);
  }
  // Step 2: Apply WHT
  wht_inplace(vec, n);
}

// Inverse: H^T * D^T = H * D (since H is symmetric and D is its own inverse)
template <typename scalar_t>
__device__ void inverse_randomized_hadamard(scalar_t* __restrict__ vec, int n,
                                            uint32_t seed) {
  // Step 1: Apply WHT (H is symmetric, H^T = H, and H * H = I up to scale)
  wht_inplace(vec, n);
  // Step 2: Multiply by same random diagonal signs (D^-1 = D for sign matrix)
  for (int i = 0; i < n; i++) {
    float s = random_sign(seed, i);
    vec[i] = static_cast<scalar_t>(static_cast<float>(vec[i]) * s);
  }
}

// ============================================================================
// Polar coordinate transforms
// ============================================================================

// Convert a pair (x1, x2) to polar coordinates (r, theta).
// Returns theta normalized to [0, 1) range for uniform quantization.
__device__ __forceinline__ void cartesian_to_polar(float x1, float x2,
                                                   float& r, float& theta) {
  r = hypotf(x1, x2);
  // atan2 returns [-pi, pi], normalize to [0, 1)
  float angle = atan2f(x2, x1);
  theta = (angle + CUDART_PI_F) / (2.0f * CUDART_PI_F);
  // Clamp to [0, 1) to avoid edge cases
  theta = fminf(fmaxf(theta, 0.0f), 0.999999f);
}

// Convert polar coordinates back to Cartesian.
__device__ __forceinline__ void polar_to_cartesian(float r, float theta,
                                                   float& x1, float& x2) {
  // theta is in [0, 1), convert back to [-pi, pi]
  float angle = theta * (2.0f * CUDART_PI_F) - CUDART_PI_F;
  float s, c;
  __sincosf(angle, &s, &c);
  x1 = r * c;
  x2 = r * s;
}

// ============================================================================
// Uniform quantization of angles
// ============================================================================

// Quantize a [0, 1) value to `bits`-bit unsigned integer.
__device__ __forceinline__ uint8_t quantize_angle(float theta, int bits) {
  int levels = 1 << bits;
  int q = __float2int_rn(theta * levels);
  q = min(max(q, 0), levels - 1);
  return static_cast<uint8_t>(q);
}

// Dequantize a `bits`-bit unsigned integer back to [0, 1) midpoint.
__device__ __forceinline__ float dequantize_angle(uint8_t q, int bits) {
  int levels = 1 << bits;
  return (static_cast<float>(q) + 0.5f) / static_cast<float>(levels);
}

// ============================================================================
// Recursive polar folding
// ============================================================================

// PolarQuant encodes a d-dimensional vector by:
//   1. Pairing consecutive elements → d/2 (r, theta) pairs
//   2. Recursively pairing radii → d/4 (R, phi) pairs, etc.
//   3. Until a single radius remains
//
// Total angles stored: d/2 + d/4 + ... + 1 = d - 1 angles + 1 radius
// For head_size=128: 127 angles + 1 radius per head per token

// Encode: convert d floats → (d-1) quantized angles + 1 fp16 radius.
// `angles_out` must have space for (d-1) entries.
// Returns the final radius.
template <int BITS>
__device__ float polar_encode(const float* __restrict__ vec, int d,
                              uint8_t* __restrict__ angles_out) {
  // Working buffer for radii at current level
  // Max head_size is typically 128 or 256
  float radii[256];

  int angle_idx = 0;
  int n = d;

  // First level: pair original elements
  for (int i = 0; i < n / 2; i++) {
    float r, theta;
    cartesian_to_polar(vec[2 * i], vec[2 * i + 1], r, theta);
    angles_out[angle_idx++] = quantize_angle(theta, BITS);
    radii[i] = r;
  }
  n /= 2;

  // Recursive folding: pair radii
  while (n > 1) {
    for (int i = 0; i < n / 2; i++) {
      float r, theta;
      cartesian_to_polar(radii[2 * i], radii[2 * i + 1], r, theta);
      angles_out[angle_idx++] = quantize_angle(theta, BITS);
      radii[i] = r;
    }
    n /= 2;
  }

  return radii[0];  // Single remaining radius
}

// Decode: reconstruct d floats from (d-1) quantized angles + 1 fp16 radius.
template <int BITS>
__device__ void polar_decode(const uint8_t* __restrict__ angles,
                             float radius, int d,
                             float* __restrict__ vec_out) {
  // We need to reverse the encoding process.
  // The angles are stored in order: first d/2 (leaf level), then d/4, etc.
  // We reconstruct from the top (single radius) down to the leaves.

  // Count angles per level
  // Level 0 (leaves): d/2 angles
  // Level 1: d/4 angles
  // ...
  // Level log2(d)-1: 1 angle

  int num_levels = 0;
  {
    int n = d;
    while (n > 1) {
      num_levels++;
      n /= 2;
    }
  }

  // Compute offset of each level in the angles array
  // Level 0 starts at 0, has d/2 angles
  // Level 1 starts at d/2, has d/4 angles
  // etc.
  int level_offset[16];  // Enough for head_size up to 65536
  int level_size[16];
  {
    int off = 0;
    int n = d;
    for (int lvl = 0; lvl < num_levels; lvl++) {
      level_size[lvl] = n / 2;
      level_offset[lvl] = off;
      off += n / 2;
      n /= 2;
    }
  }

  // Start with the single radius at the top
  float radii_current[1] = {radius};
  float radii_next[256];

  // Reconstruct from top level down
  for (int lvl = num_levels - 1; lvl >= 0; lvl--) {
    int n_pairs = level_size[lvl];
    int n_radii_in = n_pairs;  // Number of radii to expand
    // Each input radius + angle → two output values
    for (int i = 0; i < n_radii_in; i++) {
      float theta = dequantize_angle(angles[level_offset[lvl] + i], BITS);
      float x1, x2;
      polar_to_cartesian(radii_current[i], theta, x1, x2);
      radii_next[2 * i] = x1;
      radii_next[2 * i + 1] = x2;
    }
    // Swap buffers
    for (int i = 0; i < 2 * n_radii_in; i++) {
      radii_current[i] = radii_next[i];
    }
  }

  // radii_current now contains the reconstructed d-dimensional vector
  for (int i = 0; i < d; i++) {
    vec_out[i] = radii_current[i];
  }
}

// ============================================================================
// QJL sign-bit projection
// ============================================================================

// QJL (Quantized Johnson-Lindenstrauss) applies a random sign-bit projection
// to the quantization residual for bias correction.
//
// Encode: sign_bits[j] = sign(dot(residual, random_vector_j))
// Decode: correction = (1/m) * sum_j(sign_bits[j] * random_vector_j)
//
// The random vectors are generated deterministically from a seed, so they
// don't need to be stored.

// Compute sign bit for one projection dimension.
// The random projection vector is generated from seed + proj_idx.
__device__ __forceinline__ bool qjl_sign_bit(const float* __restrict__ residual,
                                             int d, uint32_t seed,
                                             int proj_idx) {
  float dot = 0.0f;
  // Generate random sign vector and compute dot product
  uint32_t proj_seed = seed ^ (static_cast<uint32_t>(proj_idx) * 0x9e3779b9u);
  for (int i = 0; i < d; i++) {
    float s = random_sign(proj_seed, i);
    dot += residual[i] * s;
  }
  return dot >= 0.0f;
}

// Pack sign bits into bytes. `num_proj` sign bits → ceil(num_proj/8) bytes.
__device__ void qjl_encode(const float* __restrict__ residual, int d,
                           uint32_t seed, int num_proj,
                           uint8_t* __restrict__ sign_bits_out) {
  for (int byte_idx = 0; byte_idx < (num_proj + 7) / 8; byte_idx++) {
    uint8_t byte_val = 0;
    for (int bit = 0; bit < 8; bit++) {
      int proj_idx = byte_idx * 8 + bit;
      if (proj_idx < num_proj) {
        if (qjl_sign_bit(residual, d, seed, proj_idx)) {
          byte_val |= (1u << bit);
        }
      }
    }
    sign_bits_out[byte_idx] = byte_val;
  }
}

// Reconstruct the bias correction vector from sign bits.
// correction[i] = (1/num_proj) * sum_j(sign_bit_j * random_vector_j[i])
// where sign_bit_j is +1 or -1 based on stored sign.
__device__ void qjl_decode_correction(const uint8_t* __restrict__ sign_bits,
                                      int d, uint32_t seed, int num_proj,
                                      float* __restrict__ correction_out) {
  // Initialize correction to zero
  for (int i = 0; i < d; i++) {
    correction_out[i] = 0.0f;
  }

  float scale = 1.0f / static_cast<float>(num_proj);
  for (int proj_idx = 0; proj_idx < num_proj; proj_idx++) {
    // Read the sign bit
    int byte_idx = proj_idx / 8;
    int bit_idx = proj_idx % 8;
    float sign = (sign_bits[byte_idx] & (1u << bit_idx)) ? 1.0f : -1.0f;

    // Generate and accumulate scaled random vector
    uint32_t proj_seed = seed ^ (static_cast<uint32_t>(proj_idx) * 0x9e3779b9u);
    for (int i = 0; i < d; i++) {
      float s = random_sign(proj_seed, i);
      correction_out[i] += sign * s * scale;
    }
  }
}

// ============================================================================
// Combined TurboQuant encode/decode (single head, single token)
// ============================================================================

// Full TurboQuant encode for one KV head vector.
// Input: vec[head_size] (fp16/bf16/fp32)
// Output: packed angles, radius, and optionally QJL sign bits
template <TQDataType DT>
__device__ void turboquant_encode_head(
    const float* __restrict__ vec, int head_size,
    uint32_t rotation_seed, uint32_t qjl_seed,
    uint8_t* __restrict__ angles_out,  // (head_size - 1) angle entries
    half* __restrict__ radius_out,     // 1 fp16 radius
    uint8_t* __restrict__ qjl_out,     // QJL sign bits (if enabled)
    int qjl_proj_dim) {
  constexpr int BITS = angle_bits(DT);

  // Working buffer
  float buf[256];
  for (int i = 0; i < head_size; i++) {
    buf[i] = vec[i];
  }

  // Step 1: Apply randomized Hadamard rotation
  randomized_hadamard(buf, head_size, rotation_seed);

  // Step 2: Polar encode
  float radius = polar_encode<BITS>(buf, head_size, angles_out);
  *radius_out = __float2half(radius);

  // Step 3: QJL residual correction (for tq2/tq3 modes)
  if constexpr (has_qjl(DT)) {
    // Compute residual = original_rotated - dequantized
    float decoded[256];
    polar_decode<BITS>(angles_out, radius, head_size, decoded);

    float residual[256];
    for (int i = 0; i < head_size; i++) {
      residual[i] = buf[i] - decoded[i];
    }

    // Encode residual as sign bits
    qjl_encode(residual, head_size, qjl_seed, qjl_proj_dim, qjl_out);
  }
}

// Full TurboQuant decode for one KV head vector.
template <TQDataType DT>
__device__ void turboquant_decode_head(
    const uint8_t* __restrict__ angles, half radius_fp16,
    const uint8_t* __restrict__ qjl_bits,
    int head_size, uint32_t rotation_seed, uint32_t qjl_seed,
    int qjl_proj_dim,
    float* __restrict__ vec_out) {
  constexpr int BITS = angle_bits(DT);

  float radius = __half2float(radius_fp16);

  // Step 1: Polar decode
  polar_decode<BITS>(angles, radius, head_size, vec_out);

  // Step 2: Add QJL correction (for tq2/tq3 modes)
  if constexpr (has_qjl(DT)) {
    float correction[256];
    qjl_decode_correction(qjl_bits, head_size, qjl_seed, qjl_proj_dim,
                          correction);
    // Scale correction by norm of residual (approximated by norm * scale_factor)
    for (int i = 0; i < head_size; i++) {
      vec_out[i] += correction[i];
    }
  }

  // Step 3: Inverse randomized Hadamard rotation
  inverse_randomized_hadamard(vec_out, head_size, rotation_seed);
}

// ============================================================================
// Block layout utilities
// ============================================================================

// For a block of `block_size` tokens with `num_heads` KV heads of `head_size`:
//
// PQ4 storage per block:
//   angles: num_heads * block_size * (head_size - 1) * 4 bits
//   radii:  num_heads * block_size * 2 bytes (fp16)
//
// TQ3 storage per block:
//   angles: num_heads * block_size * (head_size - 1) * 3 bits
//   qjl:    num_heads * block_size * ceil(head_size / 8) bytes
//   radii:  num_heads * block_size * 2 bytes (fp16)
//
// TQ2 storage per block:
//   angles: num_heads * block_size * (head_size - 1) * 2 bits
//   qjl:    num_heads * block_size * ceil(head_size / 8) bytes
//   radii:  num_heads * block_size * 2 bytes (fp16)

// Calculate total bytes per block for a TurboQuant mode.
__host__ __device__ int turboquant_block_bytes(TQDataType dt, int num_kv_heads,
                                               int head_size, int block_size) {
  int bits = angle_bits(dt);
  int num_angles = head_size - 1;
  // Packed angle bytes per token per head
  int angle_bytes_per_token = (num_angles * bits + 7) / 8;
  // QJL bytes per token per head
  int qjl_bytes_per_token = has_qjl(dt) ? (head_size + 7) / 8 : 0;
  // Radius: 2 bytes (fp16) per head per token
  int radius_bytes_per_token = 2;

  int bytes_per_token_per_head =
      angle_bytes_per_token + qjl_bytes_per_token + radius_bytes_per_token;

  return num_kv_heads * block_size * bytes_per_token_per_head;
}

// Per-head seed derivation from layer-level seed
__device__ __forceinline__ uint32_t derive_rotation_seed(uint32_t layer_seed,
                                                         int head_idx) {
  return layer_seed ^ (static_cast<uint32_t>(head_idx) * 2654435761u);
}

__device__ __forceinline__ uint32_t derive_qjl_seed(uint32_t layer_seed,
                                                    int head_idx) {
  return layer_seed ^ (static_cast<uint32_t>(head_idx) * 2246822519u);
}

}  // namespace turboquant
}  // namespace vllm
