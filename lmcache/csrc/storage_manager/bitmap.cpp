// SPDX-License-Identifier: Apache-2.0

#include "bitmap.h"

#include <algorithm>

namespace lmcache {
namespace storage_manager {

namespace {

constexpr unsigned kBitsPerByte = 8;

inline size_t byte_index(size_t bit_index) { return bit_index / kBitsPerByte; }
inline unsigned bit_offset(size_t bit_index) {
  return static_cast<unsigned>(bit_index % kBitsPerByte);
}

}  // namespace

Bitmap::Bitmap(size_t size)
    : size_(size), data_(size == 0 ? 0 : (size - 1) / kBitsPerByte + 1, 0) {}

Bitmap::Bitmap(size_t size, size_t prefix_bits) : Bitmap(size) {
  if (prefix_bits == 0 || size == 0) return;
  if (prefix_bits > size) prefix_bits = size;

  // Set full bytes to 0xFF
  const size_t full_bytes = prefix_bits / kBitsPerByte;
  for (size_t i = 0; i < full_bytes; ++i) {
    data_[i] = 0xFF;
  }

  // Set remaining bits in the partial byte
  const unsigned remaining = prefix_bits % kBitsPerByte;
  if (remaining > 0) {
    data_[full_bytes] = static_cast<uint8_t>((1u << remaining) - 1);
  }
}

void Bitmap::set(size_t index) {
  if (index >= size_) return;
  data_[byte_index(index)] |= static_cast<uint8_t>(1u << bit_offset(index));
}

void Bitmap::batched_set(const std::vector<size_t>& indices) {
  for (size_t idx : indices) set(idx);
}

void Bitmap::set_range(size_t start, size_t end) {
  if (end > size_) end = size_;
  if (start >= end) return;

  const size_t first_byte = start / kBitsPerByte;
  const size_t last_byte = (end - 1) / kBitsPerByte;  // inclusive
  const unsigned head_low = bit_offset(start);

  if (first_byte == last_byte) {
    // Whole range lives in one byte: set bits [head_low, head_low + width).
    const unsigned width = static_cast<unsigned>(end - start);  // 1..8
    const uint8_t mask = static_cast<uint8_t>(((1u << width) - 1u) << head_low);
    data_[first_byte] |= mask;
    return;
  }

  // Head partial byte: bits [head_low, 8).
  data_[first_byte] |= static_cast<uint8_t>(0xFFu << head_low);
  // Middle full bytes.
  std::fill(data_.begin() + static_cast<std::ptrdiff_t>(first_byte + 1),
            data_.begin() + static_cast<std::ptrdiff_t>(last_byte),
            static_cast<uint8_t>(0xFF));
  // Tail partial byte: bits [0, tail_width).
  const unsigned tail_width =
      static_cast<unsigned>(end - last_byte * kBitsPerByte);  // 1..8
  data_[last_byte] |= static_cast<uint8_t>((1u << tail_width) - 1u);
}

void Bitmap::clear(size_t index) {
  if (index >= size_) return;
  data_[byte_index(index)] &= static_cast<uint8_t>(~(1u << bit_offset(index)));
}

bool Bitmap::test(size_t index) const {
  if (index >= size_) return false;
  return (data_[byte_index(index)] >> bit_offset(index)) & 1u;
}

size_t Bitmap::popcount() const {
  if (data_.empty()) return 0;

  // process full bytes
  size_t count = 0;
  const size_t num_full_bytes = size_ / kBitsPerByte;
  for (size_t i = 0; i < num_full_bytes; ++i) {
    count += static_cast<size_t>(
        __builtin_popcount(static_cast<unsigned>(data_[i])));
  }

  // process remaining bits in the last byte
  const unsigned remaining_bits = size_ % kBitsPerByte;
  if (remaining_bits > 0) {
    uint8_t last_byte = data_.back();
    uint8_t mask = static_cast<uint8_t>((1u << remaining_bits) - 1);
    last_byte &= mask;
    count += static_cast<size_t>(
        __builtin_popcount(static_cast<unsigned>(last_byte)));
  }
  return count;
}

size_t Bitmap::clz() const {
  if (size_ == 0) return 0;

  const size_t num_full_bytes = size_ / kBitsPerByte;
  size_t count = 0;

  for (size_t i = 0; i < num_full_bytes; ++i) {
    uint8_t b = data_[i];
    if (b == 0) {
      count += kBitsPerByte;
    } else {
      count += static_cast<size_t>(__builtin_ctz(static_cast<unsigned>(b)));
      return count;
    }
  }

  const unsigned remaining_bits = size_ % kBitsPerByte;
  if (remaining_bits > 0) {
    uint8_t last_byte = data_.back();
    uint8_t mask = static_cast<uint8_t>((1u << remaining_bits) - 1);
    last_byte &= mask;
    if (last_byte == 0) {
      count += remaining_bits;
    } else {
      count +=
          static_cast<size_t>(__builtin_ctz(static_cast<unsigned>(last_byte)));
    }
  }

  return count;
}

size_t Bitmap::clo() const {
  const Bitmap inverted{~(*this)};
  return inverted.clz();
}

int64_t Bitmap::highest_set_bit() const {
  // Scan bytes from the most significant down; the first non-zero byte holds
  // the highest set bit. Bits beyond size_ are never set, so the result is
  // always < size_.
  for (size_t byte = data_.size(); byte-- > 0;) {
    const uint8_t value = data_[byte];
    if (value != 0) {
      const unsigned highest =
          31u -
          static_cast<unsigned>(__builtin_clz(static_cast<unsigned>(value)));
      return static_cast<int64_t>(byte * kBitsPerByte + highest);
    }
  }
  return -1;
}

Bitmap Bitmap::operator&(const Bitmap& other) const {
  const size_t result_size = (size_ <= other.size_) ? size_ : other.size_;
  Bitmap result(result_size);
  for (size_t i = 0; i < result.data_.size(); ++i) {
    result.data_[i] = data_[i] & other.data_[i];
  }
  return result;
}

Bitmap Bitmap::operator|(const Bitmap& other) const {
  const size_t result_size = (size_ <= other.size_) ? size_ : other.size_;
  Bitmap result(result_size);
  for (size_t i = 0; i < result.data_.size(); ++i) {
    result.data_[i] = data_[i] | other.data_[i];
  }
  return result;
}

std::vector<size_t> Bitmap::get_indices() const {
  std::vector<size_t> indices;
  indices.reserve(popcount());
  for (size_t i = 0; i < data_.size(); ++i) {
    uint8_t byte = data_[i];
    while (byte != 0) {
      unsigned bit =
          static_cast<unsigned>(__builtin_ctz(static_cast<unsigned>(byte)));
      size_t idx = i * kBitsPerByte + bit;
      if (idx < size_) {
        indices.push_back(idx);
      }
      byte &= static_cast<uint8_t>(byte - 1);  // clear lowest set bit
    }
  }
  return indices;
}

std::unordered_set<size_t> Bitmap::get_indices_set() const {
  auto vec = get_indices();
  return std::unordered_set<size_t>(vec.begin(), vec.end());
}

std::string Bitmap::to_string() const {
  if (size_ == 0) return "";

  std::string result(size_, '0');
  for (size_t i = 0; i < data_.size(); ++i) {
    uint8_t byte = data_[i];
    for (unsigned j = 0; j < kBitsPerByte; ++j) {
      size_t bit_index = i * kBitsPerByte + j;
      if (bit_index >= size_) {
        break;
      }
      if ((byte >> j) & 1u) {
        result[bit_index] = '1';
      }
    }
  }

  return result;
}

Bitmap Bitmap::operator~() const {
  Bitmap result(size_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = ~data_[i];
  }

  // Clear bits that are out of range in the last byte
  const unsigned remaining_bits = size_ % kBitsPerByte;
  if (remaining_bits > 0) {
    uint8_t mask = static_cast<uint8_t>((1u << remaining_bits) - 1);
    result.data_.back() &= mask;
  }
  return result;
}

Bitmap::~Bitmap() = default;

}  // namespace storage_manager
}  // namespace lmcache
