/*!
 * Copyright (c) 2024 by Contributors
 * \file xgrammar/support/dynamic_bitset.h
 * \brief The header for utilities used in grammar-guided generation.
 */
#ifndef XGRAMMAR_SUPPORT_DYNAMIC_BITSET_H_
#define XGRAMMAR_SUPPORT_DYNAMIC_BITSET_H_

#include <picojson.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// For __popcnt
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "json_serializer.h"
#include "logging.h"

namespace xgrammar {

/*!
 * \brief A bitset whose length is specified at runtime. Note the size cannot be changed after
 * construction.
 * \details The buffer of the bitset is a uint32_t array. There are two uses for this class:
 * - When passing nullptr to data, it maintains an internal buffer for the bitset.
 * - When passing a pointer to a buffer with enough size, it uses the external buffer for the
 *   bitset.
 * \details Part of the implementation is adopted from Boost::dynamic_bitset.
 */
class DynamicBitset {
 public:
  /*!
   * \brief Calculate the minimal size of the uint32_t buffer for the bitset with the given size.
   * \param element_size The size of the bitset.
   * \return The minimal buffer size.
   */
  static int GetBufferSize(int element_size) { return (element_size + 31) / 32; }

  /*!
   * \brief Construct a empty bitset. This object should be assigned to a valid bitset before using.
   */
  DynamicBitset() : size_(0), buffer_size_(0), data_(nullptr), is_internal_(true) {}

  /*!
   * \brief Construct a bitset with the given size.
   * \param size The size of the bitset.
   * \param data The buffer for the bitset. If nullptr, the bitset will maintain an internal buffer.
   */
  DynamicBitset(int size, uint32_t* data = nullptr)
      : size_(size), buffer_size_(GetBufferSize(size)) {
    if (data == nullptr) {
      internal_buffer_.resize(buffer_size_, 0);
      data_ = internal_buffer_.data();
      is_internal_ = true;
    } else {
      data_ = data;
      is_internal_ = false;
    }
  }

  /*! \brief Copy constructor. Copy the buffer and manage the memory internally. */
  DynamicBitset(const DynamicBitset& other)
      : size_(other.size_),
        buffer_size_(other.buffer_size_),
        data_(),
        internal_buffer_(),
        is_internal_(other.is_internal_) {
    if (other.is_internal_) {
      // copy the internal buffer
      internal_buffer_ = other.internal_buffer_;
      data_ = internal_buffer_.data();
    } else {
      // simply point to the same external buffer
      data_ = other.data_;
    }
  }

  /*! \brief Move constructor. Reset other and take ownership of its buffer. */
  DynamicBitset(DynamicBitset&& other) noexcept
      : size_(std::exchange(other.size_, 0)),
        buffer_size_(std::exchange(other.buffer_size_, 0)),
        data_(std::exchange(other.data_, nullptr)),
        internal_buffer_(std::move(other.internal_buffer_)),
        is_internal_(std::exchange(other.is_internal_, true)) {}

  /*! \brief Copy assignment. */
  DynamicBitset& operator=(const DynamicBitset& other) {
    XGRAMMAR_DCHECK(is_internal_ || size_ >= other.size_)
        << "Expanding bitset size is not allowed when the "
           "memory of the bitset is externally managed";
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    if (is_internal_) {
      internal_buffer_.resize(buffer_size_);
      data_ = internal_buffer_.data();
    }
    if (data_ != other.data_) {
      std::memcpy(data_, other.data_, buffer_size_ * sizeof(uint32_t));
    }
    return *this;
  }

  /*! \brief Move assignment. */
  DynamicBitset& operator=(DynamicBitset&& other) noexcept {
    size_ = other.size_;
    buffer_size_ = other.buffer_size_;
    is_internal_ = other.is_internal_;
    if (is_internal_) {
      internal_buffer_ = std::move(other.internal_buffer_);
      data_ = internal_buffer_.data();
    } else {
      data_ = other.data_;
    }
    return *this;
  }

  /*! \brief Get the value of the bit at the given index. */
  bool operator[](int index) const {
    XGRAMMAR_DCHECK(data_ && index >= 0 && index < size_);
    return (data_[index / 32] >> (index % 32)) & 1;
  }

  /*! \brief Get the size of the bitset. */
  int Size() const { return size_; }

  /*! \brief Set the whole bitset to true. */
  void Set() {
    XGRAMMAR_DCHECK(data_);
    std::memset(data_, 0xFF, buffer_size_ * sizeof(uint32_t));
  }

  /*! \brief Set the bit at the given index to the given value. */
  void Set(int index, bool value = true) {
    XGRAMMAR_DCHECK(data_ && index >= 0 && index < size_);
    if (value) {
      data_[index / 32] |= 1 << (index % 32);
    } else {
      data_[index / 32] &= ~(1 << (index % 32));
    }
  }

  /*! \brief Set the whole bitset to false. */
  void Reset() {
    XGRAMMAR_DCHECK(data_);
    std::memset(data_, 0, buffer_size_ * sizeof(uint32_t));
  }

  /*! \brief Set the bit at the given index to false. */
  void Reset(int index) { Set(index, false); }

  /*! \brief Perform a bitwise OR operation between the current bitset and another bitset. */
  DynamicBitset& operator|=(const DynamicBitset& other) {
    XGRAMMAR_DCHECK(buffer_size_ <= other.buffer_size_);
    for (int i = 0; i < buffer_size_; ++i) {
      data_[i] |= other.data_[i];
    }
    return *this;
  }

  int FindFirstOne() const { return DoFindOneFrom(0); }

  int FindNextOne(int pos) const {
    if (pos >= size_ - 1 || size_ == 0) return -1;
    ++pos;
    int blk = pos / BITS_PER_BLOCK;
    int ind = pos % BITS_PER_BLOCK;
    uint32_t fore = data_[blk] >> ind;
    int result = fore ? pos + LowestBit(fore) : DoFindOneFrom(blk + 1);
    return result < size_ ? result : -1;
  }

  int FindFirstZero() const { return DoFindZeroFrom(0); }

  int FindNextZero(int pos) const {
    if (pos >= size_ - 1 || size_ == 0) return -1;
    ++pos;
    int blk = pos / BITS_PER_BLOCK;
    int ind = pos % BITS_PER_BLOCK;
    uint32_t fore = (~data_[blk]) >> ind;
    int result = fore ? pos + LowestBit(fore) : DoFindZeroFrom(blk + 1);
    return result < size_ ? result : -1;
  }

  int Count() const {
    int count = 0;
    for (int i = 0; i < buffer_size_; ++i) {
      count += PopCount(data_[i]);
    }
    return count;
  }

  bool All() const {
    if (size_ == 0) return true;
    // Check all complete blocks except the last one
    for (int i = 0; i < buffer_size_ - 1; ++i) {
      if (data_[i] != ~static_cast<uint32_t>(0)) {
        return false;
      }
    }
    // For the last block, create a mask for valid bits only
    int remaining_bits = size_ % BITS_PER_BLOCK;
    uint32_t last_block_mask = remaining_bits ? (static_cast<uint32_t>(1) << remaining_bits) - 1
                                              : ~static_cast<uint32_t>(0);
    return (data_[buffer_size_ - 1] & last_block_mask) == last_block_mask;
  }

  bool Any() const {
    if (size_ == 0) return false;
    // Check all complete blocks except the last one
    for (int i = 0; i < buffer_size_ - 1; ++i) {
      if (data_[i] != 0) {
        return true;
      }
    }
    // For the last block, only consider the valid bits
    int remaining_bits = size_ % BITS_PER_BLOCK;
    uint32_t last_block_mask = remaining_bits ? (static_cast<uint32_t>(1) << remaining_bits) - 1
                                              : ~static_cast<uint32_t>(0);
    return (data_[buffer_size_ - 1] & last_block_mask) != 0;
  }

  static constexpr int BITS_PER_BLOCK = 32;

  friend std::size_t MemorySize(const DynamicBitset& bitset) {
    return bitset.buffer_size_ * sizeof(bitset.data_[0]);
  }

  friend picojson::value SerializeJSONValue(const DynamicBitset& bitset) {
    XGRAMMAR_DCHECK(bitset.buffer_size_ == GetBufferSize(bitset.size_));
    picojson::array result;
    result.reserve(2 + bitset.buffer_size_);
    result.emplace_back(picojson::value(static_cast<int64_t>(bitset.size_)));
    result.emplace_back(picojson::value(static_cast<int64_t>(bitset.buffer_size_)));
    for (int i = 0; i < bitset.buffer_size_; ++i) {
      result.emplace_back(picojson::value(static_cast<int64_t>(bitset.data_[i])));
    }
    return picojson::value(std::move(result));
  }

  friend std::optional<SerializationError> DeserializeJSONValue(
      DynamicBitset* bitset, const picojson::value& value, const std::string& type_name
  ) {
    if (!value.is<picojson::array>()) {
      return ConstructDeserializeError("Expect an array", type_name);
    }
    const auto& arr = value.get<picojson::array>();
    if (arr.size() < 2) {
      return ConstructDeserializeError("Except at least 2 elements in the array", type_name);
    }
    if (!arr[0].is<int64_t>()) {
      return ConstructDeserializeError("Expect an integer for size", type_name);
    }
    int size = static_cast<int>(arr[0].get<int64_t>());
    if (!arr[1].is<int64_t>()) {
      return ConstructDeserializeError("Expect an integer for buffer_size", type_name);
    }
    int buffer_size = static_cast<int>(arr[1].get<int64_t>());
    if (size < 0 || buffer_size != GetBufferSize(size)) {
      return ConstructDeserializeError(
          "Invalid buffer_size. Buffer size should be ceil(size / 32)", type_name
      );
    }
    if (static_cast<int64_t>(arr.size()) != static_cast<int64_t>(buffer_size) + 2) {
      return ConstructDeserializeError(
          "Expect exactly buffer_size + 2 elements in the array", type_name
      );
    }

    DynamicBitset result(size);
    for (int i = 0; i < buffer_size; ++i) {
      if (!arr[i + 2].is<int64_t>()) {
        return ConstructDeserializeError("Expect an integer in the array", type_name);
      }
      int64_t value = arr[i + 2].get<int64_t>();
      if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
        return ConstructDeserializeError(
            "Integer in the array is " + std::to_string(value) + " and out of the uint32_t range",
            type_name
        );
      }
      result.data_[i] = static_cast<uint32_t>(value);
    }
    *bitset = std::move(result);
    return std::nullopt;
  }

  bool operator==(const DynamicBitset& other) const {
    if (size_ != other.size_) return false;
    if (buffer_size_ != other.buffer_size_) return false;
    for (int i = 0; i < buffer_size_; ++i) {
      if (data_[i] != other.data_[i]) return false;
    }
    return true;
  }

 private:
  static int LowestBit(uint32_t value) {
#ifdef __GNUC__
    return __builtin_ctz(value);
#else   // __GNUC__
    // From https://stackoverflow.com/a/757266
    static const int MultiplyDeBruijnBitPosition[32] = {0,  1,  28, 2,  29, 14, 24, 3,  30, 22, 20,
                                                        15, 25, 17, 4,  8,  31, 27, 13, 23, 21, 19,
                                                        16, 7,  26, 12, 18, 6,  11, 5,  10, 9};
    return MultiplyDeBruijnBitPosition[((uint32_t)((value & -value) * 0x077CB531U)) >> 27];
#endif  // __GNUC__
  }

  static int PopCount(uint32_t value) {
#ifdef __GNUC__
    return __builtin_popcount(value);
#elif defined(_MSC_VER)
    return __popcnt(value);
#else
    XGRAMMAR_LOG(FATAL) << "PopCount is not supported on this platform";
#endif
  }

  int DoFindZeroFrom(int first_block) const {
    int position = -1;
    for (int i = first_block; i < buffer_size_; ++i) {
      if (data_[i] != ~static_cast<uint32_t>(0)) {
        position = i;
        break;
      }
    }
    if (position == -1) return -1;
    return position * BITS_PER_BLOCK + LowestBit(~data_[position]);
  }

  int DoFindOneFrom(int first_block) const {
    int position = -1;
    for (int i = first_block; i < buffer_size_; ++i) {
      if (data_[i] != 0) {
        position = i;
        break;
      }
    }
    if (position == -1) return -1;
    return position * BITS_PER_BLOCK + LowestBit(data_[position]);
  }

  // The size of the bitset.
  int size_;
  // The size of the buffer.
  int buffer_size_;
  // The buffer for the bitset.
  uint32_t* data_;
  // The internal buffer. It is empty if not needed.
  std::vector<uint32_t> internal_buffer_;
  // Whether the buffer is internally managed.
  bool is_internal_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_DYNAMIC_BITSET_H_
