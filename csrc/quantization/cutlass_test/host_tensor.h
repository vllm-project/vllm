/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

/*! \file
  \brief HostTensor contributes management for both host and device memory.

  HostTensor allocates host and device memory upon construction. Basic element-wise operations on
  host memory synchronize device memory automatically. Explicit copy operations provide abstractions
  for CUDA memcpy operations.

  Call {host, device}_{data, ref, view}() for accessing host or device memory.

  See cutlass/tensor_ref.h and cutlass/tensor_view.h for more details.
*/

#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/fast_math.h"

#include "device_memory.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Host tensor
template <
  /// Data type of element stored within tensor (concept: NumericType)
  typename Element_,
  /// Defines a mapping from logical coordinate to linear memory (concept: Layout)
  typename Layout_
>
class HostTensor {
public:

  /// Data type of individual access
  using Element = Element_;

  /// Mapping function from logical coordinate to linear memory
  using Layout = Layout_;

  /// Logical rank of tensor index space
  static int const kRank = Layout::kRank;

  /// Index type
  using Index = typename Layout::Index;

  /// Long index used for pointer offsets
  using LongIndex = typename Layout::LongIndex;

  /// Coordinate in logical tensor space
  using TensorCoord = typename Layout::TensorCoord;

  /// Layout's stride vector
  using Stride = typename Layout::Stride;

  /// Tensor reference to device memory
  using TensorRef = TensorRef<Element, Layout>;

  /// Tensor reference to constant device memory
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  /// Tensor reference to device memory
  using TensorView = TensorView<Element, Layout>;

  /// Tensor reference to constant device memory
  using ConstTensorView = typename TensorView::ConstTensorView;

  /// Reference to element in tensor
  using Reference = typename TensorRef::Reference;

  /// Constant reference to element in tensor
  using ConstReference = typename ConstTensorRef::Reference;

private:
  using StorageUnit = typename platform::conditional_t<std::is_same_v<Element, bool>, uint8_t,            // Avoid the std::vector<bool> specialization
                                  typename platform::conditional_t<sizeof_bits<Element>::value % 8 == 0,  // Handle subbyte types
                                      Element, uint8_t>>;
  using StorageContainerCalculator = cutlass::detail::StorageContainerCalculator<Element, StorageUnit>;
  static constexpr int kContainerTypeNumBits = StorageContainerCalculator::kContainerTypeNumBits;
  static constexpr int kContainerTypeNumLogicalElements = StorageContainerCalculator::kContainerTypeNumLogicalElements;
  static constexpr int kContainerTypeNumBytes = StorageContainerCalculator::kContainerTypeNumBytes;
  static constexpr int kContainerTypeNumStorageUnit = StorageContainerCalculator::kContainerTypeNumStorageUnit;

  //
  // Data members
  //

  /// Extent of tensor in logical dimensions
  TensorCoord extent_;

  /// Layout object
  Layout layout_;

  /// Host-side memory allocation
  std::vector<StorageUnit> host_;

  /// Device-side memory
  device_memory::allocation<StorageUnit> device_;

  /// number of containers 
  size_t count_to_container_storage_unit_count(size_t count) {
    return (count + kContainerTypeNumLogicalElements - 1) / kContainerTypeNumLogicalElements * kContainerTypeNumStorageUnit;
  }

public:
  //
  // Device and Host Methods
  //

  /// Default constructor
  HostTensor() {}

  /// Constructs a tensor given an extent. Assumes a packed layout
  HostTensor(
    TensorCoord const &extent,
    bool device_backed = true
  ) {

    this->reset(extent, Layout::packed(extent), device_backed);
  }

  /// Constructs a tensor given an extent and layout
  HostTensor(
    TensorCoord const &extent,
    Layout const &layout,
    bool device_backed = true
  ) {

    this->reset(extent, layout, device_backed);
  }

  ~HostTensor() { }

  /// Clears the HostTensor allocation to size/capacity = 0
  void reset() {
    extent_ = TensorCoord();
    layout_ = Layout::packed(extent_);

    host_.clear();
    device_.reset();
  }

  /// Resizes internal memory allocations without affecting layout or extent
  void reserve(
    size_t count,                                        ///< size of tensor in elements
    bool device_backed_ = true) {                        ///< if true, device memory is also allocated
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("cutlass::HostTensor::reserve(count=" << count << ", device_backed_=" << (device_backed_ ? "true" : "false") << ")");
#endif

    device_.reset();
    host_.clear();

    size_t count_container = count_to_container_storage_unit_count(count);
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("cutlass::HostTensor::reserve: host_.resize(" << count_container << ")");
#endif    
    host_.resize(count_container);

    // Allocate memory
    StorageUnit* device_memory = nullptr;
    if (device_backed_) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("cutlass::HostTensor::reserve: device_memory::allocate(" << count_container << ")");
#endif
      device_memory = device_memory::allocate<StorageUnit>(count_container);
    }
    device_.reset(device_memory, device_backed_ ? count_container : 0);
  }

  /// Updates the extent and layout of the HostTensor. Allocates memory according to the new
  /// extent and layout.
  void reset(
    TensorCoord const &extent,                           ///< extent of logical tensor
    Layout const &layout,                                ///< layout object of tensor
    bool device_backed_ = true) {                        ///< if true, device memory is also allocated. 

    extent_ = extent;
    layout_ = layout;

    reserve(size_t(layout_.capacity(extent_)), device_backed_);
  }

  /// Updates the extent and layout of the HostTensor. Allocates memory according to the new
  /// extent and layout. Assumes a packed tensor configuration.
  void reset(
    TensorCoord const &extent,                           ///< extent of logical tensor
    bool device_backed_ = true) {                        ///< if true, device memory is also allocated. 

    reset(extent, Layout::packed(extent), device_backed_);
  }

  /// Changes the size of the logical tensor. Only allocates memory if new capacity exceeds reserved capacity.
  /// To force allocation, call reset().
  void resize(
    TensorCoord const &extent,                           ///< extent of logical tensor
    Layout const &layout,                                ///< layout object of tensor
    bool device_backed_ = true) {                        ///< if true, device memory is also allocated. 

    extent_ = extent;
    layout_ = layout;

    LongIndex new_size = size_t(layout_.capacity(extent_));
    LongIndex new_size_container = count_to_container_storage_unit_count((layout_.capacity(extent_)));

    if (static_cast<decltype(host_.size())>(new_size_container) > host_.size()) {
      reserve(new_size, device_backed_);
    }
  }

  /// Changes the size of the logical tensor. Only allocates memory if new capacity exceeds reserved capacity.
  /// To force allocation, call reset(). Note, this form of resize() assumes a packed tensor configuration.
  void resize(
    TensorCoord const &extent,                           ///< extent of logical tensor
    bool device_backed_ = true) {                        ///< if true, device memory is also allocated. 

    resize(extent, Layout::packed(extent), device_backed_);
  }

  /// Returns the logical number of elements stored in the host tensor
  size_t size() const {
    return layout_.capacity(extent_);
  }

  /// Returns the logical capacity in terms of number of elements. May be larger than the size().
  LongIndex capacity() const {
    return host_.size() / kContainerTypeNumStorageUnit * kContainerTypeNumLogicalElements;
  }

  /// Gets pointer to host data
  Element * host_data() { return reinterpret_cast<Element *>(host_.data()); }

  /// Gets pointer to host data with a pointer offset
  Element * host_data_ptr_offset(LongIndex ptr_element_offset) { return &ReferenceFactory<Element>::get(host_data(), ptr_element_offset); }

  /// Gets a reference to an element in host memory
  Reference host_data(LongIndex idx) {
    return ReferenceFactory<Element>::get(host_data(), idx);
  }

  /// Gets pointer to host data
  Element const * host_data() const { return reinterpret_cast<Element const *>(host_.data()); }

  /// Gets pointer to host data with a pointer offset
  Element const * host_data_ptr_offset(LongIndex ptr_element_offset) const { return &ReferenceFactory<Element>::get(host_data(), ptr_element_offset); }

  /// Gets a constant reference to an element in host memory
  ConstReference host_data(LongIndex idx) const {
    return ReferenceFactory<Element const>::get(host_data(), idx);
  }

  /// Gets pointer to device data
  Element * device_data() { return reinterpret_cast<Element *>(device_.get()); }

  /// Gets pointer to device data
  Element const * device_data() const { return reinterpret_cast<Element const *>(device_.get()); }

  /// Gets pointer to device data with a pointer offset
  Element * device_data_ptr_offset(LongIndex ptr_element_offset) { return &ReferenceFactory<Element>::get(device_data(), ptr_element_offset); }

  /// Gets pointer to device data with a pointer offset
  Element const * device_data_ptr_offset(LongIndex ptr_element_offset) const { return &ReferenceFactory<Element>::get(device_data(), ptr_element_offset); }

  /// Accesses the tensor reference pointing to data
  TensorRef host_ref(LongIndex ptr_element_offset=0) { return TensorRef(host_data_ptr_offset(ptr_element_offset), layout_); }

  /// Accesses the tensor reference pointing to data
  ConstTensorRef host_ref(LongIndex ptr_element_offset=0) const { return ConstTensorRef(host_data_ptr_offset(ptr_element_offset), layout_); }

  /// Accesses the tensor reference pointing to data
  TensorRef device_ref(LongIndex ptr_element_offset=0) {
    return TensorRef(device_data_ptr_offset(ptr_element_offset), layout_);
  }

  /// Accesses the tensor reference pointing to data
  ConstTensorRef device_ref(LongIndex ptr_element_offset=0) const {
    return TensorRef(device_data_ptr_offset(ptr_element_offset), layout_);
  }

  /// Accesses the tensor reference pointing to data
  TensorView host_view(LongIndex ptr_element_offset=0) {
    return TensorView(host_data_ptr_offset(ptr_element_offset), layout_, extent_);
  }

  /// Accesses the tensor reference pointing to data
  ConstTensorView host_view(LongIndex ptr_element_offset=0) const {
    return ConstTensorView(host_data_ptr_offset(ptr_element_offset), layout_, extent_);
  }

  /// Accesses the tensor reference pointing to data
  TensorView device_view(LongIndex ptr_element_offset=0) {
    return TensorView(device_data_ptr_offset(ptr_element_offset), layout_, extent_);
  }

  /// Accesses the tensor reference pointing to data
  ConstTensorView device_view(LongIndex ptr_element_offset=0) const {
    return ConstTensorView(device_data_ptr_offset(ptr_element_offset), layout_, extent_);
  }

  /// Returns true if device memory is allocated
  bool device_backed() const {
    return (device_.get() == nullptr) ? false : true;
  }


  /// Returns the layout object
  Layout & layout() {
    return layout_;
  }

  /// Returns the layout object
  Layout layout() const {
    return layout_;
  }

  /// Returns the layout object's stride vector
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the layout object's stride vector
  Stride & stride() {
    return layout_.stride();
  }

  /// Returns the layout object's stride in a given physical dimension
  LongIndex stride(int dim) const {
    return layout_.stride().at(dim);
  }

  /// Returns the layout object's stride in a given physical dimension
  LongIndex & stride(int dim) {
    return layout_.stride().at(dim);
  }

  /// Computes the offset of an index from the origin of the tensor
  LongIndex offset(TensorCoord const& coord) const {
    return layout_(coord);
  }

  /// Returns a reference to the element at the logical Coord in host memory
  Reference at(TensorCoord const& coord) {
    return host_data(offset(coord));
  }

  /// Returns a const reference to the element at the logical Coord in host memory
  ConstReference at(TensorCoord const& coord) const {
    return host_data(offset(coord));
  }

  /// Returns the extent of the tensor
  TensorCoord extent() const {
    return extent_;
  }

  /// Returns the extent of the tensor
  TensorCoord & extent() {
    return extent_;
  }

  /// Copies data from device to host
  void sync_host() {
    if (device_backed()) {
      device_memory::copy_to_host(
          host_.data(), device_.get(), device_.size());
    }
  }

  /// Copies data from host to device
  void sync_device() {
    if (device_backed()) {
      device_memory::copy_to_device(
          device_.get(), host_.data(), host_.size());
    }
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_in_device_to_host(
    Element const* ptr_device,        ///< source device memory
    LongIndex count = -1) {           ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_to_host(
      host_.data(), reinterpret_cast<StorageUnit const *>(ptr_device), container_count);
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_in_device_to_device(
    Element const* ptr_device,        ///< source device memory
    LongIndex count = -1) {           ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_device_to_device(
      device_.get(), reinterpret_cast<StorageUnit const *>(ptr_device), container_count);
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_in_host_to_device(
    Element const* ptr_host,          ///< source host memory
    LongIndex count = -1) {           ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_to_device(
      device_.get(), reinterpret_cast<StorageUnit const *>(ptr_host), container_count);
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_in_host_to_host(
    Element const* ptr_host,          ///< source host memory
    LongIndex count = -1) {           ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_host_to_host(
      host_.data(), reinterpret_cast<StorageUnit const *>(ptr_host), container_count);
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_out_device_to_host(
    Element * ptr_host,               ///< source device memory
    LongIndex count = -1) const {     ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_to_host(
      reinterpret_cast<StorageUnit *>(ptr_host), device_.get(), container_count);
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_out_device_to_device(
    Element * ptr_device,             ///< source device memory
    LongIndex count = -1) const {     ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_device_to_device(
      reinterpret_cast<StorageUnit *>(ptr_device), device_.get(), container_count);
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_out_host_to_device(
    Element * ptr_device,             ///< source host memory
    LongIndex count = -1) const {     ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_to_device(
      reinterpret_cast<StorageUnit *>(ptr_device), host_.data(), container_count);
  }

  /// Copy data from a caller-supplied device pointer into host memory.
  void copy_out_host_to_host(
    Element * ptr_host,               ///< source host memory
    LongIndex count = -1) const {     ///< number of elements to transfer; if negative, entire tensor is overwritten.

    if (count < 0) {
      count = capacity();
    }
    else {
      count = __NV_STD_MIN(capacity(), count);
    }
    size_t container_count = count_to_container_storage_unit_count(count);
    device_memory::copy_host_to_host(
      reinterpret_cast<StorageUnit *>(ptr_host), host_.data(), container_count);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
