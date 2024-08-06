#pragma once

#ifndef USE_ROCM
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

namespace detail {

template <typename... Ts>
union MultiUnion;

template <typename T>
union MultiUnion<T> {
  using type = T;
  type data;

  constexpr bool is_last() { return true; }

  template <size_t offset>
  constexpr T& get() {
    static_assert(offset == 0);
    return data;
  }
};

template <typename T, typename... Ts>
union MultiUnion<T, Ts...> {
  MultiUnion<T> head;
  MultiUnion<Ts...> tail;

  template <size_t offset>
  constexpr auto& get() {
    if constexpr (offset == 0) {
      return head.template get<0>();
    } else {
      return tail.template get<offset - 1>();
    }
  }
};

template <typename T, T... sizes>
struct is_ascending {
  static constexpr bool value = true;
};

template <typename T, T size1, T size2, T... sizes>
struct is_ascending<T, size1, size2, sizes...> {
  static constexpr bool value =
      size1 < size2 && is_ascending<T, size2, sizes...>::value;
};

template <typename T, T... sizes>
static constexpr bool is_ascending_v = is_ascending<T, sizes...>::value;

// Example usage/tests:
static_assert(is_ascending_v<size_t, 32, 64, 128, 256, 512, 1024>);
static_assert(!is_ascending_v<size_t, 64, 64>);
static_assert(!is_ascending_v<size_t, 64, 32, 80>);

}  // namespace detail

// BlockReduceMulti is a helper class that allows runtime dispatching to
// multiple block sizes for block reductions. When the number of threads
// participating in the reduction is not known at compile time, can select the
// smallest available block size that exceeds the number of threads.
//
// It uses a union to represent its shared storage, as only one block size is
// used at a time. This way no memory is wasted for the unused block sizes.
template <typename T, size_t... BlockSizes>
class BlockReduceMulti {
  static_assert(sizeof...(BlockSizes) > 0, "At least one block size required");
  static_assert(detail::is_ascending_v<size_t, BlockSizes...>,
                "Block sizes must be in ascending order");

  template <size_t I, size_t I0, size_t... Is>
  __device__ __host__ static constexpr size_t get() {
    static_assert(I < sizeof...(Is) + 1, "Index out of bounds");
    if constexpr (I == 0) {
      return I0;
    } else {
      return get<I - 1, Is...>();
    }
  }

 public:
  template <size_t BlockSize>
  using BlockReduce = cub::BlockReduce<T, BlockSize>;

  using TempStorage =
      detail::MultiUnion<typename BlockReduce<BlockSizes>::TempStorage...>;

  template <size_t I, typename ReductionOp>
  __device__ T reduce_impl(T input, ReductionOp op, size_t num_valid) {
    constexpr size_t block_size = get<I, BlockSizes...>();
    // If larger blocks are available and num_valid is larger than current,
    // try the next block size
    if constexpr (I < sizeof...(BlockSizes) - 1) {
      if (num_valid > block_size) {
        return reduce_impl<I + 1>(input, op, num_valid);
      }
    }

    // Either this is the last block size or num_valid is smaller than
    // block_size, so use it
    return BlockReduce<block_size>(storage.template get<I>())
        .Reduce(input, op, num_valid);
  }

  template <typename ReductionOp>
  __device__ T Reduce(T input, ReductionOp op, size_t num_valid) {
    return reduce_impl<0>(input, op, num_valid);
  }

  __device__ BlockReduceMulti(TempStorage& storage) : storage(storage) {}

 private:
  TempStorage& storage;
};

}  // namespace vllm
