#pragma once

// Registers the inputs a CUDA graph captured with every peer.

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"

ErrorCode hip_custom_communicator_register_graph(int64_t state) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;
  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);

  constexpr size_t kHandleBytes = sizeof(hipIpcMemHandle_t);
  constexpr size_t kEntryBytes = kHandleBytes + sizeof(int64_t);
  bool made = true;
  std::string mine;
  for (const auto& [slot, input] : s.input_tables.pending) {
    void* base = nullptr;
    hipIpcMemHandle_t h{};
    made = made &&
           hipPointerGetAttribute(
               &base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
               reinterpret_cast<hipDeviceptr_t>(const_cast<void*>(input))) ==
               hipSuccess &&
           hipIpcGetMemHandle(&h, base) == hipSuccess;
    const int64_t offset =
        static_cast<const char*>(input) - static_cast<const char*>(base);
    mine.append(reinterpret_cast<const char*>(&h), kHandleBytes);
    mine.append(reinterpret_cast<const char*>(&offset), sizeof(offset));
  }
  (void)hipGetLastError();

  const int64_t count = static_cast<int64_t>(s.input_tables.pending.size());
  std::string header(1, made ? '1' : '0');
  header.append(reinterpret_cast<const char*>(&count), sizeof(count));
  const std::optional<std::vector<std::string>> headers =
      all_gathered(*s.group, header, s.timeout_seconds);
  if (!headers) return error_code(Error::exchange_failed);
  if (!made) return error_code(Error::hip_failed);
  for (const std::string& h : *headers)
    if (h != header) return error_code(Error::ranks_disagree);
  const std::optional<std::vector<std::string>> everyone =
      count == 0 ? std::vector<std::string>(s.world)
                 : all_gathered(*s.group, mine, s.timeout_seconds);
  if (!everyone) return error_code(Error::exchange_failed);
  for (int64_t i = 0; i < count; ++i) {
    const auto& [slot, input] = s.input_tables.pending[i];
    std::vector<const void*> table(s.world, nullptr);
    for (int r = 0; r < s.world; ++r) {
      if (r == s.rank) {
        table[r] = input;
        continue;
      }
      const char* entry = (*everyone)[r].data() + i * kEntryBytes;
      const std::vector<char> key(entry, entry + kHandleBytes);
      int64_t offset = 0;
      std::memcpy(&offset, entry + kHandleBytes, sizeof(offset));
      auto it = s.input_tables.opened.find(key);
      if (it == s.input_tables.opened.end()) {
        hipIpcMemHandle_t h;
        std::memcpy(&h, key.data(), sizeof(h));
        void* base = nullptr;
        if (hipIpcOpenMemHandle(&base, h, hipIpcMemLazyEnablePeerAccess) !=
            hipSuccess) {
          (void)hipGetLastError();
          return error_code(Error::hip_failed);
        }
        it = s.input_tables.opened.emplace(key, static_cast<char*>(base)).first;
      }
      table[r] = it->second + offset;
    }
    if (hipMemcpy(s.input_tables.device + slot * s.world, table.data(),
                  table.size() * sizeof(void*),
                  hipMemcpyHostToDevice) != hipSuccess) {
      (void)hipGetLastError();
      return error_code(Error::hip_failed);
    }
  }
  s.input_tables.pending.clear();
  return std::nullopt;
}
