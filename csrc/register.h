#pragma once

#include <iostream>

#include <torch/extension.h>

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

namespace vllm {

template <typename FnType>
void def(torch::Library& lib, std::string const& name, FnType* fn,
         std::initializer_list<int> mutating_arg_indices = {}) {
#if 1
  auto raw_schema =
      c10::detail::inferFunctionSchemaFromFunctor<std::decay_t<FnType>>();
  auto named_schema = raw_schema->cloneWithName(name, "");

  if (mutating_arg_indices.size() != 0) {
    std::vector<c10::Argument> const& args = named_schema.arguments();
    std::vector<c10::Argument> new_args;
    for (size_t i = 0; i < args.size(); ++i) {
      auto const& arg = args[i];
      if (std::find(mutating_arg_indices.begin(), mutating_arg_indices.end(),
                    i) == mutating_arg_indices.end()) {
        new_args.push_back(arg);
      } else {
        c10::AliasInfo new_alias_info;
        if (arg.alias_info()) {
          new_alias_info = *arg.alias_info();
        }
        new_alias_info.setIsWrite(true);

        new_args.emplace_back(arg.name(), arg.type(), arg.real_type(), arg.N(),
                              arg.default_value(), arg.kwarg_only(),
                              new_alias_info);
      }
    }

    named_schema = named_schema.cloneWithArguments(std::move(new_args));
  }

  lib.def(std::move(named_schema));
#else
  lib.def(name.c_str(), fn);
#endif
}

}  // namespace vllm
