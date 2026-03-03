/**
 * binding_registry.h
 *
 * Lightweight singleton registry for PyBind11 kernel bindings.
 * Allows individual .cu kernel files to self-register their bindings
 * without requiring modification of binding.cpp.
 *
 * Do NOT modify this file.
 *
 * Usage in kernel binding files:
 *   #include "../binding_registry.h"
 *   REGISTER_BINDING(my_op, [](pybind11::module& m) {
 *       m.def("my_op_forward", &my_op_forward, ...);
 *   });
 */

#pragma once
#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <vector>

class BindingRegistry {
public:
    using BindingFn = std::function<void(pybind11::module&)>;

    static BindingRegistry& getInstance() {
        static BindingRegistry instance;
        return instance;
    }

    void registerBinding(const std::string& name, BindingFn fn) {
        bindings_.emplace_back(name, std::move(fn));
    }

    void applyBindings(pybind11::module& m) {
        for (auto& [name, fn] : bindings_) {
            fn(m);
        }
    }

private:
    BindingRegistry() = default;
    std::vector<std::pair<std::string, BindingFn>> bindings_;
};

/** Helper: auto-registers a binding function at static initialisation time. */
class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name,
                     BindingRegistry::BindingFn fn) {
        BindingRegistry::getInstance().registerBinding(name, std::move(fn));
    }
};

/**
 * REGISTER_BINDING(name, fn)
 *
 * Convenience macro. Creates a static BindingRegistrar that calls
 * BindingRegistry::getInstance().registerBinding at program start,
 * so that binding.cpp's PYBIND11_MODULE can apply them all via
 * BindingRegistry::getInstance().applyBindings(m).
 */
#define REGISTER_BINDING(name, fn)                                    \
    static BindingRegistrar _registrar_##name(#name, fn)
