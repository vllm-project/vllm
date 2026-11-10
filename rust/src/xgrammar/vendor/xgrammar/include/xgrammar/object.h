/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/object.h
 * \brief Utilities for creating objects.
 */

#ifndef XGRAMMAR_OBJECT_H_
#define XGRAMMAR_OBJECT_H_

#include <memory>   // IWYU pragma: keep
#include <utility>  // IWYU pragma: keep

namespace xgrammar {

/*!
 * \brief A tag type for creating a null object.
 */
struct NullObj {};

/*!
 * \brief This macro defines the methods for the PImpl classes.
 * \details Many classes in xgrammar are PImpl classes. PImpl classes only stores a shared pointer
 * to the implementation. This allows reference-counter-based memory management and efficient
 * object copy and passing. We always expose PImpl classes to Python to control over object sharing
 * and memory management. Note simple and critical classes should not be defined as PImpl classes,
 * but as normal classes for better efficiency.
 */
#define XGRAMMAR_DEFINE_PIMPL_METHODS(TypeName)                                \
 public:                                                                       \
  class Impl;                                                                  \
  /* Construct a null object. Note operating on a null object will fail. */    \
  explicit TypeName(NullObj) : pimpl_(nullptr) {}                              \
  /* Construct object with a shared pointer to impl. */                        \
  explicit TypeName(std::shared_ptr<Impl> pimpl) : pimpl_(std::move(pimpl)) {} \
  TypeName(const TypeName& other) = default;                                   \
  TypeName(TypeName&& other) noexcept = default;                               \
  TypeName& operator=(const TypeName& other) = default;                        \
  TypeName& operator=(TypeName&& other) noexcept = default;                    \
  bool IsNull() const { return pimpl_ == nullptr; }                            \
  /* Access the impl pointer. Useful in implementation. */                     \
  Impl* ImplPtr() { return pimpl_.get(); }                                     \
  const Impl* ImplPtr() const { return pimpl_.get(); }                         \
  Impl* operator->() { return pimpl_.get(); }                                  \
  const Impl* operator->() const { return pimpl_.get(); }                      \
                                                                               \
 private:                                                                      \
  std::shared_ptr<Impl> pimpl_

}  // namespace xgrammar

#endif  // XGRAMMAR_OBJECT_H_
