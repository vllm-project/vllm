/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/container.h
 * \brief The header for container.
 */
#ifndef XGRAMMAR_SUPPORT_CONTAINER_H_
#define XGRAMMAR_SUPPORT_CONTAINER_H_
#include <vector>

#include "logging.h"

namespace xgrammar {

namespace details {

template <typename Node>
class NodePool {
 public:
  NodePool() = default;

  void Reserve(int n) { node_pool_.reserve(n); }

  [[nodiscard]]
  int Allocate() {
    if (free_list_.empty()) {
      int node = Size();
      node_pool_.emplace_back();
      return node;
    } else {
      int node = free_list_.back();
      free_list_.pop_back();
      return node;
    }
  }

  void Deallocate(int node) { free_list_.push_back(node); }

  void Clear() {
    node_pool_.clear();
    free_list_.clear();
  }

  Node& operator[](int node) {
    XGRAMMAR_DCHECK(0 <= node && node < Size());
    return node_pool_[node];
  }

  int Size() const { return static_cast<int>(node_pool_.size()); }

 private:
  std::vector<Node> node_pool_;
  std::vector<int> free_list_;
};

}  // namespace details

template <typename Value>
class List {
 private:
  struct Node {
    int prev;
    int next;
    Value value;
  };

 public:
  struct iterator {
   public:
    iterator(int n, List& c) : node_(n), list_(&c) {
      XGRAMMAR_DCHECK(0 <= node_ && node_ < list_->node_pool_.Size());
    }
    iterator& operator++() {
      node_ = GetNode().next;
      return *this;
    }
    iterator operator++(int) {
      iterator tmp = *this;
      ++*this;
      return tmp;
    }
    Value& operator*() const { return GetNode().value; }
    Value* operator->() const { return &GetNode().value; }
    bool operator==(const iterator& rhs) const {
      XGRAMMAR_DCHECK(list_ == rhs.list_) << "compare different container is UB";
      return node_ == rhs.node_;  // compare different container is UB
    }
    bool operator!=(const iterator& rhs) const {
      XGRAMMAR_DCHECK(list_ == rhs.list_) << "compare different container is UB";
      return node_ != rhs.node_;  // compare different container is UB
    }

    int Index() const { return node_; }

   private:
    friend class List;
    Node& GetNode() const { return list_->node_pool_[node_]; }

    int node_;
    List* list_;
  };

  List(int reserved = 0) {
    node_pool_.Reserve(reserved);
    InitGuard();
  }

  iterator PushBack(const Value& value) {
    int node = node_pool_.Allocate();
    XGRAMMAR_DCHECK(0 < node && node < node_pool_.Size());
    node_pool_[node].value = value;
    LinkBefore(node, 0);
    return iterator(node, *this);
  }

  void MoveBack(int node) {
    XGRAMMAR_DCHECK(0 < node && node < node_pool_.Size());
    Unlink(node);
    LinkBefore(node, 0);
  }

  iterator Erase(iterator it) {
    int node = it.Index();
    XGRAMMAR_DCHECK(0 < node && node < node_pool_.Size());
    int next = node_pool_[node].next;
    Unlink(node);
    node_pool_.Deallocate(node);
    return iterator(next, *this);
  }

  void Clear() {
    node_pool_.Clear();
    InitGuard();
  }

  iterator begin() { return iterator(node_pool_[0].next, *this); }
  iterator end() { return iterator(0, *this); }

 private:
  void InitGuard() {
    int node_id = node_pool_.Allocate();
    XGRAMMAR_DCHECK(node_id == 0) << "node 0 should be reserved as guard node";
    node_pool_[0].prev = 0;
    node_pool_[0].next = 0;
  }

  void LinkBefore(int node, int next) {
    int prev = node_pool_[next].prev;
    node_pool_[node].prev = prev;
    node_pool_[node].next = next;
    node_pool_[prev].next = node;
    node_pool_[next].prev = node;
  }

  void Unlink(int node) {
    int prev = node_pool_[node].prev;
    int next = node_pool_[node].next;
    node_pool_[prev].next = next;
    node_pool_[next].prev = prev;
  }

  details::NodePool<Node> node_pool_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_CONTAINER_H_
