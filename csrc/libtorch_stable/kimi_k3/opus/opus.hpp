/***************************************************************************************************
 * OPUS, AI (O)(P)erator Micro(U) (S)TD
 *
 * Crafting the micro standard templates for AI Operators on ROCm
 *
 * MIT License
 * Copyright (C) 2025-2026 carlus.huang@amd.com
 *
 **************************************************************************************************/
#pragma once

// clang-format off
#include <type_traits>
#include <utility>

#ifndef OPUS_ENABLE_RUNTIME_QUERY
#define OPUS_ENABLE_RUNTIME_QUERY 0
#endif

#if OPUS_ENABLE_RUNTIME_QUERY && defined(__HIPCC__) && !defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime_api.h>
#endif

#ifdef __HIPCC__
#define OPUS_H inline __host__
#define OPUS_D inline __device__
#define OPUS_H_D inline __host__ __device__
#define OPUS_D_EXTERN __device__
#define OPUS_H_D_EXTERN __host__ __device__
#else
#define OPUS_H inline
#define OPUS_D
#define OPUS_H_D inline
#define OPUS_D_EXTERN
#define OPUS_H_D_EXTERN
#endif

#ifndef OPUS_FP32_to_BF16_DEFAULT
#define OPUS_FP32_to_BF16_DEFAULT 2 // truncate, valid on gfx94* and before
#endif

#ifndef OPUS_TILE_CONTAINER
#define OPUS_TILE_CONTAINER 0 // 0:vector, 1:array of vector, 2:flattened array
#endif

namespace opus {
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// type traits
using std::remove_reference; using std::remove_reference_t; using std::remove_cv; using std::remove_cv_t; using std::is_same; using std::is_same_v;
template<typename T> struct remove_cvref { using type = remove_cv_t<remove_reference_t<T>>; };
template<typename T> using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// constant
using index_t = int;
using long_index_t = long long;

template<index_t I> struct number : public std::integral_constant<index_t, I> {};
template<bool B>    struct bool_constant : public std::bool_constant<B> {};
typedef bool_constant<true>  true_type;
typedef bool_constant<false> false_type;

template<typename>           struct is_constant : public false_type {};
template<typename T, auto I> struct is_constant<std::integral_constant<T, I>> : true_type {};
template<auto I>             struct is_constant<number<I>> : true_type {};
template<auto I>             struct is_constant<bool_constant<I>> : true_type {};
template <class T> static constexpr bool is_constant_v = is_constant<remove_cvref_t<T>>::value;    // prefer use this

// using opus::operator""_I; // => add this in your code to utilize the literal cast, e.g. 2_I, 3_I
template <char... Ds>
OPUS_H_D constexpr decltype(auto) operator""_I() {
    constexpr auto to_number_ = []() { index_t v = 0; ((v = v * 10 + (Ds - '0')), ...); return v; }; return number<to_number_()>{};
}

#define OPUS_LEFT_UNARY_OP(OP) template <auto x>         OPUS_H_D constexpr auto operator OP(number<x>)            { return number<(OP x)>{};   }
#define OPUS_BINARY_OP(OP)     template <auto x, auto y> OPUS_H_D constexpr auto operator OP(number<x>, number<y>) { return number<(x OP y)>{}; }

OPUS_LEFT_UNARY_OP(+) OPUS_LEFT_UNARY_OP(-) OPUS_LEFT_UNARY_OP(~) OPUS_LEFT_UNARY_OP(!)
OPUS_BINARY_OP(+)   OPUS_BINARY_OP(-)   OPUS_BINARY_OP(*)   OPUS_BINARY_OP(/)
OPUS_BINARY_OP(%)   OPUS_BINARY_OP(&)   OPUS_BINARY_OP(|)   OPUS_BINARY_OP(^)
OPUS_BINARY_OP(<<)  OPUS_BINARY_OP(>>)  OPUS_BINARY_OP(&&)  OPUS_BINARY_OP(||)
OPUS_BINARY_OP(==)  OPUS_BINARY_OP(!=)  OPUS_BINARY_OP(>)   OPUS_BINARY_OP(<)
OPUS_BINARY_OP(>=)  OPUS_BINARY_OP(<=)

#undef OPUS_LEFT_UNARY_OP
#undef OPUS_BINARY_OP

template<class T, class... R> constexpr bool is_any_of() noexcept { return (std::is_same_v<T, R> || ...); }
template<class T, class... R> static constexpr bool is_any_of_v = is_any_of<T, R...>();
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// underscore, useful struture to mock
struct underscore { /*who am I*/ };
static constexpr underscore _;
template <typename T> struct is_underscore : false_type {};
template <> struct is_underscore<underscore> : true_type {};
template <typename T> static constexpr bool is_underscore_v = is_underscore<T>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// constexpr functional math
struct plus       { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a + b; } };
struct minus      { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a - b; } };
struct multiplies { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a * b; } };
struct divides    { template<typename X, typename Y=X> OPUS_H_D constexpr decltype(auto) operator()(X a, Y b) const { return a / b; } };

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// seq
template <index_t... Is>
class seq {
public:
    using value_type = index_t;
    OPUS_H_D static constexpr index_t size() { return sizeof...(Is);}
    OPUS_H_D constexpr value_type operator[](index_t i) const { return data[i]; }
    OPUS_H_D static constexpr value_type at(index_t i) { return data[i]; }
    template <index_t I> OPUS_H_D static constexpr value_type at()          { return data[I]; }
    template <index_t I> OPUS_H_D static constexpr value_type at(number<I>) { return data[I]; }
private:
    static constexpr value_type data[sizeof...(Is) + 1] = {Is..., value_type{}};
};

template <index_t I, index_t... Is> OPUS_H_D constexpr auto seq_pop_front(seq<I, Is...>) { return seq<Is...>{}; }

template <index_t I, index_t... Is> OPUS_H_D constexpr decltype(auto) get(seq<Is...>const& ) { static_assert(I < sizeof...(Is)); return seq<Is...>::at(number<I>{}); }
template <index_t I, index_t... Is> OPUS_H_D constexpr decltype(auto) get(seq<Is...>& )      { static_assert(I < sizeof...(Is)); return seq<Is...>::at(number<I>{}); }
template <index_t I, index_t... Is> OPUS_H_D constexpr decltype(auto) get(seq<Is...>&& )     { static_assert(I < sizeof...(Is)); return seq<Is...>::at(number<I>{}); }

namespace impl {
template <typename T, T... Is> struct __integer_sequence;
template <index_t... Is>       struct __integer_sequence<index_t, Is...> { using seq_type = seq<Is...>; };
template<index_t, index_t, typename>                 struct __steped_integer_seq;
template<index_t Start, index_t Step, index_t... Is> struct __steped_integer_seq<Start, Step, seq<Is...>> { using seq_type = seq<(Start + Is * Step) ... >; };

template<typename>                   struct __make_index_seq;
template <index_t N>                 struct __make_index_seq<seq<N>>          { using seq_type = typename __make_integer_seq<__integer_sequence, index_t, N>::seq_type; };
template<index_t Start, index_t End> struct __make_index_seq<seq<Start, End>> { using seq_type = typename __steped_integer_seq<Start, 1, typename __make_index_seq< seq<(End-Start)/1> >::seq_type>::seq_type; };
template<index_t Start, index_t End, index_t Step>  struct __make_index_seq<seq<Start, End, Step>> {
    using seq_type = typename __steped_integer_seq<Start, Step, typename __make_index_seq< seq<(End-Start)/Step> >::seq_type>::seq_type;
};
} // namespace impl
// make_index_seq<5> -> seq<0,1,2,3,4> | make_index_seq<4, 9> -> seq<4,5,6,7,8> | make_index_seq<4, 8, 2> -> seq<4, 6>
template<index_t...Is> using make_index_seq = typename impl::__make_index_seq<seq<Is...>>::seq_type;

namespace impl {
template<index_t Value, index_t N>
struct __make_repeated_seq {
    template<index_t... I> static constexpr auto __make(seq<I...>) { return seq<(void(I), Value)...>{}; }
    using seq_type = decltype(__make(make_index_seq<N>{}));
};
} // namespace impl
template<index_t V, index_t N> using make_repeated_seq = typename impl::__make_repeated_seq<V, N>::seq_type;

template<index_t...Xs, index_t...Ys> OPUS_H_D constexpr auto concat_seq(seq<Xs...>, seq<Ys...>) { return seq<Xs..., Ys...>{}; }

namespace impl {
template<typename, typename>                                 struct reduce_seq_impl;
template <typename R, index_t I0, index_t I1, index_t... Is> struct reduce_seq_impl<R, seq<I0, I1, Is...>> { using type = typename reduce_seq_impl<R, seq<R{}(I0, I1), Is...>>::type; };
template <typename R, index_t I>                             struct reduce_seq_impl<R, seq<I>> { using type = seq<I>; };
template <typename R>                                        struct reduce_seq_impl<R, seq<>>  { using type = seq<>;  };
}
template<typename R, index_t...Xs> OPUS_H_D constexpr auto reduce_seq(seq<Xs...>) { return typename impl::reduce_seq_impl<R, seq<Xs...>>::type{}; }
template<index_t...Xs> OPUS_H_D constexpr auto reduce_seq_sum(seq<Xs...>) { if constexpr (sizeof...(Xs) == 0) return seq<>{}; else return seq<(Xs + ...)>{}; }
template<index_t...Xs> OPUS_H_D constexpr auto reduce_seq_mul(seq<Xs...>) { if constexpr (sizeof...(Xs) == 0) return seq<>{}; else return seq<(Xs * ...)>{}; }

template<typename T> struct is_seq : false_type {};
template<index_t... Is> struct is_seq<seq<Is...>> : true_type {};
template<typename T> constexpr bool is_seq_v = is_seq<remove_cvref_t<T>>::value;

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_seq_v<T>, index_t> size(T&&) { return remove_cvref_t<T>::size(); /* tuple size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_seq_v<T>, index_t> size()    { return remove_cvref_t<T>::size(); /* tuple size */}

template <index_t I, typename T, std::enable_if_t<is_seq_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T const& t) { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_seq_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&  t)      { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_seq_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&& t)      { static_assert(I < T::size()); return t[number<I>{}]; }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// functional
namespace impl {
template <class T>       struct static_for_impl;
template <index_t... Is> struct static_for_impl<seq<Is...>> { template <class F> OPUS_H_D constexpr void operator()(F&& f) const { (f(number<Is>{}), ...); } };
}   // namespace impl
template<index_t N, typename F> OPUS_H_D constexpr void static_for(F f) { impl::static_for_impl<make_index_seq<N>>{}(f); }

template<typename F, typename... R, std::enable_if_t<(is_constant_v<R> && ...), bool> = true>
OPUS_H_D constexpr void static_for(F f, R...) { impl::static_for_impl<make_index_seq<R::value...>>{}(f); }

namespace impl {
// Flat static_ford: single-level static_for, non-recursive compile-time index decomposition via fold expressions
template <index_t D, index_t... Is, index_t... Ns> constexpr index_t ford_stride(seq<Is...>, seq<Ns...>) { return ((Is > D ? Ns : index_t(1)) * ... * index_t(1)); }
template <index_t D, index_t... Is, index_t... Ns> constexpr index_t ford_dim(seq<Is...>, seq<Ns...>) { return ((Is == D ? Ns : index_t(1)) * ...); }
template <index_t I, index_t D, index_t... Ns> constexpr index_t ford_at() { return (I / ford_stride<D>(make_index_seq<sizeof...(Ns)>{}, seq<Ns...>{})) % ford_dim<D>(make_index_seq<sizeof...(Ns)>{}, seq<Ns...>{}); }
template <typename Seq> struct static_ford_impl;
template <index_t... Ns> struct static_ford_impl<seq<Ns...>> {
    template <typename F, index_t I, index_t... Ds> OPUS_H_D static constexpr void call_one(F& f, number<I>, seq<Ds...>) { f(number<ford_at<I, Ds, Ns...>()>{}...); }
    template <typename F> OPUS_H_D constexpr void operator()(F f) const { static_for<(Ns * ... * 1)>([&](auto I) { call_one(f, I, make_index_seq<sizeof...(Ns)>{}); }); }
};
template <> struct static_ford_impl<seq<>> { template <typename F> OPUS_H_D constexpr void operator()(F f) const { f(); } };
}

template<index_t... N, typename F> OPUS_H_D constexpr void static_ford(F f) { impl::static_ford_impl<seq<N...>>{}(f); }
template<index_t... N, typename F> OPUS_H_D constexpr void static_ford(seq<N...>, F f) { impl::static_ford_impl<seq<N...>>{}(f); }
template <class... T> struct tuple;
template<index_t... N, typename F> OPUS_H_D constexpr void static_ford(tuple<number<N>...>, F f) { impl::static_ford_impl<seq<N...>>{}(f); }

template<typename T, typename R = void> struct get_value_type { using type = remove_cvref_t<T>; };
template<typename T, typename R = void> using get_value_t = typename get_value_type<T, R>::type;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// array, enhanced C like array style
template <typename T, index_t N>
struct array {
    using value_type = remove_cvref_t<T>;
    using type = array<value_type, N>;
#if 0   // don't define following, just let me be trivially copyable class
    OPUS_H_D constexpr array() = default;
    OPUS_H_D constexpr array(const type& o) { static_for<N>([&](auto i){ content[i.value] = o[i.value]; }); }
    OPUS_H_D constexpr type& operator=(const type o) { static_for<N>([&](auto i){ content[i.value] = o[i.value]; }); return *this; }
    template<typename...Z, std::enable_if_t<(std::is_same_v<remove_cvref_t<Z>, value_type> && ...), bool> = true>
    OPUS_H_D constexpr array(Z&&... zs) : content{zs...}  { /* used for make_array */ }
#endif
    OPUS_H_D constexpr value_type& operator[](index_t pos) { return content[pos]; }
    OPUS_H_D constexpr const value_type& operator[](index_t pos) const { return content[pos]; }
    template<index_t I> OPUS_H_D constexpr value_type& operator[](number<I>) { return content[I]; }
    template<index_t I> OPUS_H_D constexpr const value_type& operator[](number<I>) const { return content[I]; }
    OPUS_H_D constexpr void fill(const T& value) { static_for<N>([&](auto i){ content[i.value] = value; }); }
    OPUS_H_D constexpr void clear() { fill(static_cast<T>(0)); }
    OPUS_H_D static constexpr bool empty() { return size() == 0; }
    OPUS_H_D static constexpr index_t size() { return N; }

    // we need this "content" member to have a default value, so that the implicitly defined constructor could be constexpr
    // see: https://en.cppreference.com/w/cpp/language/constexpr.html#constexpr_constructor
    value_type content[N] {};
};

template <typename T, index_t N>
OPUS_H_D constexpr bool operator==(const array<T,N>& x, const array<T,N>& y) { for (index_t i = 0; i < N; ++i) { if (x[i] != y[i]) { return false; } } return true; }

template <typename T, index_t N> OPUS_H_D constexpr void clear(array<T,N>& a) { a.clear(); }
template <typename T, index_t N> OPUS_H_D constexpr void fill(array<T,N>& a, T const& value) { a.fill(value); }

template<typename T> struct is_array : false_type {};
template<typename T, index_t N> struct is_array<array<T, N>> : true_type {};
template<typename T> constexpr bool is_array_v = is_array<remove_cvref_t<T>>::value;
template<typename T> struct get_value_type<T, std::enable_if_t<is_array_v<T>>> { using type = typename T::value_type; };

namespace impl {
template<typename> struct is_ref_wrapper : std::false_type{};
template<typename T> struct is_ref_wrapper<std::reference_wrapper<T>> : std::true_type{};
template<typename T> using not_ref_wrapper = std::negation<is_ref_wrapper<std::decay_t<T>>>;

template<typename D, typename...> struct array_return_type_helper { using type = D; };
template<typename... Types>
struct array_return_type_helper<void, Types...> : std::common_type<Types...> {
    static_assert(std::conjunction_v<not_ref_wrapper<Types>...>, "Types cannot contain reference_wrappers when D is void");
};
template<typename D, typename... Types> using array_return_type = opus::array<typename array_return_type_helper<D, Types...>::type, sizeof...(Types)>;
}
template<typename D = void, typename... Types> OPUS_H_D constexpr impl::array_return_type<D, Types...> make_array(Types&&... t) { return {std::forward<Types>(t)...}; }

template <index_t I, typename T, std::enable_if_t<is_array_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T const& t) { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_array_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&  t)      { static_assert(I < T::size()); return t[number<I>{}]; }
template <index_t I, typename T, std::enable_if_t<is_array_v<T>, bool> = true> OPUS_H_D constexpr decltype(auto) get(T&& t)      { static_assert(I < T::size()); return t[number<I>{}]; }

namespace impl {
template <class T0, class T1, index_t... I0, index_t... I1>
OPUS_H_D constexpr auto concat_array(T0 const& t0, T1 const& t1, seq<I0...>, seq<I1...>) { return opus::make_array(get<I0>(t0)..., get<I1>(t1)...); }
template <class T0, class T1, class T2, index_t... I0, index_t... I1, index_t...I2>
OPUS_H_D constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, seq<I0...>, seq<I1...>, seq<I2...>) { return opus::make_array(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...); }
template <class T0, class T1, class T2, class T3, index_t... I0, index_t... I1, index_t...I2, index_t...I3>
OPUS_H_D constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, seq<I0...>, seq<I1...>, seq<I2...>, seq<I3...>) { return opus::make_array(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...); }
}
template <class T0> OPUS_H_D  constexpr auto concat_array(T0 const& t0) { return t0; }
template <class T0, class T1>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1) { return impl::concat_array(t0, t1, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}); }
template <class T0, class T1, class T2>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2) { return impl::concat_array(t0, t1, t2, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}); }
template <class T0, class T1, class T2, class T3>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3) {
                                            return impl::concat_array(t0, t1, t2, t3, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}, make_index_seq<T3::size()>{}); }
template <class T0, class T1, class T2, class T3, class T4, class... Ts>
OPUS_H_D  constexpr auto concat_array(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, Ts const&... ts) { return concat_array(concat_array(t0, t1, t2, t3), concat_array(t4, ts...)); }

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_array_v<T>, index_t> size(T&&) { return remove_cvref_t<T>::size(); /* tuple size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_array_v<T>, index_t> size()    { return remove_cvref_t<T>::size(); /* tuple size */}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// tuple
namespace impl {
template <index_t idx, typename T, bool is_empty = (std::is_empty_v<T> || std::is_void_v<T>)> struct tuple_object {}; // the place where content is stored

template <index_t idx, typename T>
struct tuple_object<idx, T, true> {
    OPUS_H_D constexpr tuple_object() {}
    template <typename U, typename std::enable_if<!std::is_same<remove_cvref_t<U>, tuple_object>::value, bool>::type = false>
    OPUS_H_D constexpr tuple_object(U&&) {}
};
template <index_t idx, typename T>
struct tuple_object<idx, T, false> {
    OPUS_H_D constexpr tuple_object() : element{} {}
    template <typename U, typename std::enable_if<!std::is_same<remove_cvref_t<U>, tuple_object>::value, bool>::type = false>
    OPUS_H_D constexpr tuple_object(U&& e) : element(std::forward<U>(e)) {}
    T element;
};

// NOTE: we return a instance(not a reference) if content is empty
template <index_t I, class T> OPUS_H_D constexpr T        getv(const tuple_object<I, T, true>&)    { return {}; }
template <index_t I, class T> OPUS_H_D constexpr const T& getv(const tuple_object<I, T, false>& x) { return x.element; }
template <index_t I, class T> OPUS_H_D constexpr T&       getv(tuple_object<I, T, false>& x)       { return x.element; }
template <index_t I, class T> OPUS_H_D constexpr T&&      getv(tuple_object<I, T, false>&& x)      { return static_cast<T&&>(x.element); }

template <typename index_seq, typename... T> struct tuple_base;

template <index_t... I, typename... T>
struct tuple_base<seq<I...>, T...> : tuple_object<I, T>... {
    OPUS_H_D constexpr tuple_base() = default;

    template <class U, typename std::enable_if<sizeof...(I) == 1 && sizeof...(T) == 1 && !std::is_same<remove_cvref_t<U>, tuple_base>::value, bool>::type = false>
    OPUS_H_D constexpr tuple_base(U&& u) : tuple_object<I, T>(std::forward<U>(u))... {}

    template <typename... U, typename std::enable_if<sizeof...(U) >= 2, bool>::type = false>
    OPUS_H_D constexpr tuple_base(U&&... u) : tuple_object<I, T>(std::forward<U>(u))... { static_assert(sizeof...(I) == sizeof...(T) && sizeof...(I) == sizeof...(U), "wrong!"); }
};
} // namespace impl
template <class... T>
struct tuple : impl::tuple_base<make_index_seq<sizeof...(T)>, T...> {
    OPUS_H_D static constexpr index_t size() { return sizeof...(T); }
    using base = impl::tuple_base<make_index_seq<sizeof...(T)>, T...>;
    OPUS_H_D constexpr tuple() = default;

    template <typename U, typename std::enable_if<sizeof...(T) == 1 && !std::is_same<remove_cvref_t<U>, tuple>::value, bool>::type = false>
    OPUS_H_D constexpr tuple(U&& u) : base(std::forward<U>(u)) {}

    template <typename... U, typename std::enable_if<sizeof...(U) == sizeof...(T) && sizeof...(U) >= 2, bool>::type = false>
    OPUS_H_D constexpr tuple(U&&... u) : base(std::forward<U>(u)...) {}
};
template<typename... T> __host__ __device__ tuple(T&&...) -> tuple<remove_cvref_t<T>...>;

namespace impl {
template<typename T, typename S> struct tuple_array_helper;
template<typename T, index_t... Is> struct tuple_array_helper<T, seq<Is...>> { using type = tuple<decltype((Is, remove_cvref_t<T>{}))...>; };
}
template<typename T, index_t N> using tuple_array = typename impl::tuple_array_helper<T, make_index_seq<N>>::type;  // alias for tuple<T, T....>, Nx Ts

// get the I-th type within the tuple, O(1) via compiler intrinsic
template<index_t I, class T>     struct tuple_element;
template<index_t I, class... Ts> struct tuple_element<I, opus::tuple<Ts...>> { using type = __type_pack_element<I, Ts...>; };
template<index_t I, class T> using tuple_element_t = typename tuple_element<I, T>::type;

template <index_t I, class... T> OPUS_H_D constexpr decltype(auto) get(tuple<T...> const& t) { static_assert(I < sizeof...(T)); return impl::getv<I>(t); }
template <index_t I, class... T> OPUS_H_D constexpr decltype(auto) get(tuple<T...>& t)       { static_assert(I < sizeof...(T)); return impl::getv<I>(t); }
template <index_t I, class... T> OPUS_H_D constexpr decltype(auto) get(tuple<T...>&& t)      { static_assert(I < sizeof...(T)); return impl::getv<I>(std::move(t)); }

template <index_t I0, index_t I1, index_t... Is, class T>  /*recursive get*/
OPUS_H_D constexpr decltype(auto) get(T&& t) { return get<I1, Is...>(get<I0>(std::move(t))); }

template <typename... T> OPUS_H_D constexpr auto make_tuple(T&&... xs) { return tuple<remove_cvref_t<T>...>(std::forward<T>(xs)...); }

template<typename F, typename... R, std::enable_if_t<(std::is_integral_v<R> && ...), bool> = true>  // const integer based static_for loop
OPUS_H_D constexpr void static_for(F f, R... range) {
    if      constexpr (sizeof...(range) == 1) { auto end = get<0>(make_tuple(range...));        for(index_t i = 0; i < end; i++) { f(i); } }
    else if constexpr (sizeof...(range) == 2) { auto [start, end] = make_tuple(range...);       for(index_t i = start; i < end; i++) { f(i); } }
    else if constexpr (sizeof...(range) == 3) { auto [start, end, step] = make_tuple(range...); for(index_t i = start; i < end; i += step) { f(i); } }
}

namespace impl {
template <typename T, index_t... Is> OPUS_H_D constexpr auto make_repeated_tuple(T&& x, seq<Is...>) { return opus::make_tuple((void(Is), std::forward<T>(x))...); }
} // namespace impl
template <index_t N, typename T> OPUS_H_D constexpr auto make_repeated_tuple(T&& x) { return impl::make_repeated_tuple(std::forward<T>(x), make_index_seq<N>{}); }
template <typename T, index_t N> OPUS_H_D constexpr auto make_repeated_tuple(T&& x, number<N>) { return impl::make_repeated_tuple(std::forward<T>(x), make_index_seq<N>{}); }

namespace impl {
template <class T0, class T1, index_t... I0, index_t... I1>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, seq<I0...>, seq<I1...>) { return opus::make_tuple(get<I0>(t0)..., get<I1>(t1)...); }
template <class T0, class T1, class T2, index_t... I0, index_t... I1, index_t...I2>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, seq<I0...>, seq<I1...>, seq<I2...>) { return opus::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...); }
template <class T0, class T1, class T2, class T3, index_t... I0, index_t... I1, index_t...I2, index_t...I3>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, seq<I0...>, seq<I1...>, seq<I2...>, seq<I3...>) { return opus::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...); }
}
template <class T0> OPUS_H_D  constexpr auto concat_tuple(T0 const& t0) { return t0; }
template <class T0, class T1>
OPUS_H_D  constexpr auto concat_tuple(T0 const& t0, T1 const& t1) { return impl::concat_tuple(t0, t1, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}); }
template <class T0, class T1, class T2>
OPUS_H_D  constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2) { return impl::concat_tuple(t0, t1, t2, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}); }
template <class T0, class T1, class T2, class T3>
OPUS_H_D  constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3) {
                                            return impl::concat_tuple(t0, t1, t2, t3, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}, make_index_seq<T3::size()>{}); }
namespace impl { template <class T0, class T1, class T2, class T3, class T4, index_t... I0, index_t... I1, index_t... I2, index_t... I3, index_t... I4>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, seq<I0...>, seq<I1...>, seq<I2...>, seq<I3...>, seq<I4...>) { return opus::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)..., get<I4>(t4)...); } }
template <class T0, class T1, class T2, class T3, class T4>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4) { return impl::concat_tuple(t0, t1, t2, t3, t4, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}, make_index_seq<T3::size()>{}, make_index_seq<T4::size()>{}); }
template <class T0, class T1, class T2, class T3, class T4, class T5, class... Ts>
OPUS_H_D constexpr auto concat_tuple(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, T5 const& t5, Ts const&... ts) { return concat_tuple(concat_tuple(t0, t1, t2, t3, t4), concat_tuple(t5, ts...)); }

template <typename> struct is_tuple : false_type {};
template <typename... T> struct is_tuple<opus::tuple<T...>> : true_type {};
template <typename T> static constexpr bool is_tuple_v = is_tuple<remove_cvref_t<T>>::value;
template <typename T> struct is_static_tuple : is_constant<remove_cvref_t<T>> {};
template <> struct is_static_tuple<underscore> : true_type {};
template <typename... T> struct is_static_tuple<opus::tuple<T...>> : bool_constant<(is_static_tuple<T>::value && ...)> {};
template <typename T> static constexpr bool is_static_tuple_v = is_static_tuple<remove_cvref_t<T>>::value;
template<typename T> struct get_value_type<T, std::enable_if_t<is_tuple_v<T>>> { using type = tuple_element_t<0, T>; };   // TODO: get the first element type

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_tuple_v<T>, index_t> size(T&&) { return remove_cvref_t<T>::size(); /* tuple size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_tuple_v<T>, index_t> size()    { return remove_cvref_t<T>::size(); /* tuple size */}

template <typename T, std::enable_if_t<!is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto explode_tuple(const T& t) { return opus::make_tuple(t); }
template <typename T, index_t... Is> OPUS_H_D constexpr auto                                 explode_tuple(const T&, seq<Is...>);
template <typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto  explode_tuple(const T& t) { return explode_tuple(t, make_index_seq<size<T>()>{}); }
template <typename T, index_t... Is> OPUS_H_D constexpr auto                                 explode_tuple(const T& t, seq<Is...>) { return concat_tuple(explode_tuple(get<Is>(t))...); }

template <typename T, index_t... Is> OPUS_H_D constexpr auto flatten_tuple_general(const T& t, seq<Is...>) { return concat_tuple(explode_tuple(get<Is>(t))...); }
template <typename T, std::enable_if_t<is_tuple_v<T> && !(is_tuple_v<tuple_element_t<0, remove_cvref_t<T>>>), bool> = true>
OPUS_H_D constexpr auto flatten_tuple(const T& t) { return t; }  // already flat
template <typename T, std::enable_if_t<!is_tuple_v<T>, bool> = true>
OPUS_H_D constexpr auto flatten_tuple(const T& t) { return flatten_tuple_general(t, make_index_seq<size<T>()>{}); }  // non-tuple (e.g. seq)
namespace impl { // direct flatten for 1-level nested tuples -- bypasses concat_tuple + explode_tuple
template<typename T, index_t... Gs> constexpr auto group_sizes(seq<Gs...>) { return seq<size<tuple_element_t<Gs, T>>()...>{}; }
template<typename T, index_t... Gs> constexpr index_t group_total(seq<Gs...>) { return (size<tuple_element_t<Gs, T>>() + ...); }
template<index_t J, index_t... Gs, index_t... Ns> constexpr index_t flat_group(seq<Gs...>, seq<Ns...>) { index_t acc = 0, r = 0; ((void)(acc += Ns, (acc <= J ? (void)(r = Gs + 1) : (void)0)), ...); return r; }
template<typename T, index_t G, index_t... Gs> constexpr index_t group_offset(seq<Gs...>) { return ((Gs < G ? size<tuple_element_t<Gs, T>>() : 0) + ...); }
template<typename T, index_t J, typename GS> OPUS_H_D constexpr auto flatten_at(const T& t) {
    constexpr auto gs = make_index_seq<size<T>()>{}; constexpr index_t G = flat_group<J>(gs, GS{}); return get<J - group_offset<T, G>(gs)>(get<G>(t)); }
template<typename T, typename GS, index_t... Js> OPUS_H_D constexpr auto flatten_tuple_impl(const T& t, seq<Js...>) { return opus::make_tuple(flatten_at<T, Js, GS>(t)...); }
}
template <typename T, std::enable_if_t<is_tuple_v<T> && (is_tuple_v<tuple_element_t<0, remove_cvref_t<T>>>), bool> = true>
OPUS_H_D constexpr auto flatten_tuple(const T& t) { using U = remove_cvref_t<T>; constexpr auto gs = make_index_seq<size<U>()>{}; return impl::flatten_tuple_impl<U, decltype(impl::group_sizes<U>(gs))>(t, make_index_seq<impl::group_total<U>(gs)>{}); }

namespace impl {
template<typename Outer, typename Inner, index_t...Is>
OPUS_H_D constexpr auto embed_nested_tuple_impl(const Outer& ot, const Inner& it, seq<Is...>) { return opus::make_tuple(concat_tuple(get<Is>(ot), get<Is>(it))...); }

template<typename TargetType, typename T, index_t...Is>
OPUS_H_D constexpr auto tuple_count_impl(seq<Is...>) { return (number<std::is_same_v<remove_cvref_t<decltype(get<Is>(T{}))>, remove_cvref_t<TargetType>> ? 1 : 0>{} + ...); }
}
// Outer: tuple<tuple<X, X>, tuple<Y>>,  Inner: tuple<tuple<Z>, tuple<W>> => tuple<tuple<X, X, Z>, tuple<Y, W>>
template<typename Outer, typename Inner>
OPUS_H_D constexpr auto embed_nested_tuple(const Outer& ot, const Inner& it) {
    static_assert(size<Outer>() == size<Inner>());
    return impl::embed_nested_tuple_impl(ot, it, make_index_seq<size<Outer>()>{});
}

template< typename TargetType, typename T, std::enable_if_t<is_tuple_v<T>, bool> = true>
OPUS_H_D constexpr index_t tuple_count(const T& /*t*/) { return impl::tuple_count_impl<TargetType, remove_cvref_t<T>>(make_index_seq<size<T>()>{}).value; }

template< typename TargetType, typename T, std::enable_if_t<is_tuple_v<T>, bool> = true>
OPUS_H_D constexpr index_t tuple_count() { return impl::tuple_count_impl<TargetType, remove_cvref_t<T>>(make_index_seq<size<T>()>{}).value; }

template<index_t...Is> OPUS_H_D constexpr auto seq_to_tuple(seq<Is...>) { return opus::make_tuple(number<Is>{}...); }

template<index_t...Is>                                             OPUS_H_D constexpr auto to_tuple(seq<Is...>) { return opus::make_tuple(number<Is>{}...); }
template<typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto to_tuple(const T& t) { return t; }

namespace impl {
template <typename R, typename T>            OPUS_H_D constexpr auto reduce_tuple_impl(const T& t, seq<>)  { return t; }
template <typename R, typename T, index_t I> OPUS_H_D constexpr auto reduce_tuple_impl(const T& t, seq<I>) { return t; }

template <typename R, typename T, index_t I0, index_t I1, index_t... Is>
OPUS_H_D constexpr auto reduce_tuple_impl(const T& t, seq<I0, I1, Is...>) {
    return reduce_tuple_impl<R>(opus::make_tuple(R{}(get<I0>(t), get<I1>(t)), get<Is>(t)...), make_index_seq<sizeof...(Is) + 1>{});
}
}
template<typename R, typename T, std::enable_if_t<is_tuple_v<T>, bool> = true>
OPUS_H_D constexpr auto reduce_tuple(const T & t) { return  impl::reduce_tuple_impl<R>(t, make_index_seq<size<T>()>{}); }
template<typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto reduce_tuple_sum(const T & t) { return reduce_tuple<opus::plus>(t); }
template<typename T, std::enable_if_t<is_tuple_v<T>, bool> = true> OPUS_H_D constexpr auto reduce_tuple_mul(const T & t) { return reduce_tuple<opus::multiplies>(t); }
// Fast path: fold expression for tuple of number<> types (avoids N-1 intermediate tuple types)
template<typename... Ns, std::enable_if_t<sizeof...(Ns) != 0 && (is_constant_v<Ns> && ...), bool> = true>
OPUS_H_D constexpr auto reduce_tuple_mul(const tuple<Ns...>&) { return opus::tuple<number<(Ns::value * ...)>>{}; }

namespace impl {
template<typename PT, index_t... Js>
OPUS_H_D constexpr index_t underscore_count_in(seq<Js...>) { return ((is_underscore_v<remove_cvref_t<decltype(get<Js>(PT{}))>> ? 1 : 0) + ... + 0); }

template<typename PT, typename MaxN, index_t I>
OPUS_H_D constexpr index_t peephole_idx() { constexpr index_t c = underscore_count_in<PT>(make_index_seq<I>{}); return c < MaxN::value ? c : MaxN::value - 1; }

template<typename PT, typename MaxN, index_t... Is>
OPUS_H_D constexpr auto to_peepholed_seq_impl(seq<Is...>) { return seq<peephole_idx<PT, MaxN, Is>()...>{}; }

template<typename PeepholedTuple, typename IncomTuple, index_t...Ps,  index_t...Is>
OPUS_H_D constexpr decltype(auto) merge_peepholed_tuple_impl(PeepholedTuple&& pt, IncomTuple&& it, seq<Ps...>, seq<Is...>) {
    return opus::make_tuple([&](){ if constexpr (is_underscore_v<remove_cvref_t<decltype(get<Ps>(pt))>>) return get<Is>(it);
                                   else return get<Ps>(pt);}()... );
}
}
// (Peepholed)tuple<*, *, _, *, _> + (Income)tuple<#, @> -> tuple<*, *, #, *, @>.  "_"(underscore) indicate a peephole for income tuple to chime in
template<typename PeepholedTuple, typename IncomeTuple>
OPUS_H_D constexpr decltype(auto) merge_peepholed_tuple(PeepholedTuple&& pt, IncomeTuple&& it) {
    if constexpr (tuple_count<underscore, PeepholedTuple>() == 0) return pt;
    else {
        constexpr auto income_seq = impl::to_peepholed_seq_impl< remove_cvref_t<PeepholedTuple>,
                                                                 number<opus::size<IncomeTuple>()> >(make_index_seq<opus::size<PeepholedTuple>()>{});
        return impl::merge_peepholed_tuple_impl(std::forward<PeepholedTuple>(pt), std::forward<IncomeTuple>(it), make_index_seq<opus::size<PeepholedTuple>()>{}, income_seq);
    }
}
} // namespace opus

// implementing the "tuple-like binding protocol", don't use below directly
namespace std {
template <typename... Ts> struct tuple_size<opus::tuple<Ts...>>       : std::integral_constant<std::size_t, sizeof...(Ts)> {};
template <typename... Ts> struct tuple_size<const opus::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};
template <std::size_t I, typename... Ts> struct tuple_element<I, opus::tuple<Ts...>>       { using type = __type_pack_element<I, Ts...>; };
template <std::size_t I, typename... Ts> struct tuple_element<I, const opus::tuple<Ts...>> { using type = const __type_pack_element<I, Ts...>; };
} // namespace std

namespace opus {
// fwd-decl mma adaptors -- needed so make_tiled_mma() parses on archs (gfx12) where full defs are gated out.
struct mfma_adaptor; struct mfma_adaptor_swap_ab; struct wmma_adaptor; struct wmma_adaptor_swap_ab;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// transforms
template<typename X, typename Y, index_t... Is> constexpr auto embed(const X& x, const Y& y, seq<Is...>) { return ( ... + (get<Is>(x) * get<Is>(y))); }
template<typename X, typename Y>                constexpr auto embed(const X& x, const Y& y) { return embed(x, y, make_index_seq<X::size()>{}); }

namespace impl {
template <typename F, typename X, index_t... Is> OPUS_H_D constexpr auto transform_tuple_impl(F f, const X& x, seq<Is...>) { return opus::make_tuple(f(get<Is>(x))...); }
template <typename F, typename X, index_t... Is> OPUS_H_D constexpr auto transform_tuple_with_idx_impl(F f, const X& x, seq<Is...>) { return opus::make_tuple(f(get<Is>(x), number<Is>{})...); }
} // namespace impl
// f(auto item)
template <typename F, typename X> OPUS_H_D constexpr auto transform_tuple(F f, const X& x) { return impl::transform_tuple_impl(f, x, make_index_seq<size<X>()>{}); }
// f(auto item, auto index)
template <typename F, typename X> OPUS_H_D constexpr auto transform_tuple_with_idx(F f, const X& x) { return impl::transform_tuple_with_idx_impl(f, x, make_index_seq<size<X>()>{}); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// layout, simple linear nd layout with stride, static or dynamic supported
namespace impl {
template<typename Shape, index_t I, index_t... Js>
OPUS_H_D constexpr auto packed_stride_at(seq<Js...>) { return (get<I + 1 + Js>(Shape{}) * ... * number<1>{}); }

template<typename Shape, index_t... Is>
OPUS_H_D constexpr auto packed_shape_to_stride_impl(seq<Is...>) { return opus::make_tuple(packed_stride_at<Shape, Is>(make_index_seq<Shape::size() - Is - 1>{})...); }
}

template<typename Shape>
OPUS_H_D constexpr auto packed_shape_to_stride(const Shape&) { return impl::packed_shape_to_stride_impl<Shape>(make_index_seq<Shape::size()>{}); }

template<typename Layout, typename Coord>
OPUS_H_D constexpr decltype(auto) coord_to_linear(const Layout& layout, const Coord& coord) { static_assert(size<decltype(layout.stride())>() == size<Coord>()); return embed(layout.stride(), coord); }

// Shape/Stride/Coord, they are all tuples. if Coord is not false_type, will use merge_peepholed_tuple() to construct real coord
template<typename Shape_, typename Stride_, typename Coord_ = false_type>
struct layout : public tuple<remove_cvref_t<Shape_>, remove_cvref_t<Stride_>, remove_cvref_t<Coord_>> {
    using base   = tuple<remove_cvref_t<Shape_>, remove_cvref_t<Stride_>, remove_cvref_t<Coord_>>;
    using Shape  = remove_cvref_t<Shape_>;
    using Stride = remove_cvref_t<Stride_>;
    using Coord  = remove_cvref_t<Coord_>;  // peepholed coord

    static constexpr index_t rank = Shape::size();
    static_assert(Shape::size() == Stride::size());
    static_assert(std::is_same_v<Coord, false_type> || size<std::conditional_t<std::is_same_v<Coord, false_type>, Shape, Coord>>() == rank, "Coord should be either false_type or a tuple with same size as Shape");
    static constexpr index_t coord_rank = [](){
        if constexpr (std::is_same_v<Coord, false_type>) return rank;
        else          return rank - tuple_count<underscore>(Coord{});
    }();

    OPUS_H_D constexpr layout(const Shape& shape, const Stride& stride, const Coord& coord = {}) : base(shape, stride, coord){}
    OPUS_H_D constexpr layout(Shape&& shape, Stride&& stride, Coord&& coord = {}) : base(shape, stride, coord){}

    // get ith element from shape/stride. if no I, then get the shape/stride as tuple
    template <int... I> OPUS_H_D constexpr decltype(auto) shape()        { return get<0,I...>(static_cast<base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) shape()  const { return get<0,I...>(static_cast<const base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) stride()       { return get<1,I...>(static_cast<base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) stride() const { return get<1,I...>(static_cast<const base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) coord()        { return get<2,I...>(static_cast<base&>(*this)); }
    template <int... I> OPUS_H_D constexpr decltype(auto) coord() const  { return get<2,I...>(static_cast<const base&>(*this)); }

    template <typename... Cs, std::enable_if_t<(!is_tuple_v<Cs> && ...), bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(Cs&&... cs) const { return this->operator()(opus::make_tuple(std::forward<Cs>(cs)...)); }

    template <typename InCoord, std::enable_if_t<is_tuple_v<InCoord>, bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(InCoord&& c) const {
        if constexpr (std::is_same_v<Coord, false_type>) return coord_to_linear(*this, c);
        else                                             return coord_to_linear(*this, merge_peepholed_tuple(coord(), c)); }
};

template <typename Layout> struct layout_linear;
template<index_t cached_vec_, typename Layout> struct layout_cached;

// use cached_vec to dispatch which layout implementation. cached_vec < 0 : "layout", cached_vec == 0 : "layout_linear", cached_vec > 0 : "layout_cached"
template <index_t cached_vec = 0, typename Sx, typename Sy> OPUS_H_D constexpr auto make_layout(Sx&& s, Sy&& t) {
    if      constexpr (cached_vec < 0)  return layout<Sx, Sy>(std::forward<Sx>(s), std::forward<Sy>(t));
    else if constexpr (cached_vec == 0) return layout_linear<layout<Sx, Sy>>(std::forward<Sx>(s), std::forward<Sy>(t));
    else                                return layout_cached<cached_vec, layout<Sx, Sy>>(std::forward<Sx>(s), std::forward<Sy>(t)); }
template <index_t cached_vec = 0, typename Sx, typename Sy, typename Sz>
OPUS_H_D constexpr auto                       make_layout(Sx&& s, Sy&& t, Sz&& c) {
    if constexpr (cached_vec < 0)  return layout<Sx, Sy, Sz>(std::forward<Sx>(s), std::forward<Sy>(t), std::forward<Sz>(c));
    if constexpr (cached_vec == 0) return layout_linear<layout<Sx, Sy, Sz>>(std::forward<Sx>(s), std::forward<Sy>(t), std::forward<Sz>(c));
    else                           return layout_cached<cached_vec, layout<Sx, Sy, Sz>>(std::forward<Sx>(s), std::forward<Sy>(t), std::forward<Sz>(c)); }
template <index_t cached_vec = 0, typename... Ts, std::enable_if_t<(!is_tuple_v<Ts> && ...), bool> = true>
OPUS_H_D constexpr auto                       make_layout(Ts&&... ss) { return make_layout<cached_vec>(opus::make_tuple(ss...), packed_shape_to_stride(opus::make_tuple(ss...))); }
template <index_t cached_vec = 0, typename S> OPUS_H_D constexpr auto make_layout(S&& s) { return make_layout<cached_vec>(std::forward<S>(s), packed_shape_to_stride(s)); }

template <index_t cached_vec = 0, typename S> OPUS_H_D constexpr auto               make_layout_packed(S&& s) { return make_layout<cached_vec>(std::forward<S>(s), packed_shape_to_stride(s)); } // same as single arg make_layout
template <index_t cached_vec = 0, typename Sx, typename Sz> OPUS_H_D constexpr auto make_layout_packed(Sx&& s, Sz&& c) { return make_layout<cached_vec>(std::forward<Sx>(s), packed_shape_to_stride(s), std::forward<Sz>(c)); }

template <typename Layout>
struct layout_linear : public remove_cvref_t<Layout>{
    using base = remove_cvref_t<Layout>;

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_linear(const Shape& shape, const Stride& stride, const Coord& coord = {}) : base(shape, stride, coord), linear_offset(0){}

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_linear(Shape&& shape, Stride&& stride, Coord&& coord = {}) : base(shape, stride, coord), linear_offset(0){}

    template <typename... Cs, std::enable_if_t<(!is_tuple_v<Cs> && ...), bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(Cs&&... cs) const { return this->operator()(opus::make_tuple(std::forward<Cs>(cs)...)); }

    template <typename InCoord, std::enable_if_t<is_tuple_v<InCoord>, bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(InCoord&& c) const {
        if constexpr (std::is_same_v<typename base::Coord, false_type>) return linear_offset + coord_to_linear(*this, c);
        else                                             return linear_offset + coord_to_linear(*this, merge_peepholed_tuple(base::coord(), c)); }

    OPUS_H_D constexpr void inc(index_t offset) { linear_offset += offset; }
    OPUS_H_D constexpr layout_linear& operator+=(index_t offset) { inc(offset); return *this; }
    OPUS_H_D constexpr layout_linear operator+(index_t offset) const { layout_linear result(*this); result += offset; return result; }

    index_t linear_offset;
};

template <index_t vec, typename Layout> OPUS_H_D constexpr auto layout_to_vectorized_issue_space();
template<index_t vec, typename Layout> OPUS_H_D constexpr auto layout_to_offsets(const Layout& u);

template<index_t cached_vec_, typename Layout>
struct layout_cached : public remove_cvref_t<Layout> {
    using base = remove_cvref_t<Layout>;
    static constexpr index_t cached_vec = cached_vec_;

    static constexpr auto issue_space_vec = layout_to_vectorized_issue_space<cached_vec, base>();
    static constexpr index_t num_issues = get<0>(reduce_tuple_mul(issue_space_vec)).value;

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_cached(const Shape& shape, const Stride& stride, const Coord& coord = {}) : base(shape, stride, coord), offsets{layout_to_offsets<cached_vec>(static_cast<base>(*this))}{}

    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_cached(Shape&& shape, Stride&& stride, Coord&& coord = {}) : base(shape, stride, coord), offsets{layout_to_offsets<cached_vec>(static_cast<base>(*this))}{}

    template <typename... Cs, std::enable_if_t<(!is_tuple_v<Cs> && ...), bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(Cs&&... cs) const { return this->operator()(opus::make_tuple(std::forward<Cs>(cs)...)); }

    template <typename InCoord, std::enable_if_t<is_tuple_v<InCoord>, bool> = true>
    OPUS_H_D constexpr decltype(auto) operator()(InCoord&& c) const { constexpr auto u_linear = make_layout<-1>(issue_space_vec); return offsets[u_linear(c)]; }

    OPUS_H_D constexpr void inc(index_t offset) { static_for<num_issues>([&](auto i){ offsets[i] += offset; }); }
    OPUS_H_D constexpr layout_cached& operator+=(index_t offset) { inc(offset); return *this; }
    OPUS_H_D constexpr layout_cached operator+(index_t offset) const { layout_cached result(*this); result += offset; return result; }

    array<index_t, num_issues> offsets;
};

template<typename Layout, index_t N>
struct layout_shifted : public remove_cvref_t<Layout> {
    using base = remove_cvref_t<Layout>;
    OPUS_H_D constexpr layout_shifted(const base& layout) : base(layout) {}
    template<typename Shape, typename Stride, typename Coord = false_type>
    OPUS_H_D constexpr layout_shifted(const Shape& shape, const Stride& stride, const Coord& coord = {}) : base(shape, stride, coord) {}
    template <typename... Cs>
    OPUS_H_D constexpr auto operator()(Cs&&... cs) const { return base::operator()(std::forward<Cs>(cs)...) + N; }
};

template<typename T> struct is_layout : false_type {};
template<typename X, typename Y, typename Z> struct is_layout<layout<X, Y, Z>> : true_type {};
template<index_t cached_vec, typename Layout> struct is_layout<layout_cached<cached_vec, Layout>> : true_type {};
template<typename Layout> struct is_layout<layout_linear<Layout>> : true_type {};
template<typename Layout, index_t N> struct is_layout<layout_shifted<Layout, N>> : true_type {};
template<typename T> constexpr bool is_layout_v = is_layout<remove_cvref_t<T>>::value;

template<typename Layout> struct layout_shift_traits { using base = Layout; static constexpr index_t value = 0; };
template<typename Layout, index_t N> struct layout_shift_traits<layout_shifted<Layout, N>> { using base = remove_cvref_t<Layout>; static constexpr index_t value = N; };
template<typename Layout, index_t N, std::enable_if_t<is_layout_v<remove_cvref_t<Layout>>, bool> = true>
OPUS_H_D constexpr auto operator+(const Layout& layout, number<N>) {
    using shift = layout_shift_traits<remove_cvref_t<Layout>>;
    if constexpr (N == 0) return layout;
    else                  return layout_shifted<typename shift::base, shift::value + N>(static_cast<const typename shift::base&>(layout));
}

template <typename Layout>
OPUS_H_D constexpr auto layout_to_issue_space() {
    using maybe_coord = std::conditional_t<std::is_same_v<typename Layout::Coord, false_type>, typename Layout::Shape, typename Layout::Coord>;
    using issue_space_y = remove_cvref_t<decltype(pickup_shape(typename Layout::Shape{}, maybe_coord{}, underscore{}))>;
    using single_issue_space = remove_cvref_t<decltype(make_repeated_tuple(number<1>{}, number<size<typename Layout::Shape>()>{}))>;
    using fallback_issue_space_y = std::conditional_t<std::is_same_v<issue_space_y, opus::tuple<>>, single_issue_space, issue_space_y>;
    using issue_space = std::conditional_t<std::is_same_v<typename Layout::Coord, false_type>, single_issue_space, fallback_issue_space_y>;
    return issue_space{};
}
template<typename issue_space, int vec = 1>
OPUS_H_D constexpr auto vectorize_issue_space(issue_space, number<vec> = {}) {
    constexpr index_t vec_from_issue_space = get<size<issue_space>() - 1>(issue_space{}).value;     // here we get the original last dim length(which should be y dim)
    static_assert(vec_from_issue_space % vec == 0, "please make sure requested vec size can be dividable of vec from issue space");

    constexpr auto issue_space_vec = transform_tuple_with_idx([&](auto item, auto index){           // modify the last dim, divide it by vec. Result is still a tuple
        if constexpr (index.value == size<issue_space>() - 1) return number<item.value / vec>{};
        else                                                  return item;    }, issue_space{});
    return issue_space_vec;
}
template <index_t vec, typename Layout>
OPUS_H_D constexpr auto layout_to_vectorized_issue_space() {
    constexpr auto issue_space = layout_to_issue_space<Layout>();
    constexpr auto issue_space_vec = vectorize_issue_space(issue_space, number<vec>{});
    return issue_space_vec;
}

// Cache issue-space computations for load/store (avoids redundant evaluation across methods)
template<typename Layout, index_t vec = 1> struct layout_load_traits {
    static constexpr auto issue_space = layout_to_issue_space<Layout>();
    static constexpr auto issue_space_vec = vectorize_issue_space(issue_space, number<vec>{}); static constexpr auto r_elem = get<0>(reduce_tuple_mul(issue_space_vec));
};
// Runtime flat index -> multi-index tuple (all index_t) -- avoids per-iteration template instantiation
template<index_t... Is, index_t... Ns> OPUS_H_D constexpr auto flat_to_coords(index_t flat, seq<Is...>, tuple<number<Ns>...>) {
    constexpr index_t strides[] = {impl::ford_stride<Is>(make_index_seq<sizeof...(Ns)>{}, seq<Ns...>{})...}, dims[] = {Ns...};
    return opus::make_tuple(static_cast<index_t>((flat / strides[Is]) % dims[Is])...); }
// Pre-compute offsets via runtime loop -- 1 coord_to_linear instantiation per layout instead of N
template<index_t vec, typename Layout> OPUS_H_D constexpr auto layout_to_offsets(const Layout& u) {
    using LT = layout_load_traits<Layout, vec>; constexpr auto issue_space_vec = LT::issue_space_vec;
    constexpr index_t num_issues = LT::r_elem.value, ndim = size<remove_cvref_t<decltype(issue_space_vec)>>();
    array<index_t, num_issues> offsets;
    for (index_t i = 0; i < num_issues; i++) offsets[i] = u(flat_to_coords(i, make_index_seq<ndim>{}, issue_space_vec));
    return offsets; }
// Compile-time per-issue offsets for layouts with static Shape/Stride
template<typename Layout, index_t vec>
inline constexpr auto layout_imm_offsets_v = layout_to_offsets<vec>(Layout{typename Layout::Shape{}, typename Layout::Stride{}, typename Layout::Coord{}});
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// vector, a wrapper for __attribute__((ext_vector_type(*)))
template <typename V_, index_t N_> // V_ must be literal type, otherwise clang ext_vector_type will not recognize
struct vector {
    static constexpr index_t N = N_;
    using value_type           = remove_cvref_t<V_>;
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};
template <typename T, index_t N> using vector_t = typename vector<T, N>::type;

template <typename> struct is_vector : false_type {};
template <typename T, index_t N> struct is_vector<T __attribute((ext_vector_type(N)))> : true_type {};
template <typename T, index_t N> struct is_vector<T __attribute((ext_vector_type(N)))&> : true_type {};
template <typename T, index_t N> struct is_vector<const T __attribute((ext_vector_type(N)))&> : true_type {};
template <typename T, index_t N> struct is_vector<T __attribute((ext_vector_type(N)))&&> : true_type {};
template <typename E> static constexpr bool is_vector_v = is_vector<E>::value;

namespace impl {
template <typename T>            struct vector_traits_impl { using dtype = remove_cvref_t<T>; static constexpr index_t size() { return 1; } };
template <typename T, index_t N> struct vector_traits_impl<T __attribute__((ext_vector_type(N)))> { using dtype = T; static constexpr index_t size() { return N; } };
template <typename T, index_t N> struct vector_traits_impl<array<T, N>> { using dtype = T; static constexpr index_t size() { return N; } };
template <typename... T>         struct vector_traits_impl<tuple<T...>> { using dtype = __type_pack_element<0, T...> /*TODO: use first type*/; static constexpr index_t size() { return sizeof...(T); } };
}
template <typename T> struct vector_traits : public impl::vector_traits_impl<remove_cvref_t<T>> {};

template<typename T> OPUS_H_D constexpr std::enable_if_t<is_vector_v<T>, index_t> size(T&&) {  return vector_traits<T>::size();   /* vector size */}
template<typename T> OPUS_H_D constexpr std::enable_if_t<is_vector_v<T>, index_t> size()    {  return vector_traits<T>::size();   /* vector size */}

template<typename T> struct get_value_type<T, std::enable_if_t<is_vector_v<T>>> { using type = typename vector_traits<T>::dtype; };

namespace impl {
template<typename D, typename...> struct vector_return_type_helper { using type = D; };
template<typename... Types>
struct vector_return_type_helper<void, Types...> : std::common_type<Types...> { static_assert(std::conjunction_v<not_ref_wrapper<Types>...>, "Types cannot contain reference_wrappers when D is void"); };
template<typename D, typename... Types> using vector_return_type = opus::vector_t<typename vector_return_type_helper<D, Types...>::type, sizeof...(Types)>;
}
template<typename D = void, typename... Types> constexpr impl::vector_return_type<D, Types...> make_vector(Types&&... t) { return {std::forward<Types>(t)...}; }

namespace impl {
template <typename T, index_t... Is> OPUS_H_D constexpr auto make_repeated_vector(T&& x, seq<Is...>) { return opus::make_vector((void(Is), std::forward<T>(x))...); }
} // namespace impl
template <index_t N, typename T> OPUS_H_D constexpr auto make_repeated_vector(T&& x) { return impl::make_repeated_vector(std::forward<T>(x), make_index_seq<N>{}); }
template <typename T, index_t N> OPUS_H_D constexpr auto make_repeated_vector(T&& x, number<N>) { return impl::make_repeated_vector(std::forward<T>(x), make_index_seq<N>{}); }

// vector type can't return reference! error: non-const reference cannot bind to vector element
template <index_t I, typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr typename vector_traits<T>::dtype get(T const& t) { static_assert(I < vector_traits<T>::size()); return t[I]; }
template <index_t I, typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr typename vector_traits<T>::dtype get(T&& t)      { static_assert(I < vector_traits<T>::size()); return t[I]; }

namespace impl {
template <class T0, class T1, index_t... I0, index_t... I1>
OPUS_H_D constexpr auto concat_vector(T0 const& t0, T1 const& t1, seq<I0...>, seq<I1...>) {
    if constexpr (std::is_same_v<remove_cvref_t<T0>, remove_cvref_t<T1>> && sizeof...(I0) > 1) {
        using R = vector_t<typename vector_traits<remove_cvref_t<T0>>::dtype, sizeof...(I0) + sizeof...(I1)>;
        return __builtin_bit_cast(R, __builtin_shufflevector(t0, t1, I0..., (sizeof...(I0) + I1)...));
    } else { return opus::make_vector(get<I0>(t0)..., get<I1>(t1)...); }
}
template <class T0, class T1, class T2, index_t... I0, index_t... I1, index_t...I2>
OPUS_H_D constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, seq<I0...>, seq<I1...>, seq<I2...>) { return opus::make_vector(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...); }
template <class T0, class T1, class T2, class T3, index_t... I0, index_t... I1, index_t...I2, index_t...I3>
OPUS_H_D constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, seq<I0...>, seq<I1...>, seq<I2...>, seq<I3...>) { return opus::make_vector(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...); }
}
template <class T0> OPUS_H_D  constexpr auto concat_vector(T0 const& t0) { return t0; }
template <class T0, class T1>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1) { return impl::concat_vector(t0, t1, make_index_seq<size<T0>()>{}, make_index_seq<size<T1>()>{}); }
template <class T0, class T1, class T2>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2) { return impl::concat_vector(t0, t1, t2, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}); }
template <class T0, class T1, class T2, class T3>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3) {
                                            return impl::concat_vector(t0, t1, t2, t3, make_index_seq<T0::size()>{}, make_index_seq<T1::size()>{}, make_index_seq<T2::size()>{}, make_index_seq<T3::size()>{}); }
template <class T0, class T1, class T2, class T3, class T4, class... Ts>
OPUS_H_D  constexpr auto concat_vector(T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4, Ts const&... ts) { return concat_vector(concat_vector(t0, t1, t2, t3), concat_vector(t4, ts...)); }

template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr void fill(T& a, typename vector_traits<T>::dtype const& value) {
    if constexpr (size<T>() <= 4) { static_for<size<T>()>([&](auto i){ a[i.value] = value; }); }
    else { for (index_t i = 0; i < size<T>(); ++i) a[i] = value; }  // runtime loop for large vectors
}
template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true> OPUS_H_D constexpr void clear(T& a) { a = {}; }

namespace impl {
template<typename T, index_t... Is, std::enable_if_t<is_vector_v<T>, bool> = true>
OPUS_H_D constexpr auto to_array_impl(const T& t, seq<Is...>) { return opus::make_array(t[Is]...); }

template<typename T, index_t... Is, std::enable_if_t<is_array_v<T>, bool> = true>
OPUS_H_D constexpr auto to_array_impl(const T& t, seq<Is...>) { return opus::concat_array(to_array_impl(get<Is>(t), make_index_seq< size(get<Is>(T{})) >{})...); }

template<typename T, index_t... Is, std::enable_if_t<is_array_v<T> && !is_vector_v<typename T::value_type>, bool> = true>
OPUS_H_D constexpr vector_t<typename T::value_type, T::size()> to_vector_impl(const T& t, seq<Is...>) { return {get<Is>(t)...}; }

template<typename T, index_t... Is, std::enable_if_t<is_array_v<T> && is_vector_v<typename T::value_type>, bool> = true>
OPUS_H_D constexpr vector_t<typename T::value_type, T::size()> to_vector_impl(const T& t, seq<Is...>) { return opus::concat_vector(to_vector_impl(get<Is>(t))...); }
}

template<typename T, std::enable_if_t<is_vector_v<T>, bool> = true> // vector type to array
OPUS_H_D constexpr auto to_array(const T& t) { return impl::to_array_impl(t, make_index_seq<size<T>()>{}); }

template<typename T, std::enable_if_t<is_array_v<T>, bool> = true>  // array of vector type to array
OPUS_H_D constexpr auto to_array(const T& t) { return impl::to_array_impl(t, make_index_seq<size<T>()>{}); }

template<typename T, std::enable_if_t<is_array_v<T>, bool> = true>
OPUS_H_D constexpr auto to_vector(const T& t) { return impl::to_vector_impl(t, make_index_seq<size<T>()>{}); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// slice
namespace impl {
template<typename C, index_t...Is, std::enable_if_t<is_vector_v<C>, bool> = true> OPUS_H_D constexpr auto slice_impl(C&& c, seq<Is...>) {
    if constexpr (sizeof...(Is) == 1) return opus::make_vector(get<Is>(c)...);
    else { using R = vector_t<typename vector_traits<remove_cvref_t<C>>::dtype, sizeof...(Is)>; return __builtin_bit_cast(R, __builtin_shufflevector(c, c, Is...)); }
}
template<typename C, index_t...Is, std::enable_if_t<is_array_v<C>, bool> = true>  OPUS_H_D constexpr auto slice_impl(C&& c, seq<Is...>) { return opus::make_array(get<Is>(c)...); }
template<typename C, index_t...Is, std::enable_if_t<is_tuple_v<C>, bool> = true>  OPUS_H_D constexpr auto slice_impl(C&& c, seq<Is...>) { return opus::make_tuple(get<Is>(c)...); }

template<index_t len, typename C, typename...Ts, std::enable_if_t<is_vector_v<C>, bool> = true>
OPUS_H_D constexpr auto slice_impl_i(C&& c, Ts... ss) { vector_t<typename vector_traits<C>::dtype, len> r;  index_t d = 0;  static_for([&](auto i){r[d++] = c[i]; }, ss...);  return r; }

template<index_t len, typename C, typename...Ts, std::enable_if_t<is_array_v<C>, bool> = true>
OPUS_H_D constexpr auto slice_impl_i(C&& c, Ts... ss) { array<typename C::value_type, len> r;  index_t d = 0;  static_for([&](auto i){r[d++] = c[i]; }, ss...);  return r; }

template<index_t... Is>
OPUS_H_D constexpr bool is_contiguous_seq(seq<Is...>) {
    if constexpr (sizeof...(Is) < 2) return true;
    else { constexpr index_t idx[] = {Is...}; for (index_t i = 1; i < sizeof...(Is); ++i) { if (idx[i] != idx[i - 1] + 1) return false; } return true; }
}

template<typename C, typename V, index_t...Ds, index_t...Ss, std::enable_if_t<(is_vector_v<C> || is_array_v<C> || is_tuple_v<C>), bool> = true>
OPUS_H_D constexpr auto set_slice_impl(C&& dst_c, V&& src_c, seq<Ds...>, seq<Ss...>) {
    using dst_t = remove_cvref_t<C>; using src_t = remove_cvref_t<V>; using scalar = typename vector_traits<dst_t>::dtype;
    constexpr index_t len = sizeof...(Ds);
    // Copy at dword granularity for sub-dword scalar types with dword-aligned contiguous slices
    if constexpr ((is_vector_v<dst_t> || is_array_v<dst_t>) && (is_vector_v<src_t> || is_array_v<src_t>) && is_contiguous_seq(seq<Ds...>{}) && is_contiguous_seq(seq<Ss...>{}) && sizeof(scalar) < 4 && len > 1) {
        constexpr index_t epd = 4 / sizeof(scalar);
        constexpr index_t d0 = seq<Ds...>::at(number<0>{}), s0 = seq<Ss...>::at(number<0>{}), dn = vector_traits<dst_t>::size(), sn = vector_traits<src_t>::size();
        if constexpr (d0 % epd == 0 && s0 % epd == 0 && len % epd == 0 && dn % epd == 0 && sn % epd == 0) {
            auto dst_i32 = __builtin_bit_cast(vector_t<int, dn / epd>, dst_c);
            const auto src_i32 = __builtin_bit_cast(vector_t<int, sn / epd>, src_c);
            static_for<len / epd>([&](auto i) { dst_i32[d0 / epd + i.value] = src_i32[s0 / epd + i.value]; });
            dst_c = __builtin_bit_cast(dst_t, dst_i32); return;
        }
    }
    if constexpr (is_contiguous_seq(seq<Ds...>{}) && is_contiguous_seq(seq<Ss...>{}) && (is_vector_v<dst_t> || is_array_v<dst_t>) && len > 2) {
        constexpr index_t d0 = seq<Ds...>::at(number<0>{}), s0 = seq<Ss...>::at(number<0>{});
        for (index_t i = 0; i < len; ++i) dst_c[d0 + i] = src_c[s0 + i];  // runtime loop avoids N-element fold instantiation
    } else { ((dst_c[Ds] = src_c[Ss]), ...); }
}
}

// static/dynamic slice. SS could be either number<x>, or const integer. Note tuple type does not support dynamic slice (ss is integral)
// (1).[end] : 0.... end, (2).[start, end] : start...end, (3).[start, end, step], start...end but with step as interval (default is 1)
template<typename C, typename... S, std::enable_if_t<is_vector_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&.../*ss*/) { return impl::slice_impl(std::forward<C>(c), make_index_seq<(S::value) ...>{}); }
template<index_t len, typename C, typename... S, std::enable_if_t<is_vector_v<C> && (std::is_integral_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&...ss) { return impl::slice_impl_i<len>(std::forward<C>(c), ss...); }
template<typename C, typename... S, std::enable_if_t<is_array_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&.../*ss*/) { return impl::slice_impl(std::forward<C>(c), make_index_seq<(S::value) ...>{}); }
template<index_t len, typename C, typename... S, std::enable_if_t<is_array_v<C> && (std::is_integral_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&...ss) { return impl::slice_impl_i<len>(std::forward<C>(c), ss...); }
template<typename C, typename... S, std::enable_if_t<is_tuple_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& c, S&&.../*ss*/) { return impl::slice_impl(std::forward<C>(c), make_index_seq<(S::value) ...>{}); }

template<typename C, typename V, typename... S, std::enable_if_t<(is_vector_v<C> || is_array_v<C> || is_tuple_v<C>) && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto set_slice(C&& dst_c, V&& src_c, S&&.../*ss*/) {
    static_assert(std::is_same_v<typename vector_traits<C>::dtype, typename vector_traits<V>::dtype>);
    using dst_seq = make_index_seq<(S::value) ...>;
    return impl::set_slice_impl(std::forward<C>(dst_c), std::forward<V>(src_c), dst_seq{}, make_index_seq<size<dst_seq>()>{});
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// BELOW IS AMDGPU SPECIFIC TYPES/ARCH/INTRINSICS
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// address space attribute
#if defined(__HIP_DEVICE_COMPILE__)
    #define OPUS_LDS_ADDR __attribute__((address_space(3)))
#else
    #define OPUS_LDS_ADDR
#endif

// dtype, suffix is "_t", and register corresponding ext_vector_type, and a specialization of is_dtype
#define REGISTER_DTYPE(dtype_base_, dtype_impl_)        \
    using dtype_base_ ## _t    = dtype_impl_;           \
    using dtype_base_ ## x1_t  = dtype_base_ ## _t __attribute__((ext_vector_type(1 )));    \
    using dtype_base_ ## x2_t  = dtype_base_ ## _t __attribute__((ext_vector_type(2 )));    \
    using dtype_base_ ## x4_t  = dtype_base_ ## _t __attribute__((ext_vector_type(4 )));    \
    using dtype_base_ ## x8_t  = dtype_base_ ## _t __attribute__((ext_vector_type(8 )));    \
    using dtype_base_ ## x16_t = dtype_base_ ## _t __attribute__((ext_vector_type(16)));    \
    using dtype_base_ ## x32_t = dtype_base_ ## _t __attribute__((ext_vector_type(32)));    \
    using dtype_base_ ## x64_t = dtype_base_ ## _t __attribute__((ext_vector_type(64)));    \
    template<> struct is_dtype<dtype_base_ ## _t> : true_type {};

template<typename T> struct is_dtype : false_type {};
template<typename T> constexpr bool is_dtype_v = is_dtype<remove_cvref_t<T>>::value;    // use this!

REGISTER_DTYPE(fp32, float)
#if __clang_major__ >= 20   // enable for rocm 7.0+
REGISTER_DTYPE(bf16, __bf16)
REGISTER_DTYPE(fp16, __fp16)
#else
REGISTER_DTYPE(bf16, unsigned short)
REGISTER_DTYPE(fp16, _Float16)
#endif
REGISTER_DTYPE(fp8 , _BitInt(8))
REGISTER_DTYPE(bf8 , unsigned _BitInt(8))
REGISTER_DTYPE(i32 , int)
REGISTER_DTYPE(u32 , unsigned int)
REGISTER_DTYPE(i16 , short)
#if __clang_major__ >= 20
REGISTER_DTYPE(u16 , unsigned short)
#endif
REGISTER_DTYPE(i8  , signed char)
REGISTER_DTYPE(u8  , unsigned char)

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// numeric_limits -- returns min/max/lowest/quiet_nan/infinity in the *original* dtype
// (see finfo below for float-valued properties like eps/max/min/tiny)
template<typename T> struct numeric_limits;

template<> struct numeric_limits<fp32_t> {
    static constexpr unsigned int bin_min = 0x00800000, bin_max = 0x7F7FFFFF, bin_lowest = 0xFF7FFFFF, bin_qnan = 0x7FC00000, bin_inf = 0x7F800000;
    OPUS_H_D static constexpr fp32_t min()       { return __builtin_bit_cast(fp32_t, bin_min); }
    OPUS_H_D static constexpr fp32_t max()       { return __builtin_bit_cast(fp32_t, bin_max); }
    OPUS_H_D static constexpr fp32_t lowest()    { return __builtin_bit_cast(fp32_t, bin_lowest); }
    OPUS_H_D static constexpr fp32_t quiet_nan() { return __builtin_bit_cast(fp32_t, bin_qnan); }
    OPUS_H_D static constexpr fp32_t infinity()  { return __builtin_bit_cast(fp32_t, bin_inf); }
};
template<> struct numeric_limits<fp16_t> {
    static constexpr unsigned short bin_min = 0x0400, bin_max = 0x7BFF, bin_lowest = 0xFBFF, bin_qnan = 0x7E00, bin_inf = 0x7C00;
    OPUS_H_D static constexpr fp16_t min()       { return __builtin_bit_cast(fp16_t, bin_min); }
    OPUS_H_D static constexpr fp16_t max()       { return __builtin_bit_cast(fp16_t, bin_max); }
    OPUS_H_D static constexpr fp16_t lowest()    { return __builtin_bit_cast(fp16_t, bin_lowest); }
    OPUS_H_D static constexpr fp16_t quiet_nan() { return __builtin_bit_cast(fp16_t, bin_qnan); }
    OPUS_H_D static constexpr fp16_t infinity()  { return __builtin_bit_cast(fp16_t, bin_inf); }
};
template<> struct numeric_limits<bf16_t> {
    static constexpr unsigned short bin_min = 0x0080, bin_max = 0x7F7F, bin_lowest = 0xFF7F, bin_qnan = 0x7FC0, bin_inf = 0x7F80;
    OPUS_H_D static constexpr bf16_t min()       { return __builtin_bit_cast(bf16_t, bin_min); }
    OPUS_H_D static constexpr bf16_t max()       { return __builtin_bit_cast(bf16_t, bin_max); }
    OPUS_H_D static constexpr bf16_t lowest()    { return __builtin_bit_cast(bf16_t, bin_lowest); }
    OPUS_H_D static constexpr bf16_t quiet_nan() { return __builtin_bit_cast(bf16_t, bin_qnan); }
    OPUS_H_D static constexpr bf16_t infinity()  { return __builtin_bit_cast(bf16_t, bin_inf); }
};
// fp8 E4M3: gfx950=OCP(ieee-like, NaN=0x7F), gfx942=fnuz(NaN=0x80). No infinity in either format.
// NOTE: __builtin_bit_cast with _BitInt(8) is not yet constexpr in clang, so use static_cast via signed char.
template<> struct numeric_limits<fp8_t> {
#if defined(__gfx942__)
    static constexpr unsigned char bin_min = 0x08, bin_max = 0x7F, bin_lowest = 0xFF, bin_qnan = 0x80, bin_inf = 0x00;
#else
    static constexpr unsigned char bin_min = 0x08, bin_max = 0x7E, bin_lowest = 0xFE, bin_qnan = 0x7F, bin_inf = 0x00;
#endif
    OPUS_H_D static constexpr fp8_t min()       { return static_cast<fp8_t>(static_cast<signed char>(bin_min)); }
    OPUS_H_D static constexpr fp8_t max()       { return static_cast<fp8_t>(static_cast<signed char>(bin_max)); }
    OPUS_H_D static constexpr fp8_t lowest()    { return static_cast<fp8_t>(static_cast<signed char>(bin_lowest)); }
    OPUS_H_D static constexpr fp8_t quiet_nan() { return static_cast<fp8_t>(static_cast<signed char>(bin_qnan)); }
    OPUS_H_D static constexpr fp8_t infinity()  { return static_cast<fp8_t>(static_cast<signed char>(bin_inf)); }
};
// bf8 E5M2: gfx950=OCP(ieee, has inf=0x7C, NaN=0x7E), gfx942=fnuz(no inf, NaN=0x80)
template<> struct numeric_limits<bf8_t> {
#if defined(__gfx942__)
    static constexpr unsigned char bin_min = 0x04, bin_max = 0x7F, bin_lowest = 0xFF, bin_qnan = 0x80, bin_inf = 0x00;
#else
    static constexpr unsigned char bin_min = 0x04, bin_max = 0x7B, bin_lowest = 0xFB, bin_qnan = 0x7F, bin_inf = 0x7C;
#endif
    OPUS_H_D static constexpr bf8_t min()       { return static_cast<bf8_t>(bin_min); }
    OPUS_H_D static constexpr bf8_t max()       { return static_cast<bf8_t>(bin_max); }
    OPUS_H_D static constexpr bf8_t lowest()    { return static_cast<bf8_t>(bin_lowest); }
    OPUS_H_D static constexpr bf8_t quiet_nan() { return static_cast<bf8_t>(bin_qnan); }
    OPUS_H_D static constexpr bf8_t infinity()  { return static_cast<bf8_t>(bin_inf); }
};
template<> struct numeric_limits<i32_t> {
    OPUS_H_D static constexpr i32_t min()       { return -2147483647 - 1; }
    OPUS_H_D static constexpr i32_t max()       { return  2147483647; }
    OPUS_H_D static constexpr i32_t lowest()    { return -2147483647 - 1; }
    OPUS_H_D static constexpr i32_t quiet_nan() { return 0; }
    OPUS_H_D static constexpr i32_t infinity()  { return 0; }
};
template<> struct numeric_limits<u32_t> {
    OPUS_H_D static constexpr u32_t min()       { return 0; }
    OPUS_H_D static constexpr u32_t max()       { return 4294967295U; }
    OPUS_H_D static constexpr u32_t lowest()    { return 0; }
    OPUS_H_D static constexpr u32_t quiet_nan() { return 0; }
    OPUS_H_D static constexpr u32_t infinity()  { return 0; }
};
template<> struct numeric_limits<i16_t> {
    OPUS_H_D static constexpr i16_t min()       { return -32768; }
    OPUS_H_D static constexpr i16_t max()       { return  32767; }
    OPUS_H_D static constexpr i16_t lowest()    { return -32768; }
    OPUS_H_D static constexpr i16_t quiet_nan() { return 0; }
    OPUS_H_D static constexpr i16_t infinity()  { return 0; }
};
#if __clang_major__ >= 20
template<> struct numeric_limits<u16_t> {
    OPUS_H_D static constexpr u16_t min()       { return 0; }
    OPUS_H_D static constexpr u16_t max()       { return 65535; }
    OPUS_H_D static constexpr u16_t lowest()    { return 0; }
    OPUS_H_D static constexpr u16_t quiet_nan() { return 0; }
    OPUS_H_D static constexpr u16_t infinity()  { return 0; }
};
#endif
template<> struct numeric_limits<i8_t> {
    OPUS_H_D static constexpr i8_t min()       { return -128; }
    OPUS_H_D static constexpr i8_t max()       { return  127; }
    OPUS_H_D static constexpr i8_t lowest()    { return -128; }
    OPUS_H_D static constexpr i8_t quiet_nan() { return 0; }
    OPUS_H_D static constexpr i8_t infinity()  { return 0; }
};
template<> struct numeric_limits<u8_t> {
    OPUS_H_D static constexpr u8_t min()       { return 0; }
    OPUS_H_D static constexpr u8_t max()       { return 255; }
    OPUS_H_D static constexpr u8_t lowest()    { return 0; }
    OPUS_H_D static constexpr u8_t quiet_nan() { return 0; }
    OPUS_H_D static constexpr u8_t infinity()  { return 0; }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// finfo -- like torch.finfo: eps/max/min/tiny as float, bits as int
template<typename T> struct finfo;

template<> struct finfo<fp32_t> {
    static constexpr int bits = 32;
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x34000000u); }  // 2^-23
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x7F7FFFFFu); }  // 3.4028235e+38
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xFF7FFFFFu); }  // -3.4028235e+38
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x00800000u); }  // 2^-126
};
template<> struct finfo<fp16_t> {
    static constexpr int bits = 16;
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x3A800000u); }  // 2^-10 = 9.765625e-4
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x477FE000u); }  // 65504.0
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xC77FE000u); }  // -65504.0
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x38800000u); }  // 2^-14
};
template<> struct finfo<bf16_t> {
    static constexpr int bits = 16;
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x3C000000u); }  // 2^-7 = 0.0078125
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x7F7F0000u); }  // 3.389531e+38
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xFF7F0000u); }  // -3.389531e+38
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x00800000u); }  // 2^-126
};
// fp8 E4M3: gfx950=OCP(float8_e4m3fn, bias=7), gfx942=fnuz(float8_e4m3fnuz, bias=8)
template<> struct finfo<fp8_t> {
    static constexpr int bits = 8;
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x3E000000u); }  // 2^-3 = 0.125
#if defined(__gfx942__)
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x43700000u); }  // 240.0
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xC3700000u); }  // -240.0
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x3C000000u); }  // 2^-7 = 0.0078125
#else
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x43E00000u); }  // 448.0
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xC3E00000u); }  // -448.0
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x3C800000u); }  // 2^-6 = 0.015625
#endif
};
// bf8 E5M2: gfx950=OCP(float8_e5m2, bias=15), gfx942=fnuz(float8_e5m2fnuz, bias=16)
template<> struct finfo<bf8_t> {
    static constexpr int bits = 8;
#if defined(__gfx942__)
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x3E000000u); }  // 2^-3 = 0.125
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x47600000u); }  // 57344.0
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xC7600000u); }  // -57344.0
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x38000000u); }  // 2^-15
#else
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x3E800000u); }  // 2^-2 = 0.25
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x47600000u); }  // 57344.0
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xC7600000u); }  // -57344.0
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x38800000u); }  // 2^-14
#endif
};
template<> struct finfo<i8_t> {
    static constexpr int bits = 8;
    OPUS_H_D static constexpr float max()  { return 127.0f; }
    OPUS_H_D static constexpr float min()  { return -128.0f; }
};

template<typename C, typename... S, std::enable_if_t<is_dtype_v<C> && (is_constant_v<S> && ...), bool> = true>
OPUS_H_D constexpr auto slice(C&& container, S&&.../*ss*/) { return container; }    // TODO: fallback slice a normal value does nonthing
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// type cast
OPUS_D bf16_t fp32_to_bf16_rtn_asm(const float& x) {
    union { float f; u32_t i; } u = {x}; constexpr u32_t f32_nan = 0x7fff0000; constexpr u32_t round_bias = 0x7fff; u32x2_t check_nan; u32_t tmp;
    asm volatile("\nv_cmp_u_f32 %0, %2, %2 \nv_bfe_u32 %1, %2, 16, 1 \nv_add3_u32 %1, %2, %1, %3 \nv_cndmask_b32 %2, %1, %4, %0 \nv_lshrrev_b32 %2, 16, %2 \n"
                 : "=s"(check_nan), "+v"(tmp), "+v"(u.f) : "v"(round_bias), "v"(f32_nan));
    return bf16_t(u.i);
}

OPUS_D constexpr auto fp16_to_fp32(const fp16_t& x) { return static_cast<fp32_t>(x); }
OPUS_D constexpr auto fp32_to_fp16(const fp32_t& x) { return static_cast<fp16_t>(x); }
OPUS_D constexpr auto bf16_to_fp32(const bf16_t& x) { union { u32_t i; float f; } u = {static_cast<u32_t>(__builtin_bit_cast(unsigned short, x)) << 16}; return u.f;}
OPUS_D constexpr unsigned short fp32_to_bf16_rtn_raw(float f)
{
    unsigned int bits = __builtin_bit_cast(unsigned int, f);
    if(~bits & 0x7f800000) { bits += 0x7fff + ((bits >> 16) & 1); /* Round to nearest even */ }
    else if(bits & 0xffff) { bits |= 0x10000; /* Preserve signaling NaN */ }
    return static_cast<unsigned short>(bits >> 16);
}
#if (defined(__gfx950__) || defined(__gfx1250__) || defined(__gfx1201__) || defined(__gfx1200__)) && __clang_major__ >= 20
template<index_t rm = OPUS_FP32_to_BF16_DEFAULT> // gfx950/gfx1250/gfx12 has instruction conversion, leave 'rm' here for compatiblity
OPUS_D constexpr auto fp32_to_bf16(const fp32_t& x, number<rm> = {}) { return static_cast<bf16_t>(x); }
#else
template<index_t rm = OPUS_FP32_to_BF16_DEFAULT> // 0:standard, 1:truncate_with_nan, 2:truncate, 3:standard asm 4:rta_asm(round to nearest away)
OPUS_D constexpr auto fp32_to_bf16(const fp32_t& x, number<rm> = {}) {
    if      constexpr (rm == 0) {return __builtin_bit_cast(bf16_t, fp32_to_bf16_rtn_raw(x)); }
    else if constexpr (rm == 1) {u32_t z = __builtin_bit_cast(u32_t, x); return __builtin_bit_cast(bf16_t, static_cast<unsigned short>(z | (!(~z & 0x7f800000) && (z & 0xffff) ? 0x10000 : 0) >> 16)); }
    else if constexpr (rm == 2) {u32_t z = __builtin_bit_cast(u32_t, x); return __builtin_bit_cast(bf16_t, static_cast<unsigned short>(z >> 16)); }
    else if constexpr (rm == 3) { return fp32_to_bf16_rtn_asm(x); }
}
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#pragma clang diagnostic ignored "-Wc++20-extensions"
// scalar fp8 <-> fp32 via packed intrinsics (lo slot only). NOT constexpr: clang eagerly rejects non-template
// constexpr functions containing GPU builtins (__builtin_amdgcn_cvt_*) that can never be compile-time evaluated.
// Template constexpr (packed variants, OPUS_CAST_DEFINE) survives because the check is deferred to instantiation.
// TODO: we may remove constexpr from cast in the future
OPUS_D auto fp32_to_fp8(const fp32_t& x) {
#if defined(__HIP_DEVICE_COMPILE__) && !(defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))
    // RDNA3/3.5 (gfx1100/gfx115x) lack fp8-conversion-insts; compile-only
    // stub so headers build. BF16 code paths never invoke fp8 conversion.
    (void)x; return __builtin_bit_cast(fp8_t, static_cast<signed char>(0));
#else
    int w; w = __builtin_amdgcn_cvt_pk_fp8_f32(x, 0.0f, w, /*sel=lo*/0);
    return __builtin_bit_cast(fp8_t, static_cast<signed char>(w));
#endif
}
OPUS_D auto fp8_to_fp32(const fp8_t& x) {
#if defined(__HIP_DEVICE_COMPILE__) && !(defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))
    (void)x; return fp32_t(0.0f);
#else
    int w = static_cast<int>(__builtin_bit_cast(unsigned char, x));
    return __builtin_amdgcn_cvt_f32_fp8(w, /*byte=*/0);
#endif
}
OPUS_D constexpr auto fp32_to_fp32(const fp32_t& x) { return x; }
OPUS_D constexpr auto fp32_to_i8(const fp32_t& x) { return static_cast<i8_t>(x); }
OPUS_D constexpr auto i8_to_fp32(const i8_t& x) { return static_cast<fp32_t>(x); }
#pragma clang diagnostic pop

#define OPUS_CAST_DEFINE(d_, s_) template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, s_ ## _t> && std::is_same_v<D, d_ ## _t>, bool> = true> \
                                    OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return s_ ## _to_ ## d_(s, std::forward<Aux>(aux)...); }
OPUS_CAST_DEFINE(fp16, fp32)
OPUS_CAST_DEFINE(fp32, fp16)
OPUS_CAST_DEFINE(bf16, fp32)
OPUS_CAST_DEFINE(fp32, bf16)
OPUS_CAST_DEFINE(fp8, fp32)
OPUS_CAST_DEFINE(fp32, fp8)
OPUS_CAST_DEFINE(fp32, fp32)
OPUS_CAST_DEFINE(i8, fp32)
OPUS_CAST_DEFINE(fp32, i8)

namespace impl {
// implement a "pack" of data, storage should pad to multiple of byte(8bit)
template<typename storage_, unsigned int bits_, bool is_signed_ = true>
struct dpacks {
    using storage = remove_cvref_t<storage_>;
    static constexpr unsigned int bits = bits_;
    static constexpr unsigned int mask = (1 << bits) - 1;
    static constexpr bool is_signed = is_signed_;
    static constexpr unsigned int num_packs = sizeof(storage) * 8 / bits;   // we will not check if evenly divided or not here
    OPUS_H_D                     constexpr storage operator[](index_t i) const { return (value >> (i * bits)) & mask; } // NOTE: not efficient, better use v_bfi/v_bfe/v_perm on device
    template<index_t I> OPUS_H_D constexpr storage operator[](number<I>) const { return (value >> (I * bits)) & mask; } // NOTE: not efficient, better use v_bfi/v_bfe/v_perm on device
    storage value;
};

template<typename storage_, unsigned int bits_, unsigned int exp_bits_, unsigned int mantissa_bits_, bool is_signed_ = true>
struct fpacks : dpacks<storage_, bits_, is_signed_> {
    static constexpr unsigned int exp_bits = exp_bits_;
    static constexpr unsigned int mantissa_bits = mantissa_bits_;
};
} // namespace impl

template <typename> struct is_packs : false_type {};
template <typename S, unsigned int B, bool X> struct is_packs<impl::dpacks<S, B, X>> : true_type {};
template <typename S, unsigned int B, unsigned int E, unsigned int M, bool X> struct is_packs<impl::fpacks<S, B, E, M, X>> : true_type {};
template <typename T> static constexpr bool is_packs_v = is_packs<remove_cvref_t<T>>::value;

// how many real data within one byte
template <typename T, typename = void> struct num_packs { static constexpr int value = 1; };
template <typename T> struct num_packs<T, std::enable_if_t<is_packs_v<T>>> { static constexpr int value = T::num_packs; };
template <typename T> static constexpr int num_packs_v = num_packs<T>::value;

template <typename T> struct sizeof_bits { static constexpr int value = int(sizeof(T) * 8); };
template <> struct sizeof_bits<void> { static constexpr int value = 0; };
template <typename S, unsigned int B, bool X> struct sizeof_bits<impl::dpacks<S, B, X>> { static constexpr int value = impl::dpacks<S, B, X>::bits; };
template <typename S, unsigned int B, unsigned int E, unsigned int M, bool X> struct sizeof_bits<impl::fpacks<S, B, E, M, X>> { static constexpr int value = impl::fpacks<S, B, E, M, X>::bits; };
template <class T> static constexpr auto sizeof_bits_v = sizeof_bits<T>::value;

#define OPUS_DEFINE_DPACKS(name_, storage_, bits_, is_signed_) \
    struct name_ : opus::impl::dpacks<storage_, bits_, is_signed_> { using base = opus::impl::dpacks<storage_, bits_, is_signed_>; };  \
    template<> struct sizeof_bits<name_> { static constexpr int value = name_::bits; }; template<> struct is_packs<name_> : true_type {}; template<> struct is_dtype<name_> : true_type {};

#define OPUS_DEFINE_FPACKS(name_, storage_, bits_, exp_bits_, mantissa_bits_, is_signed_) \
    struct name_ : opus::impl::fpacks<storage_, bits_, exp_bits_, mantissa_bits_, is_signed_> {using base = opus::impl::fpacks<storage_, bits_, exp_bits_, mantissa_bits_, is_signed_>; };  \
    template<> struct sizeof_bits<name_> { static constexpr int value = name_::bits; }; template<> struct is_packs<name_> : true_type {}; template<> struct is_dtype<name_> : true_type {};

// NOTE: convention here. The subbyte type below is indeed "packed" data. e.g. fp4_t, underneath it is fp4x2 in one byte, but we don't name it this way
// This is different from cutlass convention (e.g float4_e2m1_t, but storage is unsigned char, hence an array of float4_e2m1_t will be expanded), and different from ck convention(explicitly name it fp4x2_t)
OPUS_DEFINE_DPACKS(int4_t , unsigned char, 4, true)           // int4x2
OPUS_DEFINE_DPACKS(uint4_t, unsigned char, 4, false)          // uint4x2
OPUS_DEFINE_FPACKS(fp4_t,   unsigned char, 4, 2, 1, true)     // fp4x2
OPUS_DEFINE_FPACKS(e8m0_t,  unsigned char, 8, 8, 0, false)    // fp4x2

// finfo specializations for subbyte/packed types (defined after OPUS_DEFINE_FPACKS)
// fp4 E2M1: 1 sign, 2 exp, 1 mantissa, bias=1
template<> struct finfo<fp4_t> {
    static constexpr int bits = 4;
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x3F000000u); }  // 2^-1 = 0.5
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x40C00000u); }  // 6.0
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0xC0C00000u); }  // -6.0
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x3F800000u); }  // 1.0
};
// e8m0: 8-bit exponent only, unsigned, bias=127
template<> struct finfo<e8m0_t> {
    static constexpr int bits = 8;
    OPUS_H_D static constexpr float eps()  { return __builtin_bit_cast(float, 0x3F800000u); }  // 1.0
    OPUS_H_D static constexpr float max()  { return __builtin_bit_cast(float, 0x7F000000u); }  // 2^127
    OPUS_H_D static constexpr float min()  { return __builtin_bit_cast(float, 0x00400000u); }  // 2^-127 (unsigned, no negative)
    OPUS_H_D static constexpr float tiny() { return __builtin_bit_cast(float, 0x00400000u); }  // 2^-127
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#pragma clang diagnostic ignored "-Wc++20-extensions"
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp8_packed_x2(const S& s, number<sel> = {}) {
    int w ; w = __builtin_amdgcn_cvt_pk_fp8_f32(s[0], s[1], w, sel);
    return __builtin_bit_cast(fp8x2_t, static_cast<short>(w));
}
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp8_packed_x4(const S& s) {
    int w ; w = __builtin_amdgcn_cvt_pk_fp8_f32(s[0], s[1], w, 0); w = __builtin_amdgcn_cvt_pk_fp8_f32(s[2], s[3], w, 1);
    return __builtin_bit_cast(fp8x4_t, w);
}
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp8x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp8_to_fp32_packed_x2(const S& s, number<sel> = {}) {
    union { int bitwise; S f8_packs[2]; } value; value.f8_packs[0] = s;
    return __builtin_amdgcn_cvt_pk_f32_fp8(value.bitwise, sel);
}
template<typename S, std::enable_if_t<std::is_same_v<S, fp8x4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp8_to_fp32_packed_x4(const S& s) {
    int bitwise = __builtin_bit_cast(int, s);
    auto x = __builtin_amdgcn_cvt_pk_f32_fp8(bitwise, 0); auto y = __builtin_amdgcn_cvt_pk_f32_fp8(bitwise, 1);
    return fp32x4_t{x[0], x[1], y[0], y[1]};
}

namespace impl {
template<typename S, index_t... Xs>     OPUS_D constexpr decltype(auto) fold_as_tuple_of_vec(const S& s, seq<Xs...>) {
    static_assert(size<S>() % sizeof...(Xs) == 0);
    constexpr index_t Y_len = size<S>() / sizeof...(Xs);
    auto gen_ = [&]<index_t X, index_t... Ys>(number<X>, seq<Ys...>){ return vector_t<get_value_t<S>, Y_len>{get<X * Y_len + Ys>(s)...}; };
    return make_tuple(gen_(number<Xs>{}, make_index_seq<Y_len>{})...);
}
template<typename S, index_t... Xs>     OPUS_D constexpr decltype(auto) fold_as_tuple_of_arr(const S& s, seq<Xs...>) {
    static_assert(size<S>() % sizeof...(Xs) == 0);
    constexpr index_t Y_len = size<S>() / sizeof...(Xs);
    auto gen_ = [&]<index_t X, index_t... Ys>(number<X>, seq<Ys...>){ return array<get_value_t<S>, Y_len>{get<X * Y_len + Ys>(s)...}; };
    return make_tuple(gen_(number<Xs>{}, make_index_seq<Y_len>{})...);
}
template<typename S, index_t fold_size, std::enable_if_t<is_tuple_v<S> || is_vector_v<S> || is_array_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) fold_as_container_of_vec(const S& s, number<fold_size>) {
    static_assert(size<S>() % fold_size == 0);
    return fold_as_tuple_of_vec(s, make_index_seq<size<S>() / fold_size>{});
}
template<typename S, index_t fold_size, std::enable_if_t<is_tuple_v<S> || is_vector_v<S> || is_array_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) fold_as_container_of_arr(const S& s, number<fold_size>) {
    static_assert(size<S>() % fold_size == 0);
    return fold_as_tuple_of_arr(s, make_index_seq<size<S>() / fold_size>{});
}

// Unfold a tuple-of-sub-results (produced by auto-fold cast) back into a flat container. Used in pair with above
// matching the original input container type OrigS.
//   OrigS is vector  -> flat vector_t<elem, total>
//   OrigS is array   -> flat array<elem, total>
//   OrigS is tuple   -> flat tuple<elem, elem, ...>
template<typename Tup, index_t inner_n, index_t... Flat>
OPUS_D constexpr auto unfold_as_tuple(const Tup& tup, number<inner_n>, seq<Flat...>) { return make_tuple(get<Flat % inner_n>(getv<Flat / inner_n>(tup))...); }

template<typename OrigS, typename Tup, std::enable_if_t<is_tuple_v<Tup>, bool> = true>
OPUS_D constexpr auto unfold_from_container(const Tup& tup) {
    using inner_t = remove_cvref_t<decltype(getv<0>(tup))>;
    using elem_t = get_value_t<inner_t>;
    constexpr index_t outer_n = opus::size<Tup>();
    constexpr index_t inner_n = opus::size<inner_t>();
    constexpr index_t total_n = outer_n * inner_n;
    if constexpr (is_vector_v<OrigS>) { vector_t<elem_t, total_n> r; static_for<total_n>([&](auto f) { r[f.value] = get<f.value % inner_n>(getv<f.value / inner_n>(tup)); }); return r; }
    else if constexpr (is_array_v<OrigS>) { array<elem_t, total_n> r; static_for<total_n>([&](auto f) { r[f.value] = get<f.value % inner_n>(getv<f.value / inner_n>(tup)); }); return r; }
    else { /* tuple */  return unfold_as_tuple(tup, number<inner_n>{}, make_index_seq<total_n>{}); }
}
} // namespace impl
#if defined(__gfx950__)
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x2(const S& s, float scale = 1.0f, number<sel> = {}) {
    u32_t w; w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, s[0], s[1], scale, sel);
    return __builtin_bit_cast(array<fp4_t, 1>, static_cast<u8_t>(w));
}
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x4(const S& s, float scale = 1.0f) {
    u32_t w; w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, s[0], s[1], scale, 0); w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, s[2], s[3], scale, 1);
    return __builtin_bit_cast(array<fp4_t, 2>, static_cast<u16_t>(w));
}
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x8_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x8(const S& s, float scale = 1.0f) {
    u32_t w; w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, s[0], s[1], scale, 0); w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, s[2], s[3], scale, 1);
    w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, s[4], s[5], scale, 2); w = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(w, s[6], s[7], scale, 3);
    return __builtin_bit_cast(array<fp4_t, 4>, w);
}
template<typename S, index_t sel = 0, std::enable_if_t<is_any_of_v<S, fp4_t, array<fp4_t, 1>>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x2(const S& s, float scale = 1.0f, number<sel> = {}) {
    return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(static_cast<u32_t>(__builtin_bit_cast(u8_t, s)), scale, sel);
}
template<typename S, std::enable_if_t<std::is_same_v<S, array<fp4_t, 2>>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x4(const S& s, float scale = 1.0f) {
    auto ss = static_cast<u32_t>(__builtin_bit_cast(u16_t, s));
    auto x = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(ss, scale, 0); auto y = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(ss, scale, 1);
    return fp32x4_t{x[0], x[1], y[0], y[1]};
}
template<typename S, std::enable_if_t<std::is_same_v<S, array<fp4_t, 4>>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x8(const S& s, float scale = 1.0f) {
    auto ss = static_cast<u32_t>(__builtin_bit_cast(u32_t, s));
    auto x = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(ss, scale, 0); auto y = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(ss, scale, 1);
    auto z = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(ss, scale, 2); auto w = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(ss, scale, 3);
    return fp32x8_t{x[0], x[1], y[0], y[1], z[0], z[1], w[0], w[1]};
}

template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, bf16x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) bf16_to_fp4_packed_x2(const S& s, float scale = 1.0f, number<sel> = {}) {
    union { unsigned int bitwise; fp4_t fp4_pack[4]; } value;
    value.bitwise = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(value.bitwise, s, scale, sel);
    return value.fp4_pack[0];
}
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_bf16_packed_x2(const S& s, float scale = 1.0f, number<sel> = {}) { return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(s, scale, sel); }
#elif defined(__gfx1250__)
// gfx1250: pk8 builtins convert 8 fp4 <-> 8 f32 at once
// f32->fp4: __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(v8f32 src, float scale) -> i32
// fp4->f32: __builtin_amdgcn_cvt_scale_pk8_f32_fp4(i32 src, i32 scale_sel, i32 imm) -> v8f32
//   scale_sel = e8m0 scale byte (imm selects which byte), e8m0: val = 2^(byte-127), so 1.0 = 0x7F
//   extract e8m0 from float: biased exponent = (float_bits >> 23) & 0xFF
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x2(const S& s, float scale = 1.0f, number<sel> = {}) {
    fp32x8_t v{s[0], s[1], 0, 0, 0, 0, 0, 0};
    u32_t w = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(v, scale);
    return __builtin_bit_cast(array<fp4_t, 1>, static_cast<u8_t>(w));
}
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x4(const S& s, float scale = 1.0f) {
    fp32x8_t v{s[0], s[1], s[2], s[3], 0, 0, 0, 0};
    u32_t w = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(v, scale);
    return __builtin_bit_cast(array<fp4_t, 2>, static_cast<u16_t>(w));
}
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x8_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x8(const S& s, float scale = 1.0f) {
    u32_t w = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(s, scale);
    return __builtin_bit_cast(array<fp4_t, 4>, w);
}
template<typename S, index_t sel = 0, std::enable_if_t<is_any_of_v<S, fp4_t, array<fp4_t, 1>>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x2(const S& s, float scale = 1.0f, number<sel> = {}) {
    i32_t e = (__builtin_bit_cast(i32_t, scale) >> 23) & 0xFF;
    i32_t scale_e8m0 = e * static_cast<i32_t>(0x01010101);
    fp32x8_t r = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(static_cast<i32_t>(__builtin_bit_cast(u8_t, s)), scale_e8m0, 0);
    return fp32x2_t{r[0], r[1]};
}
template<typename S, std::enable_if_t<std::is_same_v<S, array<fp4_t, 2>>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x4(const S& s, float scale = 1.0f) {
    i32_t e = (__builtin_bit_cast(i32_t, scale) >> 23) & 0xFF;
    i32_t scale_e8m0 = e * static_cast<i32_t>(0x01010101);
    fp32x8_t r = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(static_cast<i32_t>(__builtin_bit_cast(u16_t, s)), scale_e8m0, 0);
    return fp32x4_t{r[0], r[1], r[2], r[3]};
}
template<typename S, std::enable_if_t<std::is_same_v<S, array<fp4_t, 4>>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x8(const S& s, float scale = 1.0f) {
    i32_t e = (__builtin_bit_cast(i32_t, scale) >> 23) & 0xFF;
    i32_t scale_e8m0 = e * static_cast<i32_t>(0x01010101);
    fp32x8_t r = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(static_cast<i32_t>(__builtin_bit_cast(u32_t, s)), scale_e8m0, 0);
    return fp32x8_t{r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]};
}
// bf16<->fp4 stubs for gfx1250 (no pk bf16<->fp4 builtins available)
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, bf16x2_t>, bool> = true>
OPUS_D constexpr decltype(auto) bf16_to_fp4_packed_x2(const S& /*s*/, float /*scale*/ = 1.0f, number<sel> = {}) { return fp4_t{}; }
template<typename S, index_t sel = 0, std::enable_if_t<std::is_same_v<S, fp4_t>, bool> = true>
OPUS_D constexpr decltype(auto) fp4_to_bf16_packed_x2(const S& /*s*/, float /*scale*/ = 1.0f, number<sel> = {}) { return bf16x2_t{}; }
#else
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x2_t>, bool> = true>  OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x2(const S& /*s*/, float /*scale*/ = 1.0f) { return array<fp4_t, 1>{}; }
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x4_t>, bool> = true>  OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x4(const S& /*s*/, float /*scale*/ = 1.0f) { return array<fp4_t, 2>{}; }
template<typename S, std::enable_if_t<std::is_same_v<S, fp32x8_t>, bool> = true>  OPUS_D constexpr decltype(auto) fp32_to_fp4_packed_x8(const S& /*s*/, float /*scale*/ = 1.0f) { return array<fp4_t, 4>{}; }
template<typename S, std::enable_if_t<is_any_of_v<S, fp4_t, array<fp4_t, 1>>, bool> = true>     OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x2(const S& /*s*/, float /*scale*/ = 1.0f) { return fp32x2_t{}; }
template<typename S, std::enable_if_t<std::is_same_v<S, array<fp4_t, 2>>, bool> = true>     OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x4(const S& /*s*/, float /*scale*/ = 1.0f) { return fp32x4_t{}; }
template<typename S, std::enable_if_t<std::is_same_v<S, array<fp4_t, 4>>, bool> = true>     OPUS_D constexpr decltype(auto) fp4_to_fp32_packed_x8(const S& /*s*/, float /*scale*/ = 1.0f) { return fp32x8_t{}; }
template<typename S, std::enable_if_t<std::is_same_v<S, bf16x2_t>, bool> = true>  OPUS_D constexpr decltype(auto) bf16_to_fp4_packed_x2(const S& /*s*/, float /*scale*/ = 1.0f) { return fp4_t{}; }
template<typename S, std::enable_if_t<std::is_same_v<S, fp4_t>, bool> = true>     OPUS_D constexpr decltype(auto) fp4_to_bf16_packed_x2(const S& /*s*/, float /*scale*/ = 1.0f) { return bf16x2_t{}; }
#endif
#pragma clang diagnostic pop

template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, fp32x2_t> && std::is_same_v<D, fp8_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp32_to_fp8_packed_x2(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, fp32x4_t> && std::is_same_v<D, fp8_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp32_to_fp8_packed_x4(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, fp8x2_t> && std::is_same_v<D, fp32_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp8_to_fp32_packed_x2(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, fp8x4_t> && std::is_same_v<D, fp32_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp8_to_fp32_packed_x4(s, std::forward<Aux>(aux)...); }

template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, fp32x2_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp32_to_fp4_packed_x2(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, fp32x4_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp32_to_fp4_packed_x4(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, fp32x8_t> && std::is_same_v<D, fp4_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp32_to_fp4_packed_x8(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<is_any_of_v<S, fp4_t, array<fp4_t, 1>> && std::is_same_v<D, fp32_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp4_to_fp32_packed_x2(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, array<fp4_t, 2>> && std::is_same_v<D, fp32_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp4_to_fp32_packed_x4(s, std::forward<Aux>(aux)...); }
template<typename D, typename S, typename... Aux, std::enable_if_t<std::is_same_v<S, array<fp4_t, 4>> && std::is_same_v<D, fp32_t>, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) { return fp4_to_fp32_packed_x8(s, std::forward<Aux>(aux)...); }

namespace impl {
// rocm-7.1.1, when there are multiple invokes of this kernel (across different __global__ in same compile target ?) will fail to inline below function
template<typename D, typename S, index_t... Is, typename... Aux, std::enable_if_t<is_vector_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast_impl(const S& s, seq<Is...>, Aux&&... aux) {
    return impl::vector_return_type<D, decltype(cast<D>(get<Is>(s), std::forward<Aux>(aux)...))...>{cast<D>(get<Is>(s), std::forward<Aux>(aux)...)...}; }
    //return opus::make_vector(cast<D>(get<Is>(s), std::forward<Aux>(aux)...)...); }
template<typename D, typename S, index_t... Is, typename... Aux, std::enable_if_t<is_tuple_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast_impl(const S& s, seq<Is...>, Aux&&... aux) {
    return tuple<remove_cvref_t<decltype(cast<D>(get<Is>(s), std::forward<Aux>(aux)...))>...>(cast<D>(get<Is>(s), std::forward<Aux>(aux)...)...); }
    // return opus::make_tuple(cast<D>(get<Is>(s), std::forward<Aux>(aux)...) ...   ); }
template<typename D, typename S, index_t... Is, typename... Aux, std::enable_if_t<is_array_v<S>, bool> = true>
OPUS_D constexpr decltype(auto) cast_impl(const S& s, seq<Is...>, Aux&&... aux) {
    return impl::array_return_type<void, decltype(cast<D>(get<Is>(s), std::forward<Aux>(aux)...))...>{cast<D>(get<Is>(s), std::forward<Aux>(aux)...)...}; }
    // return opus::make_array(cast<D>(get<Is>(s), std::forward<Aux>(aux)...)...); }
}

// entry point for vectorized cast(), non-dpacks
template<typename D, typename S, typename... Aux, std::enable_if_t<((is_vector_v<S> || is_tuple_v<S> || is_array_v<S>) && !is_packs_v<D> && !is_packs_v<get_value_t<S>>)
    && !(is_any_of_v<S, fp32x2_t, fp32x4_t>&& std::is_same_v<D, fp8_t >)
    && !(is_any_of_v<S, fp8x2_t , fp8x4_t >&& std::is_same_v<D, fp32_t>)
, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) {
    if      constexpr (std::is_same_v<get_value_t<S>, fp32_t> && size<S>() % 4 == 0 && std::is_same_v<D, fp8_t>) { // fp32 -> fp8 , x4N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<4>{}), make_index_seq<size<S>() / 4>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (std::is_same_v<get_value_t<S>, fp32_t> && size<S>() % 2 == 0 && std::is_same_v<D, fp8_t>) { // fp32 -> fp8 , x2N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<2>{}), make_index_seq<size<S>() / 2>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (std::is_same_v<get_value_t<S>, fp8_t>  && size<S>() % 4 == 0 && std::is_same_v<D, fp32_t>) { // fp8 -> fp32, x4N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<4>{}), make_index_seq<size<S>() / 4>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (std::is_same_v<get_value_t<S>, fp8_t>  && size<S>() % 2 == 0 && std::is_same_v<D, fp32_t>) { // fp8 -> fp32, x2N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<2>{}), make_index_seq<size<S>() / 2>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (is_vector_v<S> && size<S>() > 16 && sizeof...(Aux) == 0) {
        return __builtin_convertvector(s, vector_t<D, size<S>()>); }
    else   return impl::cast_impl<D>(s, make_index_seq<size<S>()>{}, std::forward<Aux>(aux)...); }

// entry point for vectorized cast(), for dpacks
template<typename D, typename S, typename... Aux, std::enable_if_t<((is_vector_v<S> || is_tuple_v<S> || is_array_v<S>) && (is_packs_v<D> || is_packs_v<get_value_t<S>>))
    && !(is_any_of_v<S, fp32x2_t, fp32x4_t, fp32x8_t> && std::is_same_v<D, fp4_t >)         // fp32
    && !(is_any_of_v<S, fp4_t, array<fp4_t, 1>, array<fp4_t, 2>, array<fp4_t, 4>, tuple_array<fp4_t, 1>, tuple_array<fp4_t, 2>, tuple_array<fp4_t, 4>> && std::is_same_v<D, fp32_t>)
, bool> = true>
OPUS_D constexpr decltype(auto) cast(const S& s, Aux&&... aux) {
    constexpr index_t num_packs_ = [&](){   // TODO: how to consider both D and S are packs?
        if constexpr (is_packs_v<D>) { static_assert(size<S>() % D::num_packs == 0); return D::num_packs; } // TODO: do not support cast pack data one by one
        else                         { return get_value_t<S>::num_packs; } }();
    if      constexpr (std::is_same_v<get_value_t<S>, fp32_t> && size<S>() % 8 == 0 && std::is_same_v<D, fp4_t>) { // fp32 -> fp4 , x8N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<8>{}), make_index_seq<size<S>() / 8>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (std::is_same_v<get_value_t<S>, fp32_t> && size<S>() % 4 == 0 && std::is_same_v<D, fp4_t>) { // fp32 -> fp4 , x4N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<4>{}), make_index_seq<size<S>() / 4>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (std::is_same_v<get_value_t<S>, fp32_t> && size<S>() % 2 == 0 && std::is_same_v<D, fp4_t>) { // fp32 -> fp4 , x2N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<2>{}), make_index_seq<size<S>() / 2>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (std::is_same_v<get_value_t<S>, fp4_t> && size<S>() % 4 == 0) { // fp4 -> fp32 , x8N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_arr(s, number<4>{}), make_index_seq<size<S>() / 4>{}, std::forward<Aux>(aux)...)); }
    else if constexpr (std::is_same_v<get_value_t<S>, fp4_t> && size<S>() % 2 == 0) { // fp4 -> fp32 , x4N
                    return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_arr(s, number<2>{}), make_index_seq<size<S>() / 2>{}, std::forward<Aux>(aux)...)); }
    else            return impl::unfold_from_container<S>(impl::cast_impl<D>(impl::fold_as_container_of_vec(s, number<num_packs_>{}), make_index_seq<size<S>() / num_packs_>{}, std::forward<Aux>(aux)...));
}

#undef OPUS_DEFINE_DPACKS
#undef OPUS_DEFINE_FPACKS
#undef OPUS_CAST_DEFINE
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// arch
//
// ---- HIPCC compilation model (clang-based)  ----
//   hipcc compiles each translation unit in TWO passes: host pass, then device pass.
//
//   Host pass  : __device__ functions are fully parsed, name-resolved, template-instantiated, and constexpr/static_assert evaluated. Only machine code generation is skipped.
//   Device pass: __host__ functions are truly skipped -- not parsed, not instantiated, not checked.
//
//   Key consequences:
//     1. Architecture macros (__GFX9__, __gfx950__, etc.) are defined ONLY during the device pass. Any #if guard on them will take the #else branch during the host pass.
//     2. __device__ constexpr variables and static_asserts inside __device__ templates are still evaluated during the host pass (since templates may be instantiated from __global__).
//     3. If your device code relies on arch-specific preprocessor branches, consider guarding the entire implementation with #if defined(__HIP_DEVICE_COMPILE__) to skip the host pass.
//
// ---- get_warp_size() / get_smem_size() ----
//   OPUS_H_D constexpr -- safe to use everywhere: template defaults, static_assert, constexpr variables, __shared__ array sizes, host launch-parameter calculations, etc.
//   During the host pass (arch macros absent), they return safe defaults:
//     get_warp_size() -> 64 (GFX9 default), 32 for gfx1250 (wave32)
//     get_smem_size() -> 65536 (64 KB, non-gfx950 default)
//   Note: __builtin_amdgcn_wavefrontsize() is NOT constexpr in clang, so it cannot be used in template arguments, static_assert, or if constexpr. Prefer get_warp_size() which uses
//   preprocessor arch detection to provide a constexpr result.
//
// ---- query_warp_size() / query_smem_size() ----
//   OPUS_H only -- runtime HIP API queries (hipGetDeviceProperties). Use when you need the true hardware value on the host (e.g. occupancy calculations).
//   Guarded by OPUS_ENABLE_RUNTIME_QUERY (default 0). Define OPUS_ENABLE_RUNTIME_QUERY=1 before
//   including opus.hpp (or via compiler flag) to enable these functions and the hip_runtime_api.h include.
//
// gfx12 wave32/64 detection: __AMDGCN_WAVEFRONT_SIZE__ removed in ROCm 7.2; _w32 builtins are gated by wavefrontsize32 target feature, so __has_builtin is a constexpr proxy.
#if (defined(__gfx1201__) || defined(__gfx1200__)) && defined(__HIP_DEVICE_COMPILE__)
#  if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#    define OPUS_GFX120X_IS_WAVE32 1
#  else
#    define OPUS_GFX120X_IS_WAVE32 0
#  endif
#endif

OPUS_H_D constexpr index_t get_warp_size()
{
#if defined(__gfx1250__)
    return 32;
#elif defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
    return 64;
#elif defined(OPUS_GFX120X_IS_WAVE32) && !OPUS_GFX120X_IS_WAVE32
    return 64;
#else
    return 32;
#endif
}
OPUS_H_D constexpr index_t get_smem_size()
{
#if defined(__gfx950__)
    return 163840;  // 160KB (CDNA4)
#else
    return 65536;   // 64KB
#endif
}

// ---- Device intrinsic wrappers ----
// Replace HIP runtime macros (threadIdx.x, __syncthreads, __all, etc.) so kernels compile
// with just #include <opus/opus.hpp> -- no <hip/hip_runtime.h> needed.
OPUS_D index_t thread_id_x() { return __builtin_amdgcn_workitem_id_x(); }
OPUS_D index_t thread_id_y() { return __builtin_amdgcn_workitem_id_y(); }
OPUS_D index_t thread_id_z() { return __builtin_amdgcn_workitem_id_z(); }
OPUS_D index_t block_id_x()  { return __builtin_amdgcn_workgroup_id_x(); }
OPUS_D index_t block_id_y()  { return __builtin_amdgcn_workgroup_id_y(); }
OPUS_D index_t block_id_z()  { return __builtin_amdgcn_workgroup_id_z(); }
OPUS_D index_t block_size_x() { return __builtin_amdgcn_workgroup_size_x(); }
OPUS_D index_t block_size_y() { return __builtin_amdgcn_workgroup_size_y(); }
OPUS_D index_t block_size_z() { return __builtin_amdgcn_workgroup_size_z(); }
OPUS_D index_t grid_size_x()  { return __builtin_amdgcn_grid_size_x(); }
OPUS_D index_t grid_size_y()  { return __builtin_amdgcn_grid_size_y(); }
OPUS_D index_t grid_size_z()  { return __builtin_amdgcn_grid_size_z(); }
OPUS_D void    sync_threads() { __builtin_amdgcn_s_barrier(); }
#if !defined(HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H)
extern "C" __device__ int __ockl_wfall_i32(int);
#endif
#if !defined(HIP_INCLUDE_HIP_AMD_DETAIL_WARP_FUNCTIONS_H)
OPUS_D int     warp_all(int predicate) { return __ockl_wfall_i32(predicate); }
#endif

#if OPUS_ENABLE_RUNTIME_QUERY
OPUS_H index_t query_warp_size() { int d; (void)hipGetDevice(&d); hipDeviceProp_t p; (void)hipGetDeviceProperties(&p, d); return static_cast<index_t>(p.warpSize); }
OPUS_H index_t query_smem_size() { int d; (void)hipGetDevice(&d); hipDeviceProp_t p; (void)hipGetDeviceProperties(&p, d); return static_cast<index_t>(p.sharedMemPerBlock); }
OPUS_H index_t query_num_cu()    { int d; (void)hipGetDevice(&d); hipDeviceProp_t p; (void)hipGetDeviceProperties(&p, d); return static_cast<index_t>(p.multiProcessorCount); }
#endif

// Uses compiler builtins (__builtin_amdgcn_*) instead of HIP runtime APIs, so no <hip/hip_runtime.h> dependency.
#ifdef __HIPCC__
struct workgroup_barrier {
    OPUS_D workgroup_barrier(unsigned int* ptr) : base_ptr(ptr) {}
    OPUS_D unsigned int ld(unsigned int offset = 0) { return __atomic_load_n(base_ptr + offset, __ATOMIC_RELAXED); }
    OPUS_D void wait_eq(unsigned int value, unsigned int offset = 0) { if (__builtin_amdgcn_workitem_id_x() == 0) while (ld(offset) != value) {} __builtin_amdgcn_s_barrier(); }
    OPUS_D void wait_lt(unsigned int value, unsigned int offset = 0) { if (__builtin_amdgcn_workitem_id_x() == 0) while (ld(offset) < value) {} __builtin_amdgcn_s_barrier(); }
    OPUS_D void inc(unsigned int offset = 0) { __builtin_amdgcn_s_barrier(); if (__builtin_amdgcn_workitem_id_x() == 0) __atomic_fetch_add(base_ptr + offset, 1u, __ATOMIC_RELAXED); }
    unsigned int* base_ptr;
};
#endif

// NOTE: all data in unsigned int. Prefer usage, construct a mdiv structure on host, pass the structure to kernel, and use div/divmod
struct mdiv {
    unsigned int divisor;   unsigned int multiplier;    unsigned int shift;
    OPUS_H_D mdiv() : divisor(0), multiplier(0), shift(0) {}
    OPUS_H_D mdiv(unsigned int divisor_) : divisor(divisor_) {
        unsigned int shift_u32 = 0;
        while ((1U << shift_u32) < divisor_) shift_u32++;
        unsigned long long tmp_u64 = static_cast<unsigned long long>((1UL << shift_u32) - divisor_) << 32;
        multiplier       = static_cast<unsigned int>(tmp_u64 / divisor_ + 1);
        shift            = shift_u32;
    }
    // previously we use __umulhi(), which is defined in <hip/hip_runtime.hpp>, for __device__ compilation. Today compiler is smart enough to generate s_mul_hi_u32 / v_mul_hi_u32
    OPUS_H_D unsigned int div(unsigned int dividend) const { unsigned int tmp = static_cast<unsigned int>((static_cast<unsigned long long>(dividend) * multiplier) >> 32); return (tmp + dividend) >> shift; }
    OPUS_H_D void divmod(unsigned int dividend, unsigned int& quotient, unsigned int& remainder) const { quotient  = div(dividend);  remainder = dividend - (quotient * divisor); }
    OPUS_H_D unsigned int get() const { return divisor; }
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// math
template <typename T, int dpp_i, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
OPUS_D T mov_dpp(T x, number<dpp_i>, number<row_mask> = {}, number<bank_mask> = {}, bool_constant<bound_ctrl> = {}) {
    static_assert(sizeof(T) == 4); return __builtin_bit_cast(T, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), dpp_i, row_mask, bank_mask, bound_ctrl));
}

template<typename O, typename T, int dpp_i, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
OPUS_D T upd_dpp(const O& old, T x, number<dpp_i>, number<row_mask> = {}, number<bank_mask> = {}, bool_constant<bound_ctrl> = {}) {
    static_assert(sizeof(T) == 4); return __builtin_bit_cast(T, __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, old), __builtin_bit_cast(int, x), dpp_i, row_mask, bank_mask, bound_ctrl));
}

// lane index within wavefront (threadIdx.x % warp_size, e.g. wave64: tid=3->3, tid=70->6)
OPUS_D unsigned int lane_id() {
    if constexpr (get_warp_size() == 32) return __builtin_amdgcn_mbcnt_lo(-1, 0);
    else return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

// cross-lane shuffle via ds_bpermute (no hip_runtime.h dependency)
template<typename T>
OPUS_D T shfl(T var, int src_lane, int width = get_warp_size()) {
    static_assert(sizeof(T) == 4);  int self = lane_id();   int index = (src_lane & (width - 1)) + (self & ~(width - 1));
    return __builtin_bit_cast(T, __builtin_amdgcn_ds_bpermute(index << 2, __builtin_bit_cast(int, var)));
}

template<typename T> OPUS_D T max(const T&a, const T&b)                { return a > b ? a : b; }
template<> OPUS_D float       max<float>(const float&a, const float&b) { return __builtin_fmaxf(a, b); }
template<typename T> OPUS_D T min(const T&a, const T&b)                { return a > b ? b : a; }
template<> OPUS_D float       min<float>(const float&a, const float&b) { return __builtin_fminf(a, b); }

template<typename T> OPUS_D T med3(const T&a, const T&b, const T&c) { auto max_0 = max(a, b); auto min_0 = min(a, b); return min(max_0, max(min_0, c)); }
template<> OPUS_D float       med3<float>(const float&a, const float&b, const float&c) { return __builtin_amdgcn_fmed3f(a, b, c); }
template<> OPUS_D fp16_t      med3<fp16_t>(const fp16_t&a, const fp16_t&b, const fp16_t&c) { return __builtin_amdgcn_fmed3h(a, b, c); }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// buffer load/store related
OPUS_D constexpr auto buffer_default_config() {
#if defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || defined(__gfx9_4_generic__)
    return 0x00020000;
#elif defined(__gfx103__)
    return 0x31014000;
// gfx1200/gfx1201 (Navi 44/48) listed explicitly -- __gfx11__/__gfx12__ are typos (clang predefines only uppercase __GFX11__/__GFX12__), so without this gfx1200/gfx1201 fall into the 0xffffffff sentinel and make_gmem<> stores silently drop.
#elif defined(__gfx11__) || defined(__gfx12__) || defined(__gfx1250__) || defined(__gfx1201__) || defined(__gfx1200__)
    return 0x31004000;
#else
    return 0xffffffff;
#endif
}
OPUS_D __amdgpu_buffer_rsrc_t make_buffer_rsrc(const void* ptr, unsigned int size = 0xffffffff, unsigned int config = buffer_default_config()) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(static_cast<const void*>(ptr)), 0, size, config); // void *p, short stride, int num, int flags
}
#if __clang_major__ < 20
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
OPUS_D void llvm_amdgcn_raw_buffer_load_lds(i32x4_t r, OPUS_LDS_ADDR unsigned int* p, index_t size, index_t vos, index_t sos, index_t ios, index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
#pragma clang diagnostic pop
#endif
template<typename T_>
struct gmem {
    using T = remove_cvref_t<T_>;
    using scalar_type = typename vector_traits<T>::dtype;
    static constexpr index_t vector_size = vector_traits<T>::size();
    template<index_t vec = 1> using vector_type = vector_t<scalar_type, vec * vector_size>;

    OPUS_D gmem(const void* ptr, unsigned int size = 0xffffffff, unsigned int config = buffer_default_config())
        : cached_rsrc(make_buffer_rsrc(ptr, size, config))
#if defined(__gfx1250__)
        , raw_ptr(static_cast<const char*>(ptr))
#endif
    {}

    template<index_t vec = 1, index_t aux = 0>   // os in unit of byte
    OPUS_D auto _load(int v_os, int s_os = 0, number<aux> = {}) {
        using type = vector_type<vec>;
        if      constexpr (sizeof(type) == 1)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b8  (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 2)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b16 (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 4)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b32 (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 8)  { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b64 (cached_rsrc, v_os, s_os, aux)); }
        else if constexpr (sizeof(type) == 16) { return __builtin_bit_cast(type, __builtin_amdgcn_raw_buffer_load_b128(cached_rsrc, v_os, s_os, aux)); }
    }

    // Async copy global -> LDS. Offsets are in bytes.
    // IMPORTANT - i_os shifts BOTH ends of the copy, not just the source.
    template<index_t vec = 1, index_t i_os = 0, index_t aux = 0>
    OPUS_D void _async_load(OPUS_LDS_ADDR void* dst, int v_os, int s_os = 0, number<i_os> = {}, number<aux> = {}) {
        using type = vector_type<vec>;
#if defined(__gfx1250__)
        static_assert(i_os >= 0 && i_os <= (1 << 23) - 1, "_async_load: i_os exceeds gfx1250 VGLOBAL IOFFSET range [0, 2^23 - 1]");
        #define GPTR_(T, p) ((__attribute__((address_space(1))) T*)(p))
        #define LPTR_(T, p) ((OPUS_LDS_ADDR T*)(p))
        {
            auto* src = raw_ptr + v_os + s_os;
            if      constexpr (sizeof(type) == 1)  { __builtin_amdgcn_global_load_async_to_lds_b8  (GPTR_(char, src), LPTR_(char, dst), i_os, 0); }
            else if constexpr (sizeof(type) == 2)  { __builtin_amdgcn_global_load_async_to_lds_b8  (GPTR_(char, src), LPTR_(char, dst), i_os, 0);
                                                     __builtin_amdgcn_global_load_async_to_lds_b8  (GPTR_(char, src + 1), LPTR_(char, (char*)dst + 1), i_os, 0); }
            else if constexpr (sizeof(type) == 4)  { __builtin_amdgcn_global_load_async_to_lds_b32 (GPTR_(int, src), LPTR_(int, dst), i_os, 0); }
            else if constexpr (sizeof(type) == 8)  { __builtin_amdgcn_global_load_async_to_lds_b64 (GPTR_(i32x2_t, src), LPTR_(i32x2_t, dst), i_os, 0); }
            else if constexpr (sizeof(type) == 16) { __builtin_amdgcn_global_load_async_to_lds_b128(GPTR_(i32x4_t, src), LPTR_(i32x4_t, dst), i_os, 0); }
        }
        #undef GPTR_
        #undef LPTR_
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)   // CDNA: buffer-load lands directly in LDS (vmem-to-lds-load-insts).
        static_assert(i_os >= 0 && i_os <= (1 << 12) - 1, "_async_load: i_os exceeds CDNA MUBUF OFFSET range [0, 2^12 - 1]");
#if __clang_major__ >= 20    // clang 20+ raw_ptr builtin (rocm 7.0+)
        if      constexpr (sizeof(type) == 1)  { __builtin_amdgcn_raw_ptr_buffer_load_lds(cached_rsrc, dst,  1, v_os, s_os, i_os, aux); }
        else if constexpr (sizeof(type) == 2)  { __builtin_amdgcn_raw_ptr_buffer_load_lds(cached_rsrc, dst,  2, v_os, s_os, i_os, aux); }
        else if constexpr (sizeof(type) == 4)  { __builtin_amdgcn_raw_ptr_buffer_load_lds(cached_rsrc, dst,  4, v_os, s_os, i_os, aux); }
#if  defined(__gfx950__)
        else if constexpr (sizeof(type) == 12) { __builtin_amdgcn_raw_ptr_buffer_load_lds(cached_rsrc, dst, 12, v_os, s_os, i_os, aux); }
        else if constexpr (sizeof(type) == 16) { __builtin_amdgcn_raw_ptr_buffer_load_lds(cached_rsrc, dst, 16, v_os, s_os, i_os, aux); }
#endif
#else
        i32x4_t cached_rsrc_;
        __builtin_memcpy(&cached_rsrc_, &cached_rsrc, sizeof(i32x4_t));   // builtin memcpy, __builtin_bit_cast() can not use here due to __amdgpu_buffer_rsrc_t is non copyable
        if      constexpr (sizeof(type) == 1)  {llvm_amdgcn_raw_buffer_load_lds(cached_rsrc_, reinterpret_cast<OPUS_LDS_ADDR u32_t*>(dst),  1, v_os, s_os, i_os, aux); }
        else if constexpr (sizeof(type) == 2)  {llvm_amdgcn_raw_buffer_load_lds(cached_rsrc_, reinterpret_cast<OPUS_LDS_ADDR u32_t*>(dst),  2, v_os, s_os, i_os, aux); }
        else if constexpr (sizeof(type) == 4)  {llvm_amdgcn_raw_buffer_load_lds(cached_rsrc_, reinterpret_cast<OPUS_LDS_ADDR u32_t*>(dst),  4, v_os, s_os, i_os, aux); }
#if  defined(__gfx950__)
        else if constexpr (sizeof(type) == 12) {llvm_amdgcn_raw_buffer_load_lds(cached_rsrc_, reinterpret_cast<OPUS_LDS_ADDR u32_t*>(dst), 12, v_os, s_os, i_os, aux); }
        else if constexpr (sizeof(type) == 16) {llvm_amdgcn_raw_buffer_load_lds(cached_rsrc_, reinterpret_cast<OPUS_LDS_ADDR u32_t*>(dst), 16, v_os, s_os, i_os, aux); }
#endif
#endif
#else
        auto val = _load<vec, aux>(v_os + i_os, s_os, number<aux>{});
        dst = reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst) + i_os);
        *reinterpret_cast<OPUS_LDS_ADDR type*>(dst) = val;
#endif
    }

    template<index_t vec = 1, typename V, index_t aux = 0>   // os in unit of byte
    OPUS_D void _store(const V& x, int v_os, int s_os = 0, number<aux> = {}) {
        static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check");
        if      constexpr (sizeof(vector_type<vec>) == 1)  { __builtin_amdgcn_raw_buffer_store_b8  (__builtin_bit_cast(i8_t,    x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 2)  { __builtin_amdgcn_raw_buffer_store_b16 (__builtin_bit_cast(i16_t,   x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 4)  { __builtin_amdgcn_raw_buffer_store_b32 (__builtin_bit_cast(i32_t,   x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 8)  { __builtin_amdgcn_raw_buffer_store_b64 (__builtin_bit_cast(i32x2_t, x), cached_rsrc, v_os, s_os, aux); }
        else if constexpr (sizeof(vector_type<vec>) == 16) { __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(i32x4_t, x), cached_rsrc, v_os, s_os, aux); }
    }

    template<index_t vec = 1, index_t aux = 0>   // os in unit of T and cast to vector with vec
    OPUS_D auto load(int v_os, int s_os = 0, number<aux> = {}) { return _load<vec>(v_os * sizeof(T), s_os * sizeof(T), number<aux>{}); }

    template<index_t vec = 1, index_t i_os = 0, index_t aux = 0>   // os in unit of T and cast to vector with vec; i_os = compile-time immediate byte offset (see _async_load notes; applies to both src and dst)
    OPUS_D void async_load(void* dst, int v_os, int s_os = 0, number<i_os> = {}, number<aux> = {}) { _async_load<vec>(reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)), v_os * sizeof(T), s_os * sizeof(T), number<i_os>{}, number<aux>{}); }

    template<index_t vec = 1, typename V, index_t aux = 0, std::enable_if_t<(is_vector_v<V> || is_dtype_v<V> || is_array_v<V>), bool> = true>   // os in unit of T and cast to vector with vec
    OPUS_D void store(const V& x, int v_os, int s_os = 0, number<aux> = {}) {
        static_assert(std::is_same_v<typename vector_traits<V>::dtype, scalar_type>, "scalar type must be same for the data to be stored" );
        if constexpr (is_dtype_v<V> && (vec * vector_size) % vector_traits<V>::size() == 0) {
            _store<vec>(make_repeated_vector(x, number<vec * vector_size / vector_traits<V>::size()>{}), v_os * sizeof(T), s_os * sizeof(T), number<aux>{});
        } else {
            static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check" );
            _store<vec>(x, v_os * sizeof(T), s_os * sizeof(T), number<aux>{});
        }
    }

    // bulk load API, give me a Shape of this tile, will issue multiple load instruction based on the y-shape space
    template<index_t vec = 1, typename Layout, index_t aux = 0, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto load(const Layout& u, int s_os = 0/* do we really need this? */, number<aux> = {})
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = layout_to_offsets<vec>(u);

#if OPUS_TILE_CONTAINER == 0
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;
        for (index_t i = 0; i < r_elem.value; i++) {
            auto tmp = load<vec>(offsets[i], s_os, number<aux>{});
            for (index_t j = 0; j < vec * vector_size; j++) r[i * vec * vector_size + j] = tmp[j];
        }
        return r;
#elif OPUS_TILE_CONTAINER == 1
        array<vector_type<vec>, r_elem.value> r;
        for (index_t i = 0; i < r_elem.value; i++) r[i] = load<vec>(offsets[i], s_os, number<aux>{});
        return r;
#endif
    }

    template<index_t vec = 1, typename V, typename Layout, index_t aux = 0, std::enable_if_t<((is_array_v<V> || is_vector_v<V>) && is_layout_v<Layout>), bool> = true>
    OPUS_D void store(const V& x, const Layout& u, int s_os = 0/* do we really need this? */, number<aux> = {})
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = layout_to_offsets<vec>(u);

#if OPUS_TILE_CONTAINER == 0
        auto a_ = [&](){ if constexpr (is_array_v<V>) return to_vector(x);
                         else if constexpr (is_dtype_v<V>) return make_repeated_vector(x, number<r_elem.value>{});
                         else if constexpr (is_vector_v<V>) return x; }();
#elif OPUS_TILE_CONTAINER == 1
        auto a_ = to_array(x);
#endif
        for (index_t i = 0; i < r_elem.value; i++) {
            vector_type<vec> v_;
            for (index_t j = 0; j < vec * vector_size; j++) v_[j] = a_[i * vec * vector_size + j];
            store<vec>(v_, offsets[i], s_os, number<aux>{});
        }
    }

    template<index_t vec = 1, typename LayoutG, typename LayoutS, index_t i_os = 0, index_t aux = 0, std::enable_if_t<is_layout_v<LayoutG> && is_layout_v<LayoutS>, bool> = true>
    OPUS_D void async_load(void* smem_base, const LayoutG& u_gmem, const LayoutS& u_smem, int s_os = 0, number<i_os> = {}, number<aux> = {}) {
        using LT = layout_load_traits<LayoutG, vec>;
        constexpr auto r_elem = LT::r_elem;
        auto gmem_offsets = layout_to_offsets<vec>(u_gmem);
        auto smem_offsets = layout_to_offsets<vec>(u_smem);
        auto smem_ptr = reinterpret_cast<OPUS_LDS_ADDR scalar_type*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_base));
        for (index_t i = 0; i < r_elem.value; i++) {
            async_load<vec>(reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_ptr + smem_offsets[i])), gmem_offsets[i], s_os, number<i_os>{}, number<aux>{});
        }
    }

    template<index_t vec = 1, typename Predicate, typename Layout, index_t aux = 0, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto load_if(const Predicate& pred, const Layout& u, int s_os = 0, number<aux> = {})
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto issue_space = LT::issue_space;
        constexpr auto issue_space_vec = LT::issue_space_vec;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = layout_to_offsets<vec>(u);
        constexpr auto u_r = make_layout<-1>(issue_space_vec);

#if OPUS_TILE_CONTAINER == 0
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;
        static_ford(issue_space_vec, [&](auto ... ids){
            constexpr index_t idx = u_r(ids...);
            auto tmp = pred(ids...) ? load<vec>(offsets[idx], s_os, number<aux>{}) : vector_type<vec>{0};
            set_slice(r, tmp, number<idx * vec>{}, number<(idx + 1) * vec>{});
        });
        return r;
#elif OPUS_TILE_CONTAINER == 1
        array<vector_type<vec>, r_elem.value> r;
        static_ford(issue_space_vec, [&](auto ... ids){ r[u_r(ids...)] = pred(ids...) ? load<vec>(offsets[u_r(ids...)], s_os, number<aux>{}) : vector_type<vec>{0}; });
        return r;
#endif
    }

    template<index_t vec = 1, typename Predicate, typename V, typename Layout, index_t aux = 0, std::enable_if_t<((is_array_v<V> || is_vector_v<V>) && is_layout_v<Layout>), bool> = true>
    OPUS_D void store_if(const Predicate& pred, const V& x, const Layout& u, int s_os = 0, number<aux> = {})
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto issue_space = LT::issue_space;
        constexpr auto issue_space_vec = LT::issue_space_vec;
        auto offsets = layout_to_offsets<vec>(u);
        constexpr auto u_r = make_layout<-1>(issue_space);

#if OPUS_TILE_CONTAINER == 0
        auto a_ = [&](){ if constexpr (is_array_v<V>) return to_vector(x);
                         else if constexpr (is_dtype_v<V>) return make_repeated_vector(x, number<get<0>(reduce_tuple_mul(issue_space)).value>{});
                         else if constexpr (is_vector_v<V>) return x; }();
#elif OPUS_TILE_CONTAINER == 1
        auto a_ = to_array(x);
#endif
        static_ford(issue_space_vec, [&](auto ... ids){
            if (pred(ids...)) {
                constexpr index_t idx = u_r(ids...);
                auto v_ = slice(a_, number<idx>{}, number<idx + vec>{});
                store<vec>(v_, offsets[make_layout<-1>(issue_space_vec)(ids...)], s_os, number<aux>{});
            }
        });
    }

    template<index_t vec = 1, typename Predicate, typename LayoutG, typename LayoutS, index_t i_os = 0, index_t aux = 0, std::enable_if_t<is_layout_v<LayoutG> && is_layout_v<LayoutS>, bool> = true>
    OPUS_D void async_load_if(const Predicate& pred, void* smem_base, const LayoutG& u_gmem, const LayoutS& u_smem, int s_os = 0, number<i_os> = {}, number<aux> = {}) {
        using LT = layout_load_traits<LayoutG, vec>;
        constexpr auto issue_space_vec = LT::issue_space_vec;
        auto gmem_offsets = layout_to_offsets<vec>(u_gmem);
        auto smem_offsets = layout_to_offsets<vec>(u_smem);
        auto smem_ptr = reinterpret_cast<OPUS_LDS_ADDR scalar_type*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_base));
        constexpr auto u_r = make_layout<-1>(issue_space_vec);

        static_ford(issue_space_vec, [&](auto... ids) {
            constexpr index_t idx = u_r(ids...);
            if (pred(ids...)) {
                async_load<vec>(reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_ptr + smem_offsets[idx])), gmem_offsets[idx], s_os, number<i_os>{}, number<aux>{});
            } else {
                using type = vector_type<vec>;
                type z = {0};
                *reinterpret_cast<OPUS_LDS_ADDR type*>(smem_ptr + smem_offsets[idx]) = z;
            }
        });
    }

    __amdgpu_buffer_rsrc_t cached_rsrc;
#if defined(__gfx1250__)
    const char* raw_ptr;  // flat pointer for global_load_async_to_lds (gfx1250 uses global addressing, not buffer rsrc)
#endif
};

template<typename T_> OPUS_D decltype(auto) make_gmem(const T_* ptr, unsigned int size = 0xffffffff, unsigned int config = buffer_default_config()) { return gmem<T_>{ptr, size, config}; }
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// smem load/store related
template<typename T_>
struct smem {
    using T = remove_cvref_t<T_>;
    using scalar_type = typename vector_traits<T>::dtype;
    static constexpr index_t vector_size = vector_traits<T>::size();
    template<index_t vec = 1> using vector_type = vector_t<scalar_type, vec * vector_size>;

    OPUS_D smem(void* ptr_) : ptr(reinterpret_cast<OPUS_LDS_ADDR char*>(reinterpret_cast<__UINTPTR_TYPE__>(ptr_))) {}

    template<index_t vec = 1> OPUS_D auto _load(int v_os/* in unit of byte*/) { using type = vector_type<vec>; return *reinterpret_cast<OPUS_LDS_ADDR type*>(ptr + v_os); }

    template<index_t vec = 1, int imm_offset = 0> OPUS_D auto _tr_load(int v_os/* in unit of byte*/) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
        using type = vector_type<vec>;
        static_assert(sizeof(type) == 8, "DS_READ_B64_TR requires 8-byte (64-bit) load");
        constexpr index_t elem_bits = sizeof_bits_v<scalar_type>;
        i32x2_t raw;
        const u32_t addr = static_cast<u32_t>(reinterpret_cast<__UINTPTR_TYPE__>(ptr + v_os));
        if      constexpr (elem_bits == 16) { asm volatile("ds_read_b64_tr_b16 %0, %1 offset:%2\n" : "=v"(raw) : "v"(addr), "i"(imm_offset) : "memory"); }
        else if constexpr (elem_bits == 8)  { asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2\n" : "=v"(raw) : "v"(addr), "i"(imm_offset) : "memory"); }
        else if constexpr (elem_bits == 4)  { asm volatile("ds_read_b64_tr_b4 %0, %1 offset:%2\n" : "=v"(raw) : "v"(addr), "i"(imm_offset) : "memory"); }
        else { static_assert(sizeof(T_) == 0, "smem::_tr_load: unsupported scalar type"); }
        return __builtin_bit_cast(type, raw);
#elif defined(__HIP_DEVICE_COMPILE__)
        static_assert(sizeof(T_) == 0, "smem::_tr_load requires __gfx950__");
        return _load<vec>(v_os + imm_offset);
#else
        return _load<vec>(v_os + imm_offset);
#endif
    }

    template<index_t vec = 1, typename V>
    OPUS_D void _store(const V& x, int v_os/* in unit of byte*/) {
        static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check");
        using type = vector_type<vec>;
        *reinterpret_cast<OPUS_LDS_ADDR type*>(ptr + v_os) = __builtin_bit_cast(type, x);
    }

    template<index_t vec = 1> OPUS_D auto load(int v_os) { return _load<vec>(v_os * sizeof(T)); }

    template<index_t vec = 1> OPUS_D auto tr_load(int v_os) { return _tr_load<vec>(v_os * sizeof(T)); }

    template<index_t vec = 1, typename V, std::enable_if_t<(is_vector_v<V> || is_dtype_v<V> || is_array_v<V>), bool> = true>
    OPUS_D void store(const V& x, int v_os) {
        static_assert(std::is_same_v<typename vector_traits<V>::dtype, scalar_type>, "scalar type must be same for the data to be stored" );
        if constexpr (is_dtype_v<V> && (vec * vector_size) % vector_traits<V>::size() == 0) {
            _store<vec>(make_repeated_vector(x, number<vec * vector_size / vector_traits<V>::size()>{}), v_os * sizeof(T));
        } else {
            static_assert((vec * vector_size) == vector_traits<V>::size(), "vector size need to be same, please check" );
            _store<vec>(x, v_os * sizeof(T));
        }
    }

    // bulk load API, give me a Shape of this tile, will issue multiple load instruction based on the y-shape space
    template<index_t vec = 1, typename Layout, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto load(const Layout& u)
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = layout_to_offsets<vec>(u);

#if OPUS_TILE_CONTAINER == 0
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;
        for (index_t i = 0; i < r_elem.value; i++) {
            auto tmp = load<vec>(offsets[i]);
            for (index_t j = 0; j < vec * vector_size; j++) r[i * vec * vector_size + j] = tmp[j];
        }
        return r;
#elif OPUS_TILE_CONTAINER == 1
        array<vector_type<vec>, r_elem.value> r;
        for (index_t i = 0; i < r_elem.value; i++) r[i] = load<vec>(offsets[i]);
        return r;
#endif
    }

    template<index_t vec = 1, typename Layout, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto tr_load(const Layout& u)
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto r_elem = LT::r_elem;
        constexpr bool is_static_layout = is_static_tuple_v<typename Layout::Shape> && is_static_tuple_v<typename Layout::Stride>;
        [[maybe_unused]] auto offsets = layout_to_offsets<vec>(u);

        auto issue = [&](auto i) {
            if constexpr (is_static_layout) {
                using shift = layout_shift_traits<remove_cvref_t<Layout>>;
                constexpr int off = layout_imm_offsets_v<Layout, vec>[i.value] * sizeof(T);
                if constexpr (off >= 0 && off <= 0xffff) {
                    const auto& ub = static_cast<const typename shift::base&>(u);
                    const int base = ub(make_repeated_tuple(number<0>{}, number<size<decltype(LT::issue_space_vec)>()>{})) * sizeof(T);
                    return _tr_load<vec, off>(base);
                }
            }
            return tr_load<vec>(offsets[i.value]);
        };

#if OPUS_TILE_CONTAINER == 0
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;
        static_for<r_elem.value>([&](auto i){
            set_slice(r, issue(i), number<i.value * vec>{}, number<(i.value + 1) * vec>{});
        });
        return r;
#elif OPUS_TILE_CONTAINER == 1
        array<vector_type<vec>, r_elem.value> r;
        static_for<r_elem.value>([&](auto i){ r[i.value] = issue(i); });
        return r;
#endif
    }

    template<index_t vec = 1, typename V, typename Layout, std::enable_if_t<((is_array_v<V> || is_dtype_v<V> || is_vector_v<V>) && is_layout_v<Layout>), bool> = true>
    OPUS_D void store(const V& x, const Layout& u)
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = layout_to_offsets<vec>(u);

#if OPUS_TILE_CONTAINER == 0
        auto a_ = [&](){ if constexpr (is_array_v<V>) return to_vector(x);
                         else if constexpr (is_dtype_v<V>) return make_repeated_vector(x, number<r_elem.value>{});
                         else if constexpr (is_vector_v<V>) return x; }();
#elif OPUS_TILE_CONTAINER == 1
        auto a_ = to_array(x);
#endif
        for (index_t i = 0; i < r_elem.value; i++) {
            vector_type<vec> v_;
            for (index_t j = 0; j < vec * vector_size; j++) v_[j] = a_[i * vec * vector_size + j];
            store<vec>(v_, offsets[i]);
        }
    }

    template<index_t vec = 1, typename Predicate, typename Layout, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto load_if(const Predicate& pred, const Layout& u)
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto issue_space_vec = LT::issue_space_vec;
        constexpr auto r_elem = LT::r_elem;
        auto offsets = layout_to_offsets<vec>(u);
        constexpr auto u_r = make_layout<-1>(issue_space_vec);

#if OPUS_TILE_CONTAINER == 0
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;
        static_ford(issue_space_vec, [&](auto ... ids){
            constexpr index_t idx = u_r(ids...);
            auto tmp = pred(ids...) ? load<vec>(offsets[idx]) : vector_type<vec>{0};
            set_slice(r, tmp, number<idx * vec>{}, number<(idx + 1) * vec>{});
        });
        return r;
#elif OPUS_TILE_CONTAINER == 1
        array<vector_type<vec>, r_elem.value> r;
        static_ford(issue_space_vec, [&](auto ... ids){ r[u_r(ids...)] = pred(ids...) ? load<vec>(offsets[u_r(ids...)]) : vector_type<vec>{0}; });
        return r;
#endif
    }

    template<index_t vec = 1, typename Predicate, typename Layout, std::enable_if_t<is_layout_v<Layout>, bool> = true>
    OPUS_D auto tr_load_if(const Predicate& pred, const Layout& u)
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto issue_space_vec = LT::issue_space_vec;
        constexpr auto r_elem = LT::r_elem;
        constexpr bool is_static_layout = is_static_tuple_v<typename Layout::Shape> && is_static_tuple_v<typename Layout::Stride>;
        [[maybe_unused]] auto offsets = layout_to_offsets<vec>(u);
        constexpr auto u_r = make_layout<-1>(issue_space_vec);

        auto issue = [&](auto i) {
            if constexpr (is_static_layout) {
                using shift = layout_shift_traits<remove_cvref_t<Layout>>;
                constexpr int off = layout_imm_offsets_v<Layout, vec>[i.value] * sizeof(T);
                if constexpr (off >= 0 && off <= 0xffff) {
                    const auto& ub = static_cast<const typename shift::base&>(u);
                    const int base = ub(make_repeated_tuple(number<0>{}, number<size<decltype(LT::issue_space_vec)>()>{})) * sizeof(T);
                    return _tr_load<vec, off>(base);
                }
            }
            return tr_load<vec>(offsets[i.value]);
        };

#if OPUS_TILE_CONTAINER == 0
        vector_t<scalar_type, vec * vector_size * r_elem.value> r;
        static_ford(issue_space_vec, [&](auto ... ids){
            constexpr index_t idx = u_r(ids...);
            auto tmp = pred(ids...) ? issue(number<idx>{}) : vector_type<vec>{0};
            set_slice(r, tmp, number<idx * vec>{}, number<(idx + 1) * vec>{});
        });
        return r;
#elif OPUS_TILE_CONTAINER == 1
        array<vector_type<vec>, r_elem.value> r;
        static_ford(issue_space_vec, [&](auto ... ids){ constexpr index_t idx = u_r(ids...); r[idx] = pred(ids...) ? issue(number<idx>{}) : vector_type<vec>{0}; });
        return r;
#endif
    }

    template<index_t vec = 1, typename Predicate, typename V, typename Layout, std::enable_if_t<((is_array_v<V> || is_dtype_v<V> || is_vector_v<V>) && is_layout_v<Layout>), bool> = true>
    OPUS_D void store_if(const Predicate& pred, const V& x, const Layout& u)
    {
        using LT = layout_load_traits<Layout, vec>;
        constexpr auto issue_space = LT::issue_space;
        constexpr auto issue_space_vec = LT::issue_space_vec;
        auto offsets = layout_to_offsets<vec>(u);
        constexpr auto u_r = make_layout<-1>(issue_space);

#if OPUS_TILE_CONTAINER == 0
        auto a_ = [&](){ if constexpr (is_array_v<V>) return to_vector(x);
                         else if constexpr (is_dtype_v<V>) return make_repeated_vector(x, number<get<0>(reduce_tuple_mul(issue_space)).value>{});
                         else if constexpr (is_vector_v<V>) return x; }();
#elif OPUS_TILE_CONTAINER == 1
        auto a_ = to_array(x);
#endif
        static_ford(issue_space_vec, [&](auto ... ids){
            if (pred(ids...)) {
                constexpr index_t idx = u_r(ids...);
                auto v_ = slice(a_, number<idx>{}, number<idx + vec>{});
                store<vec>(v_, offsets[make_layout<-1>(issue_space_vec)(ids...)]);
            }
        });
    }

    OPUS_LDS_ADDR char* ptr; // in unit of byte
};

template<typename T_> OPUS_D decltype(auto) make_smem(T_* ptr) { return smem<T_>{ptr}; }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// mem type traits & free function wrappers (eliminate .template syntax in dependent context)
template<typename>   struct is_gmem : false_type {};
template<typename T> struct is_gmem<gmem<T>> : true_type {};
template<typename T> constexpr bool is_gmem_v = is_gmem<remove_cvref_t<T>>::value;

template<typename>   struct is_smem : false_type {};
template<typename T> struct is_smem<smem<T>> : true_type {};
template<typename T> constexpr bool is_smem_v = is_smem<remove_cvref_t<T>>::value;

template<typename T> constexpr bool is_mem_v = is_gmem_v<T> || is_smem_v<T>;

template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_mem_v<Mem>, bool> = true>
OPUS_D auto load(Mem& mem, Args&&... args) { return mem.template load<vec>(std::forward<Args>(args)...); }
template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_mem_v<Mem>, bool> = true>
OPUS_D void store(Mem& mem, Args&&... args) { mem.template store<vec>(std::forward<Args>(args)...); }
template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_gmem_v<Mem>, bool> = true>
OPUS_D void async_load(Mem& mem, Args&&... args) { mem.template async_load<vec>(std::forward<Args>(args)...); }

template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_mem_v<Mem>, bool> = true>
OPUS_D auto load_if(Mem& mem, Args&&... args) { return mem.template load_if<vec>(std::forward<Args>(args)...); }
template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_smem_v<Mem>, bool> = true>
OPUS_D auto tr_load(Mem& mem, Args&&... args) { return mem.template tr_load<vec>(std::forward<Args>(args)...); }
template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_smem_v<Mem>, bool> = true>
OPUS_D auto tr_load_if(Mem& mem, Args&&... args) { return mem.template tr_load_if<vec>(std::forward<Args>(args)...); }
template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_mem_v<Mem>, bool> = true>
OPUS_D void store_if(Mem& mem, Args&&... args) { mem.template store_if<vec>(std::forward<Args>(args)...); }
template<index_t vec = 1, typename Mem, typename... Args, std::enable_if_t<is_gmem_v<Mem>, bool> = true>
OPUS_D void async_load_if(Mem& mem, Args&&... args) { mem.template async_load_if<vec>(std::forward<Args>(args)...); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// waitcnt
#if defined(__gfx1250__)
// gfx1250: split wait counters, exposed as native instruction wrappers via LLVM IR intrinsics.
// s_wait_expcnt/s_wait_samplecnt/s_wait_bvhcnt do NOT exist on gfx1250.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
OPUS_D void llvm_s_wait_loadcnt(short cnt)    __asm("llvm.amdgcn.s.wait.loadcnt");
OPUS_D void llvm_s_wait_dscnt(short cnt)      __asm("llvm.amdgcn.s.wait.dscnt");
OPUS_D void llvm_s_wait_storecnt(short cnt)   __asm("llvm.amdgcn.s.wait.storecnt");
OPUS_D void llvm_s_wait_kmcnt(short cnt)      __asm("llvm.amdgcn.s.wait.kmcnt");
OPUS_D void llvm_s_wait_asynccnt(short cnt)   __asm("llvm.amdgcn.s.wait.asynccnt");
OPUS_D void llvm_s_wait_tensorcnt(short cnt)  __asm("llvm.amdgcn.s.wait.tensorcnt");
#pragma clang diagnostic pop

template <index_t cnt> OPUS_D void s_wait_loadcnt(number<cnt> = {})   { llvm_s_wait_loadcnt(cnt); }
template <index_t cnt> OPUS_D void s_wait_dscnt(number<cnt> = {})     { llvm_s_wait_dscnt(cnt); }
template <index_t cnt> OPUS_D void s_wait_storecnt(number<cnt> = {})  { llvm_s_wait_storecnt(cnt); }
template <index_t cnt> OPUS_D void s_wait_kmcnt(number<cnt> = {})     { llvm_s_wait_kmcnt(cnt); }
template <index_t cnt> OPUS_D void s_wait_asynccnt(number<cnt> = {})  { llvm_s_wait_asynccnt(cnt); }
template <index_t cnt> OPUS_D void s_wait_tensorcnt(number<cnt> = {}) { llvm_s_wait_tensorcnt(cnt); }
#else
// gfx9: combined s_waitcnt instruction
template <index_t vmcnt, index_t lgkmcnt, index_t expcnt = 7>
OPUS_D void s_waitcnt(number<vmcnt>, number<lgkmcnt>, number<expcnt> = {})
{   __builtin_amdgcn_s_waitcnt((((0b110000 & vmcnt) << (14 - 4)) | (0b1111 & vmcnt)) | ((0b111 & expcnt) << 4) | ((0b1111 & lgkmcnt) << 8)); }

template <index_t vmcnt>   OPUS_D void s_waitcnt_vmcnt(number<vmcnt>) { s_waitcnt(number<vmcnt>{}, number<15>{}); }
template <index_t lgkmcnt> OPUS_D void s_waitcnt_lgkmcnt(number<lgkmcnt>) { s_waitcnt(number<63>{}, number<lgkmcnt>{}); }
#endif

// Helper: resolve vtype for MFMA/WMMA registers. Packed types (fp4_t etc.) use underlying storage since ext_vector_type requires scalar types.
namespace impl { template<typename T, index_t N, typename = void> struct mfma_vtype { using type = vector_t<T, N>; };
template<typename T, index_t N> struct mfma_vtype<T, N, std::enable_if_t<is_packs_v<T>>> { using type = vector_t<typename T::storage, N>; }; }
template<typename T, index_t N> using mfma_vtype_t = typename impl::mfma_vtype<T, N>::type;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// mfma (GFX9: gfx942, gfx950)
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define DISPATCH_MFMA_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_) {  return inst_(a, b, c, cbsz, abid, blgp); }

#define DISPATCH_MFMA_STEP_K_(ta_, tb_, tc_, wm_, wn_, wk_, inst_k_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    constexpr index_t steps = wk_ / inst_k_;  constexpr index_t e_a = elem_a / steps; constexpr index_t e_b = elem_b / steps;   \
    auto tmp = inst_(slice(a, number<0>{}, number<e_a>{}), slice(b, number<0>{}, number<e_b>{}), c, cbsz, abid, blgp);          \
    static_for<steps - 1>([&](auto i){ tmp = inst_(slice(a, number<e_a * (i+1)>{}, number<e_a*(i+2)>{}), slice(b, number<e_b * (i+1)>{}, number<e_b * (i+2)>{}), tmp, cbsz, abid, blgp); });  \
    return tmp; }

// f32 MFMA: inputs are scalar floats (elem_a = elem_b = 1), extract via [0] from vector_t<fp32_t, 1>
#define DISPATCH_MFMA_F32_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { return inst_(a[0], b[0], c, cbsz, abid, blgp); }

// gfx942 _1k bf16 intrinsics require short vectors; bitcast bf16 -> short before calling
#define DISPATCH_MFMA_BF16_1K_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    using _sa = short __attribute__((ext_vector_type(elem_a))); using _sb = short __attribute__((ext_vector_type(elem_b))); \
    return inst_(__builtin_bit_cast(_sa, a), __builtin_bit_cast(_sb, b), c, cbsz, abid, blgp); }

#define DISPATCH_MFMA_STEP_K_BF16_1K_(ta_, tb_, tc_, wm_, wn_, wk_, inst_k_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    constexpr index_t steps = wk_ / inst_k_;  constexpr index_t e_a = elem_a / steps; constexpr index_t e_b = elem_b / steps;   \
    using _sa = short __attribute__((ext_vector_type(e_a))); using _sb = short __attribute__((ext_vector_type(e_b))); \
    auto tmp = inst_(__builtin_bit_cast(_sa, slice(a, number<0>{}, number<e_a>{})), __builtin_bit_cast(_sb, slice(b, number<0>{}, number<e_b>{})), c, cbsz, abid, blgp);          \
    static_for<steps - 1>([&](auto i){ tmp = inst_(__builtin_bit_cast(_sa, slice(a, number<e_a * (i+1)>{}, number<e_a*(i+2)>{})), __builtin_bit_cast(_sb, slice(b, number<e_b * (i+1)>{}, number<e_b * (i+2)>{})), tmp, cbsz, abid, blgp); });  \
    return tmp; }

// fp8/bf8 intrinsics expect packed long (8 x 8-bit = 64-bit); bitcast ext_vector -> long
#define DISPATCH_MFMA_8BIT_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    return inst_(__builtin_bit_cast(long, a), __builtin_bit_cast(long, b), c, cbsz, abid, blgp); }

// scaled MFMA (f8f6f4): input bitcast to i32x8_t (256 bits); uses format codes, runtime scale, and 2-bit byte-selectors (scale_op_sel_a/b) into the packed-int32 scale word.
#define DISPATCH_MFMA_SCALE_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    return inst_(__builtin_bit_cast(i32x8_t, a), __builtin_bit_cast(i32x8_t, b), c, fmt_a, fmt_b, scale_op_sel_a, scale_a, scale_op_sel_b, scale_b); }

// prefer use make_mfma() to create instance, which will return impl::mfma_adaptor_xxx. In this way we can access layout info from the "mma"
//
// Scaled MFMA (gfx950: __builtin_amdgcn_mfma_scale_f32_{32x32x64,16x16x128}_f8f6f4)
// is also dispatched from this struct via the operator()(a, b, c, int scale_a, int scale_b) overload.
// Input registers are always 256 bits (i32x8_t) regardless of element type; bitcast is done internally.
// Format codes (Atype / Btype): 0=fp8(E4M3), 1=bf8(E5M2), 2=fp6(E2M3), 3=bf6(E3M2), 4=fp4(E2M1)
// scale_a, scale_b: E8M0 exponent values (int); actual_scale = 2^(value - 127). Use 127 for no scaling.
template<typename dtype_a_, typename dtype_b_, typename dtype_c_, index_t wave_m_, index_t wave_n_, index_t wave_k_, index_t warp_size_ = get_warp_size()>
struct mfma {
    using dtype_a = remove_cvref_t<dtype_a_>;
    using dtype_b = remove_cvref_t<dtype_b_>;
    using dtype_c = remove_cvref_t<dtype_c_>;
    static constexpr index_t wave_m = wave_m_;
    static constexpr index_t wave_n = wave_n_;
    static constexpr index_t wave_k = wave_k_;
    static constexpr index_t warp_size = warp_size_;
    static constexpr index_t elem_a = wave_m * wave_k / warp_size;
    static constexpr index_t elem_b = wave_n * wave_k / warp_size;
    static constexpr index_t elem_c = wave_m * wave_n / warp_size;

    using vtype_a = mfma_vtype_t<dtype_a, elem_a>;
    using vtype_b = mfma_vtype_t<dtype_b, elem_b>;
    using vtype_c = vector_t<dtype_c, elem_c>;

    // Format code for scaled MFMA (f8f6f4); -1 for types that don't support scaling
    static constexpr int fmt_a = std::is_same_v<dtype_a, fp8_t> ? 0 : std::is_same_v<dtype_a, bf8_t> ? 1 : std::is_same_v<dtype_a, fp4_t> ? 4 : -1;
    static constexpr int fmt_b = std::is_same_v<dtype_b, fp8_t> ? 0 : std::is_same_v<dtype_b, bf8_t> ? 1 : std::is_same_v<dtype_b, fp4_t> ? 4 : -1;

    // Regular MFMA dispatch (cbsz/abid/blgp are compile-time parameters)
    template<typename VA, typename VB, typename VC, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) -> vtype_c {
        (void)a; (void)b; (void)c; // used by DISPATCH_MFMA_ macros; suppress -Wunused-parameter on host
        if      constexpr (false) {} // in case of macro not defined
#if defined(__gfx942__) || defined(__gfx9_4_generic__) || defined(__gfx950__)
        else if constexpr DISPATCH_MFMA_(fp16_t, fp16_t, fp32_t, 32, 32,  8, __builtin_amdgcn_mfma_f32_32x32x8f16)
        else if constexpr DISPATCH_MFMA_(fp16_t, fp16_t, fp32_t, 16, 16, 16, __builtin_amdgcn_mfma_f32_16x16x16f16)
        else if constexpr DISPATCH_MFMA_BF16_1K_(bf16_t, bf16_t, fp32_t, 32, 32,  8, __builtin_amdgcn_mfma_f32_32x32x8bf16_1k)
        else if constexpr DISPATCH_MFMA_BF16_1K_(bf16_t, bf16_t, fp32_t, 16, 16, 16, __builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
        else if constexpr DISPATCH_MFMA_8BIT_(fp8_t , fp8_t , fp32_t, 32, 32, 16, __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8)
        else if constexpr DISPATCH_MFMA_8BIT_(fp8_t , fp8_t , fp32_t, 16, 16, 32, __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
        else if constexpr DISPATCH_MFMA_8BIT_(bf8_t , bf8_t , fp32_t, 32, 32, 16, __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8)
        else if constexpr DISPATCH_MFMA_8BIT_(bf8_t , bf8_t , fp32_t, 16, 16, 32, __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8)
        else if constexpr DISPATCH_MFMA_F32_(fp32_t, fp32_t, fp32_t, 32, 32,  2, __builtin_amdgcn_mfma_f32_32x32x2f32)
        else if constexpr DISPATCH_MFMA_F32_(fp32_t, fp32_t, fp32_t, 16, 16,  4, __builtin_amdgcn_mfma_f32_16x16x4f32)
#endif
#if defined(__gfx942__) || defined(__gfx9_4_generic__)
        else if constexpr DISPATCH_MFMA_STEP_K_(fp16_t, fp16_t, fp32_t, 32, 32, 16,  8, __builtin_amdgcn_mfma_f32_32x32x8f16)
        else if constexpr DISPATCH_MFMA_STEP_K_(fp16_t, fp16_t, fp32_t, 16, 16, 32, 16, __builtin_amdgcn_mfma_f32_16x16x16f16)
        else if constexpr DISPATCH_MFMA_STEP_K_BF16_1K_(bf16_t, bf16_t, fp32_t, 32, 32, 16,  8, __builtin_amdgcn_mfma_f32_32x32x8bf16_1k)
        else if constexpr DISPATCH_MFMA_STEP_K_BF16_1K_(bf16_t, bf16_t, fp32_t, 16, 16, 32, 16, __builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
#endif
#if defined(__gfx950__)
        else if constexpr DISPATCH_MFMA_(fp16_t, fp16_t, fp32_t, 32, 32, 16, __builtin_amdgcn_mfma_f32_32x32x16_f16)
        else if constexpr DISPATCH_MFMA_(fp16_t, fp16_t, fp32_t, 16, 16, 32, __builtin_amdgcn_mfma_f32_16x16x32_f16)
        else if constexpr DISPATCH_MFMA_(bf16_t, bf16_t, fp32_t, 32, 32, 16, __builtin_amdgcn_mfma_f32_32x32x16_bf16)
        else if constexpr DISPATCH_MFMA_(bf16_t, bf16_t, fp32_t, 16, 16, 32, __builtin_amdgcn_mfma_f32_16x16x32_bf16)
#endif
        __builtin_unreachable();    // supprize warning for return type deduction
    }

    template<typename VA, typename VB, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        vtype_c c{0}; return operator()(a, b, c, number<cbsz>{}, number<abid>{}, number<blgp>{});
    }

    // Scaled MFMA dispatch (gfx950: f8f6f4 with E8M0 block exponent scaling)
    // scale_a, scale_b are runtime E8M0 exponent values; 127 = no scaling (2^0 = 1.0).
    // scale_op_sel_a/b: compile-time 2-bit byte-selectors picking which byte of the packed-int32 scale word feeds this MFMA; 0 = low byte = legacy scalar behavior.
    template<typename VA, typename VB, typename VC, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) -> vtype_c {
        (void)a; (void)b; (void)c; (void)scale_a; (void)scale_b;  // used by DISPATCH_MFMA_SCALE_; suppress -Wunused-parameter on host
        if      constexpr (false) {}
#if defined(__gfx950__)
        else if constexpr DISPATCH_MFMA_SCALE_(fp8_t, fp8_t, fp32_t, 32, 32,  64, __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4)
        else if constexpr DISPATCH_MFMA_SCALE_(fp8_t, fp8_t, fp32_t, 16, 16, 128, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
        else if constexpr DISPATCH_MFMA_SCALE_(fp4_t, fp4_t, fp32_t, 32, 32,  64, __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4)
        else if constexpr DISPATCH_MFMA_SCALE_(fp4_t, fp4_t, fp32_t, 16, 16, 128, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
#endif
        __builtin_unreachable();
    }

    template<typename VA, typename VB, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        vtype_c c{0}; return operator()(a, b, c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{});
    }
};
#undef DISPATCH_MFMA_
#undef DISPATCH_MFMA_F32_
#undef DISPATCH_MFMA_STEP_K_
#undef DISPATCH_MFMA_BF16_1K_
#undef DISPATCH_MFMA_STEP_K_BF16_1K_
#undef DISPATCH_MFMA_8BIT_
#undef DISPATCH_MFMA_SCALE_

using mfma_f32_32x32x2_f32      = mfma<fp32_t, fp32_t, fp32_t, 32, 32,  2>;
using mfma_f32_16x16x4_f32      = mfma<fp32_t, fp32_t, fp32_t, 16, 16,  4>;
using mfma_f32_32x32x8_f16      = mfma<fp16_t, fp16_t, fp32_t, 32, 32,  8>;
using mfma_f32_16x16x16_f16     = mfma<fp16_t, fp16_t, fp32_t, 16, 16, 16>;
using mfma_f32_32x32x8_bf16     = mfma<bf16_t, bf16_t, fp32_t, 32, 32,  8>;
using mfma_f32_16x16x16_bf16    = mfma<bf16_t, bf16_t, fp32_t, 16, 16, 16>;
using mfma_f32_32x32x16_f16     = mfma<fp16_t, fp16_t, fp32_t, 32, 32, 16>;
using mfma_f32_16x16x32_f16     = mfma<fp16_t, fp16_t, fp32_t, 16, 16, 32>;
using mfma_f32_32x32x16_bf16    = mfma<bf16_t, bf16_t, fp32_t, 32, 32, 16>;
using mfma_f32_16x16x32_bf16    = mfma<bf16_t, bf16_t, fp32_t, 16, 16, 32>;
using mfma_f32_32x32x16_fp8_fp8 = mfma<fp8_t , fp8_t , fp32_t, 32, 32, 16>;
using mfma_f32_16x16x32_fp8_fp8 = mfma<fp8_t , fp8_t , fp32_t, 16, 16, 32>;
using mfma_f32_32x32x16_bf8_bf8 = mfma<bf8_t , bf8_t , fp32_t, 32, 32, 16>;
using mfma_f32_16x16x32_bf8_bf8 = mfma<bf8_t , bf8_t , fp32_t, 16, 16, 32>;
// Scaled MFMA type aliases (gfx950 only, unified into struct mfma)
using mfma_f32_32x32x64_fp8_fp8   = mfma<fp8_t, fp8_t, fp32_t, 32, 32,  64>;
using mfma_f32_16x16x128_fp8_fp8  = mfma<fp8_t, fp8_t, fp32_t, 16, 16, 128>;
using mfma_f32_32x32x64_fp4_fp4   = mfma<fp4_t, fp4_t, fp32_t, 32, 32,  64>;
using mfma_f32_16x16x128_fp4_fp4  = mfma<fp4_t, fp4_t, fp32_t, 16, 16, 128>;
// Backward-compatible aliases (deprecated: prefer mfma_f32_* above)
using mfma_scale_f32_32x32x64_fp8_fp8   = mfma_f32_32x32x64_fp8_fp8;
using mfma_scale_f32_16x16x128_fp8_fp8  = mfma_f32_16x16x128_fp8_fp8;
using mfma_scale_f32_32x32x64_fp4_fp4   = mfma_f32_32x32x64_fp4_fp4;
using mfma_scale_f32_16x16x128_fp4_fp4  = mfma_f32_16x16x128_fp4_fp4;
#endif // __GFX9__ (mfma)

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// wmma (RDNA4 / wave32) -- gfx1250 uses wmma-256b builtins (16x16x{4,32,64,128}); gfx1200/gfx1201 (Navi 44/48) use wmma-128b _w32_gfx12 builtins (16x16x16). Dispatch macros for the two arg-list shapes differ -- gfx12 set is DISPATCH_WMMA_GFX12_*.
#if defined(__gfx1250__) || defined(__gfx1201__) || defined(__gfx1200__) || !defined(__HIP_DEVICE_COMPILE__)
// f16/bf16/f32 builtins: (neg_a, A, neg_b, B, matrix_fmts, C, clamp, neg_c)
#define DISPATCH_WMMA_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && \
  wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    return inst_(false, a, false, b, static_cast<short>(0), c, false, false); }
// bf16f32 special: accumulator is f32 but output is bf16 => (neg_a, A, neg_b, B, fmts, C_f32, clamp, neg_c)
// The builtin takes f32 accumulator and returns bf16 output; we store the f32 accum but return bf16.
#define DISPATCH_WMMA_BF16F32_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && \
  wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    return inst_(false, a, false, b, static_cast<short>(0), c, false, false); }
// fp8/bf8 builtins: (A, B, matrix_fmts, C, clamp, neg_c)  -- no neg_a/neg_b
// A/B are packed as _ExtVector<N, int>; bitcast from the fp8/bf8 vector
#define DISPATCH_WMMA_8BIT_(ta_, tb_, tc_, wm_, wn_, wk_, inst_) \
 (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && \
  wave_m == wm_ && wave_n == wn_ && wave_k == wk_) { \
    constexpr index_t i32_a = elem_a * static_cast<index_t>(sizeof(dtype_a)) / static_cast<index_t>(sizeof(i32_t)); \
    constexpr index_t i32_b = elem_b * static_cast<index_t>(sizeof(dtype_b)) / static_cast<index_t>(sizeof(i32_t)); \
    return inst_(__builtin_bit_cast(vector_t<i32_t, i32_a>, a), \
                 __builtin_bit_cast(vector_t<i32_t, i32_b>, b), \
                 static_cast<short>(0), c, false, false); }

// gfx12 builtins: 3-arg (A, B, C) -- no matrix_fmts/neg_c. ws_ selects _w32/_w64 (same {wm,wn,wk} triple).
#define DISPATCH_WMMA_GFX12_MATCH_(ta_, tb_, tc_, wm_, wn_, wk_, ws_) \
    (std::is_same_v<dtype_a, ta_> && std::is_same_v<dtype_b, tb_> && std::is_same_v<dtype_c, tc_> && wave_m == wm_ && wave_n == wn_ && wave_k == wk_ && warp_size == ws_)
#define DISPATCH_WMMA_GFX12_F32_(ta_, tb_, tc_, wm_, wn_, wk_, ws_, inst_) \
    DISPATCH_WMMA_GFX12_MATCH_(ta_, tb_, tc_, wm_, wn_, wk_, ws_) { return inst_(a, b, c); }
#define DISPATCH_WMMA_GFX12_8BIT_(ta_, tb_, tc_, wm_, wn_, wk_, ws_, inst_) /* fp8/bf8: A/B bitcast to i32 */ \
    DISPATCH_WMMA_GFX12_MATCH_(ta_, tb_, tc_, wm_, wn_, wk_, ws_) { \
    constexpr index_t i32_a = elem_a * static_cast<index_t>(sizeof(dtype_a)) / static_cast<index_t>(sizeof(i32_t)); \
    constexpr index_t i32_b = elem_b * static_cast<index_t>(sizeof(dtype_b)) / static_cast<index_t>(sizeof(i32_t)); \
    return inst_(__builtin_bit_cast(vector_t<i32_t, i32_a>, a), __builtin_bit_cast(vector_t<i32_t, i32_b>, b), c); }
#define DISPATCH_WMMA_GFX12_F32_STEP_K_(ta_, tb_, tc_, wm_, wn_, wk_, ws_, inst_k_, inst_) /* chain Nxinst for wider K */ \
    DISPATCH_WMMA_GFX12_MATCH_(ta_, tb_, tc_, wm_, wn_, wk_, ws_) { \
    constexpr index_t steps = wk_ / inst_k_, e_a = elem_a / steps, e_b = elem_b / steps; \
    auto tmp = inst_(slice(a, number<0>{}, number<e_a>{}), slice(b, number<0>{}, number<e_b>{}), c); \
    static_for<steps - 1>([&](auto i){ tmp = inst_(slice(a, number<e_a*(i+1)>{}, number<e_a*(i+2)>{}), slice(b, number<e_b*(i+1)>{}, number<e_b*(i+2)>{}), tmp); }); \
    return tmp; }

template<typename dtype_a_, typename dtype_b_, typename dtype_c_, index_t wave_m_, index_t wave_n_, index_t wave_k_, index_t warp_size_ = get_warp_size()>
struct wmma {
    using dtype_a = remove_cvref_t<dtype_a_>;
    using dtype_b = remove_cvref_t<dtype_b_>;
    using dtype_c = remove_cvref_t<dtype_c_>;
    static constexpr index_t wave_m = wave_m_;
    static constexpr index_t wave_n = wave_n_;
    static constexpr index_t wave_k = wave_k_;
    static constexpr index_t warp_size = warp_size_;  // 32 for gfx1250
    static constexpr index_t elem_a = wave_m * wave_k / warp_size;
    static constexpr index_t elem_b = wave_n * wave_k / warp_size;
    static constexpr index_t elem_c = wave_m * wave_n / warp_size;

    // For packed types (fp4), the hardware register packs multiple elements per byte.
    // elem counts logical elements; the register holds elem * bits_per_element / 8 bytes.
    // For non-packed types, sizeof(T) gives bytes per element directly.
    static constexpr index_t reg_bytes_a = is_packs_v<dtype_a> ? (elem_a * sizeof_bits<dtype_a>::value / 8) : (elem_a * static_cast<index_t>(sizeof(dtype_a)));
    static constexpr index_t reg_bytes_b = is_packs_v<dtype_b> ? (elem_b * sizeof_bits<dtype_b>::value / 8) : (elem_b * static_cast<index_t>(sizeof(dtype_b)));

    // vtype: for packed types, use i32 dword vector matching the hardware register size.
    // For non-packed types, use mfma_vtype_t (which gives ext_vector of the element type).
    using vtype_a = std::conditional_t<is_packs_v<dtype_a>, vector_t<i32_t, reg_bytes_a / static_cast<index_t>(sizeof(i32_t))>, mfma_vtype_t<dtype_a, elem_a>>;
    using vtype_b = std::conditional_t<is_packs_v<dtype_b>, vector_t<i32_t, reg_bytes_b / static_cast<index_t>(sizeof(i32_t))>, mfma_vtype_t<dtype_b, elem_b>>;
    using vtype_c = vector_t<dtype_c, elem_c>;

    // Format code for scaled WMMA (f8f6f4); -1 for types that don't support scaling
    static constexpr int fmt_a = std::is_same_v<dtype_a, fp8_t> ? 0 : std::is_same_v<dtype_a, bf8_t> ? 1 : std::is_same_v<dtype_a, fp4_t> ? 4 : -1;
    static constexpr int fmt_b = std::is_same_v<dtype_b, fp8_t> ? 0 : std::is_same_v<dtype_b, bf8_t> ? 1 : std::is_same_v<dtype_b, fp4_t> ? 4 : -1;

    // Regular (non-scaled) dispatch
    template<typename VA, typename VB, typename VC>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c) -> vtype_c {
        (void)a; (void)b; (void)c;
        if      constexpr (false) {}
#if defined(__gfx1250__)
        // f16/bf16 16x16x32
        else if constexpr DISPATCH_WMMA_(fp16_t, fp16_t, fp32_t, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x32_f16)
        else if constexpr DISPATCH_WMMA_(fp16_t, fp16_t, fp16_t, 16, 16, 32, __builtin_amdgcn_wmma_f16_16x16x32_f16)
        else if constexpr DISPATCH_WMMA_(bf16_t, bf16_t, fp32_t, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x32_bf16)
        else if constexpr DISPATCH_WMMA_(bf16_t, bf16_t, bf16_t, 16, 16, 32, __builtin_amdgcn_wmma_bf16_16x16x32_bf16)
        // f32 16x16x4
        else if constexpr DISPATCH_WMMA_(fp32_t, fp32_t, fp32_t, 16, 16,  4, __builtin_amdgcn_wmma_f32_16x16x4_f32)
        // fp8/bf8 16x16x64 -> f32
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, fp8_t, fp32_t, 16, 16, 64, __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, bf8_t, fp32_t, 16, 16, 64, __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, fp8_t, fp32_t, 16, 16, 64, __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, bf8_t, fp32_t, 16, 16, 64, __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8)
        // fp8/bf8 16x16x64 -> f16
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, fp8_t, fp16_t, 16, 16, 64, __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, bf8_t, fp16_t, 16, 16, 64, __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, fp8_t, fp16_t, 16, 16, 64, __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, bf8_t, fp16_t, 16, 16, 64, __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8)
        // fp8/bf8 16x16x128 -> f32
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, fp8_t, fp32_t, 16, 16, 128, __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, bf8_t, fp32_t, 16, 16, 128, __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, fp8_t, fp32_t, 16, 16, 128, __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, bf8_t, fp32_t, 16, 16, 128, __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8)
        // fp8/bf8 16x16x128 -> f16
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, fp8_t, fp16_t, 16, 16, 128, __builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(fp8_t, bf8_t, fp16_t, 16, 16, 128, __builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, fp8_t, fp16_t, 16, 16, 128, __builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8)
        else if constexpr DISPATCH_WMMA_8BIT_(bf8_t, bf8_t, fp16_t, 16, 16, 128, __builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8)
#endif
#if defined(__gfx1201__) || defined(__gfx1200__)
        // _w32/_w64 builtins gated by wavefrontsize target feature; OPUS_GFX120X_IS_WAVE32 selects the active wave mode.
#  if OPUS_GFX120X_IS_WAVE32
        // gfx12 wave32 16x16x16: f16/bf16/fp8/bf8 -> f32 + same-type acc. iu8/iu4 deferred.
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (fp16_t, fp16_t, fp32_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (bf16_t, bf16_t, fp32_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (fp16_t, fp16_t, fp16_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (bf16_t, bf16_t, bf16_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_8BIT_(fp8_t , fp8_t , fp32_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_8BIT_(fp8_t , bf8_t , fp32_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_8BIT_(bf8_t , fp8_t , fp32_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_8BIT_(bf8_t , bf8_t , fp32_t , 16, 16, 16, 32, __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12)
        // gfx12 wave32 16x16x32 synthetic: 2x stacked 16x16x16 along K.
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(fp16_t, fp16_t, fp32_t , 16, 16, 32, 32, 16, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(bf16_t, bf16_t, fp32_t , 16, 16, 32, 32, 16, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(fp16_t, fp16_t, fp16_t , 16, 16, 32, 32, 16, __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(bf16_t, bf16_t, bf16_t , 16, 16, 32, 32, 16, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12)
#  endif // wave32 builtin available
#  if !OPUS_GFX120X_IS_WAVE32
        // gfx12 wave64 16x16x16: native _w64_gfx12 builtins (4 elem/lane). Requires -mwavefrontsize64.
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (fp16_t, fp16_t, fp32_t , 16, 16, 16, 64, __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (bf16_t, bf16_t, fp32_t , 16, 16, 16, 64, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (fp16_t, fp16_t, fp16_t , 16, 16, 16, 64, __builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_ (bf16_t, bf16_t, bf16_t , 16, 16, 16, 64, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12)
        // gfx12 wave64 16x16x32 synthetic: 2x stacked _w64 16x16x16 (8 elem/lane = 1 b128 load).
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(fp16_t, fp16_t, fp32_t , 16, 16, 32, 64, 16, __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(bf16_t, bf16_t, fp32_t , 16, 16, 32, 64, 16, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(fp16_t, fp16_t, fp16_t , 16, 16, 32, 64, 16, __builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12)
        else if constexpr DISPATCH_WMMA_GFX12_F32_STEP_K_(bf16_t, bf16_t, bf16_t , 16, 16, 32, 64, 16, __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12)
#  endif // wave64 builtin available
#endif // __gfx1201__ / __gfx1200__
        __builtin_unreachable();
    }

    template<typename VA, typename VB>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b) {
        vtype_c c{0}; return operator()(a, b, c);
    }

    // Scaled WMMA dispatch (gfx1250: f8f6f4 / f4 with E8M0 block-scale)
    // scale_a, scale_b are per-lane E8M0 exponent values; 127 = no scaling (2^0 = 1.0).
    // BX32: int -- 4 packed E8M0 bytes (byte 0 used with scale_sel=0, scale_fmt=0).
    // BX16: long -- 8 packed E8M0 bytes.
    // matrix_a_scale_sel controls OPSEL: 0=scale from lanes 0-15, 1=scale from lanes 16-31.

    // BX32 scaled dispatch
    template<typename VA, typename VB, typename VC, index_t a_scale_sel = 0, index_t b_scale_sel = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c,
                                     int scale_a, int scale_b,
                                     number<a_scale_sel> = {}, number<b_scale_sel> = {}) -> vtype_c {
        (void)a; (void)b; (void)c; (void)scale_a; (void)scale_b;
        if constexpr (false) {}
#if defined(__gfx1250__)
        // 16x16x128 f8f6f4 (fp8/fp4 via format code): builtin always takes i32x16
        else if constexpr (fmt_a >= 0 && fmt_b >= 0 && std::is_same_v<dtype_c, fp32_t> && wave_m == 16 && wave_n == 16 && wave_k == 128) {
            // For packed types (fp4), vtype may be smaller than i32x16; zero-pad via union.
            auto pad_to_i32x16 = [](const auto& v) {
                if constexpr (sizeof(v) == sizeof(i32x16_t)) return __builtin_bit_cast(i32x16_t, v);
                else { union { i32x16_t w; char z[sizeof(i32x16_t)]; } u{}; __builtin_memcpy(&u, &v, sizeof(v)); return u.w; }
            };
            return __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                fmt_a, pad_to_i32x16(a),
                fmt_b, pad_to_i32x16(b),
                static_cast<short>(0), c,
                a_scale_sel, 0, scale_a, b_scale_sel, 0, scale_b, false, false);
        }
        // 32x16x128 f4 (dedicated fp4 instruction): A=i32x16, B=i32x8
        else if constexpr (std::is_same_v<dtype_a, fp4_t> && std::is_same_v<dtype_b, fp4_t> && std::is_same_v<dtype_c, fp32_t> && wave_m == 32 && wave_n == 16 && wave_k == 128) {
            return __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(
                __builtin_bit_cast(i32x16_t, a),
                __builtin_bit_cast(i32x8_t, b),
                static_cast<short>(0), c,
                a_scale_sel, 0, scale_a, b_scale_sel, 0, scale_b, false, false);
        }
#endif
        __builtin_unreachable();
    }

    template<typename VA, typename VB, index_t a_scale_sel = 0, index_t b_scale_sel = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, int scale_a, int scale_b,
                                     number<a_scale_sel> = {}, number<b_scale_sel> = {}) {
        vtype_c c{0}; return operator()(a, b, c, scale_a, scale_b, number<a_scale_sel>{}, number<b_scale_sel>{});
    }

    // BX16 scaled dispatch (scale exponent is long = 64 bits = 8 packed E8M0 bytes)
    template<typename VA, typename VB, typename VC, index_t a_scale_sel = 0, index_t b_scale_sel = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c,
                                     long scale_a, long scale_b,
                                     number<a_scale_sel> = {}, number<b_scale_sel> = {}) -> vtype_c {
        (void)a; (void)b; (void)c; (void)scale_a; (void)scale_b;
        if constexpr (false) {}
#if defined(__gfx1250__)
        // 16x16x128 f8f6f4 BX16
        else if constexpr (fmt_a >= 0 && fmt_b >= 0 && std::is_same_v<dtype_c, fp32_t> && wave_m == 16 && wave_n == 16 && wave_k == 128) {
            auto pad_to_i32x16 = [](const auto& v) {
                if constexpr (sizeof(v) == sizeof(i32x16_t)) return __builtin_bit_cast(i32x16_t, v);
                else { union { i32x16_t w; char z[sizeof(i32x16_t)]; } u{}; __builtin_memcpy(&u, &v, sizeof(v)); return u.w; }
            };
            return __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                fmt_a, pad_to_i32x16(a),
                fmt_b, pad_to_i32x16(b),
                static_cast<short>(0), c,
                a_scale_sel, 0, scale_a, b_scale_sel, 0, scale_b, false, false);
        }
        // 32x16x128 f4 BX16
        else if constexpr (std::is_same_v<dtype_a, fp4_t> && std::is_same_v<dtype_b, fp4_t> && std::is_same_v<dtype_c, fp32_t> && wave_m == 32 && wave_n == 16 && wave_k == 128) {
            return __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(
                __builtin_bit_cast(i32x16_t, a),
                __builtin_bit_cast(i32x8_t, b),
                static_cast<short>(0), c,
                a_scale_sel, 0, scale_a, b_scale_sel, 0, scale_b, false, false);
        }
#endif
        __builtin_unreachable();
    }

    template<typename VA, typename VB, index_t a_scale_sel = 0, index_t b_scale_sel = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, long scale_a, long scale_b,
                                     number<a_scale_sel> = {}, number<b_scale_sel> = {}) {
        vtype_c c{0}; return operator()(a, b, c, scale_a, scale_b, number<a_scale_sel>{}, number<b_scale_sel>{});
    }
};
#undef DISPATCH_WMMA_
#undef DISPATCH_WMMA_BF16F32_
#undef DISPATCH_WMMA_8BIT_
#undef DISPATCH_WMMA_GFX12_MATCH_
#undef DISPATCH_WMMA_GFX12_F32_
#undef DISPATCH_WMMA_GFX12_8BIT_
#undef DISPATCH_WMMA_GFX12_F32_STEP_K_

// gfx12 wave32 16x16x16 aliases (default warp_size). wave64 + synthetic 16x16x32 aliases below.
using wmma_f32_16x16x16_f16     = wmma<fp16_t, fp16_t, fp32_t, 16, 16, 16>;
using wmma_f16_16x16x16_f16     = wmma<fp16_t, fp16_t, fp16_t, 16, 16, 16>;
using wmma_f32_16x16x16_bf16    = wmma<bf16_t, bf16_t, fp32_t, 16, 16, 16>;
using wmma_bf16_16x16x16_bf16   = wmma<bf16_t, bf16_t, bf16_t, 16, 16, 16>;
using wmma_f32_16x16x16_fp8_fp8 = wmma<fp8_t , fp8_t , fp32_t, 16, 16, 16>;
using wmma_f32_16x16x16_fp8_bf8 = wmma<fp8_t , bf8_t , fp32_t, 16, 16, 16>;
using wmma_f32_16x16x16_bf8_fp8 = wmma<bf8_t , fp8_t , fp32_t, 16, 16, 16>;
using wmma_f32_16x16x16_bf8_bf8 = wmma<bf8_t , bf8_t , fp32_t, 16, 16, 16>;
// gfx12 wave32 16x16x32 synthetic (2x 16x16x16).
using wmma_f32_16x16x32_f16_w32   = wmma<fp16_t, fp16_t, fp32_t, 16, 16, 32, 32>;
using wmma_f32_16x16x32_bf16_w32  = wmma<bf16_t, bf16_t, fp32_t, 16, 16, 32, 32>;
using wmma_f16_16x16x32_f16_w32   = wmma<fp16_t, fp16_t, fp16_t, 16, 16, 32, 32>;
using wmma_bf16_16x16x32_bf16_w32 = wmma<bf16_t, bf16_t, bf16_t, 16, 16, 32, 32>;
// gfx12 wave64 16x16x16 (-mwavefrontsize64).
using wmma_f32_16x16x16_f16_w64   = wmma<fp16_t, fp16_t, fp32_t, 16, 16, 16, 64>;
using wmma_f32_16x16x16_bf16_w64  = wmma<bf16_t, bf16_t, fp32_t, 16, 16, 16, 64>;
using wmma_f16_16x16x16_f16_w64   = wmma<fp16_t, fp16_t, fp16_t, 16, 16, 16, 64>;
using wmma_bf16_16x16x16_bf16_w64 = wmma<bf16_t, bf16_t, bf16_t, 16, 16, 16, 64>;
// gfx12 wave64 16x16x32 synthetic (2x 16x16x16_w64).
using wmma_f32_16x16x32_f16_w64   = wmma<fp16_t, fp16_t, fp32_t, 16, 16, 32, 64>;
using wmma_f32_16x16x32_bf16_w64  = wmma<bf16_t, bf16_t, fp32_t, 16, 16, 32, 64>;
using wmma_f16_16x16x32_f16_w64   = wmma<fp16_t, fp16_t, fp16_t, 16, 16, 32, 64>;
using wmma_bf16_16x16x32_bf16_w64 = wmma<bf16_t, bf16_t, bf16_t, 16, 16, 32, 64>;

// f16/bf16 16x16x32
using wmma_f32_16x16x32_f16   = wmma<fp16_t, fp16_t, fp32_t, 16, 16, 32>;
using wmma_f16_16x16x32_f16   = wmma<fp16_t, fp16_t, fp16_t, 16, 16, 32>;
using wmma_f32_16x16x32_bf16  = wmma<bf16_t, bf16_t, fp32_t, 16, 16, 32>;
using wmma_bf16_16x16x32_bf16 = wmma<bf16_t, bf16_t, bf16_t, 16, 16, 32>;
// f32 16x16x4
using wmma_f32_16x16x4_f32    = wmma<fp32_t, fp32_t, fp32_t, 16, 16,  4>;
// fp8/bf8 16x16x64
using wmma_f32_16x16x64_fp8_fp8  = wmma<fp8_t, fp8_t, fp32_t, 16, 16, 64>;
using wmma_f32_16x16x64_fp8_bf8  = wmma<fp8_t, bf8_t, fp32_t, 16, 16, 64>;
using wmma_f32_16x16x64_bf8_fp8  = wmma<bf8_t, fp8_t, fp32_t, 16, 16, 64>;
using wmma_f32_16x16x64_bf8_bf8  = wmma<bf8_t, bf8_t, fp32_t, 16, 16, 64>;
using wmma_f16_16x16x64_fp8_fp8  = wmma<fp8_t, fp8_t, fp16_t, 16, 16, 64>;
using wmma_f16_16x16x64_fp8_bf8  = wmma<fp8_t, bf8_t, fp16_t, 16, 16, 64>;
using wmma_f16_16x16x64_bf8_fp8  = wmma<bf8_t, fp8_t, fp16_t, 16, 16, 64>;
using wmma_f16_16x16x64_bf8_bf8  = wmma<bf8_t, bf8_t, fp16_t, 16, 16, 64>;
// fp8/bf8 16x16x128
using wmma_f32_16x16x128_fp8_fp8 = wmma<fp8_t, fp8_t, fp32_t, 16, 16, 128>;
using wmma_f32_16x16x128_fp8_bf8 = wmma<fp8_t, bf8_t, fp32_t, 16, 16, 128>;
using wmma_f32_16x16x128_bf8_fp8 = wmma<bf8_t, fp8_t, fp32_t, 16, 16, 128>;
using wmma_f32_16x16x128_bf8_bf8 = wmma<bf8_t, bf8_t, fp32_t, 16, 16, 128>;
using wmma_f16_16x16x128_fp8_fp8 = wmma<fp8_t, fp8_t, fp16_t, 16, 16, 128>;
using wmma_f16_16x16x128_fp8_bf8 = wmma<fp8_t, bf8_t, fp16_t, 16, 16, 128>;
using wmma_f16_16x16x128_bf8_fp8 = wmma<bf8_t, fp8_t, fp16_t, 16, 16, 128>;
using wmma_f16_16x16x128_bf8_bf8 = wmma<bf8_t, bf8_t, fp16_t, 16, 16, 128>;
// Scaled WMMA (f8f6f4 unified instruction, supports fp8/bf8/fp4 via format code)
using wmma_scale_f32_16x16x128_fp8_fp8 = wmma<fp8_t, fp8_t, fp32_t, 16, 16, 128>;
using wmma_scale_f32_16x16x128_fp4_fp4 = wmma<fp4_t, fp4_t, fp32_t, 16, 16, 128>;
// Scaled WMMA (dedicated fp4 32x16x128 instruction)
using wmma_scale_f32_32x16x128_fp4_fp4 = wmma<fp4_t, fp4_t, fp32_t, 32, 16, 128>;
#endif // __gfx1250__ / __gfx1201__ / __gfx1200__ (wmma)

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// adaptor
struct p_dim {};
struct y_dim {};

namespace impl{ // utlity function to play with shape
template<typename FDim, typename Target> static constexpr auto pickup_filter(seq<>) { return seq<>{}; }
template<typename FDim, typename Target, index_t I0, index_t... Rest> static constexpr auto pickup_filter(seq<I0, Rest...>) {
    if constexpr (std::is_same_v<remove_cvref_t<decltype(get<I0>(FDim{}))>, remove_cvref_t<Target>>) return concat_seq(seq<I0>{}, pickup_filter<FDim, Target>(seq<Rest...>{}));
    else return pickup_filter<FDim, Target>(seq<Rest...>{}); }
template<typename Shape, index_t... Fs> OPUS_D static constexpr auto pickup_shape_apply(seq<Fs...>) { return opus::make_tuple(get<Fs>(Shape{})...); }
template<typename Shape, typename FDim, typename Target, index_t... Is>
OPUS_D static constexpr auto pickup_shape_impl(const Shape&, const FDim&, Target, seq<Is...>) { static_assert(size<Shape>() == size<FDim>()); return pickup_shape_apply<Shape>(pickup_filter<FDim, Target>(seq<Is...>{})); }

template<typename Dim, index_t... Js>
OPUS_D constexpr index_t dim_group_size_sum(seq<Js...>) { return (static_cast<index_t>(get<Js>(Dim{}).size()) + ... + 0); }

template<typename Shape, typename Dim, index_t DIdx, index_t... Ss>
OPUS_D constexpr auto unflatten_shape_group(seq<Ss...>) {
    constexpr index_t SStart = dim_group_size_sum<Dim>(make_index_seq<DIdx>{});
    return opus::make_tuple(get<SStart + Ss>(Shape{})...);
}

template<typename Shape, typename Dim, index_t... DIs>
OPUS_D constexpr auto unflatten_shape_impl(seq<DIs...>) { return opus::make_tuple(unflatten_shape_group<Shape, Dim, DIs>(make_index_seq<get<DIs>(Dim{}).size()>{})...); }

template<typename Dim, index_t... Js>
OPUS_D constexpr index_t p_count_in(seq<Js...>) { return ((std::is_same_v<remove_cvref_t<decltype(get<Js>(Dim{}))>, p_dim> ? 1 : 0) + ... + 0); }

template<typename Dim, typename Coord, index_t... Is>
OPUS_D constexpr auto unfold_p_coord_impl(const Coord& coord, seq<Is...>) {
    return opus::make_tuple( [&]() -> decltype(auto) {
            if constexpr (std::is_same_v<remove_cvref_t<decltype(get<Is>(Dim{}))>, p_dim>) return get< p_count_in<Dim>(make_index_seq<Is>{}) >(coord);
            else                                                                           return underscore{};
        }()...
    );
}

template<typename Dim, index_t... Js>
OPUS_D constexpr index_t dim_offset_sum(seq<Js...>) { return (static_cast<index_t>(size<decltype(get<Js>(Dim{}))>()) + ... + 0); }

template<typename Dim, typename Shape, typename Stride, index_t I>
OPUS_D constexpr auto unfold_x_stride_each(const Stride& stride) {
    constexpr index_t C = dim_offset_sum<Dim>(make_index_seq<I>{});
    constexpr index_t len = size<decltype(get<I>(Dim{}))>();
    constexpr auto current_shape = slice(Shape{}, number<C>{}, number<C + len>{});
    constexpr auto current_stride = packed_shape_to_stride(current_shape);
    return transform_tuple([&](auto i_elem){ return i_elem * get<I>(stride); }, current_stride);
}

template<typename Dim, index_t J, index_t... Gs> constexpr index_t unfold_find_group(seq<Gs...>) {
    index_t acc = 0, r = 0; ((void)(acc += size<decltype(get<Gs>(Dim{}))>(), (acc <= J ? (void)(r = Gs + 1) : (void)0)), ...); return r; }
template<typename Dim, typename Shape, typename Stride, index_t J> OPUS_D constexpr auto unfold_x_stride_at(const Stride& stride) {
    constexpr index_t G = unfold_find_group<Dim, J>(make_index_seq<size<Dim>()>{}); constexpr index_t group_end = dim_offset_sum<Dim>(make_index_seq<G + 1>{});
    return packed_stride_at<Shape, J>(make_index_seq<group_end - J - 1>{}) * get<G>(stride); }
template<typename Dim, typename Shape, typename Stride, index_t... Js> OPUS_D constexpr auto unfold_x_stride_flat(const Stride& stride, seq<Js...>) { return opus::make_tuple(unfold_x_stride_at<Dim, Shape, Stride, Js>(stride)...); }
template<typename Dim, typename Shape, typename Stride, index_t... Is> OPUS_D constexpr auto unfold_x_stride_impl(const Stride& stride, seq<Is...>) { return unfold_x_stride_flat<Dim, Shape, Stride>(stride, make_index_seq<size<Shape>()>{}); }
}

template<typename Shape, typename Dim, typename Target>
OPUS_D static constexpr auto pickup_shape(const Shape&, const Dim&, Target) { return pickup_shape_impl(Shape{}, flatten_tuple(Dim{}), Target{}, make_index_seq<size<Shape>()>{}); }

// Shape : tuple<N0, N1, N2, N3, N4, N5>
// Dim   : tuple<tuple<*, *>, tuple<*, *, *>, tuple<*>>
// =>    : tuple<tuple<N0, N1>, tuple<N2, N3, N4>, tuple<N5>>
template<typename Shape, typename Dim, index_t... Ds /* index for Dim not Shape */>
OPUS_D constexpr auto unflatten_shape(const Shape&, const Dim&) {
    return impl::unflatten_shape_impl<Shape, Dim>(make_index_seq<size<Dim>()>{});
}

// coord: tuple<a, b>, dim: tuple<tuple<p_dim, y_dim>, tuple<y_dim, p_dim, y_dim>> -> tuple <a, _, _, b, _>
template<typename Dim, typename Coord>
OPUS_D constexpr auto unfold_p_coord(const Dim&, const Coord& coord) {
    constexpr auto flatten_dim = flatten_tuple(Dim{});
    using FDim = remove_cvref_t<decltype(flatten_dim)>;
    static_assert(tuple_count<opus::p_dim>(flatten_dim) == size<Coord>(), "input coord must be same size as p_dim inside Dim");
    return impl::unfold_p_coord_impl<FDim, Coord>(coord, make_index_seq<size<FDim>()>{});
}

template<typename Dim, typename Shape, typename Stride>
OPUS_D constexpr auto unfold_x_stride(const Dim&, const Shape&, const Stride& stride) {
    constexpr auto flatten_dim = flatten_tuple(Dim{});
    static_assert(size<Dim>() == size<Stride>(), "input stride must be same size as x_dim");
    static_assert(size<Shape>() == size<remove_cvref_t<decltype(flatten_dim)>>(), "input shape must be same size as flattened dim");
    return impl::unfold_x_stride_impl<Dim, Shape, Stride>(stride, make_index_seq<size<Dim>()>{});
}

#define OPUS_KP_(x_) static_assert(opus::tuple_count<opus::p_dim>(opus::flatten_tuple(x_ ())) == size<C>())
// Per-axis layout API: generates y_shape_X, p_shape_X, layout_X (3 overloads), layout_X_packed, y_layout_X
#define OPUS_ADAPTOR_LAYOUT_API_DEFINE_FOR(X)                                                                                                                       \
    OPUS_D static constexpr auto y_shape_##X() { return y_shape(shape_##X(), dim_##X()); }                                                                          \
    OPUS_D static constexpr auto p_shape_##X() { return p_shape(shape_##X(), dim_##X()); }                                                                          \
    template<index_t cached_vec = 0> OPUS_D constexpr auto layout_##X() { return make_layout<cached_vec>(shape_##X());}                                             \
    template<index_t cached_vec = 0, typename S> OPUS_D constexpr auto layout_##X(S&& stride) { return make_layout<cached_vec>(shape_##X(), unfold_x_stride(dim_##X(), shape_##X(), stride));} \
    template<index_t cached_vec = 0, typename S, typename C> OPUS_D constexpr auto layout_##X(S&& stride, C&& z) { OPUS_KP_(dim_##X); return make_layout<cached_vec>(shape_##X(), unfold_x_stride(dim_##X(), shape_##X(), stride), opus::unfold_p_coord(dim_##X(), z));}  \
    template<index_t cached_vec = 0, typename C> OPUS_D constexpr auto layout_##X##_packed(C&& z) { OPUS_KP_(dim_##X); return make_layout_packed<cached_vec>(shape_##X(), opus::unfold_p_coord(dim_##X(), z));}   \
    template<index_t cached_vec = 0, typename... Ts, std::enable_if_t<(!is_tuple_v<Ts> && ...), bool> = true> OPUS_D constexpr auto layout_##X(Ts&&... strides) {return layout_##X<cached_vec>(opus::make_tuple(strides...)); }  \
    template<index_t cached_vec = 0> OPUS_D constexpr auto y_layout_##X() { return make_layout<cached_vec>(y_shape_##X());}

// any struct implement adaptor like feature must implement(or using from base) shape_a/b/c, dim_a/b/c
#define OPUS_ADAPTOR_LAYOUT_API_DEFINE                                                                                                                              \
    template<typename S, typename D> OPUS_D static constexpr auto y_shape(const S& /*shape*/, const D& /*dim*/) { return opus::pickup_shape(S{}, D{}, y_dim{}); }   \
    template<typename S, typename D> OPUS_D static constexpr auto p_shape(const S& /*shape*/, const D& /*dim*/) { return opus::pickup_shape(S{}, D{}, p_dim{}); }   \
    OPUS_ADAPTOR_LAYOUT_API_DEFINE_FOR(a)                                                                                                                           \
    OPUS_ADAPTOR_LAYOUT_API_DEFINE_FOR(b)                                                                                                                           \
    OPUS_ADAPTOR_LAYOUT_API_DEFINE_FOR(c)

// Note: any class to support adaptor need include OPUS_ADAPTOR_LAYOUT_API_DEFINE and implement shape_a()/shape_b()/shape_c()
// P indicates dim cross thread, Y indicates dim within thread, this is X layout (X=P+Y) view the tensor as a whole
// A:[(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)], MxK
// B:[(grpn_b<p>), (rept_b<y>, grpk_b<p>, pack_b<y>)], NxK
// C:[(rept_c<y>, grpm_c<p>, pack_c<y>), (grpn_c<p>)], MxN
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
namespace impl {
template<typename MFMA>
struct mfma_adaptor : public remove_cvref_t<MFMA> {
    using mfma_type = remove_cvref_t<MFMA>;

    static constexpr index_t grpm_a = mfma_type::wave_m;
    static constexpr index_t grpn_b = mfma_type::wave_n;
    static_assert(mfma_type::warp_size % grpm_a == 0 && mfma_type::warp_size % grpn_b == 0 && grpm_a == grpn_b);
    static constexpr index_t grpk_a = mfma_type::warp_size / grpm_a;
    static constexpr index_t grpk_b = grpk_a;
    static constexpr index_t grpn_c = mfma_type::wave_n;
    static constexpr index_t grpm_c = mfma_type::warp_size / grpn_c;

    static constexpr index_t max_pack_a = 16 / sizeof(typename mfma_type::dtype_a); // max 4 dwords
    static constexpr index_t max_pack_b = 16 / sizeof(typename mfma_type::dtype_b); // max 4 dwords
    static constexpr index_t max_pack_c = 16 / sizeof(typename mfma_type::dtype_c); // max 4 dwords

    // pack_* should be vector load from ds_read/global_read
    static constexpr index_t pack_a = (max_pack_a < mfma_type::elem_a ? max_pack_a : mfma_type::elem_a);
    static constexpr index_t pack_b = (max_pack_b < mfma_type::elem_b ? max_pack_b : mfma_type::elem_b);
    static constexpr index_t pack_c = (max_pack_c < mfma_type::elem_c ? max_pack_c : mfma_type::elem_c);

    static constexpr index_t rept_a = mfma_type::elem_a / pack_a;
    static constexpr index_t rept_b = mfma_type::elem_b / pack_b;
    static constexpr index_t rept_c = mfma_type::elem_c / pack_c;

    // by default, this is X shape, P + Y
    OPUS_D static constexpr auto shape_a() { return tuple<number<grpm_a>, number<rept_a>, number<grpk_a>, number<pack_a>>{}; }
    OPUS_D static constexpr auto shape_b() { return tuple<number<grpn_b>, number<rept_b>, number<grpk_b>, number<pack_b>>{}; }
    OPUS_D static constexpr auto shape_c() { return tuple<number<rept_c>, number<grpm_c>, number<pack_c>, number<grpn_c>>{}; }

    // here we describe above shape by group them into a 2d shape style, and with p/y dim. we could put into same structure, but let's make things easier
    OPUS_D static constexpr auto dim_a()   { return tuple< tuple<p_dim>,  tuple<y_dim, p_dim, y_dim> >{}; }    // dim encoding for A, MxK
    OPUS_D static constexpr auto dim_b()   { return tuple< tuple<p_dim>,  tuple<y_dim, p_dim, y_dim> >{}; }    // dim encoding for B, NxK
    OPUS_D static constexpr auto dim_c()   { return tuple< tuple<y_dim, p_dim, y_dim>,  tuple<p_dim> >{}; }    // dim encoding for C, MxN

    OPUS_ADAPTOR_LAYOUT_API_DEFINE
};

// A:[(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)], MxK
// B:[(grpn_b<p>), (rept_b<y>, grpk_b<p>, pack_b<y>)], NxK
// C:[(grpn_c<p>), (rept_c<y>, grpm_c<p>, pack_c<y>)], MxN transposed(!)
template<typename MFMA>
struct mfma_adaptor_swap_ab : mfma_adaptor<MFMA> {
    using base = mfma_adaptor<MFMA>;
    using base::shape_a; using base::shape_b; using base::dim_a; using base::dim_b; using base::y_shape; using base::p_shape;
    using base::y_shape_a; using base::y_shape_b; using base::p_shape_a; using base::p_shape_b;
    using base::layout_a; using base::layout_b; using base::layout_a_packed; using base::layout_b_packed; using base::y_layout_a; using base::y_layout_b;
    OPUS_D static constexpr auto shape_c() { return tuple<number<base::grpn_c>, number<base::rept_c>, number<base::grpm_c>, number<base::pack_c>>{}; }
    OPUS_D static constexpr auto dim_c()   { return tuple<tuple<p_dim>,  tuple<y_dim, p_dim, y_dim> >{}; }    // dim encoding for C, MxN
    // Only generate _c layout methods (shape_c/dim_c changed)
    OPUS_ADAPTOR_LAYOUT_API_DEFINE_FOR(c)

    template<typename VA, typename VB, typename VC, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        return base::operator()(b, a, c, number<cbsz>{}, number<abid>{}, number<blgp>{});
    }

    template<typename VA, typename VB, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        typename MFMA::vtype_c c{0}; return operator()(a, b, c, number<cbsz>{}, number<abid>{}, number<blgp>{});
    }

    template<typename VA, typename VB, typename VC, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        return base::operator()(b, a, c, scale_b, scale_a, number<scale_op_sel_b>{}, number<scale_op_sel_a>{});
    }

    template<typename VA, typename VB, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        typename MFMA::vtype_c c{0}; return operator()(a, b, c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{});
    }
};
}
// helper class to create adaptor instance for mfma, need be paired with make_mfma(). don't directly use it
struct mfma_adaptor         { template<typename M> OPUS_D decltype(auto) operator()(M&&) { return impl::mfma_adaptor<remove_cvref_t<M>>{};} };
struct mfma_adaptor_swap_ab { template<typename M> OPUS_D decltype(auto) operator()(M&&) { return impl::mfma_adaptor_swap_ab<remove_cvref_t<M>>{};} };

template<typename d_a, typename d_b, typename d_c, index_t w_m, index_t w_n, index_t w_k, typename A = mfma_adaptor, index_t warp_size_ = get_warp_size()>
OPUS_D decltype(auto) make_mfma(number<w_m>, number<w_n>, number<w_k>, A&& = {}, number<warp_size_> = {}) { return A{}(mfma<d_a, d_b, d_c, w_m, w_n, w_k, warp_size_>{}); }

template<typename d_a, typename d_b, typename d_c, typename WaveMNK /*seq<m, n, k>*/, typename A = mfma_adaptor, index_t warp_size_ = get_warp_size()>
OPUS_D decltype(auto) make_mfma(WaveMNK&&, A&& = {}, number<warp_size_> = {}) { return A{}(mfma<d_a, d_b, d_c, get<0>(WaveMNK{}), get<1>(WaveMNK{}), get<2>(WaveMNK{}), warp_size_>{}); }
#endif // __GFX9__

// wmma_adaptor: layout encoding for wave32 WMMA (gfx1250, gfx1200/gfx1201).
// A:[(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)], MxK
// B:[(grpn_b<p>), (rept_b<y>, grpk_b<p>, pack_b<y>)], NxK
// C:[(grpm_c<p>, rept_c<y>, pack_c<y>), (grpn_c<p>)], MxN
#if defined(__gfx1250__) || defined(__gfx1201__) || defined(__gfx1200__) || !defined(__HIP_DEVICE_COMPILE__)
namespace impl {
template<typename WMMA>
struct wmma_adaptor : public remove_cvref_t<WMMA> {
    using wmma_type = remove_cvref_t<WMMA>;

    static constexpr index_t grpm_a = wmma_type::wave_m;
    static constexpr index_t grpn_b = wmma_type::wave_n;
    static_assert(wmma_type::warp_size % grpm_a == 0 && wmma_type::warp_size % grpn_b == 0 && grpm_a == grpn_b);
    static constexpr index_t grpk_a = wmma_type::warp_size / grpm_a;
    static constexpr index_t grpk_b = grpk_a;
    static constexpr index_t grpn_c = wmma_type::wave_n;
    static constexpr index_t grpm_c = wmma_type::warp_size / grpn_c;

    static constexpr index_t max_pack_a = 16 / sizeof(typename wmma_type::dtype_a);
    static constexpr index_t max_pack_b = 16 / sizeof(typename wmma_type::dtype_b);
    static constexpr index_t max_pack_c = 16 / sizeof(typename wmma_type::dtype_c);

    static constexpr index_t pack_a = (max_pack_a < wmma_type::elem_a ? max_pack_a : wmma_type::elem_a);
    static constexpr index_t pack_b = (max_pack_b < wmma_type::elem_b ? max_pack_b : wmma_type::elem_b);
    static constexpr index_t pack_c = (max_pack_c < wmma_type::elem_c ? max_pack_c : wmma_type::elem_c);

    static constexpr index_t rept_a = wmma_type::elem_a / pack_a;
    static constexpr index_t rept_b = wmma_type::elem_b / pack_b;
    static constexpr index_t rept_c = wmma_type::elem_c / pack_c;

    OPUS_D static constexpr auto shape_a() { return tuple<number<grpm_a>, number<rept_a>, number<grpk_a>, number<pack_a>>{}; }
    OPUS_D static constexpr auto shape_b() { return tuple<number<grpn_b>, number<rept_b>, number<grpk_a>, number<pack_b>>{}; }
    OPUS_D static constexpr auto shape_c() { return tuple<number<grpm_c>, number<rept_c>, number<pack_c>, number<grpn_c>>{}; }

    OPUS_D static constexpr auto dim_a()   { return tuple< tuple<p_dim>,  tuple<y_dim, p_dim, y_dim> >{}; }
    OPUS_D static constexpr auto dim_b()   { return tuple< tuple<p_dim>,  tuple<y_dim, p_dim, y_dim> >{}; }
    OPUS_D static constexpr auto dim_c()   { return tuple< tuple<p_dim, y_dim, y_dim>,  tuple<p_dim> >{}; }

    OPUS_ADAPTOR_LAYOUT_API_DEFINE
};

template<typename WMMA>
struct wmma_adaptor_swap_ab : wmma_adaptor<WMMA> {
    using base = wmma_adaptor<WMMA>;
    using base::shape_a; using base::shape_b; using base::dim_a; using base::dim_b; using base::y_shape; using base::p_shape;
    using base::y_shape_a; using base::y_shape_b; using base::p_shape_a; using base::p_shape_b;
    using base::layout_a; using base::layout_b; using base::layout_a_packed; using base::layout_b_packed; using base::y_layout_a; using base::y_layout_b;
    OPUS_D static constexpr auto shape_c() { return tuple<number<base::grpn_c>, number<base::grpm_c>, number<base::rept_c>, number<base::pack_c>>{}; }
    OPUS_D static constexpr auto dim_c()   { return tuple<tuple<p_dim>,  tuple<p_dim, y_dim, y_dim> >{}; }
    // Only generate _c layout methods (shape_c/dim_c changed)
    OPUS_ADAPTOR_LAYOUT_API_DEFINE_FOR(c)

    template<typename VA, typename VB, typename VC>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c) {
        return base::operator()(b, a, c);
    }

    template<typename VA, typename VB>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b) {
        typename WMMA::vtype_c c{0}; return operator()(b, a, c);
    }

    // Scaled overloads (BX32 / BX16): swap a,b then forward to base
    template<typename VA, typename VB, typename VC>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, int scale_a, int scale_b) {
        return base::operator()(b, a, c, scale_a, scale_b);
    }

    template<typename VA, typename VB, typename VC>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, long scale_a, long scale_b) {
        return base::operator()(b, a, c, scale_a, scale_b);
    }
};
} // namespace impl (wmma_adaptor)

struct wmma_adaptor         { template<typename M> OPUS_D decltype(auto) operator()(M&&) { return impl::wmma_adaptor<remove_cvref_t<M>>{};} };
struct wmma_adaptor_swap_ab { template<typename M> OPUS_D decltype(auto) operator()(M&&) { return impl::wmma_adaptor_swap_ab<remove_cvref_t<M>>{};} };

template<typename d_a, typename d_b, typename d_c, index_t w_m, index_t w_n, index_t w_k, typename A = wmma_adaptor, index_t warp_size_ = get_warp_size()>
OPUS_D decltype(auto) make_wmma(number<w_m>, number<w_n>, number<w_k>, A&& = {}, number<warp_size_> = {}) { return A{}(wmma<d_a, d_b, d_c, w_m, w_n, w_k, warp_size_>{}); }

template<typename d_a, typename d_b, typename d_c, typename WaveMNK, typename A = wmma_adaptor, index_t warp_size_ = get_warp_size()>
OPUS_D decltype(auto) make_wmma(WaveMNK&&, A&& = {}, number<warp_size_> = {}) { return A{}(wmma<d_a, d_b, d_c, get<0>(WaveMNK{}), get<1>(WaveMNK{}), get<2>(WaveMNK{}), warp_size_>{}); }
#endif // __gfx1250__ / __gfx1201__ / __gfx1200__ (wmma_adaptor)

/////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace impl {
// tiled mma, warp level mfma/wmma/... EXPAND_: each wave need repeat along m/n/k dim how many times. TILE_: number of waves in m/n/k dim
// A:[(expd_m<y>, tile_m<p>), (expd_k<y>, tile_k<p>)]
// B:[(expd_n<y>, tile_n<p>), (expd_k<y>, tile_k<p>)]
// C:[(expd_m<y>, tile_m<p>), (expd_n<y>, tile_n<p>)]
template <typename MMA_, index_t EXPAND_M, index_t EXPAND_N, index_t EXPAND_K, index_t TILE_M, index_t TILE_N, index_t TILE_K>
struct tiled_mma_adaptor : public MMA_ {
    using MMA = remove_cvref_t<MMA_>;
    static constexpr index_t expd_m = EXPAND_M;
    static constexpr index_t expd_n = EXPAND_N;
    static constexpr index_t expd_k = EXPAND_K;
    static constexpr index_t tile_m = TILE_M;
    static constexpr index_t tile_n = TILE_N;
    static constexpr index_t tile_k = TILE_K;
#if OPUS_TILE_CONTAINER == 0
    using vtype_a = vector_t<typename MMA::dtype_a, expd_m * expd_k * MMA::elem_a>;
    using vtype_b = vector_t<typename MMA::dtype_b, expd_n * expd_k * MMA::elem_b>;
    using vtype_c = vector_t<typename MMA::dtype_c, expd_m * expd_n * MMA::elem_c>;
#elif OPUS_TILE_CONTAINER == 1
    using vtype_a = array<typename MMA::vtype_a, expd_m * expd_k>;
    using vtype_b = array<typename MMA::vtype_b, expd_n * expd_k>;
    using vtype_c = array<typename MMA::vtype_c, expd_m * expd_n>;
#endif
    OPUS_D static constexpr auto tile_shape_a() { return tuple<number<expd_m>, number<tile_m>, number<expd_k>, number<tile_k>>{}; }
    OPUS_D static constexpr auto tile_shape_b() { return tuple<number<expd_n>, number<tile_n>, number<expd_k>, number<tile_k>>{}; }
    OPUS_D static constexpr auto tile_shape_c() { return tuple<number<expd_m>, number<tile_m>, number<expd_n>, number<tile_n>>{}; }

    OPUS_D static constexpr auto tile_dim_a()   { return tuple< tuple<y_dim, p_dim>,  tuple<y_dim, p_dim> >{}; }    // dim encoding for A, MxK
    OPUS_D static constexpr auto tile_dim_b()   { return tuple< tuple<y_dim, p_dim>,  tuple<y_dim, p_dim> >{}; }    // dim encoding for B, NxK
    OPUS_D static constexpr auto tile_dim_c()   { return tuple< tuple<y_dim, p_dim>,  tuple<y_dim, p_dim> >{}; }    // dim encoding for C, MxN

    OPUS_D static constexpr auto shape_a() { return flatten_tuple(embed_nested_tuple(unflatten_shape(tile_shape_a(), tile_dim_a()), unflatten_shape(MMA::shape_a(), MMA::dim_a()))); }
    OPUS_D static constexpr auto shape_b() { return flatten_tuple(embed_nested_tuple(unflatten_shape(tile_shape_b(), tile_dim_b()), unflatten_shape(MMA::shape_b(), MMA::dim_b()))); }
    OPUS_D static constexpr auto shape_c() { return flatten_tuple(embed_nested_tuple(unflatten_shape(tile_shape_c(), tile_dim_c()), unflatten_shape(MMA::shape_c(), MMA::dim_c()))); }

    OPUS_D static constexpr auto dim_a()   { return embed_nested_tuple(tile_dim_a(), MMA::dim_a()); }    // dim encoding for A, MxK
    OPUS_D static constexpr auto dim_b()   { return embed_nested_tuple(tile_dim_b(), MMA::dim_b()); }    // dim encoding for A, MxK
    OPUS_D static constexpr auto dim_c()   { return embed_nested_tuple(tile_dim_c(), MMA::dim_c()); }    // dim encoding for A, MxK

    // Cached tile sizes (avoids re-evaluating y_shape + reduce_tuple_mul in every operator/step_k)
    static constexpr index_t mma_a_len = get<0>(reduce_tuple_mul(MMA::y_shape_a())).value;
    static constexpr index_t mma_b_len = get<0>(reduce_tuple_mul(MMA::y_shape_b())).value;
    static constexpr index_t mma_c_len = get<0>(reduce_tuple_mul(MMA::y_shape_c())).value;
    static constexpr index_t tile_a_len = EXPAND_M * EXPAND_K * mma_a_len;
    static constexpr index_t tile_b_len = EXPAND_N * EXPAND_K * mma_b_len;
    static constexpr index_t tile_c_len = EXPAND_M * EXPAND_N * mma_c_len;

    // input a/b/c is array of ext type e.g. "fp16x2_t a[2];", pass "a" to this function
    // #pragma unroll required: constexpr trip counts but clang's heuristic gives up on large tiles -> runtime indices into vtype_a/b/c -> N-way s_cselect_b64 chain on GFX9 -> SGPR-spill-to-VGPR-lane blow-up.
    template<typename VA, typename VB, typename VC, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0,
                    std::enable_if_t< (is_array_v< remove_cvref_t<VA> > && is_array_v< remove_cvref_t<VB> > && is_array_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        VC c_ {c};
        #pragma unroll
        for (index_t I = 0; I < EXPAND_K * EXPAND_M * EXPAND_N; I++) {
            index_t i_k = I / (EXPAND_M * EXPAND_N), i_m = (I / EXPAND_N) % EXPAND_M, i_n = I % EXPAND_N;
            auto s_a = a[i_m * EXPAND_K + i_k]; auto s_b = b[i_n * EXPAND_K + i_k]; auto s_c = c_[i_m * EXPAND_N + i_n];
            s_c = MMA{}(s_a, s_b, s_c); c_[i_m * EXPAND_N + i_n] = s_c;
        }
        return c_;
    }
    template<typename VA, typename VB, typename VC, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0,
                    std::enable_if_t< (is_vector_v< remove_cvref_t<VA> > && is_vector_v< remove_cvref_t<VB> > && is_vector_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        static_assert(size<VA>() == tile_a_len);
        static_assert(size<VB>() == tile_b_len);
        static_assert(size<VC>() == tile_c_len);

        constexpr index_t a_len = mma_a_len, b_len = mma_b_len, c_len = mma_c_len;

        VC c_ {c};
        #pragma unroll
        for (index_t I = 0; I < EXPAND_K * EXPAND_M * EXPAND_N; I++) {
            index_t i_k = I / (EXPAND_M * EXPAND_N), i_m = (I / EXPAND_N) % EXPAND_M, i_n = I % EXPAND_N;
            index_t i_a = (i_m * EXPAND_K + i_k) * a_len, i_b = (i_n * EXPAND_K + i_k) * b_len, i_c = (i_m * EXPAND_N + i_n) * c_len;
            typename MMA::vtype_a s_a;
            #pragma unroll
            for (index_t j = 0; j < a_len; j++) s_a[j] = a[i_a + j];
            typename MMA::vtype_b s_b;
            #pragma unroll
            for (index_t j = 0; j < b_len; j++) s_b[j] = b[i_b + j];
            typename MMA::vtype_c s_c;
            #pragma unroll
            for (index_t j = 0; j < c_len; j++) s_c[j] = c_[i_c + j];
            s_c = MMA{}(s_a, s_b, s_c);
            #pragma unroll
            for (index_t j = 0; j < c_len; j++) c_[i_c + j] = s_c[j];
        }
        return c_;
    }

    template<typename VA, typename VB, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        vtype_c c{0};
        return operator()(a, b, c, number<cbsz>{}, number<abid>{}, number<blgp>{});
    }

    // Scaled MFMA (f8f6f4): forward scale_a, scale_b and per-call scale_op_sel to underlying MMA; all sub-MFMAs in this call share the same scale_op_sel (same K position in the packed-int32 scale word).
    template<typename VA, typename VB, typename VC, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0,
             std::enable_if_t< (is_array_v< remove_cvref_t<VA> > && is_array_v< remove_cvref_t<VB> > && is_array_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        VC c_ {c};
        #pragma unroll
        for (index_t I = 0; I < EXPAND_K * EXPAND_M * EXPAND_N; I++) {
            index_t i_k = I / (EXPAND_M * EXPAND_N), i_m = (I / EXPAND_N) % EXPAND_M, i_n = I % EXPAND_N;
            auto s_a = a[i_m * EXPAND_K + i_k]; auto s_b = b[i_n * EXPAND_K + i_k]; auto s_c = c_[i_m * EXPAND_N + i_n];
            s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{}); c_[i_m * EXPAND_N + i_n] = s_c;
        }
        return c_;
    }

    template<typename VA, typename VB, typename VC, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0,
             std::enable_if_t< (is_vector_v< remove_cvref_t<VA> > && is_vector_v< remove_cvref_t<VB> > && is_vector_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, const VC& c, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        static_assert(size<VA>() == tile_a_len);
        static_assert(size<VB>() == tile_b_len);
        static_assert(size<VC>() == tile_c_len);

        constexpr index_t a_len = mma_a_len, b_len = mma_b_len, c_len = mma_c_len;

        VC c_ {c};
        static_ford<EXPAND_K, EXPAND_M, EXPAND_N>([&](auto i_k, auto i_m, auto i_n){
            constexpr index_t i_tile_a = i_m * EXPAND_K + i_k;
            constexpr index_t i_tile_b = i_n * EXPAND_K + i_k;
            constexpr index_t i_tile_c = i_m * EXPAND_N + i_n;
            auto s_a = slice(a, number<i_tile_a * a_len>{}, number<i_tile_a * a_len + a_len>{});
            auto s_b = slice(b, number<i_tile_b * b_len>{}, number<i_tile_b * b_len + b_len>{});
            auto s_c = slice(c_, number<i_tile_c * c_len>{}, number<i_tile_c * c_len + c_len>{});
            s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{});
            set_slice(c_, s_c, number<i_tile_c * c_len>{}, number<i_tile_c * c_len + c_len>{});
        });
        return c_;
    }

    template<typename VA, typename VB, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0>
    OPUS_D constexpr auto operator()(const VA& a, const VB& b, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        vtype_c c{0};
        return operator()(a, b, c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{});
    }

    template<index_t STEP_K, typename VA, typename VB, typename VC, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0,
                    std::enable_if_t< (is_array_v< remove_cvref_t<VA> > && is_array_v< remove_cvref_t<VB> > && is_array_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto step_k(number<STEP_K>, const VA& a, const VB& b, const VC& c, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        static_assert(STEP_K < EXPAND_K);
        VC c_ {c};
        static_for<EXPAND_M * EXPAND_N>([&](auto I){
            constexpr index_t i_m = I.value / EXPAND_N, i_n = I.value % EXPAND_N;
            auto s_a = a[i_m * EXPAND_K + STEP_K]; auto s_b = b[i_n * EXPAND_K + STEP_K]; auto s_c = c_[i_m * EXPAND_N + i_n];
            s_c = MMA{}(s_a, s_b, s_c); c_[i_m * EXPAND_N + i_n] = s_c;
        });
        return c_;
    }

    template<index_t STEP_K, typename VA, typename VB, typename VC, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0,
                    std::enable_if_t< (is_vector_v< remove_cvref_t<VA> > && is_vector_v< remove_cvref_t<VB> > && is_vector_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto step_k(number<STEP_K>, const VA& a, const VB& b, const VC& c, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        static_assert(STEP_K < EXPAND_K);
        static_assert(size<VA>() == tile_a_len);
        static_assert(size<VB>() == tile_b_len);
        static_assert(size<VC>() == tile_c_len);

        constexpr index_t a_len = mma_a_len, b_len = mma_b_len, c_len = mma_c_len;

        VC c_ {c};
        #pragma unroll
        for (index_t I = 0; I < EXPAND_M * EXPAND_N; I++) {
            index_t i_m = I / EXPAND_N, i_n = I % EXPAND_N;
            index_t i_a = (i_m * EXPAND_K + STEP_K) * a_len, i_b = (i_n * EXPAND_K + STEP_K) * b_len, i_c = (i_m * EXPAND_N + i_n) * c_len;
            typename MMA::vtype_a s_a;
            #pragma unroll
            for (index_t j = 0; j < a_len; j++) s_a[j] = a[i_a + j];
            typename MMA::vtype_b s_b;
            #pragma unroll
            for (index_t j = 0; j < b_len; j++) s_b[j] = b[i_b + j];
            typename MMA::vtype_c s_c;
            #pragma unroll
            for (index_t j = 0; j < c_len; j++) s_c[j] = c_[i_c + j];
            s_c = MMA{}(s_a, s_b, s_c);
            #pragma unroll
            for (index_t j = 0; j < c_len; j++) c_[i_c + j] = s_c[j];
        }
        return c_;
    }

    template<index_t STEP_K, typename VA, typename VB, index_t cbsz = 0, index_t abid = 0, index_t blgp = 0>
    OPUS_D constexpr auto step_k(number<STEP_K> step, const VA& a, const VB& b, number<cbsz> = {}, number<abid> = {}, number<blgp> = {}) {
        vtype_c c{0};
        return step_k(step, a, b, c, number<cbsz>{}, number<abid>{}, number<blgp>{});
    }

    template<index_t STEP_K, typename VA, typename VB, typename VC, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0,
             std::enable_if_t< (is_array_v< remove_cvref_t<VA> > && is_array_v< remove_cvref_t<VB> > && is_array_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto step_k(number<STEP_K>, const VA& a, const VB& b, const VC& c, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        static_assert(STEP_K < EXPAND_K);
        VC c_ {c};
        #pragma unroll
        for (index_t I = 0; I < EXPAND_M * EXPAND_N; I++) {
            index_t i_m = I / EXPAND_N, i_n = I % EXPAND_N;
            auto s_a = a[i_m * EXPAND_K + STEP_K]; auto s_b = b[i_n * EXPAND_K + STEP_K]; auto s_c = c_[i_m * EXPAND_N + i_n];
            s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{}); c_[i_m * EXPAND_N + i_n] = s_c;
        }
        return c_;
    }

    template<index_t STEP_K, typename VA, typename VB, typename VC, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0,
             std::enable_if_t< (is_vector_v< remove_cvref_t<VA> > && is_vector_v< remove_cvref_t<VB> > && is_vector_v< remove_cvref_t<VC> >), bool > = true>
    OPUS_D constexpr auto step_k(number<STEP_K>, const VA& a, const VB& b, const VC& c, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        static_assert(STEP_K < EXPAND_K);
        static_assert(size<VA>() == tile_a_len);
        static_assert(size<VB>() == tile_b_len);
        static_assert(size<VC>() == tile_c_len);

        constexpr index_t a_len = mma_a_len, b_len = mma_b_len, c_len = mma_c_len;

        VC c_ {c};
        #pragma unroll
        for (index_t I = 0; I < EXPAND_M * EXPAND_N; I++) {
            index_t i_m = I / EXPAND_N, i_n = I % EXPAND_N;
            index_t i_a = (i_m * EXPAND_K + STEP_K) * a_len, i_b = (i_n * EXPAND_K + STEP_K) * b_len, i_c = (i_m * EXPAND_N + i_n) * c_len;
            typename MMA::vtype_a s_a;
            #pragma unroll
            for (index_t j = 0; j < a_len; j++) s_a[j] = a[i_a + j];
            typename MMA::vtype_b s_b;
            #pragma unroll
            for (index_t j = 0; j < b_len; j++) s_b[j] = b[i_b + j];
            typename MMA::vtype_c s_c;
            #pragma unroll
            for (index_t j = 0; j < c_len; j++) s_c[j] = c_[i_c + j];
            s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{});
            #pragma unroll
            for (index_t j = 0; j < c_len; j++) c_[i_c + j] = s_c[j];
        }
        return c_;
    }

    template<index_t STEP_K, typename VA, typename VB, index_t scale_op_sel_a = 0, index_t scale_op_sel_b = 0>
    OPUS_D constexpr auto step_k(number<STEP_K> step, const VA& a, const VB& b, int scale_a, int scale_b, number<scale_op_sel_a> = {}, number<scale_op_sel_b> = {}) {
        vtype_c c{0};
        return step_k(step, a, b, c, scale_a, scale_b, number<scale_op_sel_a>{}, number<scale_op_sel_b>{});
    }

    OPUS_ADAPTOR_LAYOUT_API_DEFINE
};
}
struct tiled_mma_adaptor {
    template<typename MMA, index_t... Ts> OPUS_D decltype(auto) operator()(MMA&&, number<Ts>...) { return impl::tiled_mma_adaptor<remove_cvref_t<MMA>, Ts...>{};}
};

template<typename MMA, index_t E_M, index_t E_N, index_t E_K, index_t T_M, index_t T_N, index_t T_K, typename A = tiled_mma_adaptor>
OPUS_D decltype(auto) make_tiled_mma(MMA&& mma, number<E_M>, number<E_N>, number<E_K>, number<T_M>, number<T_N>, number<T_K>, A&& = {}) {
    return A{}(std::forward<MMA>(mma), number<E_M>{}, number<E_N>{}, number<E_K>{}, number<T_M>{}, number<T_N>{}, number<T_K>{});
}

template<typename MMA, typename ES /* expand-m/n/k */, typename TS /* tile-m/n/k */, typename A = tiled_mma_adaptor>
OPUS_D decltype(auto) make_tiled_mma(MMA&& mma, ES, TS, A&& = {}) {
    return A{}(std::forward<MMA>(mma), number<get<0>(ES{})>{}, number<get<1>(ES{})>{}, number<get<2>(ES{})>{}, number<get<0>(TS{})>{}, number<get<1>(TS{})>{}, number<get<2>(TS{})>{});
}

template<typename d_a, typename d_b, typename d_c, typename ES /* expand-m/n/k */, typename TS /* tile-m/n/k */, typename WS /* wave-m/n/k*/,
#if defined(__gfx1250__)
         typename WA = wmma_adaptor,
#else
         typename WA = mfma_adaptor,
#endif
         typename TA = tiled_mma_adaptor, index_t warp_size = get_warp_size()>
OPUS_D decltype(auto) make_tiled_mma(ES, TS, WS, WA&& = {}, TA&& = {}) {
#if defined(__gfx1250__)
    return TA{}(make_wmma<d_a, d_b, d_c>(WS{}, WA{}, number<warp_size>{}),
#else
    return TA{}(make_mfma<d_a, d_b, d_c>(WS{}, WA{}, number<warp_size>{}),
#endif
            number<get<0>(ES{})>{}, number<get<1>(ES{})>{}, number<get<2>(ES{})>{}, number<get<0>(TS{})>{}, number<get<1>(TS{})>{}, number<get<2>(TS{})>{});
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
template<index_t cached_vec = 0, typename L, typename D, typename S, typename C, std::enable_if_t<is_layout_v<L> && is_tuple_v<D> && is_tuple_v<S> && is_tuple_v<C>, bool> = true>
OPUS_D constexpr auto partition_layout(L&& layout, D&& dims, S&& shapes, C&& p_coord) {
    OPUS_KP_(dims);
    return make_layout<cached_vec>(std::forward<S>(shapes), unfold_x_stride(std::forward<D>(dims), std::forward<S>(shapes), layout.stride()), unfold_p_coord(std::forward<D>(dims), p_coord));
}
// partition, use cached_vec to dispatch which layout implementation. cached_vec < 0 : "layout", cached_vec == 0 : "layout_linear", cached_vec > 0 : "layout_cached"
template<index_t cached_vec = 0, typename M> OPUS_D constexpr auto partition_layout_a(M&& mma) { return mma.template layout_a<cached_vec>(); }
template<index_t cached_vec = 0, typename M> OPUS_D constexpr auto partition_layout_b(M&& mma) { return mma.template layout_b<cached_vec>(); }
template<index_t cached_vec = 0, typename M> OPUS_D constexpr auto partition_layout_c(M&& mma) { return mma.template layout_c<cached_vec>(); }

template<index_t cached_vec = 0, typename M, typename S, std::enable_if_t<is_tuple_v<S>, bool> = true> OPUS_D constexpr auto partition_layout_a(M&& mma, S&& x_stride) { return mma.template layout_a<cached_vec>(std::forward<S>(x_stride)); }
template<index_t cached_vec = 0, typename M, typename S, std::enable_if_t<is_tuple_v<S>, bool> = true> OPUS_D constexpr auto partition_layout_b(M&& mma, S&& x_stride) { return mma.template layout_b<cached_vec>(std::forward<S>(x_stride)); }
template<index_t cached_vec = 0, typename M, typename S, std::enable_if_t<is_tuple_v<S>, bool> = true> OPUS_D constexpr auto partition_layout_c(M&& mma, S&& x_stride) { return mma.template layout_c<cached_vec>(std::forward<S>(x_stride)); }

template<index_t cached_vec = 0, typename M, typename S, typename C, std::enable_if_t<is_tuple_v<S> && is_tuple_v<C>, bool> = true>
OPUS_D constexpr auto partition_layout_a(M&& mma, S&& x_stride, C&& p_coord) { return mma.template layout_a<cached_vec>(std::forward<S>(x_stride), std::forward<C>(p_coord)); }
template<index_t cached_vec = 0, typename M, typename S, typename C, std::enable_if_t<is_tuple_v<S> && is_tuple_v<C>, bool> = true>
OPUS_D constexpr auto partition_layout_b(M&& mma, S&& x_stride, C&& p_coord) { return mma.template layout_b<cached_vec>(std::forward<S>(x_stride), std::forward<C>(p_coord)); }
template<index_t cached_vec = 0, typename M, typename S, typename C, std::enable_if_t<is_tuple_v<S> && is_tuple_v<C>, bool> = true>
OPUS_D constexpr auto partition_layout_c(M&& mma, S&& x_stride, C&& p_coord) { return mma.template layout_c<cached_vec>(std::forward<S>(x_stride), std::forward<C>(p_coord)); }

template<index_t cached_vec = 0, typename M, typename C, std::enable_if_t<is_tuple_v<C>, bool> = true> OPUS_D constexpr auto partition_layout_a_packed(M&& mma, C&& p_coord) { return mma.template layout_a_packed<cached_vec>(std::forward<C>(p_coord)); }
template<index_t cached_vec = 0, typename M, typename C, std::enable_if_t<is_tuple_v<C>, bool> = true> OPUS_D constexpr auto partition_layout_b_packed(M&& mma, C&& p_coord) { return mma.template layout_b_packed<cached_vec>(std::forward<C>(p_coord)); }
template<index_t cached_vec = 0, typename M, typename C, std::enable_if_t<is_tuple_v<C>, bool> = true> OPUS_D constexpr auto partition_layout_c_packed(M&& mma, C&& p_coord) { return mma.template layout_c_packed<cached_vec>(std::forward<C>(p_coord)); }
#undef OPUS_KP_

} // namespace opus

// call this macro within your kernel body to have fast access to opus types
#define OPUS_USING_COMMON_TYPES  \
    using opus::operator""_I;    \
    using p_dim = opus::p_dim;   \
    using y_dim = opus::y_dim;

// call this macro in global scope (outside of your kernel function, or under structure)
#define OPUS_USING_COMMON_TYPES_ALL     \
    OPUS_USING_COMMON_TYPES             \
    template<opus::index_t I>     using num = opus::number<I>;      \
    template<typename... T>       using tup = opus::tuple<T...>;    \
    template<opus::index_t... Is> using seq = opus::seq<Is...>;

// clang-format on
