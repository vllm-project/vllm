/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/support/json_serializer.h
 * \brief A JSON-based serializer. Automatically generates serialization and deserialization logic
 * from reflection.
 */
#ifndef XGRAMMAR_SUPPORT_JSON_SERIALIZER_H_
#define XGRAMMAR_SUPPORT_JSON_SERIALIZER_H_

#include <picojson.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "encoding.h"
#include "json_parse.h"
#include "logging.h"
#include "reflection.h"
#include "utils.h"
#include "xgrammar/exception.h"
#include "xgrammar/object.h"

namespace xgrammar {

/******************** Interfaces ********************/

/*!
 * \brief Manages the version of the serialized object. The version will be added to the serialized
 * object, and during deserialization, the object's version must match the current serialization
 * version in xgrammar.
 */
class SerializeVersion {
 public:
  /*!
   * \brief Returns the current serialization version.
   */
  static std::string_view GetVersion() { return kXGrammarSerializeVersion; }

  /*!
   * \brief Adds the version info to the serialized object.
   */
  static void Apply(picojson::object* object);

  /*!
   * \brief Checks if the serialized object's version matches the current serialization version.
   * \return An error if the version does not exist or does not match.
   */
  static std::optional<SerializationError> Check(const picojson::object& object);

 private:
  /*!
   * \brief The key of the version info in the serialized object.
   */
  static constexpr const char kXGrammarSerializeVersionKey[] = "__VERSION__";

  /*!
   * \brief The current serialization version. When the serialization result of any object in
   * XGrammar is changed, this version should be bumped.
   */
  static constexpr const char kXGrammarSerializeVersion[] = "v16";
};

/*!
 * \brief Serializes a value to a JSON value.
 * \details It supports STL types, PImpl types, reflection-based types (whose members are defined
 * through XGRAMMAR_MEMBER_TABLE or XGRAMMAR_MEMBER_ARRAY), and types who have defined a global
 * SerializeJSONValue function. For reflection-based types, the serialization logic is automatically
 * generated from the defined members.
 * \param value The value to be serialized.
 * \return The serialized JSON value.
 */
template <typename T>
picojson::value AutoSerializeJSONValue(const T& value);

/*!
 * \brief Serializes a value to a JSON string.
 * \details It supports STL types, PImpl types, reflection-based types (whose members are defined
 * through XGRAMMAR_MEMBER_TABLE or XGRAMMAR_MEMBER_ARRAY), and types who have defined a global
 * SerializeJSONValue function. For reflection-based types, the serialization logic is automatically
 * generated from the defined members.
 * \param value The value to be serialized.
 * \param add_version Whether to add the version info to the serialized object. The addition is
 * valid only when the serialized result is an object.
 * \return The serialized JSON string.
 */
template <typename T>
std::string AutoSerializeJSON(const T& value, bool add_version = false);

/*!
 * \brief Deserializes a value from a JSON value.
 * \details It supports STL types, PImpl types, reflection-based types (whose members are defined
 * through XGRAMMAR_MEMBER_TABLE or XGRAMMAR_MEMBER_ARRAY), and types who have defined a global
 * DeserializeJSONValue function. For reflection-based types, the deserialization logic is
 * automatically generated from the defined members. If a reflection-based type also defines
 * `std::optional<std::string> Validate() const`, it is called after all members are deserialized,
 * and a returned message is reported as a deserialization error. This is where a type checks the
 * invariants (index ranges, CSR layout, ...) that the constructors normally guarantee.
 * \param result The pointer to the result to be deserialized.
 * \param value The JSON value to be deserialized.
 * \param type_name The name of the type to be deserialized. Used for error message.
 * \return The deserialization error if any.
 */
template <typename T>
std::optional<SerializationError> AutoDeserializeJSONValue(
    T* result, const picojson::value& value, const std::string& type_name = ""
);

/*!
 * \brief Deserializes a value from a JSON string.
 * \details It supports STL types, PImpl types, reflection-based types (whose members are defined
 * through XGRAMMAR_MEMBER_TABLE or XGRAMMAR_MEMBER_ARRAY), and types who have defined a global
 * DeserializeJSONValue function. For reflection-based types, the deserialization logic is
 * automatically generated from the defined members.
 * \param result The pointer to the result to be deserialized.
 * \param json_string The JSON string to be deserialized.
 * \param check_version Whether to check the version info in the serialized object. The check is
 * valid only when the serialized object is an object.
 * \param type_name The name of the type to be deserialized. Used for error message.
 * \return The deserialization error if any.
 */
template <typename T>
std::optional<SerializationError> AutoDeserializeJSON(
    T* result,
    const std::string& json_string,
    bool check_version = false,
    const std::string& type_name = ""
);

/*!
 * \brief Constructs a deserialize error with the given error message and type name.
 * \param error_message The error message.
 * \param type_name The name of the type.
 * \return The constructed runtime error.
 */
inline SerializationError ConstructDeserializeError(
    const std::string& error_message, const std::string& type_name
);

/******************** Implementations ********************/

inline void SerializeVersion::Apply(picojson::object* object) {
  XGRAMMAR_DCHECK(object != nullptr);
  XGRAMMAR_DCHECK(object->find(kXGrammarSerializeVersionKey) == object->end());
  (*object)[kXGrammarSerializeVersionKey] = picojson::value(std::string(GetVersion()));
}

inline std::optional<SerializationError> SerializeVersion::Check(const picojson::object& object) {
  if (object.find(kXGrammarSerializeVersionKey) == object.end()) {
    return DeserializeVersionError(
        std::string("Missing version in serialized object: ") + kXGrammarSerializeVersionKey
    );
  }
  if (object.at(kXGrammarSerializeVersionKey).get<std::string>() != GetVersion()) {
    return DeserializeVersionError(
        std::string("Wrong version in serialized object: Got ") +
        object.at(kXGrammarSerializeVersionKey).get<std::string>() + ", expected " +
        std::string(GetVersion())
    );
  }
  return std::nullopt;
}

/******************** Template Implementations ********************/

namespace detail::json_serializer {

template <typename, typename = void>
struct has_serialize_json_global : std::false_type {};

template <typename T>
struct has_serialize_json_global<
    T,
    std::void_t<decltype(SerializeJSONValue(std::declval<const T&>()))>> : std::true_type {
  static_assert(
      std::is_same_v<decltype(SerializeJSONValue(std::declval<const T&>())), picojson::value>,
      "SerializeJSONValue must be a global function returning picojson::value"
  );
};

template <typename T, typename = void>
struct has_deserialize_json_global : std::false_type {};

template <typename T>
struct has_deserialize_json_global<
    T,
    std::void_t<decltype(DeserializeJSONValue(std::declval<T*>(), picojson::value{}, std::string{})
    )>> : std::true_type {
  static_assert(
      std::is_same_v<
          decltype(DeserializeJSONValue(std::declval<T*>(), picojson::value{}, std::string{})),
          std::optional<SerializationError>>,
      "DeserializeJSONValue must be a global function returning std::optional<SerializationError>"
  );
  static_assert(
      std::is_default_constructible_v<T>,
      "global deserializer can only apply to a default constructible type"
  );
};

template <typename, typename = void>
struct has_validate : std::false_type {};

template <typename T>
struct has_validate<T, std::void_t<decltype(std::declval<const T&>().Validate())>>
    : std::true_type {
  static_assert(
      std::is_same_v<decltype(std::declval<const T&>().Validate()), std::optional<std::string>>,
      "Validate must be a const member function returning std::optional<std::string>"
  );
};

/*! \brief Runs T::Validate() on a deserialized value if the type defines one. */
template <typename T>
inline std::optional<SerializationError> ValidateDeserialized(
    const T& value, const std::string& type_name
) {
  if constexpr (has_validate<T>::value) {
    if (auto error = value.Validate()) {
      return ConstructDeserializeError(*error, type_name);
    }
  }
  return std::nullopt;
}

template <typename>
inline constexpr bool false_v = false;

template <typename T>
inline picojson::value TraitSerializeJSONValue(const T& value) {
  using Functor = member_functor<T>;
  if constexpr (Functor::value == member_type::kConfig) {
    if constexpr (Functor::has_names) {
      // normal named struct
      picojson::object obj;
      obj.reserve(Functor::member_count);
      visit_config<T>([&](auto ptr, const char* name, std::size_t) {
        XGRAMMAR_DCHECK(obj.find(name) == obj.end());
        obj[name] = AutoSerializeJSONValue(value.*ptr);
      });
      return picojson::value(std::move(obj));
    } else if constexpr (Functor::member_count == 1) {
      // optimize for single member unnamed structs
      constexpr auto member_ptr = std::get<0>(Functor::members);
      return AutoSerializeJSONValue(value.*member_ptr);
    } else {
      // normal unnamed struct
      picojson::array arr;
      arr.resize(Functor::member_count);
      visit_config<T>([&](auto ptr, const char*, std::size_t idx) {
        arr[idx] = AutoSerializeJSONValue(value.*ptr);
      });
      return picojson::value(std::move(arr));
    }
  } else {
    // should give an error in this case
    static_assert(detail::json_serializer::false_v<T>, "Invalid trait type");
    return picojson::value{};
  }
}

template <typename T>
inline std::optional<SerializationError> TraitDeserializeJSONValue(
    T* result, const picojson::value& value, const std::string& type_name
) {
  using Functor = member_functor<T>;
  if constexpr (Functor::value == member_type::kConfig) {
    if constexpr (Functor::has_names) {
      // normal named struct
      if (!value.is<picojson::object>()) {
        return ConstructDeserializeError("Expect an object", type_name);
      }
      const auto& obj = value.get<picojson::object>();
      std::optional<SerializationError> err = std::nullopt;
      visit_config<T>([&](auto ptr, const char* name, std::size_t idx) {
        if (err) {
          return;
        } else if (obj.find(name) == obj.end()) {
          err = ConstructDeserializeError("Missing member " + std::string(name), type_name);
        } else if (auto e = AutoDeserializeJSONValue(&(result->*ptr), obj.at(name), type_name)) {
          err = e;
        }
      });
      return err;
    } else if constexpr (Functor::member_count == 1) {
      // optimize for single member unnamed structs
      constexpr auto member_ptr = std::get<0>(Functor::members);
      return AutoDeserializeJSONValue(&(result->*member_ptr), value, type_name);
    } else {
      // normal unnamed struct
      if (!value.is<picojson::array>()) {
        return ConstructDeserializeError("Expect an array", type_name);
      }
      const auto& arr = value.get<picojson::array>();
      if (arr.size() != Functor::member_count) {
        return ConstructDeserializeError(
            "Wrong number of elements in array: Expected " + std::to_string(Functor::member_count) +
                ", but got " + std::to_string(arr.size()),
            type_name
        );
      }
      std::optional<SerializationError> err = std::nullopt;
      visit_config<T>([&](auto ptr, const char*, std::size_t idx) {
        if (err) {
          return;
        } else if (auto e = AutoDeserializeJSONValue(&(result->*ptr), arr[idx], type_name)) {
          err = e;
        }
      });
      return err;
    }
  } else {
    // should give an error in this case
    static_assert(detail::json_serializer::false_v<T>, "Invalid trait type");
    XGRAMMAR_UNREACHABLE();
  }
}

/******************** Customized Serialization ********************/

template <typename T, typename = is_pimpl_class<T>>
inline picojson::value AutoSerializeJSONValuePImpl(const T& value) {
  if (value.IsNull()) return picojson::value{};
  return AutoSerializeJSONValue(*value.ImplPtr());
}

template <typename T, typename = is_pimpl_class<T>>
inline std::optional<SerializationError> AutoDeserializeJSONValuePImpl(
    T* result, const picojson::value& value, const std::string& type_name
) {
  XGRAMMAR_DCHECK(result->IsNull());
  if (value.is<picojson::null>()) {
    *result = T{NullObj{}};
    return std::nullopt;
  }
  auto ptr = std::make_shared<typename T::Impl>();
  if (auto error = AutoDeserializeJSONValue(ptr.get(), value, type_name)) {
    return error;
  }
  *result = T(std::move(ptr));
  return std::nullopt;
}

}  // namespace detail::json_serializer

inline SerializationError ConstructDeserializeError(
    const std::string& error_message, const std::string& type_name
) {
  if (type_name.empty()) {
    return DeserializeFormatError("Deserialize error: " + error_message);
  } else {
    return DeserializeFormatError("Deserialize error for type " + type_name + ": " + error_message);
  }
}

template <typename T>
inline picojson::value AutoSerializeJSONValue(const T& value) {
  if constexpr (detail::json_serializer::has_serialize_json_global<T>::value) {
    // User-defined SerializeJSONValue (highest priority)
    return SerializeJSONValue(value);
  } else if constexpr (is_pimpl_class<T>::value) {
    // Library-customized serialization methods
    return detail::json_serializer::AutoSerializeJSONValuePImpl(value);
  } else if constexpr (member_trait<T>::value != member_type::kNone) {
    // Trait serialization methods
    return detail::json_serializer::TraitSerializeJSONValue(value);
  } else if constexpr (std::is_same_v<T, bool>) {
    // Below is primitive types
    return picojson::value(value);
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    return picojson::value(static_cast<int64_t>(value));
  } else if constexpr (std::is_floating_point_v<T>) {
    return picojson::value(static_cast<double>(value));
  } else if constexpr (std::is_same_v<T, std::string>) {
    std::string result;
    ByteToLatin1(value, &result);
    return picojson::value(result);
  } else if constexpr (is_std_optional<T>::value) {
    if (value.has_value()) {
      return AutoSerializeJSONValue(*value);
    } else {
      return picojson::value{};
    }
  } else if constexpr (is_std_pair<T>::value) {
    // std::pair<T1, T2>: serialize as an array of size 2
    picojson::array arr;
    arr.resize(2);
    arr[0] = AutoSerializeJSONValue(value.first);
    arr[1] = AutoSerializeJSONValue(value.second);
    return picojson::value(std::move(arr));
  } else if constexpr (is_std_vector<T>::value) {
    picojson::array arr;
    arr.reserve(value.size());
    for (const auto& item : value) {
      arr.push_back(AutoSerializeJSONValue(item));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (is_std_unordered_set<T>::value) {
    std::vector<const typename T::value_type*> ptr_vec;
    ptr_vec.reserve(value.size());
    for (const auto& item : value) {
      ptr_vec.push_back(&item);
    }
    std::sort(ptr_vec.begin(), ptr_vec.end(), [](const auto* a, const auto* b) { return *a < *b; });
    picojson::array arr;
    arr.reserve(value.size());
    for (const auto* ptr : ptr_vec) {
      arr.push_back(AutoSerializeJSONValue(*ptr));
    }
    return picojson::value(std::move(arr));
  } else if constexpr (is_std_unordered_map<T>::value) {
    if constexpr (std::is_same_v<typename T::key_type, std::string>) {
      // unordered_map<string, T>: map to json object
      picojson::object obj;
      obj.reserve(value.size());
      for (const auto& item : value) {
        obj[item.first] = AutoSerializeJSONValue(item.second);
      }
      return picojson::value(std::move(obj));
    } else {
      // unordered_map<T1, T2> (T1 is not string): map to json array of array of size 2
      std::vector<const typename T::value_type*> ptr_vec;
      ptr_vec.reserve(value.size());
      for (const auto& item : value) {
        ptr_vec.push_back(&item);
      }
      std::sort(ptr_vec.begin(), ptr_vec.end(), [](const auto* a, const auto* b) {
        return a->first < b->first;
      });
      picojson::array arr;
      arr.reserve(value.size());
      for (const auto* ptr : ptr_vec) {
        const auto& [key, item] = *ptr;
        picojson::array sub_arr{AutoSerializeJSONValue(key), AutoSerializeJSONValue(item)};
        arr.push_back(picojson::value(std::move(sub_arr)));
      }
      return picojson::value(std::move(arr));
    }
  } else {
    // should give an error in this case
    static_assert(detail::json_serializer::false_v<T>, "Cannot serialize this type");
    XGRAMMAR_UNREACHABLE();
  }
}

template <typename T>
inline std::string AutoSerializeJSON(const T& value, bool add_version) {
  picojson::value json_value = AutoSerializeJSONValue(value);
  if (add_version) {
    XGRAMMAR_DCHECK(json_value.is<picojson::object>());
    SerializeVersion::Apply(&json_value.get<picojson::object>());
  }
  return picojson::value(json_value).serialize();
}

template <typename T>
inline std::optional<SerializationError> AutoDeserializeJSONValue(
    T* result, const picojson::value& value, const std::string& type_name
) {
  static_assert(!std::is_const_v<T>, "Cannot deserialize into a const type");
  if constexpr (detail::json_serializer::has_deserialize_json_global<T>::value) {
    return DeserializeJSONValue(result, value, type_name);
  } else if constexpr (is_pimpl_class<T>::value) {
    return detail::json_serializer::AutoDeserializeJSONValuePImpl(result, value, type_name);
  } else if constexpr (member_trait<T>::value != member_type::kNone) {
    if (auto error = detail::json_serializer::TraitDeserializeJSONValue(result, value, type_name)) {
      return error;
    }
    return detail::json_serializer::ValidateDeserialized(*result, type_name);
  } else if constexpr (std::is_same_v<T, bool>) {
    if (!value.is<bool>()) {
      return ConstructDeserializeError("Expect a boolean", type_name);
    }
    *result = value.get<bool>();
    return std::nullopt;
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    if (!value.is<int64_t>()) {
      return ConstructDeserializeError("Expect an integer", type_name);
    }
    *result = static_cast<T>(value.get<int64_t>());
    return std::nullopt;
  } else if constexpr (std::is_floating_point_v<T>) {
    if (!value.is<double>()) {
      return ConstructDeserializeError("Expect a floating point number", type_name);
    }
    *result = static_cast<T>(value.get<double>());
    return std::nullopt;
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (!value.is<std::string>()) {
      return ConstructDeserializeError("Expect a string", type_name);
    }
    // Now PicoJSON will convert byte sequence to latin-1 string. Convert it back to byte sequence.
    auto error = Latin1ToBytes(value.get<std::string>(), result);
    if (error) {
      return ConstructDeserializeError(
          "XGramamr serializer will serialize byte sequence as latin-1 string, but got invalid "
          "latin-1 string",
          type_name
      );
    }
    return std::nullopt;
  } else if constexpr (is_std_optional<T>::value) {
    // for the following container<T>, T must be default constructible
    if (value.is<picojson::null>()) {
      result->reset();
      return std::nullopt;
    } else {
      return AutoDeserializeJSONValue(&(result->emplace()), value, type_name);
    }
  } else if constexpr (is_std_pair<T>::value) {
    // std::pair<T1, T2>: deserialize from an array of size 2
    if (!value.is<picojson::array>()) {
      return ConstructDeserializeError("Expect an array for deserializing pair", type_name);
    }
    const auto& arr = value.get<picojson::array>();
    if (arr.size() != 2) {
      return ConstructDeserializeError(
          "Expect an array of size 2 for deserializing pair", type_name
      );
    }
    if (auto error = AutoDeserializeJSONValue(&(result->first), arr[0], type_name)) {
      return error;
    }
    if (auto error = AutoDeserializeJSONValue(&(result->second), arr[1], type_name)) {
      return error;
    }
    return std::nullopt;
  } else if constexpr (is_std_vector<T>::value) {
    if (!value.is<picojson::array>()) {
      return ConstructDeserializeError("Expect an array", type_name);
    }
    const auto& arr = value.get<picojson::array>();
    result->clear();
    result->reserve(arr.size());
    for (const auto& item : arr) {
      if (auto error = AutoDeserializeJSONValue(&(result->emplace_back()), item, type_name)) {
        return error;
      }
    }
    return std::nullopt;
  } else if constexpr (is_std_unordered_set<T>::value) {
    if (!value.is<picojson::array>()) {
      return ConstructDeserializeError(
          "Expect an array for deserializing unordered set", type_name
      );
    }
    const auto& arr = value.get<picojson::array>();
    result->clear();
    result->reserve(arr.size());
    for (const auto& item : arr) {
      typename T::value_type item_value{};
      if (auto error = AutoDeserializeJSONValue(&item_value, item, type_name)) {
        return error;
      }
      result->emplace(std::move(item_value));
    }
    return std::nullopt;
  } else if constexpr (is_std_unordered_map<T>::value) {
    if constexpr (std::is_same_v<typename T::key_type, std::string>) {
      // unordered_map<string, T>: convert from json object
      if (!value.is<picojson::object>()) {
        return ConstructDeserializeError("Expect an object", type_name);
      }
      const auto& obj = value.get<picojson::object>();
      result->clear();
      result->reserve(obj.size());
      for (const auto& [key, item] : obj) {
        typename T::mapped_type item_value{};
        if (auto error = AutoDeserializeJSONValue(&item_value, item, type_name)) {
          return error;
        }
        result->try_emplace(key, std::move(item_value));
      }
      return std::nullopt;
    } else {
      // unordered_map<T1, T2> (T1 is not string): convert from json array of array of size 2
      if (!value.is<picojson::array>()) {
        return ConstructDeserializeError(
            "Expect an array for deserializing unordered map", type_name
        );
      }
      const auto& arr = value.get<picojson::array>();
      result->clear();
      result->reserve(arr.size());
      for (const auto& item : arr) {
        if (!item.is<picojson::array>()) {
          return ConstructDeserializeError(
              "Expect an array of array of size 2 for deserializing unordered map", type_name
          );
        }
        const auto& sub_arr = item.get<picojson::array>();
        if (sub_arr.size() != 2) {
          return ConstructDeserializeError(
              "Expect an array of array of size 2 for deserializing unordered map", type_name
          );
        }
        typename T::key_type key_value{};
        if (auto error = AutoDeserializeJSONValue(&key_value, sub_arr[0], type_name)) {
          return error;
        }
        typename T::mapped_type item_value{};
        if (auto error = AutoDeserializeJSONValue(&item_value, sub_arr[1], type_name)) {
          return error;
        }
        result->emplace(std::move(key_value), std::move(item_value));
      }
      return std::nullopt;
    }
  } else {
    // should give an error in this case
    static_assert(detail::json_serializer::false_v<T>, "Cannot deserialize this type");
    XGRAMMAR_UNREACHABLE();
  }
}

template <typename T>
inline std::optional<SerializationError> AutoDeserializeJSON(
    T* result, const std::string& json_string, bool check_version, const std::string& type_name
) {
  picojson::value json_value;
  if (auto error = ParseJSON(json_value, json_string); !error.empty()) {
    return InvalidJSONError(error);
  }
  if (check_version) {
    XGRAMMAR_DCHECK(json_value.is<picojson::object>());
    if (auto error = SerializeVersion::Check(json_value.get<picojson::object>())) {
      return error;
    }
  }
  return AutoDeserializeJSONValue(result, json_value, type_name);
}

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_JSON_SERIALIZER_H_
