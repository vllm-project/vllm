// C ABI over the vendored xgrammar C++ library, consumed by `src/ffi.rs`.
//
// Every entry point catches C++ exceptions: on failure it stores a
// malloc'ed, NUL-terminated message in `*err` (freed with `xgr_string_free`)
// and returns a null pointer or -1.

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <vector>

using namespace xgrammar;

struct xgr_tokenizer_info {
  TokenizerInfo value;
};
struct xgr_compiler {
  GrammarCompiler value;
};
struct xgr_compiled_grammar {
  CompiledGrammar value;
};
struct xgr_matcher {
  GrammarMatcher value;
};

namespace {

char* CopyString(const std::string& s) {
  char* out = static_cast<char*>(std::malloc(s.size() + 1));
  if (out != nullptr) {
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
  }
  return out;
}

void SetError(char** err, const char* message) {
  if (err != nullptr) *err = CopyString(message);
}

template <typename R, typename F>
R Guard(char** err, R on_error, F f) {
  try {
    return f();
  } catch (const std::exception& e) {
    SetError(err, e.what());
  } catch (...) {
    SetError(err, "unknown C++ exception");
  }
  return on_error;
}

std::vector<std::string> ToVocab(const char* const* tokens, const size_t* lens,
                                 size_t n) {
  std::vector<std::string> vocab;
  vocab.reserve(n);
  for (size_t i = 0; i < n; ++i) vocab.emplace_back(tokens[i], lens[i]);
  return vocab;
}

std::optional<std::vector<int32_t>> ToOptionalIds(const int32_t* ids, size_t n,
                                                  bool has_ids) {
  if (!has_ids) return std::nullopt;
  return std::vector<int32_t>(ids, ids + n);
}

xgr_compiled_grammar* Wrap(CompiledGrammar cg) {
  return new xgr_compiled_grammar{std::move(cg)};
}

}  // namespace

extern "C" {

void xgr_string_free(char* s) { std::free(s); }

int32_t xgr_get_bitmask_size(int32_t vocab_size) {
  return GetBitmaskSize(vocab_size);
}

// ---- TokenizerInfo ----

xgr_tokenizer_info* xgr_tokenizer_info_new(
    const char* const* tokens, const size_t* lens, size_t num_tokens,
    int32_t vocab_type, int32_t vocab_size, bool has_vocab_size,
    const int32_t* stop_token_ids, size_t num_stop_token_ids,
    bool has_stop_token_ids, bool add_prefix_space, char** err) {
  return Guard(err, static_cast<xgr_tokenizer_info*>(nullptr), [&] {
    if (vocab_type < 0 || vocab_type > 2) {
      throw std::runtime_error("Invalid vocab type: " +
                               std::to_string(vocab_type));
    }
    return new xgr_tokenizer_info{TokenizerInfo(
        ToVocab(tokens, lens, num_tokens), static_cast<VocabType>(vocab_type),
        has_vocab_size ? std::optional<int>(vocab_size) : std::nullopt,
        ToOptionalIds(stop_token_ids, num_stop_token_ids, has_stop_token_ids),
        add_prefix_space)};
  });
}

xgr_tokenizer_info* xgr_tokenizer_info_from_vocab_and_metadata(
    const char* const* tokens, const size_t* lens, size_t num_tokens,
    const char* metadata, size_t metadata_len, char** err) {
  return Guard(err, static_cast<xgr_tokenizer_info*>(nullptr), [&] {
    return new xgr_tokenizer_info{TokenizerInfo::FromVocabAndMetadata(
        ToVocab(tokens, lens, num_tokens),
        std::string(metadata, metadata_len))};
  });
}

char* xgr_detect_metadata_from_hf(const char* backend_str, size_t len,
                                  char** err) {
  return Guard(err, static_cast<char*>(nullptr), [&] {
    return CopyString(
        TokenizerInfo::DetectMetadataFromHF(std::string(backend_str, len)));
  });
}

void xgr_tokenizer_info_free(xgr_tokenizer_info* p) { delete p; }

int32_t xgr_tokenizer_info_vocab_size(const xgr_tokenizer_info* p) {
  return p->value.GetVocabSize();
}

int32_t xgr_tokenizer_info_vocab_type(const xgr_tokenizer_info* p) {
  return static_cast<int32_t>(p->value.GetVocabType());
}

bool xgr_tokenizer_info_add_prefix_space(const xgr_tokenizer_info* p) {
  return p->value.GetAddPrefixSpace();
}

// Returns a pointer into `p`, valid while `p` lives.
const int32_t* xgr_tokenizer_info_stop_token_ids(const xgr_tokenizer_info* p,
                                                 size_t* len) {
  const auto& ids = p->value.GetStopTokenIds();
  *len = ids.size();
  return ids.data();
}

// Returns a pointer into `p`, valid while `p` lives.
const int32_t* xgr_tokenizer_info_special_token_ids(const xgr_tokenizer_info* p,
                                                    size_t* len) {
  const auto& ids = p->value.GetSpecialTokenIds();
  *len = ids.size();
  return ids.data();
}

char* xgr_tokenizer_info_dump_metadata(const xgr_tokenizer_info* p,
                                       char** err) {
  return Guard(err, static_cast<char*>(nullptr),
               [&] { return CopyString(p->value.DumpMetadata()); });
}

char* xgr_tokenizer_info_serialize_json(const xgr_tokenizer_info* p,
                                        char** err) {
  return Guard(err, static_cast<char*>(nullptr),
               [&] { return CopyString(p->value.SerializeJSON()); });
}

// ---- GrammarCompiler ----

xgr_compiler* xgr_compiler_new(const xgr_tokenizer_info* info,
                               int32_t max_threads, bool cache_enabled,
                               int64_t cache_limit_bytes, char** err) {
  return Guard(err, static_cast<xgr_compiler*>(nullptr), [&] {
    return new xgr_compiler{GrammarCompiler(info->value, max_threads,
                                            cache_enabled, cache_limit_bytes)};
  });
}

void xgr_compiler_free(xgr_compiler* p) { delete p; }

int64_t xgr_compiler_cache_size_bytes(const xgr_compiler* p) {
  return p->value.GetCacheSizeBytes();
}

xgr_compiled_grammar* xgr_compile_json_schema(
    xgr_compiler* c, const char* schema, size_t schema_len, bool any_whitespace,
    int32_t indent, bool has_indent, const char* item_separator,
    size_t item_separator_len, const char* key_separator,
    size_t key_separator_len, bool has_separators, bool strict_mode,
    int32_t max_whitespace_cnt, bool has_max_whitespace_cnt, bool any_order,
    char** err) {
  return Guard(err, static_cast<xgr_compiled_grammar*>(nullptr), [&] {
    std::optional<std::pair<std::string, std::string>> separators;
    if (has_separators) {
      separators =
          std::make_pair(std::string(item_separator, item_separator_len),
                         std::string(key_separator, key_separator_len));
    }
    return Wrap(c->value.CompileJSONSchema(
        std::string(schema, schema_len), any_whitespace,
        has_indent ? std::optional<int>(indent) : std::nullopt, separators,
        strict_mode,
        has_max_whitespace_cnt ? std::optional<int>(max_whitespace_cnt)
                               : std::nullopt,
        any_order));
  });
}

xgr_compiled_grammar* xgr_compile_builtin_json(xgr_compiler* c, char** err) {
  return Guard(err, static_cast<xgr_compiled_grammar*>(nullptr),
               [&] { return Wrap(c->value.CompileBuiltinJSONGrammar()); });
}

xgr_compiled_grammar* xgr_compile_ebnf(xgr_compiler* c, const char* ebnf,
                                       size_t ebnf_len,
                                       const char* root_rule_name,
                                       size_t root_rule_name_len, char** err) {
  return Guard(err, static_cast<xgr_compiled_grammar*>(nullptr), [&] {
    return Wrap(c->value.CompileGrammar(
        std::string(ebnf, ebnf_len),
        std::string(root_rule_name, root_rule_name_len)));
  });
}

xgr_compiled_grammar* xgr_compile_lark(xgr_compiler* c, const char* lark,
                                       size_t lark_len, char** err) {
  return Guard(err, static_cast<xgr_compiled_grammar*>(nullptr), [&] {
    return Wrap(c->value.CompileLark(std::string(lark, lark_len)));
  });
}

xgr_compiled_grammar* xgr_compile_regex(xgr_compiler* c, const char* regex,
                                        size_t regex_len, char** err) {
  return Guard(err, static_cast<xgr_compiled_grammar*>(nullptr), [&] {
    return Wrap(c->value.CompileRegex(std::string(regex, regex_len)));
  });
}

xgr_compiled_grammar* xgr_compile_structural_tag(xgr_compiler* c,
                                                 const char* tag_json,
                                                 size_t tag_json_len,
                                                 char** err) {
  return Guard(err, static_cast<xgr_compiled_grammar*>(nullptr), [&] {
    return Wrap(
        c->value.CompileStructuralTag(std::string(tag_json, tag_json_len)));
  });
}

void xgr_compiled_grammar_free(xgr_compiled_grammar* p) { delete p; }

int64_t xgr_compiled_grammar_memory_size_bytes(const xgr_compiled_grammar* p) {
  return static_cast<int64_t>(p->value.MemorySizeBytes());
}

// ---- GrammarMatcher ----

xgr_matcher* xgr_matcher_new(const xgr_compiled_grammar* cg,
                             const int32_t* override_stop_tokens,
                             size_t num_override_stop_tokens,
                             bool has_override_stop_tokens,
                             bool terminate_without_stop_token,
                             int32_t max_rollback_tokens, char** err) {
  return Guard(err, static_cast<xgr_matcher*>(nullptr), [&] {
    std::optional<std::vector<int>> stop;
    if (has_override_stop_tokens) {
      stop = std::vector<int>(override_stop_tokens,
                              override_stop_tokens + num_override_stop_tokens);
    }
    return new xgr_matcher{GrammarMatcher(
        cg->value, stop, terminate_without_stop_token, max_rollback_tokens)};
  });
}

void xgr_matcher_free(xgr_matcher* p) { delete p; }

// Returns 1 if accepted, 0 if rejected, -1 on error.
int32_t xgr_matcher_accept_token(xgr_matcher* m, int32_t token_id, char** err) {
  return Guard(err, int32_t{-1}, [&] {
    return static_cast<int32_t>(m->value.AcceptToken(token_id));
  });
}

// Returns 1 if accepted, 0 if rejected, -1 on error.
int32_t xgr_matcher_accept_string(xgr_matcher* m, const char* s, size_t len,
                                  char** err) {
  return Guard(err, int32_t{-1}, [&] {
    return static_cast<int32_t>(m->value.AcceptString(std::string(s, len)));
  });
}

// Fills one bitmask row of `len` int32 words. Returns 1 if the mask needs to
// be applied (not all-true), 0 otherwise, -1 on error.
int32_t xgr_matcher_fill_next_token_bitmask(xgr_matcher* m, int32_t* row,
                                            int64_t len, char** err) {
  return Guard(err, int32_t{-1}, [&] {
    int64_t shape[1] = {len};
    DLTensor tensor{
        row, DLDevice{kDLCPU, 0}, 1, GetBitmaskDLType(), shape, nullptr, 0};
    return static_cast<int32_t>(m->value.FillNextTokenBitmask(&tensor, 0));
  });
}

// Returns 0 on success, -1 on error.
int32_t xgr_matcher_rollback(xgr_matcher* m, int32_t num_tokens, char** err) {
  return Guard(err, int32_t{-1}, [&] {
    m->value.Rollback(num_tokens);
    return int32_t{0};
  });
}

bool xgr_matcher_is_terminated(const xgr_matcher* m) {
  return m->value.IsTerminated();
}

bool xgr_matcher_is_completed(const xgr_matcher* m) {
  return m->value.IsCompleted();
}

void xgr_matcher_reset(xgr_matcher* m) { m->value.Reset(); }

char* xgr_matcher_find_jump_forward_string(xgr_matcher* m, char** err) {
  return Guard(err, static_cast<char*>(nullptr),
               [&] { return CopyString(m->value.FindJumpForwardString()); });
}

}  // extern "C"
