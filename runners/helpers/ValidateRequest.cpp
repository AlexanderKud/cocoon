#include "ValidateRequest.h"
#include "errorcode.h"
#include "td/utils/JsonBuilder.h"
#include "td/utils/buffer.h"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <set>

namespace cocoon {

td::Result<td::BufferSlice> validate_modify_completions_request(td::BufferSlice request, std::string *model_ptr,
                                                                td::int64 *max_completion_tokens_ptr) {
  auto S = request.as_slice();
  auto b = nlohmann::json::parse(S.begin(), S.end(), nullptr, false, false);

  if (b.is_discarded()) {
    return td::Status::Error(ton::ErrorCode::protoviolation, "expected json object");
  }
  if (!b.is_object()) {
    return td::Status::Error(ton::ErrorCode::protoviolation, "expected json object");
  }

  std::string model;
  td::int64 max_completion_tokens = 0;
  bool has_stream_options = false;
  bool has_stream = false;
  bool stream = false;
  bool has_max_tokens = false;

  std::set<td::Slice> processed_fields;

  for (auto &[name, value] : b.items()) {
    if (processed_fields.count(name) > 0) {
      return td::Status::Error(ton::ErrorCode::protoviolation, PSTRING() << "duplicate '" << name << "' field");
    }
    processed_fields.insert(name);
    if (name == "messages") {
      if (!value.is_array()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "messages must be an array");
      }

      // check?
    } else if (name == "model") {
      if (!value.is_string()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "model must be a string");
      }
      model = value.get<std::string>();
    } else if (false && name == "audio") {
    } else if (name == "frequency_penalty") {
      if (!value.is_number()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "frequency_penalty must be a number");
      }
      double v = value.get<double>();
      if (v < -2 || v > 2) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "frequency_penalty must be between -2.0 and 2.0");
      }
    } else if (false && name == "audio") {
    } else if (false && name == "logit_bias") {
    } else if (false && name == "logprobs") {
    } else if (name == "max_completion_tokens") {
      if (!value.is_number_integer()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "max_completion_tokens must be a number");
      }
      max_completion_tokens = value.get<td::int64>();
      if (max_completion_tokens < 0) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "max_completion_tokens must be non-negative");
      }
      if (max_completion_tokens_ptr && max_completion_tokens > *max_completion_tokens_ptr) {
        value = *max_completion_tokens_ptr;
      }
      has_max_tokens = true;
    } else if (name == "max_tokens") {
      if (!value.is_number_integer()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "max_completion_tokens must be a number");
      }
      max_completion_tokens = value.get<td::int64>();
      if (max_completion_tokens < 0) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "max_completion_tokens must be non-negative");
      }
      if (max_completion_tokens_ptr && max_completion_tokens > *max_completion_tokens_ptr) {
        value = *max_completion_tokens_ptr;
      }
      has_max_tokens = true;
    } else if (false && name == "metadata") {
    } else if (false && name == "modalities") {
    } else if (name == "n") {
      if (!value.is_number_integer()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "n must be a number");
      }
      auto n = value.get<td::int64>();
      if (n < 1) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "n must be positive");
      }
    } else if (name == "parallel_tool_calls") {
      if (!value.is_boolean()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "parallel_tool_calls must be a boolean");
      }
    } else if (name == "prediction") {
      if (!value.is_number()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "prediction must be a boolean");
      }
    } else if (name == "presence_penalty") {
      if (!value.is_number()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "presence_penalty must be a number");
      }
    } else if (false && name == "prompt_cache_key") {
    } else if (name == "reasoning_effort") {
      if (!value.is_string()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "reasoning_effort must be a string");
      }
    } else if (name == "response_format") {
      // do not validate
    } else if (false && name == "safety_identifier") {
    } else if (false && name == "service_tier") {
    } else if (name == "stop") {
      // allow for now, but since it's deprecated
    } else if (false && name == "store") {
    } else if (name == "stream") {
      if (!value.is_boolean()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "stream must be a boolean");
      }
      has_stream = true;
      stream = value.get<bool>();
    } else if (name == "stream_options") {
      if (!value.is_object()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "stream_options must be an object");
      }
      has_stream_options = true;
      value["include_usage"] = true;
      for (auto &[k, v] : value.items()) {
        if (k == "include_obfuscation") {
          if (!v.is_boolean()) {
            return td::Status::Error(ton::ErrorCode::protoviolation,
                                     "stream_options.include_obfuscation must be a boolean");
          }
        } else if (k == "include_usage") {
          if (!v.is_boolean()) {
            return td::Status::Error(ton::ErrorCode::protoviolation, "stream_options.include_usage must be a boolean");
          }
        } else {
          return td::Status::Error(ton::ErrorCode::protoviolation,
                                   PSTRING() << "unknown option '" << k << "' in stream_options");
        }
      }
    } else if (name == "temperature") {
      if (!value.is_number()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "temperature must be a number");
      }
    } else if (false && name == "tool_choice") {
    } else if (false && name == "tools") {
    } else if (name == "top_logprobs") {
      if (!value.is_number_unsigned()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "top_logprobs must be a number");
      }
    } else if (name == "top_p") {
      if (!value.is_number()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "top_p must be a number");
      }
    } else if (name == "verbosity") {
      if (!value.is_string()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "verbosity must be a string");
      }
    } else if (false && name == "web_search_options") {
    } else if (name == "chat_template_kwargs") {
      if (!value.is_object()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "chat_template_kwargs must be an object");
      }
      for (auto &[i_name, i_value] : value.items()) {
        if (i_name == "enable_thinking") {
          if (!i_value.is_boolean()) {
            return td::Status::Error(ton::ErrorCode::protoviolation,
                                     "chat_template_kwargs.enable_thinking must be a boolean");
          }
        } else {
          return td::Status::Error(ton::ErrorCode::protoviolation, PSTRING() << "unknown suboption '" << i_name
                                                                             << "' in chat_template_kwargs in request");
        }
      }

    } else {
      return td::Status::Error(ton::ErrorCode::protoviolation, PSTRING()
                                                                   << "unknown option '" << name << "' in request");
    }
  }

  if (processed_fields.count("messages") == 0) {
    return td::Status::Error(ton::ErrorCode::protoviolation, PSTRING() << "missing required field 'messages'");
  }
  if (processed_fields.count("model") == 0) {
    return td::Status::Error(ton::ErrorCode::protoviolation, PSTRING() << "missing required field 'model'");
  }

  if (!has_stream && has_stream_options) {
    b["stream"] = true;
    has_stream = true;
    stream = true;
  }

  if (stream && !has_stream_options) {
    b["stream_options"]["include_usage"] = true;
    has_stream_options = true;
  }

  if (!has_max_tokens && max_completion_tokens_ptr) {
    b["max_completion_tokens"] = *max_completion_tokens_ptr;
  }

  if (model_ptr) {
    *model_ptr = model;
  }

  if (max_completion_tokens_ptr) {
    *max_completion_tokens_ptr = max_completion_tokens;
  }

  return td::BufferSlice(b.dump());
}

td::Result<td::BufferSlice> validate_modify_request(std::string url, td::BufferSlice request, std::string *model,
                                                    td::int64 *max_tokens) {
  auto p = url.find('/');
  if (p != std::string::npos) {
    url = url.substr(p);
  }

  if (url == "/v1/chat/completions") {
    return validate_modify_completions_request(std::move(request), model, max_tokens);
  } else {
    return td::Status::Error(ton::ErrorCode::protoviolation, "unsupported method");
  }
}

}  // namespace cocoon
