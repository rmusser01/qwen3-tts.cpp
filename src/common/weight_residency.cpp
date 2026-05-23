#include "common/weight_residency.h"

#include <cctype>
#include <limits>
#include <utility>
#include <vector>

namespace qwen3_tts {

namespace {

struct tensor_allocation_state {
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    void * data = nullptr;
    void * extra = nullptr;
};

bool iequals(const char * a, const char * b) {
    if (!a || !b) {
        return false;
    }
    while (*a && *b) {
        if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

std::vector<tensor_allocation_state> detach_context_tensors(ggml_context * ctx) {
    std::vector<tensor_allocation_state> states;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx);
         tensor;
         tensor = ggml_get_next_tensor(ctx, tensor)) {
        states.push_back({tensor, tensor->buffer, tensor->data, tensor->extra});
        tensor->buffer = nullptr;
        tensor->data = nullptr;
        tensor->extra = nullptr;
    }
    return states;
}

void restore_context_tensors(const std::vector<tensor_allocation_state> & states) {
    for (const tensor_allocation_state & state : states) {
        state.tensor->buffer = state.buffer;
        state.tensor->data = state.data;
        state.tensor->extra = state.extra;
    }
}

} // namespace

void host_tensor_store::clear() {
    tensors.clear();
    total_bytes = 0;
}

bool backend_is_cuda_or_vulkan(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return iequals(reg_name, "CUDA") || iequals(reg_name, "Vulkan");
}

bool download_tensors_to_host(const std::map<std::string, ggml_tensor *> & tensors,
                              host_tensor_store & out,
                              std::string & error) {
    error.clear();
    out.clear();

    for (const auto & entry : tensors) {
        if (!entry.second) {
            error = "Cannot download null tensor: " + entry.first;
            out.clear();
            return false;
        }
    }

    out.tensors.reserve(tensors.size());
    for (const auto & entry : tensors) {
        ggml_tensor * tensor = entry.second;
        const size_t nbytes = ggml_nbytes(tensor);

        host_tensor_copy copy;
        copy.name = entry.first;
        copy.bytes.resize(nbytes);
        if (nbytes > 0) {
            ggml_backend_tensor_get(tensor, copy.bytes.data(), 0, nbytes);
        }

        if (nbytes > std::numeric_limits<size_t>::max() - out.total_bytes) {
            error = "Host tensor store byte count overflow";
            out.clear();
            return false;
        }
        out.total_bytes += nbytes;
        out.tensors.push_back(std::move(copy));
    }

    return true;
}

bool upload_tensors_from_host(ggml_context * ctx,
                              const std::map<std::string, ggml_tensor *> & tensors,
                              ggml_backend_t backend,
                              const host_tensor_store & store,
                              ggml_backend_buffer_t & buffer,
                              std::string & error) {
    error.clear();

    if (buffer) {
        error = "Cannot upload tensors: destination buffer must be null";
        return false;
    }
    if (!ctx) {
        error = "Cannot upload tensors: ggml context is null";
        return false;
    }
    if (!backend) {
        error = "Cannot upload tensors: backend is null";
        return false;
    }

    for (const host_tensor_copy & copy : store.tensors) {
        auto it = tensors.find(copy.name);
        if (it == tensors.end()) {
            error = "Cannot upload tensor not present in destination map: " + copy.name;
            return false;
        }
        if (!it->second) {
            error = "Cannot upload to null tensor: " + copy.name;
            return false;
        }

        const size_t expected = ggml_nbytes(it->second);
        if (copy.bytes.size() != expected) {
            error = "Cannot upload tensor with mismatched byte size: " + copy.name;
            return false;
        }
    }

    std::vector<tensor_allocation_state> old_states = detach_context_tensors(ctx);
    ggml_backend_buffer_t new_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!new_buffer) {
        restore_context_tensors(old_states);
        error = "Failed to allocate backend tensor buffer";
        return false;
    }

    for (const host_tensor_copy & copy : store.tensors) {
        ggml_tensor * tensor = tensors.find(copy.name)->second;
        if (!copy.bytes.empty()) {
            ggml_backend_tensor_set(tensor, copy.bytes.data(), 0, copy.bytes.size());
        }
    }

    buffer = new_buffer;
    return true;
}

} // namespace qwen3_tts
