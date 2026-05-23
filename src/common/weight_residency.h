#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace qwen3_tts {

enum class weight_residency {
    Unloaded,
    GpuResident,
    RamResident,
};

struct host_tensor_copy {
    std::string name;
    std::vector<uint8_t> bytes;
};

struct host_tensor_store {
    std::vector<host_tensor_copy> tensors;
    size_t total_bytes = 0;
    void clear();
};

bool backend_is_cuda_or_vulkan(ggml_backend_t backend);
bool download_tensors_to_host(const std::map<std::string, ggml_tensor *> & tensors,
                              host_tensor_store & out,
                              std::string & error);

// Allocates a new backend buffer for all tensors in ctx and uploads stored
// tensor bytes into it. Tensor maps must be keyed by each tensor's GGML
// metadata name.
// Precondition: buffer must be nullptr on entry; callers own and must free any
// previous backend buffer before reloading weights. Destination tensors must
// belong to ctx. Destination names must map one-to-one to distinct tensor
// pointers. The host store must contain exactly one copy for every destination
// tensor in tensors, and tensors must cover every tensor in ctx.
bool upload_tensors_from_host(ggml_context * ctx,
                              const std::map<std::string, ggml_tensor *> & tensors,
                              ggml_backend_t backend,
                              const host_tensor_store & store,
                              ggml_backend_buffer_t & buffer,
                              std::string & error);

} // namespace qwen3_tts
