#pragma once

#include <string>

namespace qwen3_tts {

struct gpu_offload_policy {
    bool enabled = false;
    int idle_secs = 0;
    std::string reason;
};

gpu_offload_policy parse_gpu_offload_policy(const char * idle_env, bool low_mem_enabled);

} // namespace qwen3_tts
