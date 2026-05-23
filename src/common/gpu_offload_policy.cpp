#include "common/gpu_offload_policy.h"

#include <cerrno>
#include <climits>
#include <cstdlib>

namespace qwen3_tts {

gpu_offload_policy parse_gpu_offload_policy(const char * idle_env, bool low_mem_enabled) {
    gpu_offload_policy out;
    if (low_mem_enabled) {
        out.reason = "disabled because QWEN3_TTS_LOW_MEM is enabled";
        return out;
    }
    if (!idle_env || idle_env[0] == '\0') {
        out.reason = "unset";
        return out;
    }

    errno = 0;
    char * end = nullptr;
    long parsed = std::strtol(idle_env, &end, 10);
    if (errno != 0 || end == idle_env || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        out.reason = "invalid value, disabled";
        return out;
    }
    if (parsed == 0) {
        out.reason = "disabled";
        return out;
    }

    out.enabled = true;
    out.idle_secs = (int) parsed;
    out.reason = "enabled";
    return out;
}

} // namespace qwen3_tts
