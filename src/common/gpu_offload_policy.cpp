#include "common/gpu_offload_policy.h"

#include <climits>

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

    int parsed = 0;
    for (const char * p = idle_env; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            out.reason = "invalid value, disabled";
            return out;
        }

        const int digit = *p - '0';
        if (parsed > (INT_MAX - digit) / 10) {
            out.reason = "invalid value, disabled";
            return out;
        }
        parsed = parsed * 10 + digit;
    }

    if (parsed == 0) {
        out.reason = "disabled";
        return out;
    }

    out.enabled = true;
    out.idle_secs = parsed;
    out.reason = "enabled";
    return out;
}

} // namespace qwen3_tts
