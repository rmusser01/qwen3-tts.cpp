#include "common/gpu_offload_policy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

static int expect_policy(const char * env_value, bool low_mem,
                         bool expected_enabled, int expected_secs,
                         const char * expected_reason_substr) {
    qwen3_tts::gpu_offload_policy p =
        qwen3_tts::parse_gpu_offload_policy(env_value, low_mem);
    if (p.enabled != expected_enabled) return fail("enabled mismatch");
    if (p.idle_secs != expected_secs) return fail("idle_secs mismatch");
    if (expected_reason_substr &&
        p.reason.find(expected_reason_substr) == std::string::npos) {
        return fail("reason mismatch");
    }
    return 0;
}

int main() {
    if (expect_policy(nullptr, false, false, 0, "unset") != 0) return 1;
    if (expect_policy("", false, false, 0, "unset") != 0) return 1;
    if (expect_policy("0", false, false, 0, "disabled") != 0) return 1;
    if (expect_policy("15", false, true, 15, "enabled") != 0) return 1;
    if (expect_policy("-2", false, false, 0, "invalid") != 0) return 1;
    if (expect_policy("abc", false, false, 0, "invalid") != 0) return 1;
    if (expect_policy("15", true, false, 0, "QWEN3_TTS_LOW_MEM") != 0) return 1;
    std::printf("gpu_offload_policy tests passed\n");
    return 0;
}
