#include "pipeline/qwen3_tts.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

bool equals_ignore_case(const std::string & a, const char * b) {
    if (!b) {
        return false;
    }
    size_t i = 0;
    for (; i < a.size() && b[i] != '\0'; ++i) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca = (char) (ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char) (cb - 'A' + 'a');
        if (ca != cb) {
            return false;
        }
    }
    return i == a.size() && b[i] == '\0';
}

bool env_truthy(const char * name) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    std::string text(value);
    return equals_ignore_case(text, "1") ||
           equals_ignore_case(text, "true") ||
           equals_ignore_case(text, "yes") ||
           equals_ignore_case(text, "on");
}

void set_env_if_unset(const char * name, const char * value) {
    const char * current = std::getenv(name);
    if (current && current[0] != '\0') {
        return;
    }
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int parse_positive_int_env(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 3600) {
        return -1;
    }
    return (int) parsed;
}

std::string model_dir_from_args(int argc, char ** argv) {
    if (argc > 1 && argv[1] && argv[1][0] != '\0') {
        return argv[1];
    }
    const char * env_model_dir = std::getenv("QWEN3_TTS_MODEL_DIR");
    if (env_model_dir && env_model_dir[0] != '\0') {
        return env_model_dir;
    }
    return "models";
}

} // namespace

int main(int argc, char ** argv) {
    if (!env_truthy("QWEN3_TTS_RUN_GPU_OFFLOAD_VALIDATION")) {
        std::printf("SKIP: set QWEN3_TTS_RUN_GPU_OFFLOAD_VALIDATION=1 to run the CUDA/Vulkan idle offload validation\n");
        return 0;
    }

    set_env_if_unset("QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS", "1");
    set_env_if_unset("QWEN3_TTS_LOW_MEM", "0");

    if (env_truthy("QWEN3_TTS_LOW_MEM")) {
        std::fprintf(stderr, "FAIL: QWEN3_TTS_LOW_MEM disables idle GPU RAM offload\n");
        return 1;
    }

    const char * backend = std::getenv("QWEN3_TTS_BACKEND");
    if (backend && equals_ignore_case(backend, "cpu")) {
        std::fprintf(stderr, "FAIL: QWEN3_TTS_BACKEND=cpu cannot validate CUDA/Vulkan idle offload\n");
        return 1;
    }

    const int idle_secs = parse_positive_int_env("QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS", 1);
    if (idle_secs <= 0) {
        std::fprintf(stderr, "FAIL: QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS must be a positive integer for validation\n");
        return 1;
    }

    qwen3_tts::Qwen3TTS tts;
    const std::string model_dir = model_dir_from_args(argc, argv);
    if (!tts.load_models(model_dir)) {
        std::fprintf(stderr, "FAIL: models unavailable in %s: %s\n",
                     model_dir.c_str(), tts.get_error().c_str());
        return 1;
    }

    qwen3_tts::tts_params params;
    params.max_audio_tokens = 8;
    params.temperature = 0.0f;
    params.seed = 1;
    params.print_timing = false;

    auto first = tts.synthesize("gpu idle offload validation", params);
    if (!first.success) {
        std::fprintf(stderr, "FAIL: initial synthesis failed: %s\n", first.error_msg.c_str());
        return 1;
    }
    if (first.audio.empty()) {
        std::fprintf(stderr, "FAIL: initial synthesis produced empty audio\n");
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(idle_secs * 1000 + 750));
    if (!qwen3_tts::Qwen3TTSDiagnostics::transformer_ram_offloaded(tts)) {
        std::fprintf(stderr, "FAIL: transformer did not RAM-offload after idle timeout; verify CUDA/Vulkan was selected\n");
        return 1;
    }
    if (!qwen3_tts::Qwen3TTSDiagnostics::decoder_ram_offloaded(tts)) {
        std::fprintf(stderr, "FAIL: decoder did not RAM-offload after idle timeout; verify CUDA/Vulkan was selected\n");
        return 1;
    }

    auto second = tts.synthesize("gpu idle offload reload validation", params);
    if (!second.success) {
        std::fprintf(stderr, "FAIL: synthesis after RAM offload failed: %s\n", second.error_msg.c_str());
        return 1;
    }
    if (second.audio.empty()) {
        std::fprintf(stderr, "FAIL: synthesis after RAM offload produced empty audio\n");
        return 1;
    }
    if (qwen3_tts::Qwen3TTSDiagnostics::transformer_ram_offloaded(tts) ||
        qwen3_tts::Qwen3TTSDiagnostics::decoder_ram_offloaded(tts)) {
        std::fprintf(stderr, "FAIL: synthesis after idle offload did not reload all RAM-resident weights\n");
        return 1;
    }

    std::printf("gpu idle offload validation passed\n");
    return 0;
}
