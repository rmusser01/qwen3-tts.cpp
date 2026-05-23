#include "pipeline/qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

static void set_test_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main() {
    set_test_env("QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS", "1");
    set_test_env("QWEN3_TTS_BACKEND", "cpu");
    set_test_env("QWEN3_TTS_LOW_MEM", "0");

    qwen3_tts::Qwen3TTS tts;
    if (!tts.load_models("models")) {
        std::fprintf(stderr, "SKIP: models unavailable: %s\n", tts.get_error().c_str());
        return 0;
    }

    const auto names = tts.get_speaker_names();

    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    if (qwen3_tts::Qwen3TTSDiagnostics::transformer_ram_offloaded(tts)) {
        std::fprintf(stderr, "FAIL: CPU backend production path offloaded transformer\n");
        return 1;
    }

    qwen3_tts::tts_params params;
    // Keep this short, but above the vocoder's reflect-padding minimum.
    params.max_audio_tokens = 8;
    auto result = tts.synthesize("test", params);
    if (!result.success) {
        std::fprintf(stderr, "FAIL: synthesize failed: %s\n", result.error_msg.c_str());
        return 1;
    }

    if (!names.empty()) {
        std::string err;
        if (!qwen3_tts::Qwen3TTSDiagnostics::force_transformer_offload(tts, err)) {
            std::fprintf(stderr, "FAIL: forced transformer offload failed: %s\n", err.c_str());
            return 1;
        }
        if (!qwen3_tts::Qwen3TTSDiagnostics::transformer_ram_offloaded(tts)) {
            std::fprintf(stderr, "FAIL: transformer did not report RAM-offloaded\n");
            return 1;
        }

        std::vector<float> embedding;
        if (!tts.get_speaker_embedding(names[0], embedding)) {
            std::fprintf(stderr, "FAIL: get_speaker_embedding failed after forced offload: %s\n",
                         tts.get_error().c_str());
            return 1;
        }
        if (embedding.empty()) {
            std::fprintf(stderr, "FAIL: get_speaker_embedding returned an empty embedding\n");
            return 1;
        }
        if (qwen3_tts::Qwen3TTSDiagnostics::transformer_ram_offloaded(tts)) {
            std::fprintf(stderr, "FAIL: get_speaker_embedding did not reload transformer\n");
            return 1;
        }
    }

    std::string err;
    if (!qwen3_tts::Qwen3TTSDiagnostics::force_idle_offload_once(tts, err)) {
        std::fprintf(stderr, "FAIL: forced idle offload failed: %s\n", err.c_str());
        return 1;
    }
    if (!qwen3_tts::Qwen3TTSDiagnostics::transformer_ram_offloaded(tts)) {
        std::fprintf(stderr, "FAIL: forced idle offload did not offload transformer\n");
        return 1;
    }

    result = tts.synthesize("test", params);
    if (!result.success) {
        std::fprintf(stderr, "FAIL: synthesize failed after forced idle offload: %s\n",
                     result.error_msg.c_str());
        return 1;
    }
    if (qwen3_tts::Qwen3TTSDiagnostics::transformer_ram_offloaded(tts)) {
        std::fprintf(stderr, "FAIL: guarded synthesize did not reload transformer after idle offload\n");
        return 1;
    }

    if (!tts.load_models("models")) {
        std::fprintf(stderr, "FAIL: reload after forced idle offload failed: %s\n",
                     tts.get_error().c_str());
        return 1;
    }

    std::printf("pipeline offload lifecycle test passed\n");
    return 0;
}
