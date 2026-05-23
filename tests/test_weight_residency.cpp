#include "common/weight_residency.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

static int run_backend_roundtrip(enum ggml_backend_dev_type type) {
    ggml_backend_t backend = ggml_backend_init_by_type(type, nullptr);
    if (!backend) return 0; // optional backend unavailable

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return fail("ggml_init failed");

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, "test.weight");
    std::map<std::string, ggml_tensor *> tensors;
    tensors["test.weight"] = t;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return fail("alloc buffer failed");

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ggml_backend_tensor_set(t, input, 0, sizeof(input));

    qwen3_tts::host_tensor_store store;
    std::string error;
    if (!qwen3_tts::download_tensors_to_host(tensors, store, error)) {
        return fail(error.c_str());
    }

    ggml_backend_buffer_free(buffer);
    buffer = nullptr;

    if (!qwen3_tts::upload_tensors_from_host(ctx, tensors, backend, store, buffer, error)) {
        return fail(error.c_str());
    }

    float output[4] = {};
    ggml_backend_tensor_get(t, output, 0, sizeof(output));
    for (int i = 0; i < 4; ++i) {
        if (output[i] != input[i]) return fail("roundtrip mismatch");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}

int main() {
    if (run_backend_roundtrip(GGML_BACKEND_DEVICE_TYPE_CPU) != 0) return 1;
    (void) run_backend_roundtrip(GGML_BACKEND_DEVICE_TYPE_GPU);
    std::printf("weight_residency tests passed\n");
    return 0;
}
