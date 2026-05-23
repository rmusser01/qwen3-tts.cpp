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

static int run_download_rejects_null_tensor() {
    qwen3_tts::host_tensor_store store;
    qwen3_tts::host_tensor_copy existing;
    existing.name = "old.weight";
    existing.bytes.push_back(42);
    store.tensors.push_back(existing);
    store.total_bytes = 1;

    std::map<std::string, ggml_tensor *> tensors;
    tensors["null.weight"] = nullptr;

    std::string error;
    if (qwen3_tts::download_tensors_to_host(tensors, store, error)) {
        return fail("download accepted null tensor");
    }
    if (!store.tensors.empty()) return fail("null tensor failure left tensor copies");
    if (store.total_bytes != 0) return fail("null tensor failure left total bytes");
    return 0;
}

static int run_upload_rejects_missing_destination() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return fail("backend init failed");

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return fail("ggml_init failed");
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, "test.weight");
    std::map<std::string, ggml_tensor *> tensors;
    tensors["test.weight"] = t;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return fail("alloc buffer failed");
    }

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ggml_backend_tensor_set(t, input, 0, sizeof(input));

    qwen3_tts::host_tensor_store store;
    std::string error;
    int result = 0;
    if (!qwen3_tts::download_tensors_to_host(tensors, store, error)) {
        result = fail(error.c_str());
    } else {
        std::map<std::string, ggml_tensor *> missing;
        ggml_backend_buffer_t upload_buffer = nullptr;
        if (qwen3_tts::upload_tensors_from_host(ctx, missing, backend, store, upload_buffer, error)) {
            result = fail("upload accepted missing destination tensor");
        } else if (upload_buffer != nullptr) {
            result = fail("missing destination failure changed buffer");
        }
        if (upload_buffer) ggml_backend_buffer_free(upload_buffer);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

static int run_upload_rejects_non_null_buffer() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return fail("backend init failed");

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return fail("ggml_init failed");
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(t, "test.weight");
    std::map<std::string, ggml_tensor *> tensors;
    tensors["test.weight"] = t;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return fail("alloc buffer failed");
    }

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ggml_backend_tensor_set(t, input, 0, sizeof(input));

    qwen3_tts::host_tensor_store store;
    std::string error;
    int result = 0;
    if (!qwen3_tts::download_tensors_to_host(tensors, store, error)) {
        result = fail(error.c_str());
    } else {
        ggml_backend_buffer_t original_buffer = buffer;
        if (qwen3_tts::upload_tensors_from_host(ctx, tensors, backend, store, buffer, error)) {
            result = fail("upload accepted non-null buffer");
        } else if (buffer != original_buffer) {
            result = fail("non-null buffer failure changed buffer");
        }

        if (buffer != original_buffer && buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = original_buffer;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

static int run_backend_roundtrip(enum ggml_backend_dev_type type, bool optional) {
    ggml_backend_t backend = ggml_backend_init_by_type(type, nullptr);
    if (!backend) {
        return optional ? 0 : fail("backend init failed");
    }

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
    if (store.total_bytes != sizeof(input)) return fail("host store total bytes mismatch");
    if (store.tensors.size() != 1) return fail("host store tensor count mismatch");

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
    if (run_download_rejects_null_tensor() != 0) return 1;
    if (run_upload_rejects_missing_destination() != 0) return 1;
    if (run_upload_rejects_non_null_buffer() != 0) return 1;
    if (run_backend_roundtrip(GGML_BACKEND_DEVICE_TYPE_CPU, false) != 0) return 1;
    if (run_backend_roundtrip(GGML_BACKEND_DEVICE_TYPE_GPU, true) != 0) return 1;
    std::printf("weight_residency tests passed\n");
    return 0;
}
