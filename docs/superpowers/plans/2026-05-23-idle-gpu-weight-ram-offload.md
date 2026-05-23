# Idle GPU Weight RAM Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in CUDA/Vulkan idle timeout that copies model weights from GPU memory to host RAM, frees idle GPU weight buffers, and reloads from RAM before the next backend-tensor operation.

**Architecture:** Add a reusable common tensor-residency helper, then apply it to `TTSTransformer` and `AudioTokenizerDecoder`. `Qwen3TTS` owns the policy, lifecycle mutex, worker thread, guarded public operations, and low-memory precedence; component classes own their own offload/reload mechanics.

**Tech Stack:** C++17, GGML backend APIs, CMake standalone test executables, existing `fprintf(stderr, ...)` logging style.

---

## Reference Documents

- Spec: `docs/superpowers/specs/2026-05-23-idle-gpu-weight-ram-offload-design.md`
- Backend helper patterns: `src/common/gguf_loader.{h,cpp}`
- Transformer component: `src/transformer/tts_transformer.{h,cpp}`
- Vocoder component: `src/decoder/audio_tokenizer_decoder.{h,cpp}`
- Pipeline lifecycle: `src/pipeline/qwen3_tts.{h,cpp}`

## File Structure

- Create `src/common/gpu_offload_policy.h`: policy struct and parser declaration.
- Create `src/common/gpu_offload_policy.cpp`: parse `QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS`, low-memory precedence, invalid/negative handling.
- Create `src/common/weight_residency.h`: residency enum, host tensor store types, backend eligibility declaration, tensor download/upload declarations.
- Create `src/common/weight_residency.cpp`: reusable tensor copy, transactional upload, host-store cleanup, CUDA/Vulkan backend detection.
- Modify `CMakeLists.txt`: add new common sources and new tests.
- Create `tests/test_gpu_offload_policy.cpp`: standalone parser tests with no model files.
- Create `tests/test_weight_residency.cpp`: standalone GGML tensor store/reload spike using CPU backend and an optional GPU backend branch when one is available at runtime.
- Modify `src/transformer/tts_transformer.{h,cpp}`: transformer residency state, host copies, component offload/reload API.
- Modify `src/decoder/audio_tokenizer_decoder.{h,cpp}`: decoder residency state, host copies, component offload/reload API.
- Modify `src/pipeline/qwen3_tts.{h,cpp}`: lifecycle mutex, idle worker, operation guard, public path residency guard, low-memory precedence.
- Modify `README.md`: document `QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS`, precedence with `QWEN3_TTS_LOW_MEM`, and validation notes.

## Task 1: Add Offload Policy Parser

**Files:**
- Create: `src/common/gpu_offload_policy.h`
- Create: `src/common/gpu_offload_policy.cpp`
- Create: `tests/test_gpu_offload_policy.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing policy parser tests**

Add `tests/test_gpu_offload_policy.cpp`:

```cpp
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
```

- [ ] **Step 2: Register test target and run to verify it fails**

Modify `CMakeLists.txt` temporarily enough to add:

```cmake
add_executable(test_gpu_offload_policy
    tests/test_gpu_offload_policy.cpp
)
target_link_libraries(test_gpu_offload_policy PRIVATE
    qwen3_tts_common
)
add_test(NAME gpu_offload_policy_test
    COMMAND test_gpu_offload_policy
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
```

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target test_gpu_offload_policy -j4
```

Expected: build fails because `common/gpu_offload_policy.h` does not exist.

- [ ] **Step 3: Implement minimal policy parser**

Create `src/common/gpu_offload_policy.h`:

```cpp
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
```

Create `src/common/gpu_offload_policy.cpp`:

```cpp
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
```

Add `src/common/gpu_offload_policy.cpp` to `COMMON_SOURCES` in `CMakeLists.txt`.
Add `src/common/gpu_offload_policy.h` to the installed header list.

- [ ] **Step 4: Run parser test**

Run:

```bash
cmake --build build --target test_gpu_offload_policy -j4
./build/test_gpu_offload_policy
```

Expected: `gpu_offload_policy tests passed`.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/common/gpu_offload_policy.* tests/test_gpu_offload_policy.cpp
git commit -m "feat(offload): add idle GPU offload policy parser"
```

## Task 2: Add Reusable Weight Residency Helpers

**Files:**
- Create: `src/common/weight_residency.h`
- Create: `src/common/weight_residency.cpp`
- Create: `tests/test_weight_residency.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing tensor residency test**

Create `tests/test_weight_residency.cpp` with a CPU backend spike. This verifies that GGML tensor metadata can be reused after freeing and reallocating a backend buffer. Include an optional GPU backend branch in the same test when one is available at runtime, but keep CPU as the always-runnable baseline.

```cpp
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
```

- [ ] **Step 2: Register test target and run to verify it fails**

Add `test_weight_residency` to `CMakeLists.txt`, linked with `qwen3_tts_common`.

Run:

```bash
cmake --build build --target test_weight_residency -j4
```

Expected: build fails because `common/weight_residency.h` does not exist.

- [ ] **Step 3: Implement helper types and functions**

Create `src/common/weight_residency.h`:

```cpp
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
bool upload_tensors_from_host(ggml_context * ctx,
                              const std::map<std::string, ggml_tensor *> & tensors,
                              ggml_backend_t backend,
                              const host_tensor_store & store,
                              ggml_backend_buffer_t & buffer,
                              std::string & error);

} // namespace qwen3_tts
```

Create `src/common/weight_residency.cpp` with:

- `backend_is_cuda_or_vulkan()`: inspect `ggml_backend_get_device()`, `ggml_backend_dev_backend_reg()`, and `ggml_backend_reg_name()`, accepting `CUDA` and `Vulkan`.
- `download_tensors_to_host()`: clear destination, iterate tensor map, copy `ggml_nbytes(tensor)` with `ggml_backend_tensor_get()`, track total bytes.
- `upload_tensors_from_host()`: allocate a new buffer with `ggml_backend_alloc_ctx_tensors(ctx, backend)`, upload all copies by name, and free the new buffer on any partial failure.
- `host_tensor_store::clear()`: clear vector and reset bytes.

Add `src/common/weight_residency.cpp` to `COMMON_SOURCES`.
Add `src/common/weight_residency.h` to the installed header list.

- [ ] **Step 4: Run tensor residency test**

Run:

```bash
cmake --build build --target test_weight_residency -j4
./build/test_weight_residency
```

Expected: `weight_residency tests passed`.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/common/weight_residency.* tests/test_weight_residency.cpp
git commit -m "feat(offload): add weight residency helpers"
```

## Task 3: Add Transformer Component Offload API

**Files:**
- Modify: `src/transformer/tts_transformer.h`
- Modify: `src/transformer/tts_transformer.cpp`
- Modify: `tests/test_transformer.cpp`

- [ ] **Step 1: Write failing transformer API coverage**

In `tests/test_transformer.cpp`, add coverage near the existing model-load checks:

```cpp
if (transformer.is_ram_offloaded()) {
    fprintf(stderr, "  FAIL: transformer should not start RAM-offloaded\n");
    return 1;
}
if (transformer.can_offload_to_ram()) {
    std::string offload_error;
    if (!transformer.offload_weights_to_ram(offload_error)) {
        fprintf(stderr, "  FAIL: transformer offload failed: %s\n", offload_error.c_str());
        return 1;
    }
    if (!transformer.is_ram_offloaded()) {
        fprintf(stderr, "  FAIL: transformer did not enter RAM-resident state\n");
        return 1;
    }
    if (!transformer.reload_weights_from_ram(offload_error)) {
        fprintf(stderr, "  FAIL: transformer reload failed: %s\n", offload_error.c_str());
        return 1;
    }
}
```

This test only performs the full offload/reload path when the selected backend is CUDA/Vulkan. On CPU/Metal it verifies the methods exist and the component does not falsely report RAM-offloaded state.

- [ ] **Step 2: Run test target to verify it fails**

Run:

```bash
cmake --build build --target test_transformer -j4
```

Expected: build fails because transformer offload methods do not exist.

- [ ] **Step 3: Add transformer state and declarations**

In `src/transformer/tts_transformer.h`, include `common/weight_residency.h` and add public methods:

```cpp
bool can_offload_to_ram() const;
bool offload_weights_to_ram(std::string & error);
bool reload_weights_from_ram(std::string & error);
bool is_ram_offloaded() const;
size_t ram_offloaded_bytes() const;
```

Add private/model fields:

```cpp
weight_residency residency = weight_residency::Unloaded;
host_tensor_store host_weights;
```

- [ ] **Step 4: Implement transformer offload/reload**

In `TTSTransformer::load_tensor_data()`, after successful upload, set:

```cpp
model_.host_weights.clear();
model_.residency = weight_residency::GpuResident;
```

In `free_transformer_model()`, clear host weights and set `Unloaded`.

Implement:

```cpp
bool TTSTransformer::can_offload_to_ram() const {
    return model_.residency == weight_residency::GpuResident &&
           model_.buffer != nullptr &&
           backend_is_cuda_or_vulkan(state_.backend);
}
```

`offload_weights_to_ram()` should:

- Return true immediately if already `RamResident`.
- Return false with error if not GPU-resident.
- Free talker and code predictor KV caches.
- Reset scheduler if present.
- Download `model_.tensors` into `model_.host_weights`.
- Free `model_.buffer`.
- Set `model_.residency = weight_residency::RamResident`.

`reload_weights_from_ram()` should:

- Return true immediately if already `GpuResident`.
- Allocate/upload transactionally through `upload_tensors_from_host()`.
- On success set `GpuResident` and clear host copies.
- On failure keep `RamResident`.

- [ ] **Step 5: Run transformer target**

Run:

```bash
cmake --build build --target test_transformer -j4
```

Expected: build succeeds. If model/reference files are available, run:

```bash
./build/test_transformer
```

Expected: existing transformer checks pass; offload path is skipped unless CUDA/Vulkan is active.

- [ ] **Step 6: Commit**

```bash
git add src/transformer/tts_transformer.* tests/test_transformer.cpp
git commit -m "feat(offload): add transformer RAM residency API"
```

## Task 4: Add Decoder Component Offload API

**Files:**
- Modify: `src/decoder/audio_tokenizer_decoder.h`
- Modify: `src/decoder/audio_tokenizer_decoder.cpp`
- Modify: `tests/test_decoder.cpp`

- [ ] **Step 1: Write failing decoder API coverage**

In `tests/test_decoder.cpp`, after `decoder.load_model(tokenizer_path)` succeeds, add:

```cpp
if (decoder.is_ram_offloaded()) {
    fprintf(stderr, "  FAIL: decoder should not start RAM-offloaded\n");
    return 1;
}
if (decoder.can_offload_to_ram()) {
    std::string offload_error;
    if (!decoder.offload_weights_to_ram(offload_error)) {
        fprintf(stderr, "  FAIL: decoder offload failed: %s\n", offload_error.c_str());
        return 1;
    }
    if (!decoder.is_ram_offloaded()) {
        fprintf(stderr, "  FAIL: decoder did not enter RAM-resident state\n");
        return 1;
    }
    if (!decoder.reload_weights_from_ram(offload_error)) {
        fprintf(stderr, "  FAIL: decoder reload failed: %s\n", offload_error.c_str());
        return 1;
    }
}
```

- [ ] **Step 2: Run test target to verify it fails**

Run:

```bash
cmake --build build --target test_decoder -j4
```

Expected: build fails because decoder offload methods do not exist.

- [ ] **Step 3: Add decoder state and declarations**

In `src/decoder/audio_tokenizer_decoder.h`, include `common/weight_residency.h`, add the same public API as transformer, and add `weight_residency residency` plus `host_tensor_store host_weights` to `audio_decoder_model`.

- [ ] **Step 4: Implement decoder offload/reload**

In `AudioTokenizerDecoder::load_model()`, after tensor load and codebook normalization, set residency `GpuResident` and clear host copies.

In `free_audio_decoder_model()`, clear host copies and set `Unloaded`.

Implement component methods analogous to transformer:

- `can_offload_to_ram()` requires CUDA/Vulkan, `GpuResident`, and non-null `model_.buffer`.
- `offload_weights_to_ram()` resets scheduler if present, downloads tensors, frees `model_.buffer`, marks `RamResident`.
- `reload_weights_from_ram()` uploads transactionally, clears host copies on success, remains `RamResident` on failure.

- [ ] **Step 5: Run decoder target**

Run:

```bash
cmake --build build --target test_decoder -j4
```

Expected: build succeeds. If tokenizer model and reference codes are available, run:

```bash
./build/test_decoder
```

Expected: existing decoder checks pass; offload path is skipped unless CUDA/Vulkan is active.

- [ ] **Step 6: Commit**

```bash
git add src/decoder/audio_tokenizer_decoder.* tests/test_decoder.cpp
git commit -m "feat(offload): add decoder RAM residency API"
```

## Task 5: Add Engine Lifecycle Guard And Idle Worker

**Files:**
- Modify: `src/pipeline/qwen3_tts.h`
- Modify: `src/pipeline/qwen3_tts.cpp`

- [ ] **Step 1: Add lifecycle fields and helper declarations**

In `src/pipeline/qwen3_tts.h`, include `<condition_variable>`, `<mutex>`, `<thread>`, and `<cstdint>`.

Add private fields:

```cpp
std::mutex lifecycle_mutex_;
std::condition_variable idle_cv_;
std::thread idle_worker_;
bool idle_worker_shutdown_ = false;
bool operation_active_ = false;
uint64_t idle_generation_ = 0;
int gpu_offload_idle_secs_ = 0;
bool gpu_idle_offload_enabled_ = false;
```

Add private helper declarations:

```cpp
enum class residency_component : uint32_t {
    none = 0,
    transformer = 1u << 0,
    decoder = 1u << 1,
};

bool ensure_runtime_resident_locked(uint32_t required, std::string & error);
void finish_guarded_operation_locked();
void arm_idle_worker_locked();
void start_idle_worker_locked();
void stop_idle_worker();
void idle_worker_main();
void offload_idle_components_locked();
```

Add public C++-only test/diagnostic declarations near the other public methods.
Do not expose these through `qwen3tts_c_api.h` or the Python binding:

```cpp
bool force_transformer_offload_for_test(std::string & error);
bool transformer_ram_offloaded_for_test() const;
bool force_idle_offload_once_for_test(std::string & error);
```

- [ ] **Step 2: Implement worker shutdown before teardown**

Change `Qwen3TTS::~Qwen3TTS()` from default to:

```cpp
Qwen3TTS::~Qwen3TTS() {
    stop_idle_worker();
}
```

At the start of `load_models()`, call `stop_idle_worker()` before unloading/replacing existing components.

`stop_idle_worker()` must set shutdown, increment generation, notify, unlock, and join if joinable.

- [ ] **Step 3: Parse and log policy**

In `load_models()`, after computing `low_mem_mode_`, parse:

```cpp
auto policy = parse_gpu_offload_policy(std::getenv("QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS"),
                                       low_mem_mode_);
gpu_idle_offload_enabled_ = policy.enabled;
gpu_offload_idle_secs_ = policy.idle_secs;
fprintf(stderr, "  GPU idle RAM offload: %s (%s)\n",
        gpu_idle_offload_enabled_ ? "enabled" : "disabled",
        policy.reason.c_str());
```

After loading initial components, start the worker when enabled.

- [ ] **Step 4: Implement guarded operation helper**

`ensure_runtime_resident_locked(required, error)` should:

- Increment `idle_generation_` to invalidate pending timers.
- If a required component is RAM-resident, call its reload method.
- Set `operation_active_ = true` only after required reloads have succeeded.
- Return false and leave a clear error if reload fails.

`finish_guarded_operation_locked()` should:

- Set `operation_active_ = false`.
- Increment `idle_generation_`.
- Notify the idle worker if enabled.

- [ ] **Step 5: Implement idle worker**

`idle_worker_main()` should loop:

1. Wait until enabled, not active, and not shutdown.
2. Capture current generation.
3. Wait for `gpu_offload_idle_secs_`.
4. Recheck shutdown, active state, and generation.
5. Call `offload_idle_components_locked()`.

`offload_idle_components_locked()` should call transformer/decoder offload only when loaded and `can_offload_to_ram()` returns true. Log bytes copied and warnings on failure. If idle RAM offload is enabled but a loaded component is ineligible because the backend is CPU, Metal, or unknown, log that ineligibility once per component/backend so users understand why no VRAM is released.

- [ ] **Step 6: Add deterministic test/diagnostic hooks**

Add C++-only helpers on `Qwen3TTS`; do not expose them through the C API or Python binding:

```cpp
bool Qwen3TTS::force_transformer_offload_for_test(std::string & error);
bool Qwen3TTS::transformer_ram_offloaded_for_test() const;
bool Qwen3TTS::force_idle_offload_once_for_test(std::string & error);
```

`force_transformer_offload_for_test()` should lock the lifecycle mutex and call
`transformer_.offload_weights_to_ram(error)` without requiring CUDA/Vulkan
eligibility. This is for deterministic residency guard tests; production idle
worker logic must still use `can_offload_to_ram()` so CPU/Metal do not offload
in normal operation.

`force_idle_offload_once_for_test()` should lock the lifecycle mutex, verify no
operation is active, and call the same internal offload path as the worker with
an explicit test flag that bypasses backend eligibility. This makes worker
teardown and guard tests deterministic on CPU builds.

- [ ] **Step 7: Build**

Run:

```bash
cmake --build build --target qwen3_tts -j4
cmake --build build --target qwen3tts_shared -j4
```

Expected: both targets build.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline/qwen3_tts.*
git commit -m "feat(offload): add engine idle offload lifecycle"
```

## Task 6: Guard Public Engine Paths Without Deadlocks

**Files:**
- Modify: `src/pipeline/qwen3_tts.h`
- Modify: `src/pipeline/qwen3_tts.cpp`
- Modify: `tests/test_codec_encoder.cpp` only if build fallout requires it

- [ ] **Step 1: Refactor public methods to avoid nested locks**

Public methods that call other public overloads must not acquire the lifecycle mutex twice. Introduce private unlocked helpers where needed:

```cpp
tts_result synthesize_with_voice_samples_unlocked(const std::string & text,
                                                  const float * ref_samples,
                                                  int32_t n_ref_samples,
                                                  const tts_params & params);
tts_result synthesize_internal_unlocked(const std::string & text,
                                        const float * speaker_embedding,
                                        const tts_params & params,
                                        tts_result & result,
                                        const int32_t * ref_codes = nullptr,
                                        int32_t n_ref_frames = 0);
bool extract_speaker_embedding_unlocked(const float * ref_samples,
                                        int32_t n_ref_samples,
                                        std::vector<float> & embedding,
                                        const tts_params & params);
```

At minimum, avoid deadlock in:

- `synthesize_with_voice(const std::string & text, const std::string & reference_audio, const tts_params & params)` calling the samples overload.
- `synthesize()` calling `synthesize_internal()`.
- `synthesize_with_embedding()` calling `synthesize_internal()`.

- [ ] **Step 2: Guard synthesis paths**

At the start of each public synthesis path, hold the lifecycle mutex for the full operation and require:

- `synthesize()`: transformer plus decoder if decoder is already loaded; decoder is reloaded/lazy-loaded later under the same lock.
- `synthesize_with_voice()` and ICL paths: transformer, plus lazy encoders as needed; first-pass offload only reloads transformer/decoder.
- `synthesize_with_embedding()`: transformer plus decoder.

After operation completion or early return, call `finish_guarded_operation_locked()` via RAII to ensure the idle worker is armed after failures too.

- [ ] **Step 3: Guard speaker embedding extraction**

Wrap public `extract_speaker_embedding()` with the lifecycle mutex and operation
guard. It does not need transformer/decoder residency in the first pass, but it
must mark the engine active so the idle worker cannot offload transformer or
decoder while lazy speaker encoder loading/compute is in progress.

The public method should call
`extract_speaker_embedding_unlocked(ref_samples, n_ref_samples, embedding, params)`
after the guard is established.

- [ ] **Step 4: Guard non-synthesis tensor path**

Wrap `get_speaker_embedding()` with the lifecycle mutex and `ensure_runtime_resident_locked(transformer, error_msg_)` before calling `transformer_.get_codec_embedding_row()`.

Do not reload for metadata-only accessors:

- `get_model_type()`
- `get_model_size()`
- `has_speaker_encoder()`
- `get_speaker_names()`
- `get_speaker_ids()`
- `get_speaker_dialects()`
- `get_speaker_id()`
- `is_loaded()`

- [ ] **Step 5: Build all public API targets**

Run:

```bash
cmake --build build --target qwen3-tts-cli qwen3tts_shared -j4
```

Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/qwen3_tts.*
git commit -m "feat(offload): guard engine tensor access paths"
```

## Task 7: Add Integration Coverage For Guard And Worker Behavior

**Files:**
- Create: `tests/test_pipeline_offload_lifecycle.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write model-dependent integration test**

Create `tests/test_pipeline_offload_lifecycle.cpp`. Keep it model-dependent like existing component tests and skip cleanly when model files are absent.

Test shape:

```cpp
#include "pipeline/qwen3_tts.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
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

    qwen3_tts::Qwen3TTS tts;
    if (!tts.load_models("models")) {
        std::fprintf(stderr, "SKIP: models unavailable: %s\n", tts.get_error().c_str());
        return 0;
    }

    // CPU backend should remain functional and should not offload.
    auto names = tts.get_speaker_names();

    qwen3_tts::tts_params params;
    params.max_audio_tokens = 1;
    auto result = tts.synthesize("test", params);
    if (!result.success) {
        std::fprintf(stderr, "FAIL: synthesize failed: %s\n", result.error_msg.c_str());
        return 1;
    }

    // Force an offload through the deterministic test hook, then verify a
    // public non-synthesis tensor accessor reloads before reading tensors.
    if (!names.empty()) {
        std::string err;
        if (!tts.force_transformer_offload_for_test(err)) {
            std::fprintf(stderr, "FAIL: forced transformer offload failed: %s\n", err.c_str());
            return 1;
        }
        if (!tts.transformer_ram_offloaded_for_test()) {
            std::fprintf(stderr, "FAIL: transformer did not report RAM-offloaded\n");
            return 1;
        }
        std::vector<float> embedding;
        if (!tts.get_speaker_embedding(names[0], embedding)) {
            std::fprintf(stderr, "FAIL: get_speaker_embedding failed after forced offload: %s\n",
                         tts.get_error().c_str());
            return 1;
        }
        if (embedding.empty() || tts.transformer_ram_offloaded_for_test()) {
            std::fprintf(stderr, "FAIL: get_speaker_embedding did not reload transformer\n");
            return 1;
        }
    }

    // Force the worker offload path and ensure load_models() can stop worker
    // state and reload models without racing teardown.
    std::string err;
    if (!tts.force_idle_offload_once_for_test(err)) {
        std::fprintf(stderr, "FAIL: forced idle offload failed: %s\n", err.c_str());
        return 1;
    }
    if (!tts.transformer_ram_offloaded_for_test()) {
        std::fprintf(stderr, "FAIL: forced idle offload did not offload transformer\n");
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
```

This test verifies policy plumbing, CPU no-op behavior, deterministic forced
offload/reload, guarded public tensor access, and `load_models()` worker
shutdown with the feature configured. CUDA/Vulkan device-memory release is
validated manually in Task 10.

- [ ] **Step 2: Register and build test**

Add target:

```cmake
add_executable(test_pipeline_offload_lifecycle
    tests/test_pipeline_offload_lifecycle.cpp
)
target_link_libraries(test_pipeline_offload_lifecycle PRIVATE
    qwen3_tts
    Threads::Threads
)
add_test(NAME pipeline_offload_lifecycle_test
    COMMAND test_pipeline_offload_lifecycle
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
```

Run:

```bash
cmake --build build --target test_pipeline_offload_lifecycle -j4
./build/test_pipeline_offload_lifecycle
```

Expected: either `pipeline offload lifecycle test passed` when models exist or output beginning with `SKIP: models unavailable` with exit code 0.

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt tests/test_pipeline_offload_lifecycle.cpp
git commit -m "test(offload): cover engine lifecycle guard"
```

## Task 8: Update Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add runtime variable documentation**

In README runtime environment variables, add:

```markdown
| `QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS` | `0` | CUDA/Vulkan only. Positive values enable idle RAM offload after N idle seconds. Disabled when `QWEN3_TTS_LOW_MEM=1`. |
```

- [ ] **Step 2: Add backend notes**

Near Backend Selection or Memory Management, add a short note:

```markdown
Idle GPU RAM offload is intended for long-lived CUDA/Vulkan processes that want
to release VRAM between requests. It copies weights to host RAM only after the
idle timeout fires, frees GPU weight buffers, and reloads from RAM on the next
tensor-using request. It does not stream layers during inference and is disabled
for Metal/CPU backends.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document idle GPU RAM offload"
```

## Task 9: Full Local Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Configure build**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

Expected: configure completes.

- [ ] **Step 2: Build all core/test targets**

Run:

```bash
cmake --build build -j4
```

Expected: build completes.

- [ ] **Step 3: Run model-independent tests**

Run:

```bash
./build/test_gpu_offload_policy
./build/test_weight_residency
```

Expected: both print passed messages.

- [ ] **Step 4: Run available model-dependent tests**

Run as available:

```bash
./build/test_transformer
./build/test_decoder
./build/test_pipeline_offload_lifecycle
```

Expected: tests pass when model files are present; `test_pipeline_offload_lifecycle` may skip if model files are absent.

- [ ] **Step 5: Address verification fallout in the owning task**

If verification reveals a bug, return to the task that introduced that code,
make the smallest fix there, rerun that task's tests, and commit with that
task's file-specific `git add` command. Do not make an unscoped verification
commit.

## Task 10: Manual CUDA/Vulkan Validation

**Files:**
- No source changes expected unless validation reveals a bug.

- [ ] **Step 1: CUDA VRAM validation**

With a CUDA GGML build and model files present, run:

```bash
QWEN3_TTS_BACKEND=cuda QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS=2 \
  ./build/qwen3-tts-cli -m models -t "short test" -o /tmp/qwen-offload-test.wav
```

In another terminal, observe:

```bash
nvidia-smi
```

Expected:

- VRAM rises during load/inference.
- After about 2 idle seconds in a long-lived process, logs show idle offload.
- VRAM drops after offload.
- A subsequent tensor-using request logs reload from RAM and succeeds.

For CLI, the process may exit before idle offload matters. Prefer the Python server or a tiny local harness that keeps `Qwen3TTS` alive across two requests.

- [ ] **Step 2: Vulkan validation**

With a Vulkan GGML build, run a long-lived server or harness:

```bash
QWEN3_TTS_BACKEND=auto QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS=2 \
  python server/main.py
```

Send a speech request, wait beyond the timeout, then send another request.

Expected:

- Logs identify a Vulkan backend for offloadable components.
- Logs show idle offload and RAM reload.
- The second request succeeds.

- [ ] **Step 3: Low-memory precedence validation**

Run:

```bash
QWEN3_TTS_LOW_MEM=1 QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS=2 \
  ./build/qwen3-tts-cli -m models -t "short test" -o /tmp/qwen-lowmem-test.wav
```

Expected: startup logs say idle RAM offload is disabled because low-memory mode is enabled.

- [ ] **Step 4: Address GPU validation fallout in the owning task**

If CUDA/Vulkan validation reveals a bug, return to the component or engine task
that owns the failure, make the smallest fix there, rerun relevant local tests
and GPU validation, then commit with a file-specific `git add` command.

## Final Handoff Checklist

- [ ] `git status --short` is clean.
- [ ] `git log --oneline -5` shows task commits.
- [ ] Report which tests ran and which GPU manual validations were available.
- [ ] If CUDA/Vulkan hardware was unavailable, explicitly state that GPU VRAM release still needs hardware validation.
