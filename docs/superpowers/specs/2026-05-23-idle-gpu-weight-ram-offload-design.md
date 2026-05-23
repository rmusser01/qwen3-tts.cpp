# Idle GPU Weight RAM Offload Design

## Summary

Add an engine-core idle offload feature for CUDA and Vulkan backends. When a
long-lived `Qwen3TTS` engine has been idle for a configured timeout, it copies
GPU-resident model weights into host RAM, frees the component GPU weight
buffers, and marks those components as RAM-resident. The next synthesis request
uploads the host copies back to GPU before inference.

The feature is disabled by default and is distinct from `QWEN3_TTS_LOW_MEM`.
Low-memory mode unloads components and reloads from GGUF files. This feature
keeps component metadata and host tensor bytes so reload avoids disk reads.

## Goals

- Reduce idle CUDA/Vulkan VRAM usage between requests in long-lived processes.
- Keep reload faster than disk-based unload/reload by retaining host tensor
  bytes after offload.
- Keep active inference RAM usage unchanged by creating host copies only when
  the idle offload fires.
- Put the policy in the C++ engine core so CLI, C API, and Python server users
  can all benefit.
- Preserve current behavior unless the feature is explicitly enabled.

## Non-Goals

- Layer streaming to fit models larger than available VRAM.
- Offloading weights during active inference.
- Metal support in the first version. Apple Silicon unified memory makes RAM vs
  VRAM behavior less clear, and CUDA/Vulkan are easier to validate.
- Replacing `QWEN3_TTS_LOW_MEM`.
- Preserving KV caches or transient compute buffers across idle offload.

## Configuration

Add an environment-driven configuration in the engine core:

- `QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS=0` disables idle RAM offload. This is the
  default.
- A positive value enables idle offload after that many idle seconds.

The first implementation can parse this in `Qwen3TTS::load_models()` and store
it on the engine. Later API-level configuration can be added if consumers need
non-environment control.

If `QWEN3_TTS_LOW_MEM` and `QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS` are both enabled,
`QWEN3_TTS_LOW_MEM` takes precedence and idle RAM offload is disabled entirely,
with a clear startup log. This keeps the two memory policies from fighting over
the same component lifecycle.

## Residency Model

Each offloadable component tracks a weight residency state:

- `GpuResident`: GGML tensor metadata and backend weight buffer exist, and
  weights are ready for inference.
- `RamResident`: GGML tensor metadata exists, the GPU weight buffer has been
  freed, and host byte copies exist for tensors.
- `Unloaded`: current full teardown behavior; no usable tensor data is kept.

The first pass should support the large startup-resident components:

- `TTSTransformer`
- `AudioTokenizerDecoder`

The same interface can later be applied to `AudioTokenizerEncoder` and
`AudioCodecEncoder`, but those are lazy-loaded and request-specific, so they are
not required for the initial idle VRAM reduction.

## Component Interface

Each supported component should expose a small explicit API:

- `bool can_offload_to_ram() const`
- `bool offload_weights_to_ram(std::string & error)`
- `bool reload_weights_from_ram(std::string & error)`
- `bool is_ram_offloaded() const`

`can_offload_to_ram()` returns true only when the component is using a CUDA or
Vulkan backend and is currently GPU-resident. CPU, Metal, and unknown backends
return false.

The component remains responsible for its own GGML model context, tensor map,
backend weight buffer, and host copies. `Qwen3TTS` coordinates when to call the
component methods.

## Residency Guard

Every public engine path that may read model tensors must ensure the target
component is GPU-resident before touching backend tensors. This includes
synthesis paths and non-synthesis accessors such as preset speaker embedding
lookup, which can read `TTSTransformer` codec embedding rows.

Add a shared engine helper such as `ensure_component_resident(component)` or
`ensure_runtime_resident(required_components)`. Public methods call this helper
under the lifecycle mutex before invoking component code that uses
`ggml_backend_tensor_get()`, `ggml_backend_tensor_set()`, or graph compute.

The helper cancels pending idle offload, reloads any `RamResident` required
component from host RAM, and marks the engine active for the duration of that
operation. This prevents a path that is not full synthesis from reading tensor
metadata after the GPU weight buffer has been freed.

Current public paths to guard or explicitly classify:

- `synthesize()`, `synthesize_with_voice()`, and `synthesize_with_embedding()`
  require residency for the components they use.
- `extract_speaker_embedding()` requires a residency check if speaker encoder
  offload is added later. In the first pass, it should still enter the shared
  lifecycle guard so idle offload cannot race lazy encoder loading.
- `get_speaker_embedding()` requires `TTSTransformer` residency because it reads
  a codec embedding tensor row.
- Metadata-only accessors such as `get_model_type()`, `get_model_size()`,
  `has_speaker_encoder()`, `get_speaker_names()`, `get_speaker_ids()`,
  `get_speaker_dialects()`, `get_speaker_id()`, and `is_loaded()` do not need a
  GPU residency reload because they read config data, not backend tensor data.
- C API and Python binding methods inherit the behavior of the engine methods
  they call; they should not implement separate residency policy.

## Offload Flow

When the idle timeout expires, the engine:

1. Locks the engine lifecycle mutex.
2. Confirms no synthesis call is active and the idle generation is still
   current.
3. For each GPU-resident offloadable component, clears transient state that
   should not survive offload, such as KV cache or scheduler scratch state.
4. Allocates or resizes a host byte vector for each model tensor using
   `ggml_nbytes(tensor)`.
5. Downloads tensor bytes with `ggml_backend_tensor_get()`.
6. Frees the component's backend weight buffer with
   `ggml_backend_buffer_free()`.
7. Keeps GGML tensor metadata, config, tensor maps, and host bytes.
8. Marks the component `RamResident`.

Offload does not release the shared preferred backend handle unless the
component is genuinely unloaded or destroyed. Freeing the weight buffer is the
operation intended to release idle VRAM.

## Reload Flow

At the start of any guarded public operation that needs backend tensors:

1. Lock the engine lifecycle mutex.
2. Cancel or invalidate any pending idle offload timer generation.
3. For each `RamResident` component needed by the request, allocate a fresh
   CUDA/Vulkan backend buffer with `ggml_backend_alloc_ctx_tensors()`.
4. Upload each host tensor copy with `ggml_backend_tensor_set()`.
5. Clear the host copies after successful upload to return RAM where possible.
6. Mark the component `GpuResident`.
7. Run the operation normally.

Reload should be transactional at the component level. Allocate and upload into
a new backend buffer, and only transition to `GpuResident` after all tensors
succeed. If any upload fails, free the partial backend buffer, keep the host
copies, leave the component `RamResident`, and return a clear error.

The design assumes GGML tensor metadata remains valid after freeing and
reallocating the backend buffer. If a backend requires the context and tensor
objects to be rebuilt, the fallback is to rebuild the component GGML context and
tensors from saved metadata, then upload from the host RAM copies. That fallback
still avoids rereading tensor bytes from GGUF.

## Idle Worker And Concurrency

`Qwen3TTS` owns the idle policy and synchronization:

- A lifecycle mutex protects request activity, component residency transitions,
  and idle timer state.
- A monotonic generation counter invalidates stale timers.
- Request start cancels or invalidates any pending offload and reloads
  RAM-resident weights before compute.
- Request end records `last_used`, marks the engine inactive, and wakes or arms
  the idle worker.
- The idle worker waits for the configured timeout, then rechecks the generation
  and active-request state under the lifecycle mutex before offloading.

The first implementation should hold the lifecycle mutex for the full public
operation that can access offloadable components. That preserves the current
effectively serialized behavior and avoids races between compute, tensor
accessors, and idle offload. If later work wants concurrent requests, it should
introduce a narrower read/write residency lock separately.

## Idle Worker Lifecycle

The idle worker must have explicit startup and shutdown ownership in `Qwen3TTS`.
The worker starts only when idle RAM offload is enabled and models are loaded.

`Qwen3TTS::~Qwen3TTS()` and any `load_models()` call that tears down/replaces
components must first stop the idle worker:

1. Lock the lifecycle mutex.
2. Set a shutdown flag and increment the idle generation counter.
3. Notify the worker condition variable.
4. Unlock and join the worker thread.
5. Only after the worker has stopped, unload components or release backend
   buffers.

This prevents the worker from touching GGML contexts, tensor maps, or backend
buffers after component teardown has begun.

## Error Handling

- If idle offload fails before freeing the GPU buffer, leave the component
  `GpuResident`, log a warning, and keep the engine usable.
- If offload fails after some tensors were copied but before the GPU buffer is
  freed, discard partial host copies and keep the component `GpuResident`.
- If offload fails after the GPU buffer is freed and rollback is impossible,
  mark the component `Unloaded` and make the next request fail clearly rather
  than hiding the problem.
- If reload from RAM fails at request start, return a synthesis error. Do not
  silently fall back to disk reload because that would hide correctness bugs and
  create unpredictable latency.
- If reload fails after partial upload, free the partial backend buffer, keep the
  host tensor copies, leave the component `RamResident`, and report the error.
- CPU, Metal, and unsupported GPU backends should log that idle RAM offload is
  inactive when the feature is configured.

## Observability

Add concise logs for:

- Feature enabled/disabled and configured timeout.
- Backend eligibility per component.
- Idle offload start, success, failure, and bytes copied.
- Reload start, success, failure, and bytes uploaded.

The existing memory timing fields track process memory. Device-memory
validation should be done with backend-specific tools such as `nvidia-smi` for
CUDA and Vulkan memory tooling or backend logs for Vulkan.

## Testing

Add focused tests around state transitions and request behavior:

- CPU backend with offload enabled: no component offloads, synthesis behavior is
  unchanged.
- CUDA/Vulkan-capable build: after idle timeout, transformer and vocoder become
  `RamResident`.
- Next synthesis after idle offload reloads weights and produces valid,
  non-silent audio.
- Non-synthesis tensor accessors, such as preset speaker embedding lookup,
  reload required RAM-resident weights before reading tensors.
- Back-to-back requests inside the timeout do not offload.
- A request racing with timer expiry invalidates the timer and prevents active
  offload.
- Forced offload/reload component tests verify state transitions without a full
  server lifecycle.
- Worker lifecycle tests verify that `load_models()` and `Qwen3TTS` destruction
  stop the idle worker before component buffers are freed.
- Failure-injection tests for tensor download/upload leave the engine in a clear
  usable or explicit-error state.

Manual validation should include measuring VRAM before load, after inference,
after idle offload, and after reload on CUDA. Vulkan validation can use available
backend memory logs or platform tools.

## Acceptance Criteria

- With `QWEN3_TTS_GPU_OFFLOAD_IDLE_SECS=0`, behavior matches current code.
- With a positive timeout on CUDA or Vulkan, idle transformer and vocoder weight
  buffers are copied to RAM and GPU buffers are freed after the timeout.
- The next request reloads from RAM, not from GGUF disk reads, and completes
  synthesis.
- Public non-synthesis tensor accessors also reload from RAM before reading
  offloaded tensors.
- No offload occurs during active synthesis or another guarded public operation.
- Engine teardown and model reload stop the idle worker before freeing component
  GGML resources.
- Unsupported backends remain functional and do not attempt offload.
- Errors during offload or reload are explicit and do not silently switch to a
  different loading strategy.
