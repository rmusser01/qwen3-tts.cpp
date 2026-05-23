#include "encoder/audio_tokenizer_encoder.h"

#include <cstdio>
#include <cstdint>

static ggml_tensor * fake_tensor(uintptr_t value) {
    return reinterpret_cast<ggml_tensor *>(value);
}

int main() {
    qwen3_tts::speaker_encoder_model model;

    model.conv0_w = fake_tensor(0x1001);
    model.conv0_b = fake_tensor(0x1002);
    model.blocks[0].tdnn1_w = fake_tensor(0x1003);
    model.blocks[1].res2net_w[3] = fake_tensor(0x1004);
    model.blocks[2].se_conv2_b = fake_tensor(0x1005);
    model.mfa_w = fake_tensor(0x1006);
    model.asp_conv_b = fake_tensor(0x1007);
    model.fc_b = fake_tensor(0x1008);
    model.tensors["spk_enc.conv0.weight"] = model.conv0_w;

    qwen3_tts::free_speaker_encoder_model(model);

    if (model.conv0_w || model.conv0_b || model.blocks[0].tdnn1_w ||
        model.blocks[1].res2net_w[3] || model.blocks[2].se_conv2_b ||
        model.mfa_w || model.asp_conv_b || model.fc_b || !model.tensors.empty()) {
        std::fprintf(stderr, "FAIL: speaker encoder model retained stale tensor pointers\n");
        return 1;
    }

    std::printf("audio tokenizer encoder lifecycle tests passed\n");
    return 0;
}
