#pragma once

#include <string>
#include <vector>

namespace qwen3_tts {

// Save speaker embedding to file.
// Extension determines format: .json for human-readable, .bin for raw float32 binary.
bool save_speaker_embedding(const std::string & path, const std::vector<float> & embedding);

// Load speaker embedding from file.
// Detects format by extension: .json or .bin.
bool load_speaker_embedding(const std::string & path, std::vector<float> & embedding);

} // namespace qwen3_tts
