#include "common/speaker_embedding_io.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

namespace qwen3_tts {

static bool ends_with(const std::string & s, const std::string & suffix) {
    if (suffix.size() > s.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool save_speaker_embedding(const std::string & path, const std::vector<float> & embedding) {
    if (embedding.empty()) return false;

    if (ends_with(path, ".json")) {
        FILE * f = fopen(path.c_str(), "w");
        if (!f) return false;
        fprintf(f, "{\n  \"embedding_size\": %d,\n  \"data\": [", (int)embedding.size());
        for (size_t i = 0; i < embedding.size(); ++i) {
            if (i > 0) fprintf(f, ",");
            if (i % 8 == 0) fprintf(f, "\n    ");
            fprintf(f, "%.8g", embedding[i]);
        }
        fprintf(f, "\n  ]\n}\n");
        fclose(f);
        return true;
    }

    // Default: raw float32 binary
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = fwrite(embedding.data(), sizeof(float), embedding.size(), f);
    fclose(f);
    return written == embedding.size();
}

bool load_speaker_embedding(const std::string & path, std::vector<float> & embedding) {
    embedding.clear();

    if (ends_with(path, ".json")) {
        FILE * f = fopen(path.c_str(), "r");
        if (!f) return false;

        // Simple JSON parser: find "data" array and read floats
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string content(size, '\0');
        if ((long)fread(&content[0], 1, size, f) != size) {
            fclose(f);
            return false;
        }
        fclose(f);

        // Find the "data" array
        size_t pos = content.find("\"data\"");
        if (pos == std::string::npos) return false;
        pos = content.find('[', pos);
        if (pos == std::string::npos) return false;
        pos++; // skip '['

        // Parse comma-separated floats
        while (pos < content.size()) {
            // Skip whitespace
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' ||
                   content[pos] == '\r' || content[pos] == '\t' || content[pos] == ',')) {
                pos++;
            }
            if (pos >= content.size() || content[pos] == ']') break;

            char * end = nullptr;
            float val = strtof(&content[pos], &end);
            if (end == &content[pos]) break; // parse error
            embedding.push_back(val);
            pos = end - &content[0];
        }
        return !embedding.empty();
    }

    // Default: raw float32 binary
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0 || size % sizeof(float) != 0) {
        fclose(f);
        return false;
    }
    embedding.resize(size / sizeof(float));
    size_t read = fread(embedding.data(), sizeof(float), embedding.size(), f);
    fclose(f);
    return read == embedding.size();
}

} // namespace qwen3_tts
