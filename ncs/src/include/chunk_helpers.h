#pragma once
#include <algorithm>
#include <vector>

namespace ncs_chunk {
template <typename T>
inline std::vector<std::vector<T>> split(const std::vector<T>& data, size_t max_chunk_size) {
    std::vector<std::vector<T>> chunks;
    if (data.empty()) { chunks.emplace_back(); return chunks; }
    for (size_t offset = 0; offset < data.size(); offset += max_chunk_size) {
        size_t end = std::min(offset + max_chunk_size, data.size());
        chunks.emplace_back(data.begin() + static_cast<long>(offset), data.begin() + static_cast<long>(end));
    }
    return chunks;
}
}
