#ifndef PARTITION_H
#define PARTITION_H

#include <algorithm>
#include <cstddef>
#include <vector>

inline std::vector<size_t> calculate_boundaries(
    const std::vector<long long>& prefix,
    int threads,
    size_t frontier_size
) {
    long long total = prefix.back();

    std::vector<size_t> boundaries(
        static_cast<size_t>(threads + 1)
    );

    boundaries[0] = 0;
    boundaries[threads] = frontier_size;

    for (int t = 1; t < threads; ++t) {
        long long target = total * t / threads;

        auto it = std::lower_bound(
            prefix.begin(),
            prefix.end(),
            target
        );

        size_t index = static_cast<size_t>(
            std::distance(prefix.begin(), it)
        );

        boundaries[t] = std::min(index, frontier_size);
    }

    return boundaries;
}

#endif