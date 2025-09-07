//
// Created by cshlli on 2024/12/12.
//

#include "ProximityGraph.h"

#include <chrono>
#include <set>
#include <stdexcept>
#include <cmath>
#include <ostream>

// # define USE_AVX

ProximityGraph::ProximityGraph(const size_t dimension, const size_t n) {
    this->dimension_ = dimension;
    this->n_ = n;
}

ProximityGraph::~ProximityGraph() = default;

float euclideanDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same dimension");
    }

    float sum = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float euclideanDistance(const float* vec1, const float* vec2, unsigned n) {
    auto s = std::chrono::high_resolution_clock::now();
    #ifdef USE_AVX
        // AVX
        __m256 sum = _mm256_setzero_ps();
        unsigned i = 0;
        const unsigned block_size = 8;
        const unsigned num_blocks = n / block_size;

        //
        for (; i < num_blocks * block_size; i += block_size) {
            __m256 va = _mm256_loadu_ps(vec1 + i);
            __m256 vb = _mm256_loadu_ps(vec2 + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        //
        float tail_sum = 0.0f;
        for (; i < n; ++i) {
            float diff = vec1[i] - vec2[i];
            tail_sum += diff * diff;
        }

        //
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1),
                                  _mm256_castps256_ps128(sum));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        DC_time = DC_time + diff.count();
        DC_count++;
        return std::sqrt(_mm_cvtss_f32(sum128) + tail_sum);

    #else
        float sum = 0.0;
        for (size_t i = 0; i < n; ++i)
            sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        DC_time = DC_time + diff.count();
        DC_count++;
        return std::sqrt(sum);

    #endif
}


float euclideanDistance(const uint8_t* vec1, const uint8_t* vec2, unsigned n) {
    float sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    return std::sqrt(sum);
}
