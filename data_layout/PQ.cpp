//
// Created by cshlli on 2025/2/26.
//

#include "PQ.h"
#include <omp.h>
// # define USE_AVX

uint8_t* PQ::read_pq_compressed(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "can not open: " << filename << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    num_vectors = fileSize / (code_size * sizeof(uint8_t));

    uint8_t* data = new uint8_t[num_vectors * 128];

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < 128; ++j) {
            uint8_t value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            data[i * 128 + j] = static_cast<unsigned int>(value);
        }
    }
    file.close();
    return data;
}


void PQ::init_multi_thread(unsigned int num_threads) {
    query_distance_table.resize(num_threads);
    query_table_computed.resize(num_threads, false);
}

void PQ::prepare_query(const uint8_t* query_encoded) {
    int tid = omp_get_thread_num();
    query_distance_table[tid].resize(M * K);
    for (int m = 0; m < M; ++m) {
        int query_centroid_idx = query_encoded[m];
        for (int k = 0; k < K; ++k) {
            query_distance_table[tid][m * K + k] = distance_table[m * K * K + query_centroid_idx * K + k];
        }
    }
    query_table_computed[tid] = true;
}

float PQ::pq_distance_fast(const uint8_t* encoded_vec) {
    int tid = omp_get_thread_num();
    auto s_PQ = std::chrono::steady_clock::now();
    if (!query_table_computed[tid]) {
        return pq_distance(encoded_vec, encoded_vec); //
    }

    float distance = 0.0f;

    int m = 0;
    for (; m < M - 3; m += 4) {
        distance += query_distance_table[tid][m * K + encoded_vec[m]];
        distance += query_distance_table[tid][(m+1) * K + encoded_vec[m+1]];
        distance += query_distance_table[tid][(m+2) * K + encoded_vec[m+2]];
        distance += query_distance_table[tid][(m+3) * K + encoded_vec[m+3]];
    }

    for (; m < M; ++m) {
        distance += query_distance_table[tid][m * K + encoded_vec[m]];
    }

    auto e_PQ = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_PQ = e_PQ - s_PQ;
    DC_PQ_time += diff_PQ.count();
    DC_PQ_count++;
    return distance;
}

uint8_t* PQ::read_pq_compressed_uint_8_t(const std::string& filename, unsigned PQ_length) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "can not open: " << filename << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    num_vectors = fileSize / (code_size * sizeof(uint8_t));
    uint8_t* data = new uint8_t[num_vectors * PQ_length];
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < PQ_length; ++j) {
            uint8_t value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            data[i * PQ_length + j] = value;
        }
    }
    file.close();
    return data;
}

float PQ::pq_distance(const uint8_t* vec1, const uint8_t* vec2) {
    auto s_PQ = std::chrono::steady_clock::now();
    float distance = 0.0f;
    for (int m = 0; m < M; ++m) {
        int idx1 = vec1[m];
        int idx2 = vec2[m];
        distance += distance_table[m * K * K + idx1 * K + idx2];
    }

    auto e_PQ = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_PQ = e_PQ - s_PQ;
    DC_PQ_time += diff_PQ.count();
    DC_PQ_count ++;
    return distance;
}

float PQ::pq_distance(const unsigned* vec1, const unsigned* vec2) {
    float distance = 0.0f;
    for (int m = 0; m < M; ++m) {
        int idx1 = vec1[m];
        int idx2 = vec2[m];
        distance += distance_table[m * K * K + idx1 * K + idx2];
    }
    return distance;
}


std::vector<float> PQ::read_distance_table(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "can not open : " << filename << std::endl;
        exit(1);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expected_file_size = M * K * K * sizeof(float);
    if (file_size != expected_file_size) {
        exit(1);
    }

    std::vector<float> distance_table(M * K * K);
    file.read(reinterpret_cast<char*>(distance_table.data()), file_size);
    file.close();
    return distance_table;
}

void PQ::restore_pq_codes(char *new_file_path, std::map<unsigned, unsigned> id_offsets) {
    std::ofstream file(new_file_path, std::ios::binary);
    std::vector<unsigned> old_ids(id_offsets.size());
    for (auto item: id_offsets) {
        unsigned id = item.first;
        unsigned offset = item.second;
        old_ids[offset] = id;
    }
    for (unsigned i = 0; i < old_ids.size(); ++i) {
        unsigned old_id = old_ids[i];
        file.write(reinterpret_cast<const char*>(pq_vector + old_id * M), M);
    }
    file.close();
}