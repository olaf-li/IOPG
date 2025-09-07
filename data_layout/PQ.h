//
// Created by cshlli on 2025/2/26.
//

#ifndef PQ_H
#define PQ_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <map>

#ifndef DC_PQ
#define DC_PQ

inline double DC_PQ_time = 0;
inline double DC_PQ_count = 0;

#endif

class PQ {
public:
    std::string base_pq_file;
    std::string query_pq_file;
    std::string distance_table_file;

    unsigned num_vectors{}; //
    unsigned M; //
    unsigned nbits;
    unsigned K = pow(2, nbits); //
    unsigned code_size = M * nbits / 8;

    mutable std::vector<std::vector<float>> query_distance_table;  // thread * M × K
    mutable std::vector<bool> query_table_computed;

    void prepare_query(const uint8_t* query_encoded);
    float pq_distance_fast(const uint8_t* encoded_vec);

    uint8_t* read_pq_compressed(const std::string& filename);
    uint8_t* read_pq_compressed_uint_8_t(const std::string& filename, unsigned PQ_length);
    float pq_distance(const uint8_t* vec1, const uint8_t* vec2);
    float pq_distance(const unsigned* vec1, const unsigned* vec2);
    std::vector<float> read_distance_table(const std::string& filename);

    uint8_t* pq_vector;
    uint8_t* pq_vector_query;
    std::vector<float> distance_table;

    PQ(std::string& base_pq_file, std::string& query_pq_file, std::string& distance_table_file, unsigned M, unsigned nbits):
        base_pq_file(base_pq_file), query_pq_file(query_pq_file), distance_table_file(distance_table_file), M(M), nbits(nbits) {}

    PQ(std::string& base_pq_file, unsigned M, unsigned nbits):
        base_pq_file(base_pq_file), M(M), nbits(nbits) {}

    void init() {
        pq_vector = read_pq_compressed_uint_8_t(base_pq_file, M);
        pq_vector_query = read_pq_compressed_uint_8_t(query_pq_file, M);
        distance_table = read_distance_table(distance_table_file);
    }

    void init_multi_thread(unsigned int num_threads);

    void init_distance_table() {
        distance_table = read_distance_table(distance_table_file);
    }

    void init_pq() {
        pq_vector = read_pq_compressed_uint_8_t(base_pq_file, M);
    }

    void init_pq_query() {
        pq_vector_query = read_pq_compressed_uint_8_t(query_pq_file, M);
    }

    void restore_pq_codes(char* new_file_path, std::map<unsigned, unsigned> id_offsets);
};


#endif //PQ_H
