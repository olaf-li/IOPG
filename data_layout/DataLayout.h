//
// Created by cshlli on 2024/12/30.
//

#ifndef DATALAYOUT_H
#define DATALAYOUT_H
#include <vector>
#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <unordered_set>

struct  Block {
    unsigned block_id;
    unsigned block_size;
    std::vector<unsigned> nodes;
    unsigned remain_size;

    Block(unsigned block_id, unsigned block_size): remain_size(0) {
        this->block_id = block_id;
        this->block_size = block_size;
        this->remain_size = block_size;
    }


    Block(): block_id(0), block_size(0), remain_size(0) {
    }

    void reset() {
        remain_size = block_size;
        nodes.clear();
    }


};

// struct CandidateNode {
//     unsigned id;
//     int value;
//     std::vector<unsigned> parents;
// };
//
// struct Compare {
//     bool operator()(CandidateNode a, CandidateNode b) {
//         return a.value < b.value;
//     }
// };

struct SimpleSubGraph {
    unsigned id;
    std::set<unsigned> nodes;
};

struct CandidateNode {
    unsigned id;
    int value;
    std::vector<unsigned> parents;

    bool operator<(const CandidateNode& other) const {
        return value > other.value; // For min-heap
    }
};

class DataLayout {
public:
    DataLayout() = default;
    float overlap_rate;
    float out_degree;

    void BNP(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned dimension);
    void BNP_BA(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned max_neighbor, unsigned PQ_length);
    void BNF(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned dimension);
    void BS_swap(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned dimension);

    void Save_data_layout(char* file_name, char* offset_file_name, const float* vector_data, unsigned dimension, std::vector<std::vector<unsigned>> pg);

    void Save_data_layout_dynamic_block(char* filename, char* offset_filename, const float* vector_data, unsigned dimension, std::vector<std::vector<unsigned>> pg);
    void Save_data_layout_BA(char* filename, char* offset_filename, char* memory_filename, char* raw_filename,
    const float* vector_data, unsigned dimension, std::vector<std::vector<Block*>> multi_layer_blocks,
    std::vector<std::unordered_map<unsigned, unsigned>> multi_layer_maps,
    std::vector<std::vector<std::vector<unsigned>>> multi_layer_graphs,
    std::vector<unsigned> multi_layer_start_node, uint8_t* pq_data, unsigned PQ_length, unsigned max_neighbor,
    std::map<unsigned, unsigned>& id_offsets);

    static float or_calculator_block(Block* block, const std::vector<std::vector<unsigned>>& pg);
    float or_calculator_graph(const std::vector<std::vector<unsigned>>& pg);

    float out_degree_graph(const std::vector<std::vector<unsigned>>& pg);
    static float out_degree_block(Block* block, const std::vector<std::vector<unsigned>>& pg);
    float in_degree_graph(const std::vector<std::vector<unsigned>>& pg);
    static float in_degree_block(Block* block, const std::vector<std::vector<unsigned>>& pg);
    std::vector<Block*> get_blocks() {
        return blocks;
    }

private:
    std::vector<Block*> blocks;

};
#endif //DATALAYOUT_H