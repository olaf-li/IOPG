//
// Created by cshlli on 2025/1/8.
//

#ifndef IOPG_H
#define IOPG_H
#include "NSG.h"

std::vector<unsigned> find_key_nodes(const std::vector<Block*>& blocks, const std::vector<std::vector<unsigned>>& graph);

class BAMG: public NSG{
public:
    unsigned step;
    float alpha;

    explicit BAMG(const size_t dimension, const size_t n, ProximityGraph *initializer, unsigned mbs, unsigned lne);

    std::vector<Block*> re_construct_block_aware_(const float *data, char *knn_graph_path, unsigned block_size,
        unsigned l, unsigned max_m, unsigned max_candidate, std::vector<std::vector<unsigned>>& new_graph);

    void build_pg(const float *data, char *knn_graph_path, unsigned l, unsigned max_m, unsigned max_candidate);
    std::vector<Block*> put_into_blocks(unsigned block_size, unsigned flag_in_block[], DataLayout* data_layout);
    void across_block_pruning(const unsigned* flag_in_block, const std::vector<Block*>& blocks,
        std::vector<std::vector<unsigned>>& new_graph, unsigned maxc);
    void build_block_aware_pg(const float *data, char *knn_graph_path, unsigned block_size, unsigned l, unsigned max_m,
                unsigned max_candidate, char *filename, char *offset_filename, const unsigned node_size,
                const unsigned min_block_layer_size, char* raw_filename, char* memory_filename);

private:
    // unsigned ep;
    // unsigned width;
    // maximum block size
    unsigned mbs;
    // least number of edges
    unsigned lne;
    // the size of a node: ID + PQ
    unsigned node_size;

};

#endif //IOPG_H
