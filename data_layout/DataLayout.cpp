//
// Created by cshlli on 2024/12/30.
//

#include "DataLayout.h"

#include <fstream>
#include <boost/unordered_set.hpp>
#include <iostream>
#include <numeric>
#include <utility>
#include <bits/fs_fwd.h>
#include <queue>
#include <unordered_set>
#include "../build_PG/BAMG.h"
#include <boost/graph/adjacency_list.hpp>


float DataLayout::or_calculator_block(Block *block, const std::vector<std::vector<unsigned>>& pg) {
    float over_ratio = 0;
    for (auto node : block->nodes) {
        float counter = 0;
        for (auto neighbor_node: pg[node]) {
            for (auto temp_node: block->nodes) {
                if (temp_node == neighbor_node) {
                    counter++;
                    break;
                }
            }
        }
        if (block->nodes.size() > 1) over_ratio += counter / static_cast<float>(block->nodes.size() - 1);
    }

    return over_ratio / static_cast<float>(block->nodes.size());
}

float DataLayout::or_calculator_graph(const std::vector<std::vector<unsigned>>& pg) {
    float over_ratio = 0;
    for (auto block : blocks) {
        over_ratio += or_calculator_block(block, pg);
    }
    return over_ratio / static_cast<float>(blocks.size());
}

float DataLayout::out_degree_block(Block* block, const std::vector<std::vector<unsigned>>& pg) {
    float out_degree = 0;
    for (auto node : block->nodes) {
        for (auto neighbor_node: pg[node]) {
            if (std::find(block->nodes.begin(), block->nodes.end(), neighbor_node) == block->nodes.end()) {
                out_degree++;
            }
        }
    }
    return out_degree;
}

float DataLayout::out_degree_graph(const std::vector<std::vector<unsigned>>& pg) {
    float out_degree = 0;
    for (auto block : blocks) {
        out_degree += out_degree_block(block, pg);
    }
    return out_degree / static_cast<float>(pg.size());
}

float DataLayout::in_degree_block(Block* block, const std::vector<std::vector<unsigned>>& pg) {
    float in_degree = 0;
    for (auto node : block->nodes) {
        for (auto neighbor_node: pg[node]) {
            if (std::find(block->nodes.begin(), block->nodes.end(), neighbor_node) != block->nodes.end()) {
                in_degree++;
            }
        }
    }
    return in_degree;
}

float DataLayout::in_degree_graph(const std::vector<std::vector<unsigned>>& pg) {
    float in_degree = 0;
    for (auto block : blocks) {
        in_degree += in_degree_block(block, pg);
    }
    return in_degree / static_cast<float>(pg.size());
}

void DataLayout::BNP(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned dimension) {
    boost::unordered_set<unsigned> placed_ids;
    std::cout << dimension << std::endl;
    std::cout << BLOCK_SIZE << std::endl;
    std::cout << pg.size() << std::endl;

    unsigned block_id = 0;
    auto* current_block = new Block();
    blocks.push_back(current_block);
    current_block->block_id = block_id;
    current_block->block_size = BLOCK_SIZE * 1024;
    block_id++;

    unsigned remain_size = BLOCK_SIZE * 1024 - 8;
    for (unsigned id = 0; id < pg.size(); id++) {

        if (id % 10000000 == 0) {
            std::cout << id / 10000000 << std::endl;
        }

        if (placed_ids.find(id) != placed_ids.end()) {
            continue;
        }
        unsigned needed_size = (1 + dimension + 1 + pg[id].size()) * 4;

        bool flag_enlarge = false;
        if (needed_size > current_block->block_size - 8) {
            flag_enlarge = true;
        }

        if (needed_size > remain_size) {
            current_block->remain_size = remain_size;
            current_block = new Block();
            blocks.push_back(current_block);
            current_block->block_id = block_id;
            if (flag_enlarge) {
                current_block->block_size = BLOCK_SIZE * 1024 * ( needed_size / (BLOCK_SIZE * 1024) + 1);
                remain_size = BLOCK_SIZE * 1024 * (needed_size / (BLOCK_SIZE * 1024) + 1) - 8;
            }
            else {
                current_block->block_size = BLOCK_SIZE * 1024;
                remain_size = BLOCK_SIZE * 1024 - 8;
            }
            block_id++;
        }

        current_block->nodes.push_back(id);
        remain_size = remain_size - needed_size;
        placed_ids.insert(id);

        for (unsigned j = 0; j < pg[id].size(); j++) {
            if (placed_ids.find(pg[id][j]) == placed_ids.end()) {
                needed_size = (1 + dimension + 1 + pg[pg[id][j]].size()) * 4;
                if (needed_size <= remain_size) {
                    current_block->nodes.push_back(pg[id][j]);
                    remain_size = remain_size - needed_size;
                    placed_ids.insert(pg[id][j]);
                }
                else break;
            }
        }
    }

    std::cout << "number of placed blocks: " << blocks.size() << std::endl;
}

void DataLayout::BNP_BA(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned max_neighbor, unsigned PQ_length) {
    boost::unordered_set<unsigned> placed_ids;
    unsigned block_id = 0;
    auto* current_block = new Block();
    blocks.push_back(current_block);
    current_block->block_id = block_id;
    current_block->block_size = BLOCK_SIZE;
    block_id++;
    unsigned needed_size = sizeof(unsigned) + sizeof(unsigned) + sizeof(unsigned) + sizeof(unsigned) * max_neighbor;

    unsigned remain_size = BLOCK_SIZE; // Node Size 4 byte
    unsigned total_node_size = 0;
    unsigned total_neighbor = 0;
    for (unsigned id = 0; id < pg.size(); id++) {
        if (placed_ids.find(id) != placed_ids.end()) {
            continue;
        }
        bool flag_enlarge = false;

        total_neighbor += pg[id].size();
        total_node_size += needed_size;
        if (needed_size > remain_size) {
            current_block->remain_size = remain_size;
            current_block = new Block();
            blocks.push_back(current_block);
            current_block->block_id = block_id;

            if (flag_enlarge) {
                current_block->block_size = BLOCK_SIZE * (needed_size / BLOCK_SIZE + 1);
                remain_size = BLOCK_SIZE * (needed_size / BLOCK_SIZE + 1) - sizeof(unsigned);
                std::cout << "enlarged block: " << current_block->block_size << " " << remain_size << std::endl;
            }
            else {
                current_block->block_size = BLOCK_SIZE;
                remain_size = BLOCK_SIZE;
            }
            block_id++;
        }
        current_block->nodes.push_back(id);
        remain_size = remain_size - needed_size;
        placed_ids.insert(id);
        for (unsigned j = 0; j < pg[id].size(); j++) {
            if (placed_ids.find(pg[id][j]) == placed_ids.end()) {
                total_neighbor += pg[pg[id][j]].size();
                total_node_size += needed_size;
                if (needed_size < remain_size) {
                    current_block->nodes.push_back(pg[id][j]);
                    remain_size = remain_size - needed_size;
                    placed_ids.insert(pg[id][j]);
                }
                else break;
            }
        }
    }
}


void DataLayout::BNF(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned dimension) {

    BNP(pg, BLOCK_SIZE, dimension);

    float old_or = or_calculator_graph(pg);
    unsigned count_i = 0;
    while (true) {
        count_i++;
        std::unordered_map<unsigned, unsigned> node_block_map;
        for (auto block : blocks) {
            for (const auto& node : block->nodes) {
                node_block_map.emplace(node, block->block_id);
            }
        }
        for (auto block : blocks) {
            block->remain_size = block->block_size - 8;
            block->nodes.clear();
        }
        for (unsigned id = 0; id < pg.size(); id++) {
            std::unordered_map<unsigned, unsigned> block_count_map;
            for (auto neighbor: pg[id]) {
                unsigned neighbor_block_id = node_block_map[neighbor];
                if (block_count_map.find(neighbor_block_id) == block_count_map.end()) {
                    block_count_map.emplace(neighbor_block_id, 1);
                }
                else {
                    block_count_map[neighbor_block_id] += 1;
                }
            }
            std::vector<std::pair<unsigned, unsigned>> neighbor_block_count;
            for (auto item: block_count_map) {
                neighbor_block_count.emplace_back(item.first, item.second);
            }
            std::sort(neighbor_block_count.begin(), neighbor_block_count.end(),
                [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            unsigned needed_size = (1 + dimension + 1 + pg[id].size()) * 4;

            bool flag_new_block = true;
            for (auto item: neighbor_block_count) {
                unsigned best_block_id = item.first;
                if (blocks[best_block_id]->remain_size > needed_size) {
                    blocks[best_block_id]->nodes.push_back(id);
                    blocks[best_block_id]->remain_size -= needed_size;
                    flag_new_block = false;
                    break;
                }
            }
            if (flag_new_block) {
                unsigned best_block_id = -1;
                unsigned max_remain_size = 0;
                bool flag_empty_block = false;
                for (auto block: blocks) {
                    if (block->remain_size == block->block_size - 8) {
                        block->nodes.push_back(id);
                        block->remain_size -= needed_size;
                        flag_empty_block = true;
                        break;
                    }
                    else {
                        if (block->remain_size > max_remain_size) {
                            max_remain_size = block->remain_size;
                            best_block_id = block->block_id;
                        }
                    }
                }
                if (!flag_empty_block) {
                    if (best_block_id != -1) {
                        blocks[best_block_id]->nodes.push_back(id);
                        blocks[best_block_id]->remain_size -= needed_size;
                    }
                    else {
                        auto* current_block = new Block();
                        current_block->block_id = blocks.size();
                        current_block->block_size = BLOCK_SIZE * 1024;
                        current_block->remain_size = BLOCK_SIZE * 1024 - 8;
                        current_block->nodes.push_back(id);
                        current_block->remain_size -= needed_size;
                        blocks.push_back(current_block);
                    }
                }
            }
        }

        float new_or = or_calculator_graph(pg);
        unsigned count_nodes = 0;
        for (auto block: blocks) {
            count_nodes += block->nodes.size();
        }
        if (count_i > 100) {
            break;
        }
        old_or = new_or;
    }
}


void DataLayout::BS_swap(const std::vector<std::vector<unsigned>>& pg, unsigned BLOCK_SIZE, unsigned dimension) {
    BNP(pg, BLOCK_SIZE, dimension);
    boost::unordered_map<unsigned, unsigned> node_block_map;
    boost::unordered_map<unsigned, float> block_or_rate;

    for (unsigned i=0; i<blocks.size(); i++) {
        for (auto id: blocks[i]->nodes) {
            node_block_map.emplace(id, i);
            float rate = or_calculator_block(blocks[i], pg);
            block_or_rate.emplace(i, rate);
        }
    }
    float last_rate = 0;
    std::cout<< blocks.size()<<std::endl;
    bool iteration_flag = true;
    unsigned count_iteration = 0;
    while (iteration_flag) {
        float new_rate = or_calculator_graph(pg);
        std::cout << count_iteration << ": " << new_rate << std::endl;
        if (new_rate - last_rate < 0.01) {
            break;
        }
        count_iteration++;
        iteration_flag = false;
        unsigned iteration_i = 0;
        for (auto block: blocks) {
            if (iteration_i % 10000 == 0) {
                std::cout << iteration_i << std::endl;
            }
            iteration_i++;
            std::vector<std::pair<unsigned, unsigned>> in_nodes;
            std::vector<std::pair<unsigned, unsigned>> out_nodes;
            for (auto node: block->nodes) {
                int counter = 0;
                for (auto neighbor_node: pg[node]) {
                    for (auto node_1: block->nodes) {
                        if (node_1 == neighbor_node) {
                            counter++;
                            break;
                        }
                    }
                }
                in_nodes.emplace_back(node, counter);
                for (auto neighbor_node: pg[node]) {
                    if (std::find(block->nodes.begin(), block->nodes.end(), neighbor_node) == block->nodes.end()) {
                        counter = 0;
                        for (auto neighbor_node_1: pg[neighbor_node]) {
                            for (auto node_1: block->nodes) {
                                if (node_1 == neighbor_node_1) {
                                    counter++;
                                }
                            }
                        }
                        out_nodes.emplace_back(neighbor_node, counter);
                    }
                }
            }
            std::sort(in_nodes.begin(), in_nodes.end(),
                [](const std::pair<unsigned, unsigned>& a, const std::pair<unsigned, unsigned>& b) {
                            return a.second < b.second;
            });
            std::sort(out_nodes.begin(), out_nodes.end(),
                [](const std::pair<unsigned, unsigned>& a, const std::pair<unsigned, unsigned>& b) {
                            return a.second > b.second;
            });
            unsigned swap_node_in;
            unsigned swap_node_out;
            unsigned swap_block_in;
            unsigned swap_block_out;

            unsigned swap_in_memory_count;
            unsigned swap_out_memory_count;

            float or_in;
            float or_out;

            bool swap_flag = false;
            for (unsigned i = 0; i < in_nodes.size(); i++) {
                for (unsigned j = 0; j < out_nodes.size(); j++) {
                    if (in_nodes[i].first < out_nodes[j].first) {
                        auto in_block = new Block();
                        auto out_block = new Block();

                        swap_node_in = in_nodes[i].first;
                        swap_node_out = out_nodes[j].first;
                        swap_block_in = node_block_map[swap_node_in];
                        swap_block_out = node_block_map[swap_node_out];

                        swap_in_memory_count = (1 + dimension + 1 + pg[swap_node_in].size()) * 4;
                        swap_out_memory_count = (1 + dimension + 1 + pg[swap_node_out].size()) * 4;

                        if (blocks[swap_block_in]->remain_size - swap_in_memory_count + swap_out_memory_count < 0 ||
                        blocks[swap_block_out]->remain_size - swap_out_memory_count + swap_in_memory_count < 0) {
                            swap_flag = false;
                            break;
                        }

                        for (unsigned ii = 0; ii < in_nodes.size(); ii++) {
                            if(ii!=i) in_block->nodes.push_back(in_nodes[ii].first);
                        }
                        in_block->nodes.push_back(out_nodes[j].first);
                        for (unsigned jj = 0; jj < blocks[swap_block_out]->nodes.size(); jj++) {
                            if(jj!=j) out_block->nodes.push_back(out_nodes[jj].first);
                        }
                        out_block->nodes.push_back(in_nodes[i].first);

                        or_in = or_calculator_block(in_block, pg);
                        or_out = or_calculator_block(out_block, pg);

                        if (or_in + or_out > block_or_rate[swap_block_in] + block_or_rate[swap_block_out]) {
                            swap_flag = true;
                            break;
                        }
                    }
                }
                if (swap_flag) break;
            }
            if (swap_flag) {
                unsigned in_index;
                unsigned out_index;
                for (unsigned i=0; i < block->nodes.size(); i++) {
                    if (block->nodes[i] == swap_node_in) {
                        in_index = i;
                    }
                }

                for (unsigned i=0; i < blocks[swap_block_out]->nodes.size(); i++) {
                    if (blocks[swap_block_out]->nodes[i] == swap_node_out) {
                        out_index = i;
                    }
                }
                block->nodes[in_index] = swap_node_out;
                blocks[swap_block_out]->nodes[out_index] = swap_node_in;
                blocks[swap_block_in]->remain_size = blocks[swap_block_in]->remain_size -
                    swap_in_memory_count + swap_out_memory_count;
                blocks[swap_block_out]->remain_size = blocks[swap_block_out]->remain_size -
                    swap_out_memory_count + swap_in_memory_count;

                block_or_rate[swap_block_in] = or_in;
                block_or_rate[swap_block_out] = or_out;
                node_block_map[swap_node_in] = swap_block_out;
                node_block_map[swap_node_out] = swap_block_in;
                iteration_flag = true;
            }
        }
    }
}

std::vector<unsigned> intersect(const std::vector<unsigned>& sub_graph_i, const std::vector<unsigned>& sub_graph_j) {
    std::vector<unsigned> intersection;
    std::vector<unsigned> sorted_i = sub_graph_i;
    std::vector<unsigned> sorted_j = sub_graph_j;

    std::sort(sorted_i.begin(), sorted_i.end());
    std::sort(sorted_j.begin(), sorted_j.end());
    std::set_intersection(sorted_i.begin(), sorted_i.end(),
                          sorted_j.begin(), sorted_j.end(),
                          std::back_inserter(intersection));
    return intersection;
}

struct SetHash {
    size_t operator()(const std::set<unsigned>& s) const {
        size_t hash = 0;
        for (const unsigned& elem : s) {
            hash ^= std::hash<unsigned>()(elem) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

bool isSubset(const std::set<unsigned>& a, const std::set<unsigned>& b) {
    return std::includes(b.begin(), b.end(), a.begin(), a.end());
}

std::vector<std::set<unsigned>> removeSubsets(std::vector<std::set<unsigned>>& sets) {
    std::sort(sets.begin(), sets.end(), [](const std::set<unsigned>& a, const std::set<unsigned>& b) {
        return a.size() > b.size();
    });

    std::vector<std::set<unsigned>> result;
    std::unordered_set<std::set<unsigned>, SetHash> setHash;
    unsigned i = 0;
    for (const auto& s : sets) {
        i++;
        if(i % 10000 == 0) std::cout << i << std::endl;
        bool isSub = false;
        for (const auto& r : setHash) {
            if (isSubset(s, r)) {
                isSub = true;
                break;
            }
        }
        if (!isSub) {
            result.push_back(s);
            setHash.insert(s);
        }
    }
    return result;
}

void DataLayout::Save_data_layout_dynamic_block(char* filename, char* offset_filename, const float* vector_data, unsigned dimension,
    std::vector<std::vector<unsigned>> pg) {
    std::cout<<"Saving data layout of index ..."<< filename << std::endl;
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    std::ofstream out_offset(offset_filename, std::ios::binary | std::ios::out);

    std::set<unsigned> node_set;
    for (auto block: blocks) {
        for (unsigned int node : block->nodes) {
            node_set.insert(node);
        }
    }
    std::vector<bool> test_flag (pg.size(), false);
    for (auto node : node_set) {
        test_flag[node] = true;
    }

    std::cout<< "block number: " <<blocks.size() << std::endl;
    std::cout<< "node set size: " << node_set.size() << std::endl;
    assert(node_set.size() == pg.size());

    unsigned count_nodes = 0;
    long long offset = 0;
    for (unsigned block_id = 0; block_id < blocks.size(); block_id++) {
        if (block_id % 1000000 == 0) {
            std::cout << block_id / 1000000 << std::endl;
        }
        out_offset.write((char*)&block_id, sizeof(unsigned));
        out_offset.write((char*)&offset, sizeof(long long));
        unsigned nodes_size = blocks[block_id]->nodes.size();
        out_offset.write((char*)&nodes_size, sizeof(unsigned));

        out.write((char*)&block_id, sizeof(unsigned)); offset+=sizeof(unsigned);
        out.write((char*)&nodes_size, sizeof(unsigned)); offset+=sizeof(unsigned);
        for (unsigned i = 0; i < nodes_size; i++) {
            unsigned id = blocks[block_id]->nodes[i];
            count_nodes++;
            out_offset.write((char*)&id, sizeof(unsigned));
            out.write((char*)&id, sizeof(unsigned)); offset+=sizeof(unsigned);
            out.write((char*)(vector_data + id * dimension), dimension * sizeof(float)); offset+=dimension * sizeof(float);

            unsigned GK = pg[id].size();
            out.write((char*)&GK, sizeof(unsigned)); offset+=sizeof(unsigned);
            out.write((char*)pg[id].data(), GK * sizeof(unsigned)); offset+=GK * sizeof(unsigned);
        }
    }
    std::cout<<"Saving data layout ..."<< std::endl;
}


void DataLayout::Save_data_layout_BA(char* filename, char* offset_filename, char* memory_filename, char* raw_filename,
    const float* vector_data, unsigned dimension, std::vector<std::vector<Block*>> multi_layer_blocks,
    std::vector<std::unordered_map<unsigned, unsigned>> multi_layer_maps,
    std::vector<std::vector<std::vector<unsigned>>> multi_layer_graphs,
    std::vector<unsigned> multi_layer_start_node, uint8_t* pq_data, unsigned PQ_length, unsigned max_neighbor,
    std::map<unsigned, unsigned>& id_offsets) {

    std::cout<<"Saving data layout of index ..."<< filename << std::endl;
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    std::ofstream out_memory(memory_filename, std::ios::binary | std::ios::out);
    std::ofstream out_offset(offset_filename, std::ios::binary | std::ios::out);

    unsigned number_of_layer = multi_layer_blocks.size();
    out.seekp(0);

    unsigned default_id = multi_layer_graphs[0].size();
    unsigned node_size = sizeof(unsigned) + sizeof(unsigned) + sizeof(unsigned) + sizeof(unsigned) * max_neighbor;
    unsigned num_per_block = 4096 / node_size;

    unsigned offset = 0;
    for (int layer = 0; layer >= 0; layer--) {
        auto& blocks = multi_layer_blocks[layer];
        for(auto block: blocks) {
            for (unsigned i = 0; i < block->nodes.size(); i++) {
                unsigned id = block->nodes[i];
                id_offsets.emplace(id, offset + i);
            }
            offset += num_per_block;
        }
    }

    out_offset.write((char*)&number_of_layer, sizeof(unsigned));
    for (int layer = number_of_layer - 1; layer >= 0; layer--) {
        auto& layer_map = multi_layer_maps[layer];
        unsigned num_of_nodes = multi_layer_graphs[layer].size();
        unsigned layer_start_node = id_offsets[layer_map[multi_layer_start_node[layer]]];
        out_offset.write((char*)&num_of_nodes, sizeof(unsigned));
        out_offset.write((char*)&layer_start_node, sizeof(unsigned));
    }

    long long write_offset = 0;
    out_memory.seekp(write_offset);
    for (int layer = number_of_layer - 1; layer > 0; layer--) {
        auto& blocks = multi_layer_blocks[layer];
        auto& pg = multi_layer_graphs[layer];
        auto& layer_map = multi_layer_maps[layer];
        unsigned write_layer = number_of_layer - layer - 1;

        std::cout<<"layer number: "<< write_layer << " " << multi_layer_graphs[layer].size()
        << " " << multi_layer_blocks[layer].size() << std::endl;

        for (auto block: blocks) {
            unsigned node_size = block->nodes.size();
            for (unsigned i = 0; i < node_size; i++) {
                unsigned id = block->nodes[i];
                out_memory.write((char*)&layer_map[id], sizeof(unsigned));
                out_memory.write((char*)&id_offsets[layer_map[id]], sizeof(unsigned));

                std::vector<unsigned> in_neighbors;
                std::vector<unsigned> out_neighbors;
                for (unsigned j = 0; j < pg[id].size(); j++) {
                    if (std::find(block->nodes.begin(), block->nodes.end(), pg[id][j]) != block->nodes.end()) {
                        in_neighbors.push_back(pg[id][j]);
                    }
                    else {
                        out_neighbors.push_back(pg[id][j]);
                    }
                }

                unsigned GK_in = in_neighbors.size();
                out_memory.write((char*)&GK_in, sizeof(unsigned));
                for (unsigned j = 0; j < GK_in; j++) {
                    out_memory.write((char*)&id_offsets[layer_map[in_neighbors[j]]], sizeof(unsigned));
                }
                unsigned GK_out = out_neighbors.size();
                for (unsigned j = 0; j < GK_out; j++) {
                    out_memory.write((char*)&id_offsets[layer_map[out_neighbors[j]]], sizeof(unsigned));
                }
                for (unsigned j = 0; j < max_neighbor - GK_in - GK_out; j++) {
                    out_memory.write((char*)&default_id, sizeof(unsigned));
                }
            }

            write_offset += 4096;
            out_memory.seekp(write_offset);
        }
    }

    write_offset = 0;
    out.seekp(write_offset);
    for (int layer = 0; layer >= 0; layer--) {
        auto& blocks = multi_layer_blocks[layer];
        auto& pg = multi_layer_graphs[layer];
        auto& layer_map = multi_layer_maps[layer];
        unsigned write_layer = number_of_layer - layer - 1;

        std::cout<<"layer number: "<< write_layer << " " << multi_layer_graphs[layer].size()
        << " " << multi_layer_blocks[layer].size() << std::endl;

        for (auto block: blocks) {
            unsigned node_size = block->nodes.size();
            for (unsigned i = 0; i < node_size; i++) {
                unsigned id = block->nodes[i];
                out.write((char*)&layer_map[id], sizeof(unsigned));
                out.write((char*)&id_offsets[layer_map[id]], sizeof(unsigned));
                std::vector<unsigned> in_neighbors;
                std::vector<unsigned> out_neighbors;
                for (unsigned j = 0; j < pg[id].size(); j++) {
                    if (std::find(block->nodes.begin(), block->nodes.end(), pg[id][j]) != block->nodes.end()) {
                        in_neighbors.push_back(pg[id][j]);
                    }
                    else {
                        out_neighbors.push_back(pg[id][j]);
                    }
                }

                unsigned GK_in = in_neighbors.size();
                out.write((char*)&GK_in, sizeof(unsigned));
                for (unsigned j = 0; j < GK_in; j++) {
                    out.write((char*)&id_offsets[layer_map[in_neighbors[j]]], sizeof(unsigned));
                }
                unsigned GK_out = out_neighbors.size();
                for (unsigned j = 0; j < GK_out; j++) {
                    out.write((char*)&id_offsets[layer_map[out_neighbors[j]]], sizeof(unsigned));
                }
                for (unsigned j = 0; j < max_neighbor - GK_in - GK_out; j++) {
                    out.write((char*)&default_id, sizeof(unsigned));
                }
            }
            write_offset += 4096;
            out.seekp(write_offset);
        }
        out.write((char*)&default_id, sizeof(unsigned));
    }
    std::cout<<"Saving raw data  ..." <<  std::endl;
    std::cout<< "raw_filename: " << raw_filename << std::endl;
    long long raw_offset = 0;
    std::vector<Block*> blocks = multi_layer_blocks[0];
    std::vector<std::vector<unsigned>> pg = multi_layer_graphs[0];

    std::ofstream out_raw(raw_filename, std::ios::binary | std::ios::out);

    out_raw.seekp(raw_offset);
    unsigned n_per_block = 4 * 1024 / (dimension * sizeof(float));
    std::cout << "n_per_block: " << n_per_block << std::endl;
    for (auto block: blocks) {
        if (block->nodes.empty()) {
            continue;
        }
        for (unsigned i = 0; i < block->nodes.size(); i++) {
            unsigned id = block->nodes[i];
            out_raw.write((char*)(vector_data + id * dimension), dimension * sizeof(float));
            unsigned offset_i = i % n_per_block;
            if ((i + 1) % n_per_block == 0) {
                raw_offset += 4 * 1024;
                out_raw.seekp(raw_offset);
            }
        }
        if (block->nodes.size() % n_per_block != 0) {
            raw_offset += 4 * 1024;
            out_raw.seekp(raw_offset);
        }
    }

    std::cout << "max neighbor: " << max_neighbor << std::endl;
    std::cout<<"restore PQ codes ..." <<  std::endl;
}


void DataLayout::Save_data_layout(char* filename, char* offset_filename, const float* vector_data, unsigned dimension,
    std::vector<std::vector<unsigned>> pg) {
    std::cout<<"Saving data layout of index ..."<< filename << std::endl;
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    std::ofstream out_offset(offset_filename, std::ios::binary | std::ios::out);
    std::cout << "number of blocks: " << blocks.size() << std::endl;
    std::cout <<  "number of nodes: " << pg.size() << std::endl;

    unsigned count_nodes = 0;
    long long offset = 0;
    for (unsigned block_id = 0; block_id < blocks.size(); block_id++) {
        unsigned block_size = 0;
        out.seekp(offset, std::ios::beg);
        out_offset.write((char*)&block_id, sizeof(unsigned));
        out_offset.write((char*)&offset, sizeof(long long));
        unsigned nodes_size = blocks[block_id]->nodes.size();
        out_offset.write((char*)&nodes_size, sizeof(unsigned));

        out.write((char*)&block_id, sizeof(unsigned)); block_size += sizeof(unsigned);
        out.write((char*)&nodes_size, sizeof(unsigned)); block_size += sizeof(unsigned);
        for (unsigned i = 0; i < nodes_size; i++) {
            unsigned id = blocks[block_id]->nodes[i];
            count_nodes++;
            out_offset.write((char*)&id, sizeof(unsigned));
            out.write((char*)&id, sizeof(unsigned)); block_size += sizeof(unsigned);
            out.write((char*)(vector_data + id * dimension), dimension * sizeof(float)); block_size += dimension * sizeof(float);

            unsigned GK = pg[id].size();
            out.write((char*)&GK, sizeof(unsigned)); block_size += sizeof(unsigned);
            out.write((char*)pg[id].data(), GK * sizeof(unsigned)); block_size += GK * sizeof(unsigned);
        }
        offset = offset + blocks[block_id]->block_size;
        if (block_size > 4096) {
            std::cout << "there is a bug! " << std::endl;
        }
    }
    out.seekp(offset, std::ios::beg);
    out.write((char*)&count_nodes, sizeof(unsigned));
    std::cout<<"Saving data layout ..."<< std::endl;
}