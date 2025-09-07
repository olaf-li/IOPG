//
// Created by cshlli on 2025/1/8.
//

#include "BAMG.h"

#include <queue>
#include <random>
#include <iostream>
#include <vector>
#include <limits>


BAMG::BAMG(const size_t dimension, const size_t n, ProximityGraph *initializer, unsigned mbs, unsigned lne)
    : NSG(dimension, n, initializer), mbs(mbs), lne(lne) {}


void get_cluster_from_file(std::unordered_map<unsigned, std::vector<unsigned>>& clusters, std::string file_name, int dimension, const float* & vector_data) {
    std::vector<std::vector<unsigned>> temp_clusters;

    std::cout << "cluster file path: " << file_name << std::endl;
    std::ifstream in(file_name, std::ios::binary);

    if (!in) {
        std::cerr << "Failed to open input files!" << std::endl;
    }

    unsigned last_id = 0;
    while (!in.eof()) {
        unsigned block_id;
        unsigned block_size;
        in.read((char*)&block_id, sizeof(unsigned));
        in.read((char*)&block_size, sizeof(unsigned));
        std::vector<unsigned> temp_vector;

        for (unsigned i = 0; i < block_size; i++) {
            unsigned id;
            in.read((char*)&id, sizeof(unsigned));
            in.seekg(sizeof(float) * dimension, std::ios::cur);
            unsigned neighbor_size;
            in.read((char*)&neighbor_size, sizeof(unsigned));
            in.seekg(sizeof(unsigned) * neighbor_size, std::ios::cur);
            temp_vector.push_back(id);
        }
        if (block_id > last_id or last_id == 0) {
            last_id = block_id;
            temp_clusters.push_back(temp_vector);
        }
        else {
            break;
        }
    }

    for (unsigned i = 0; i < temp_clusters.size(); i++) {

        if (temp_clusters[i].size() > 2) {
            float** distance_array = new float*[temp_clusters[i].size()];
            for (int j = 0; j < temp_clusters[i].size(); j++) {
                distance_array[j] = new float[temp_clusters[i].size()];
            }
            for (unsigned j = 0; j < temp_clusters[i].size(); j++) {
                for (unsigned k = j; k < temp_clusters[i].size(); k++) {
                    if (j == k) {
                        distance_array[j][k] = 0;
                    }
                    else {
                        unsigned id_1 = temp_clusters[i][j];
                        unsigned id_2 = temp_clusters[i][k];

                        distance_array[j][k] = distance_array[k][j] = euclideanDistance(vector_data + id_1 * dimension,
                            vector_data + id_2 * dimension, dimension);
                    }
                }
            }
            unsigned min_index = 0;
            float min_distance = 10000000.0;
            for (unsigned j = 0; j < temp_clusters[i].size(); j++) {
                float distance = 0;
                for (unsigned k = 0; k < temp_clusters[i].size(); k++) {
                    distance += distance_array[j][k];
                }
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            clusters.emplace(temp_clusters[i][min_index], temp_clusters[i]);
        }
        else {
            clusters.emplace(temp_clusters[i][0], temp_clusters[i]);
        }
    }
}

std::vector<Block*> BAMG::put_into_blocks(unsigned block_size, unsigned flag_in_block[], DataLayout* data_layout) {
    // 将构造好的 NSG 放入到 block内
    std::vector<unsigned> count_result;
    std::vector<unsigned> count_result_in_block;
    std::vector<unsigned> count_result_out_block;

    std::cout << "put_into_blocks: " << std::endl;

    data_layout->BNP_BA(final_graph, block_size, max_neighbor, pq->M);
    std::vector<Block*> blocks = data_layout->get_blocks();
    std::cout << "put the node into the blocks: " << final_graph.size() << " number of blocks: " << blocks.size()
    << " block size: " << block_size << std::endl;

    std::cout << "overlap_rate: " << data_layout->or_calculator_graph(final_graph) << std::endl;

    unsigned count_test_n = 0;
    for (const auto& block: blocks) {
        for (auto node: block->nodes) {
            flag_in_block[node] = block->block_id;
        }
        count_test_n = count_test_n + block->nodes.size();
    }

    count_result.reserve(blocks.size());
    for (const auto& block: blocks) {
        count_result.push_back(block->nodes.size());
    }
    for (unsigned id = 0; id < n_; id++) {
        const auto& this_block = blocks[flag_in_block[id]];
        unsigned count_in_block = 0;
        unsigned count_out_block = 0;
        for (auto neighbor: final_graph[id]) {
            if (flag_in_block[neighbor] == this_block->block_id) {
                count_in_block++;
            }
            else {
                count_out_block++;
            }
        }
        count_result_in_block.push_back(count_in_block);
        count_result_out_block.push_back(count_out_block);
    }
    return blocks;
}

void BAMG::build_block_aware_pg(const float *data, char *knn_graph_path, unsigned block_size, unsigned l, unsigned max_m,
                                        unsigned max_candidate, char *filename, char *offset_filename, const unsigned node_size,
                                        const unsigned min_block_layer_size, char* raw_filename, char* memory_filename) {
    auto s = std::chrono::high_resolution_clock::now();

    std::string new_filename = std::string(filename) + "_r12_" + std::to_string(step);// + "_" + std::to_string(alpha);
    std::string new_offset_filename = std::string(offset_filename) + "_r12_" + std::to_string(step);// + "_" + std::to_string(alpha);
    std::string new_memory_filename = new_filename + "_memory";
    std::string new_raw_filename = std::string(raw_filename) + "_r12_" + std::to_string(step);// + "_" + std::to_string(alpha);

    std::cout << "new_filename: " << new_filename << std::endl;
    std::cout << "new_offset_filename: " << new_offset_filename << std::endl;
    std::cout << "new_memory_filename: " << new_memory_filename << std::endl;
    std::cout << "new_raw_filename: " << new_raw_filename << std::endl;

    std::string new_file_path = pq->base_pq_file;
    std::string suffix = "_" + std::to_string(step);// + "_" + std::to_string(alpha);
    std::string extension = ".bin";

    size_t pos = new_file_path.rfind(extension);
    if (pos != std::string::npos) {
        new_file_path.insert(pos, suffix);
    }
    std::cout << "new_file_path: " << new_file_path << std::endl;

    this->node_size = node_size;
    std::cout << "node size: " << node_size << " " << max_neighbor << std::endl;
    std::vector<std::vector<unsigned>> new_graph;
    std::vector<Block*> blocks = re_construct_block_aware_(data, knn_graph_path, block_size, l, max_m, max_candidate, new_graph);

    std::cout << "multi layer navigation index: " << std::endl;
    std::vector<std::vector<Block*>> multi_layer_blocks;
    std::vector<std::unordered_map<unsigned, unsigned>> multi_layer_maps;
    std::vector<std::vector<std::vector<unsigned>>> multi_layer_graphs;
    std::vector<const float*> multi_layer_data;
    std::vector<unsigned> multi_layer_start_node;

    multi_layer_blocks.push_back(blocks);
    multi_layer_graphs.push_back(new_graph);
    multi_layer_data.push_back(data);
    multi_layer_start_node.push_back(ep);

    std::unordered_map<unsigned, unsigned> first_layer_map;
    for (unsigned node_i = 0; node_i < n_; node_i++) {
        first_layer_map[node_i] = node_i;
    }
    multi_layer_maps.push_back(first_layer_map);
    std::cout << "layer: " << 0 << ": blocks " << multi_layer_blocks[0].size()
            << ", nodes " << multi_layer_graphs[0].size() << std::endl;

    int layer = 1;
    unsigned count_layer_node = n_;
    while (layer < min_block_layer_size and count_layer_node > 1000) {
        boost::dynamic_bitset<> flags_in_out{n_, 0};
        std::unordered_map<unsigned, unsigned> reverse_layer_map;

        std::vector<unsigned> selected_nodes = find_key_nodes(multi_layer_blocks[layer - 1], multi_layer_graphs[layer - 1]);
        auto* select_data = new float[selected_nodes.size() * (size_t) dimension_];
        std::unordered_map<unsigned, unsigned> layer_node_map;
        for (unsigned node_i = 0; node_i < selected_nodes.size(); node_i++) {
            unsigned raw_node_id = multi_layer_maps[layer - 1][selected_nodes[node_i]];
            flags_in_out[raw_node_id] = true;
            layer_node_map.emplace(node_i, raw_node_id);
            reverse_layer_map.emplace(raw_node_id, node_i);
            for (unsigned d_i = 0; d_i < dimension_; d_i++) {
                select_data[node_i * dimension_ + d_i] = data[raw_node_id * dimension_ + d_i];
            }
        }

        std::vector<std::vector<unsigned>> next_layer_graph;
        BAMG this_index(dimension_, selected_nodes.size(), nullptr, max_m / 2, l);
        this_index.flag_layer = "top";
        this_index.flags_in_out = flags_in_out;
        this_index.final_graph = multi_layer_graphs[layer - 1];
        this_index.last_layer_map = multi_layer_maps[layer - 1];
        this_index.reverse_layer_map = reverse_layer_map;
        this_index.last_vector_data = multi_layer_data[layer - 1];
        this_index.node_size = node_size;
        this_index.pq = pq;
        this_index.flag_iopg = "iopg";
        this_index.last_ep = multi_layer_start_node[layer - 1];
        this_index.max_neighbor = max_neighbor;

        std::vector<Block*> next_layer_blocks = this_index.re_construct_block_aware_(select_data, nullptr,
            block_size, l, max_m, max_candidate, next_layer_graph);

        multi_layer_blocks.push_back(next_layer_blocks);
        multi_layer_graphs.push_back(next_layer_graph);
        multi_layer_maps.push_back(layer_node_map);
        multi_layer_data.push_back(select_data);
        multi_layer_start_node.push_back(this_index.ep);

        std::cout << "layer: " << layer << ": blocks " << multi_layer_blocks[layer].size()
            << ", nodes " << multi_layer_graphs[layer].size() << std::endl;
        layer++;
        count_layer_node = selected_nodes.size();
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";

    pq->init_pq();
    DataLayout* data_layout = new DataLayout();

    std::map<unsigned, unsigned> id_offsets;

    data_layout->Save_data_layout_BA(new_filename.data(), new_offset_filename.data(), new_memory_filename.data(), new_raw_filename.data(), vector_data, dimension_,
        multi_layer_blocks, multi_layer_maps, multi_layer_graphs, multi_layer_start_node, pq->pq_vector, pq->M, max_neighbor, id_offsets);

    pq->restore_pq_codes(new_file_path.data(), id_offsets);
}

std::vector<Block*> BAMG::re_construct_block_aware_(const float *data, char *knn_graph_path, unsigned block_size,
    unsigned l, unsigned max_m, unsigned max_candidate, std::vector<std::vector<unsigned>>& new_graph) {
    build_pg(data, knn_graph_path, l, max_m, max_candidate);
    auto* data_layout = new DataLayout();
    auto* flag_in_block = new unsigned[n_];
    std::vector<Block*> blocks = put_into_blocks(block_size, flag_in_block, data_layout);
    across_block_pruning(flag_in_block, blocks, new_graph, max_m);
    return blocks;
}


void BAMG::build_pg(const float *data, char *knn_graph_path, unsigned l, unsigned max_m, unsigned max_candidate) {
    if (flag_layer != "top") {
        std::cout<< "knn graph path: " << knn_graph_path << std::endl;
        Load_nn_graph(knn_graph_path);
    }
    vector_data = data;
    init_graph(l);

    std::cout << "init the graph" << std::endl;
    SimpleNeighbor* cut_graph = new SimpleNeighbor[n_ * (size_t)max_m];
    Link(max_m, l, max_candidate, cut_graph);
    final_graph.resize(n_);
    for (size_t i = 0; i < n_; i++) {
        SimpleNeighbor* pool = cut_graph + i * (size_t)max_m;
        unsigned pool_size = 0;
        for (unsigned j = 0; j < max_m; j++) {
            if (pool[j].distance == -1) break;
            pool_size = j;
        }
        pool_size++;
        final_graph[i].resize(pool_size);
        for (unsigned j = 0; j < pool_size; j++) {
            final_graph[i][j] = pool[j].id;
        }
    }
    unsigned max = 0, min = 1e6, avg = 0;
    for (size_t i = 0; i < n_; i++) {
        auto size = final_graph[i].size();
        max = max < size ? size : max;
        min = min > size ? size : min;
        avg = avg + size;
    }
    avg /= 1.0 * n_;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);
    has_built = true;
    delete cut_graph;
}

void BAMG::across_block_pruning(const unsigned* flag_in_block, const std::vector<Block*>& blocks,
    std::vector<std::vector<unsigned>>& new_graph, unsigned maxc) {
    std::cout << "begin pruning of edges across blocks " << std::endl;

    new_graph.reserve(n_);
    std::vector<std::vector<unsigned>> reverse_graph;
    for (unsigned node = 0; node < final_graph.size(); node++) {
        new_graph.push_back(std::vector<unsigned>());
        reverse_graph.push_back(std::vector<unsigned>());
    }

    for (unsigned node = 0; node < final_graph.size(); node++) {
        for (const auto& neighbor: final_graph[node]) {
            if (flag_in_block[neighbor] == flag_in_block[node]) {
                new_graph[node].push_back(neighbor);
                reverse_graph[neighbor].push_back(node);
            }
        }
    }

    std::cout << "reverse_graph" << std::endl;
    std::vector<unsigned> max_out(final_graph.size());
    for (unsigned node = 0; node < final_graph.size(); node++) {
        max_out[node] = maxc - new_graph[node].size();
    }

    unsigned max_out_candidate = 40;

    std::cout << "reverse_graph end" << std::endl;
    std::vector<std::mutex> node_mutex(final_graph.size());

#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 64)
        for (unsigned node = 0; node < final_graph.size(); node++) {
            if (node % 100000 == 0) std::cout << node << std::endl;
            std::vector<unsigned> pool_out_block;
            for (const auto& neighbor: final_graph[node]) {
                if (flag_in_block[neighbor] != flag_in_block[node]) {
                    pool_out_block.push_back(neighbor);
                }
            }
            for (const auto& neighbor: pools[node]) {
                if (flag_in_block[neighbor.id] != flag_in_block[node] &&
                    std::find(pool_out_block.begin(), pool_out_block.end(), neighbor.id) == pool_out_block.end()) {
                    pool_out_block.push_back(neighbor.id);
                    if (pool_out_block.size() > max_out_candidate) {
                        break;
                    }
                }
            }

            std::vector<std::pair<unsigned, unsigned>> added_edges;
            std::vector<unsigned> result_out_block;
            for (unsigned start = 0; start < pool_out_block.size() && result_out_block.size() < max_out[node]; start++) {
                auto &p = pool_out_block[start];
                bool occlude = false;
                for (unsigned t = 0; t < result_out_block.size(); t++) {
                    if (p == result_out_block[t]) {
                        occlude = true;
                        break;
                    }

                    unsigned node_t = result_out_block[t];

                    {
                        // rule 2, case 2
                        unsigned nearest_node = node_t;
                        float nearest_dist = euclideanDistance(vector_data + dimension_ * node_t, vector_data + dimension_ * p, dimension_);
                        bool flag_update = true;
                        unsigned step = 0;
                        while (flag_update) {
                            flag_update = false;
                            step++;

                            std::lock_guard<std::mutex> lock(node_mutex[nearest_node]);
                            for (auto neighbor: new_graph[nearest_node]) {
                                if (flag_in_block[neighbor] != flag_in_block[node_t]) continue;
                                float dist = euclideanDistance(vector_data + dimension_ * neighbor, vector_data + dimension_ * p, dimension_);
                                if (dist < nearest_dist) {
                                    nearest_node = neighbor;
                                    nearest_dist = dist;
                                    flag_update = true;
                                }
                            }
                            if (step > this->step) {
                                break;
                            }
                        }

                        float dnp = euclideanDistance(vector_data + dimension_ * node, vector_data + dimension_ * p, dimension_);
                        if (nearest_dist * this->alpha < dnp) {
                            occlude = true;
                            break;
                        }

                    }

                    {
                        // rule 2 case 1
                        bool add_flag = false;
                        if (flag_in_block[node_t] == flag_in_block[p]) {
                            std::lock_guard<std::mutex> lock(node_mutex[node_t]);
                            if (std::find(new_graph[node_t].begin(), new_graph[node_t].end(), p) != new_graph[node_t].end()) {
                                occlude = true;
                            }
                            if(!occlude) {
                                for (auto t_neighbor: final_graph[node_t]) {
                                    if (std::find(new_graph[t_neighbor].begin(), new_graph[t_neighbor].end(), p)
                                        != new_graph[t_neighbor].end()) {
                                        occlude = true;
                                        if (flag_in_block[t_neighbor] != flag_in_block[p]) {
                                            add_flag = true;
                                        }
                                    }
                                }
                            }
                        }

                        if (occlude) {
                            if (add_flag) {
                                added_edges.emplace_back(node_t, p);
                            }
                            break;
                        }
                    }
                    if (occlude) {
                        break;
                    }
                }
                if (!occlude) result_out_block.push_back(p);
            }

            for (auto node_neighbor: result_out_block) {
                {
                    std::lock_guard<std::mutex> lock(node_mutex[node]);
                    new_graph[node].push_back(node_neighbor);
                }
                {
                    std::lock_guard<std::mutex> lock(node_mutex[node_neighbor]);
                    reverse_graph[node_neighbor].push_back(node);
                }
            }

            for (auto edge: added_edges) {
                if (std::find(new_graph[edge.first].begin(), new_graph[edge.first].end(), edge.second) == new_graph[edge.first].end()) {

                    new_graph[edge.first].push_back(edge.second);
                }
            }
        }
    }
}

std::vector<unsigned> find_key_nodes(const std::vector<Block*>& blocks, const std::vector<std::vector<unsigned>>& graph) {
    unsigned flag_in_block[graph.size()];
    unsigned default_value = graph.size() + 1;
    std::fill(flag_in_block, flag_in_block + graph.size(), default_value);

    for (const auto& block: blocks) {
        for (auto node: block->nodes) {
            if (node >= graph.size()) {
                std::cout << "there is a bug: " << node << std::endl;
            };
            flag_in_block[node] = block->block_id;
        }
    }

    std::vector<int> in_degree(graph.size(), 0);
    for (const auto& block: blocks) {
        for (auto node: block->nodes) {
            for (auto neighbor: graph[node]) {
                if (flag_in_block[neighbor] == flag_in_block[node]) {
                    ++in_degree[neighbor];
                }
            }
        }
    }

    std::vector<unsigned> count_selected_nodes;
    std::vector<unsigned> count_block_size;

    std::vector<unsigned> selected_nodes;
    std::vector<bool> flag_selected(graph.size(), false);
    std::vector<bool> covered(graph.size(), false);

    for (const auto& block: blocks) {
        int count = 0;
        std::vector<unsigned> selected;
        std::queue<unsigned> visited;

        std::vector<unsigned> temp_nodes;
        std::vector<std::pair<unsigned, unsigned>> temp_nodes_out_degree;

        for (auto node: block->nodes) {
            unsigned out_degree = 0;
            for (auto neighbor: graph[node]) {
                if (flag_in_block[neighbor] == flag_in_block[node]) {
                    ++out_degree;
                }
            }
            temp_nodes_out_degree.emplace_back(node, out_degree);
        }

        std::sort(temp_nodes_out_degree.begin(), temp_nodes_out_degree.end(),
            [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        temp_nodes.reserve(temp_nodes_out_degree.size());
        for (auto item: temp_nodes_out_degree) {
            temp_nodes.push_back(item.first);
        }

        for (unsigned i = 0; i < temp_nodes.size(); i++) {
            if (in_degree[temp_nodes[i]] == 0) {
                selected.push_back(temp_nodes[i]);
                selected_nodes.push_back(temp_nodes[i]);
                flag_selected[temp_nodes[i]] = true;
                visited.push(temp_nodes[i]);
                covered[temp_nodes[i]] = true;
                count++;
            }
        }

        while (count < temp_nodes.size()) {

            while (!visited.empty()) {
                unsigned s_node = visited.front();
                visited.pop();
                for (auto neighbor: graph[s_node]) {
                    if (flag_in_block[neighbor] == flag_in_block[s_node] && covered[neighbor] == false) {
                        covered[neighbor] = true;
                        visited.push(neighbor);
                        count++;
                    }
                }
            }

            if (count < temp_nodes.size()) {
                for (const auto& s_node: temp_nodes) {
                    if (covered[s_node] == false) {
                        covered[s_node] = true;
                        selected.push_back(s_node);
                        selected_nodes.push_back(s_node);
                        flag_selected[s_node] = true;
                        visited.push(s_node);
                        count++;
                        break;
                    }
                }
            }
        }
        count_selected_nodes.push_back(selected.size());
        count_block_size.push_back(block->nodes.size());
    }

    sort(count_selected_nodes.begin(), count_selected_nodes.end());
    sort(count_block_size.begin(), count_block_size.end());
    float avg_selected_nodes = 0;
    for (unsigned i = 0; i < count_selected_nodes.size(); i++) {
        avg_selected_nodes += count_selected_nodes[i];
    }
    float avg_block_size = 0;
    for (unsigned i = 0; i < count_block_size.size(); i++) {
        avg_block_size += count_block_size[i];
    }
    return selected_nodes;
}