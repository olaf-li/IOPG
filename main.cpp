#include <iostream>
#include <ostream>
#include <string>
#include <algorithm>
#include <chrono>
#include <set>
#include <utility>
// #include <omp.h>

#include "build_PG/BAMG.h"
#include "build_PG/ProximityGraph.h"
#include "build_PG/NSG.h"
#include "data_layout/DataLayout.h"

#include <cmath>

void load_data(std::string filename, float*& data, unsigned& num,
               unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error: " << filename << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    std::cout << "dim: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
    std::cout << "number of vectors: " << num << "; dimension: " << dim << std::endl;
}

void load_data_query(std::string filename, float*& data, unsigned num, unsigned& dim, unsigned start_query) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error: " << filename << std::endl;
        exit(-1);
    }

    in.read((char*)&dim, 4);
    unsigned offset = start_query * (dim + 1) * 4;

    in.seekg(offset, std::ios::beg);
    data = new float[(size_t)num * (size_t)dim];

    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

void block_shuffling(const std::string& algorithm_name, unsigned BLOCK_SIZE, std::string output_data_layout_path, std::string node_block_path, std::string dataset_path, std::string msg_knn_graph_path) {
    float* data_load = NULL;
    unsigned points_num, dim;
    load_data(dataset_path, data_load, points_num, dim);
    NSG index(dim, points_num, nullptr);
    index.Load(msg_knn_graph_path.data());
    index.block_shuffling(algorithm_name, BLOCK_SIZE, output_data_layout_path.data(), node_block_path.data(), data_load);
}

void Starling_DiskANN_main(const std::string &dataset_path, unsigned l, unsigned range, unsigned maxc, char* output_path,
    char* knn_graph_path, char* memory_graph_path, char* disANN_output_path) {
    unsigned memory_size = 10000;
    float* data_load = NULL;
    unsigned points_num, dim;

    load_data(dataset_path, data_load, points_num, dim);

    std::cout << "load data: " << points_num << std::endl;

    NSG index(dim, points_num, nullptr);

    std::cout << "output path: " << output_path << std::endl;

    auto s = std::chrono::high_resolution_clock::now();
    std::cout << "begin building index" << std::endl;

    index.alpha_vamana = 1.05;
    index.build(points_num, data_load, range, l, maxc, knn_graph_path);

    auto e_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_1 = e_1 - s;

    std::cout << "indexing time 1: " << diff_1.count() << "\n";

    boost::dynamic_bitset<> flags_in_out{points_num, 0};
    std::unordered_map<unsigned, unsigned> reverse_layer_map;

    NSG this_index(dim, memory_size, nullptr);

    std::unordered_map<unsigned, unsigned> new_old_node_map;

    std::cout << "begin building the memory index" << std::endl;
    auto selected_nodes = this_index.select_numbers(points_num, memory_size, -1);
    auto* this_data_load = new float[memory_size * dim];
    for (unsigned i = 0; i < selected_nodes.size(); i++) {
        for (unsigned j = 0; j < dim; j++) {
            this_data_load[i * dim + j] = data_load[selected_nodes[i] * dim + j];
        }
        new_old_node_map.emplace(i, selected_nodes[i]);
        flags_in_out[selected_nodes[i]] = true;
        reverse_layer_map.emplace(selected_nodes[i], i);
    }

    this_index.flag_layer = "top";
    this_index.final_graph = index.final_graph;
    this_index.flags_in_out = flags_in_out;
    this_index.reverse_layer_map = reverse_layer_map;
    this_index.last_layer_map = new_old_node_map;
    this_index.last_ep = index.get_ep();
    this_index.last_vector_data = data_load;

    this_index.build(memory_size, this_data_load, range, l, maxc, knn_graph_path, true);
    this_index.Save_navigation_graph(memory_graph_path,new_old_node_map);

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";

    index.Save(output_path);
    index.save_diskANN(disANN_output_path, disANN_output_path);
}


void BAMG_main(const std::string &dataset_path, char* knn_graph_path, unsigned block_size, unsigned l, unsigned max_m,
    unsigned max_candidate, char* output_path, char *offset_filename, const unsigned node_size,
    const unsigned min_block_layer_size, std::string base_pq_file, std::string pq_distance_table_file, unsigned M, unsigned nbits,
    char* raw_filename, char* raw_offset_filename, unsigned max_neighbor, unsigned step, float alpha) {

    float* data_load = NULL;
    unsigned points_num, dim;

    load_data(dataset_path, data_load, points_num, dim);

    std::cout << "load data" << std::endl;

    BAMG index(dim, points_num, nullptr, block_size, max_m / 2);

    auto s = std::chrono::high_resolution_clock::now();

    std::cout << "output path: " << output_path << std::endl;
    std::cout << "offset path: " << offset_filename << std::endl;
    std::cout << "output raw data path: " << raw_filename << std::endl;
    std::string memory_filename = std::string(output_path) + "_memory";

    index.step = step;
    index.alpha = alpha;

    std::cout << "step: " << index.step << std::endl;
    std::cout << "alpha: " << index.alpha << std::endl;

    index.pq = new PQ(base_pq_file, M, nbits);
    index.pq->distance_table_file = pq_distance_table_file;

    index.flag_iopg = "iopg";
    index.max_neighbor = max_neighbor;
    index.build_block_aware_pg(data_load, knn_graph_path, block_size, l, max_m, max_candidate, output_path, offset_filename,
        node_size, min_block_layer_size, raw_filename, memory_filename.data());
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "indexing time: " << diff.count() << "\n";
}


void output_results(std::string result_output_path, std::vector<std::vector<unsigned>> search_results,
    std::string performance_output_path, unsigned l, unsigned start_query_num, double QPS, double search_time,
    double IO_time, unsigned IO_count) {

    std::ios_base::openmode mode = std::ios::binary | std::ios::out;
    if (!((l == 100 || l == 10) && start_query_num == 0)) {
        mode |= std::ios::app;

    }
    std::ofstream out(result_output_path, mode);
    std::ofstream out_performance(performance_output_path, mode); // 注意：假定你有不同的路径

    if (!out) {
        std::cerr << "Failed to open " << result_output_path << std::endl;
    }
    if (!out_performance) {
        std::cerr << "Failed to open " << performance_output_path << std::endl;
    }

    for (unsigned i = 0; i < search_results.size(); ++i) {
        out << l << " " << start_query_num + i << " ";
        for (unsigned j = 0; j < search_results[i].size(); ++j) {
            out << search_results[i][j] << " ";
        }
        out << "\n";
    }

    if ((l == 100 || l == 10) && start_query_num == 0) {
        out_performance << "l" << "\t"
    << "QPS" << "\t"
    << "search_time" << "\t"
    << "IO_time" << "\t"
    << "IO_count" << "\n";
    }

    out_performance << l << "\t"
    << QPS << "\t"
    << search_time << "\t"
    << IO_time << "\t"
    << IO_count << "\n";
}

void search_main(unsigned dim, unsigned points_num, bool flag_IO, const char* query_path, unsigned l, unsigned k,
    const char* index_offset_path, const char* data_layout_file_path, const char* data_layout_offset_file_path,
    bool flag_block, std::string index_algorithm, std::string base_pq_file, std::string query_pq_file,
    std::string distance_table_file, unsigned M, unsigned nbits, std::string raw_data_file_path,
    std::string raw_offset_file_path, unsigned memory_layer, char* memory_graph_path,
    unsigned query_num, unsigned start_query, unsigned max_neighbor, unsigned n_per_block) {

    float* query_load = NULL;
    unsigned query_dim;
    load_data_query(query_path, query_load, query_num, query_dim, start_query);

    if (l < k) {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        std::cout << l << " " << k << std::endl;
        exit(-1);
    }

    NSG index(dim, points_num, nullptr);

    index.algorithm_name = index_algorithm;
    index.max_neighbor = max_neighbor;
    index.node_size = sizeof(unsigned) * (max_neighbor + 3);
    index.raw_vector_per_block = n_per_block;
    index.num_node_per_block = 4096 / index.node_size;
    if (flag_IO) {
        index.index_offset_file_path = index_offset_path;
        index.data_layout_file_path = data_layout_file_path;
        index.data_layout_offset_file_path = data_layout_offset_file_path;
        index.memory_data_layout_file_path = (std::string(data_layout_file_path) + "_memory").data();

        if (flag_block) {
            if (index_algorithm == "BA") {
                index.raw_data_file_path = raw_data_file_path;
                index.raw_data_offset_file_path = raw_offset_file_path;
                index.pq = new PQ(base_pq_file, M, nbits);
                index.pq->distance_table_file = distance_table_file;
                index.pq->query_pq_file = query_pq_file;
                index.pq->base_pq_file = base_pq_file;
                index.pq->init();
                index.init_files();
                index.Load_offsets_NO();
                index.memory_layer = memory_layer;
                index.Load_memory_layers_NO(memory_layer);
            }
            else if (index_algorithm == "NSG") {
                index.pq = new PQ(base_pq_file, M, nbits);
                index.pq->distance_table_file = distance_table_file;
                index.pq->base_pq_file = base_pq_file;
                index.pq->query_pq_file = query_pq_file;
                index.pq->init();
                index.init_files();
                index.Load_offset_block();
            }
            else if (index_algorithm == "DiskANN") {
                index.pq = new PQ(base_pq_file, M, nbits);
                index.pq->distance_table_file = distance_table_file;
                index.pq->query_pq_file = query_pq_file;
                index.pq->init();
                index.init_files();
            }
            else {
                index.Load_offset_block();
            }
        }
    }
    if (start_query == 0) {
        std::cout<<"search begin " << std::endl;
        std::cout << "k: " << k << " l: " << l << std::endl;
        std::cout << "query path: " << query_path << std::endl;
        std::cout << "data_layout_file_path path: " << data_layout_file_path;
    }
    if (start_query % 1000 == 0) {
        std::cout << std::endl;
    }

    std::cout << start_query << " ";

    auto s = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<unsigned>> res(query_num);

    float search_time = 0.0;
    std::vector<int> vec(points_num, 0);
    index.visit_count = vec;

    auto start = std::chrono::high_resolution_clock::now();

    unsigned thread_num = 8;
    index.pq->init_multi_thread(thread_num);
#pragma omp parallel for num_threads(thread_num)
    for (unsigned i = 0; i < query_num; i++) {
        std::vector<unsigned> tmp(k);
        if (flag_IO) {
            index.debug_flag = false;
            auto s_t = std::chrono::high_resolution_clock::now();
            if (index_algorithm == "NSG") {
                uint8_t* query_pq_vec = index.pq->pq_vector_query + (i + start_query) * index.pq->M;
                index.pq->prepare_query(query_pq_vec);
                index.Search_starling(query_load + i * dim, k, l, tmp.data(), query_pq_vec);
            }
            else if (index_algorithm == "BA") {
                uint8_t* query_pq_vec = index.pq->pq_vector_query + (i + start_query) * index.pq->M;
                index.pq->prepare_query(query_pq_vec);
                index.Search_BA(query_load + i * dim, query_pq_vec, k, l, tmp.data());
            }
            else if (index_algorithm == "DiskANN") {
                uint8_t* query_pq_vec = index.pq->pq_vector_query + (i + start_query) * index.pq->M;
                index.pq->prepare_query(query_pq_vec);
                index.node_size = max_neighbor * sizeof(unsigned) + sizeof(unsigned) + sizeof(float) * dim;
                index.Search_diskANN_PQ(query_load + i * dim, k, l, tmp.data(), query_pq_vec);
            }
            auto e_t = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_t = e_t - s_t;
            search_time += diff_t.count();
        }
        res[i] = std::move(tmp);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_test = end - start;
    float QPS = query_num / diff_test.count();

    if (start_query == 0) {
        std::cout << "search time test: " << diff_test.count() << " seconds\n";
        std::cout << "search time: " << search_time << " seconds\n";
        std::cout << "IO count disk: " << IO_count_disk << "\n";
        std::cout << "IO count raw vector: " << IO_count_raw_vector << "\n";
        std::cout << "IO time disk: " << IO_time_disk << "\n";
        std::cout << "IO time raw vector: " << IO_time_raw_vector << "\n";
        std::cout << "IO time disk avg: " << IO_time_disk / IO_count_disk << "\n";
        std::cout << "IO time raw vector avg: " << IO_time_raw_vector / IO_count_raw_vector << "\n";
        std::cout << "search time memory: " << search_time_memory << "\n";
        std::cout << "search time disk: " << search_time_disk << "\n";
        std::cout << "search time reranking: " << search_time_reranking << "\n";
        std::cout << "DC count: " << DC_count << "\n";

        std::cout << "DC time: " << DC_time << "\n";
        std::cout << "DC time avg: " << DC_time / DC_count << "\n";
        std::cout << "DC count PQ: " << DC_PQ_count << "\n";
        std::cout << "DC time PQ: " << DC_PQ_time << "\n";
        std::cout << "DC time PQ avg: " << DC_PQ_time / DC_PQ_count << "\n";
        std::cout << "QPS: " << QPS << std::endl;
    }

    std::string performance_result_file;
    std::string search_result_file;
    if (index_algorithm == "NSG") {
        performance_result_file = "Starling_results.txt";
        search_result_file = "Starling_search_results.txt";
    }
    else if (index_algorithm == "DiskANN") {
        performance_result_file = "DiskANN_results.txt";
        search_result_file = "DiskANN_search_results.txt";
    }
    else {
        performance_result_file = "iopg_results.txt";
        search_result_file = "iopg_search_results.txt";
    }

    output_results(search_result_file, res, performance_result_file, l, start_query, QPS, search_time, IO_time_disk, IO_count_disk + IO_count_raw_vector);

    IO_time_disk = 0;
    IO_time_raw_vector = 0;

    IO_count_disk = 0;
    IO_count_raw_vector = 0;

    DC_count = 0;
    DC_time = 0;

    search_time_disk = 0;
    search_time_memory = 0;
    search_time_reranking = 0;
}


void temp_process() {
    std::string file_path = "/home/comp/cshlli/code/BAMG/datasets/crawl/index/DiskANN";
    std::string target_file_path = "/home/comp/cshlli/code/BAMG/datasets/crawl/IO_index/DiskANN";
    std::ifstream infile(file_path);

    if (!infile) {
        std::cerr << "can not open！" << std::endl;
    }

    unsigned n = 1989995;
    unsigned dimension = 300;
    unsigned max_k = 0;
    for (unsigned i = 0; i < n; i++) {
        unsigned node_id;
        infile.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));

        float* vec = new float[dimension];
        infile.read(reinterpret_cast<char*>(vec), sizeof(float) * dimension);

        unsigned k;
        infile.read(reinterpret_cast<char*>(&k), sizeof(unsigned));
        std::vector<unsigned> neighbors(k);
        infile.read(reinterpret_cast<char*>(neighbors.data()), sizeof(unsigned) * k);

        if (max_k < k) {
            max_k = k;
        }
    }

    infile.close();
    std::cout << "max_k: " << max_k << std::endl;

    unsigned block_per_node = 4096 / (sizeof(float) * dimension + sizeof(unsigned) + sizeof(unsigned) * max_k);
    long long offset = 0;
    infile.open(file_path);
    std::ofstream out(target_file_path, std::ios::binary | std::ios::out);
    for (unsigned i = 0; i < n; i++) {
        if (i % block_per_node == 0) {
            out.seekp(offset, std::ios::beg);
            offset += 4096;
        }
        unsigned node_id;
        infile.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));
        float* vec = new float[dimension];
        infile.read(reinterpret_cast<char*>(vec), sizeof(float) * dimension);

        unsigned k;
        infile.read(reinterpret_cast<char*>(&k), sizeof(unsigned));
        std::vector<unsigned> neighbors(k);
        infile.read(reinterpret_cast<char*>(neighbors.data()), sizeof(unsigned) * k);

        out.write(reinterpret_cast<char*>(&node_id), sizeof(node_id));
        out.write(reinterpret_cast<char*>(vec), sizeof(float) * dimension);
        out.write(reinterpret_cast<char*>(neighbors.data()), sizeof(unsigned) * k);
        for(unsigned j = 0; j < max_k - k; j++) {
            out.write(reinterpret_cast<char*>(&n), sizeof(unsigned));
        }
    }
    out.seekp(offset, std::ios::beg);
    out.write(reinterpret_cast<char*>(&n), sizeof(unsigned));
}

int main(int argc, char *argv[]) {

    std::string tmp_path = "/tmp/local/17198/";
    std::string home_path = "../../";

    std::string my_path = home_path;

    std::string dataset_name;
    std::string index_algorithm;
    std::string block_shuffling_algorithm;
    std::string dataset_path;
    std::string query_path;
    std::string ground_truth_path;
    std::string output_path;
    std::string memory_graph_path;

    std::string data_layout_path;
    std::string node_block_path;
    std::string knn_graph_path;
    std::string IO_index_path;
    std::string index_offset_path;

    std::string raw_data_file_path;
    std::string raw_offset_file_path;

    // PQ related variables
    std::string base_pq_file;
    std::string query_pq_file;
    std::string distance_table_file;
    unsigned M; 
    unsigned nbits;

    std::string cluster_path;

    bool flag_IO = true;
    bool flag_block = true;

    unsigned n_per_block;
    int min_neighbor = 50;
    int max_neighbor = 50;
    int k = 20;
    int l = 40;
    unsigned L = 40;
    unsigned range;
    unsigned maxc;
    unsigned memory_layer = 0;

    unsigned search_max_neighbor;

    unsigned dim;
    unsigned points_num;

    float alpha;
    float distance_threshold;

    unsigned BLOCK_SIZE; // unit: KB
    unsigned min_block_layer_size;

    // unsigned l_length = std::stoi(argv[2]);
    // unsigned path_flag = std::stoi(argv[6]);
    // if (path_flag == 1) {
    //     my_path = home_path;
    // }
    // else if (path_flag == 0) {
    //     my_path = tmp_path;
    // }

    // unsigned result_flag = std::stoi(argv[7]);
    // if (result_flag == 100) {
    //     result_flag = 0;
    // }


    std::string flag_build_search = argv[1];

    std::cout << flag_build_search << std::endl;

    index_algorithm = argv[2];
    dataset_name = argv[3];
    L = std::stoi(argv[4]);
    range = std::stoi(argv[5]);
    maxc = std::stoi(argv[6]);
    block_shuffling_algorithm = argv[7];

    base_pq_file = my_path + "datasets/" + dataset_name + "/" + dataset_name + "_base_pq_compressed.bin";
    query_pq_file = my_path + "datasets/" + dataset_name + "/" + dataset_name + "_query_pq_compressed.bin";
    distance_table_file = my_path + "datasets/" + dataset_name + "/" + dataset_name + "_distance_table.bin";
    dataset_path = my_path + "datasets/" + dataset_name + "/" + dataset_name + "_base.fvecs";
    query_path = my_path + "datasets/" + dataset_name + "/" + dataset_name + "_query.fvecs";
    ground_truth_path = my_path + "datasets/" + dataset_name + "/" + dataset_name + "_groundtruth.ivecs";

    output_path = my_path + "datasets/" + dataset_name + "/index/" + index_algorithm + "_" + std::to_string(L) + "_" + std::to_string(range) + "_" + std::to_string(maxc);
    IO_index_path = my_path + "datasets/" + dataset_name + "/IO_index/" + index_algorithm + "_" + std::to_string(L) + "_" + std::to_string(range) + "_" + std::to_string(maxc);
    index_offset_path = my_path + "datasets/" + dataset_name + "/IO_index/" + index_algorithm + "_offset_" + std::to_string(L) + "_" + std::to_string(range) + "_" + std::to_string(maxc);
    memory_graph_path = output_path + "_memory/";

    raw_data_file_path = my_path + "datasets/" + dataset_name + "/BA_raw_data";
    raw_offset_file_path = my_path + "datasets/" + dataset_name + "/BA_raw_offset";

    knn_graph_path = my_path + "datasets/" + dataset_name + "/index/knn_graph";

    k = 100;
    BLOCK_SIZE = 4;


    if (dataset_name == "deep1M") {
        n_per_block = 4;
        points_num = 1000000;
        dim = 256;
        M = 128;
        nbits = 8;

        if (index_algorithm == "BAMG") {
            search_max_neighbor = 65;
            L = 50;
            range = 60;
            maxc = 500;
            memory_layer = 2;
            distance_threshold = 0.6;
        }
    }
    else if (dataset_name == "sift") {
        n_per_block = 8;
        points_num = 1000000;
        dim = 128;
        M = 64; 
        nbits = 8;
    }
    else if (dataset_name == "gist") {
        n_per_block = 1;
        points_num = 1000000;
        dim = 960;
        M = 320; 
        nbits = 8;
    }
    else if (dataset_name == "msong") {
        n_per_block = 2;
        points_num = 992272;
        dim = 420;
        M = 210;
        nbits = 8;
    }
    else if (dataset_name == "glove") {
        n_per_block = 10;
        points_num = 1183514;
        dim = 100;
        M = 50; 
        nbits = 8;
    }
    else if (dataset_name == "crawl") {
        n_per_block = 3;
        points_num = 1989995;
        dim = 300;
        M = 100; 
        nbits = 8;
    }


    if (flag_build_search == "build") {

        data_layout_path = IO_index_path + "_" + block_shuffling_algorithm + "_" + std::to_string(BLOCK_SIZE);
        node_block_path = index_offset_path + "_" + block_shuffling_algorithm + "_" + std::to_string(BLOCK_SIZE);

        std::string DiskANN_path =  my_path + "datasets/" + dataset_name + "/index/DiskANN";
        if (index_algorithm == "Starling" || index_algorithm == "DsikANN") {
            Starling_DiskANN_main(dataset_path, L, range, maxc, output_path.data(), knn_graph_path.data(),
                memory_graph_path.data(), DiskANN_path.data());
            if (index_algorithm == "Starling") {

                block_shuffling(block_shuffling_algorithm, BLOCK_SIZE, data_layout_path, node_block_path, dataset_path, output_path);
            }
        }
        else if (index_algorithm == "BAMG") {
            unsigned step = std::stoi(argv[8]); // 2
            float ba_alpha = std::stof(argv[9]); // 1.1

            unsigned node_size = M * nbits / 8; // PQ_length
            std::cout << "construct graph index with algorithm Block Aware" << std::endl
            << "Node Size: " << node_size << std::endl
            << "Block Size: " << BLOCK_SIZE * 1024 << std::endl
            << "max neighbor: " << range <<std::endl
            << "max candidate set: " << maxc <<std::endl
            << "base_pq_file: "<< base_pq_file << std::endl
            << "M: " << M << std::endl
            << "nbit: " << nbits << std::endl;
            BAMG_main(dataset_path, knn_graph_path.data(), BLOCK_SIZE * 1024, L, range, maxc, data_layout_path.data(),
            node_block_path.data(), node_size, min_block_layer_size, base_pq_file, distance_table_file, M, nbits, raw_data_file_path.data(),
            raw_offset_file_path.data(), search_max_neighbor, step, ba_alpha);
        }
    }

    if (flag_build_search == "search") {
        unsigned l_length = std::stoi(argv[8]);
        unsigned query_num = std::stoi(argv[9]);
        unsigned start_query_id = std::stoi(argv[10]);
        if (index_algorithm == "Starling" or index_algorithm == "BAMG") {
            data_layout_path = IO_index_path + "_" + block_shuffling_algorithm + "_" + std::to_string(BLOCK_SIZE);
            node_block_path = index_offset_path + "_" + block_shuffling_algorithm + "_" + std::to_string(BLOCK_SIZE);
        }
        else if (index_algorithm == "DiskANN") {
            data_layout_path = IO_index_path;
            node_block_path = index_offset_path;
        }
        if (index_algorithm == "BAMG") {
            data_layout_path = data_layout_path + "_r12_2";
            node_block_path = node_block_path + "_r12_2";
            raw_data_file_path = raw_data_file_path + "_r12_2";
            std::string suffix = "_2";
            memory_graph_path = data_layout_path + "_memory";
            std::string extension = ".bin";

            size_t pos = base_pq_file.rfind(extension);
            if (pos != std::string::npos) {
                base_pq_file.insert(pos, suffix);
            }
        }
        
        search_main(dim, points_num, flag_IO, query_path.data(), l_length, k, index_offset_path.data(), data_layout_path.data(), node_block_path.data(), flag_block,
        index_algorithm, base_pq_file, query_pq_file, distance_table_file, M, nbits, raw_data_file_path,
        raw_offset_file_path, memory_layer, memory_graph_path.data(), query_num, start_query_id, search_max_neighbor,
        n_per_block);
    }
}
