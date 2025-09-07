//
// Created by cshlli on 2024/12/21.
//

#ifndef NSG_H
#define NSG_H
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <sys/mman.h>
#include <sys/stat.h>
#include <random>

#include <fcntl.h>
#include <unistd.h>
#include "ProximityGraph.h"
#include "../data_layout/DataLayout.h"
#include "../data_layout/PQ.h"

#include <boost/graph/graphviz.hpp>

#ifndef IO_COUNT
#define IO_COUNT
inline int IO_count_disk = 0;
inline int IO_count_raw_vector = 0;

inline double IO_time_disk = 0;
inline double IO_time_raw_vector = 0;

inline double search_time_memory = 0;
inline double search_time_disk = 0;
inline double search_time_reranking = 0;

#endif

namespace std {
    class mutex;
}

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

struct SimpleNeighbor{
    unsigned id;
    float distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance}{}

    inline bool operator<(const SimpleNeighbor &other) const {
        return distance < other.distance;
    }
};

template<typename DataType>
struct BlockData {
    std::vector<std::vector<unsigned>> block_graph;
    std::vector<DataType*> block_data;
    std::vector<unsigned> id_list;
    std::vector<std::vector<unsigned>> block_graph_in_block;

    BlockData() = default;
    //
    void clear_and_reuse() {
        //
        for (auto& vec_ptr : block_data) {
            if (vec_ptr) {
                //
            }
        }

        //
        block_graph.clear();
        block_graph_in_block.clear();
        id_list.clear();

        //
    }

    DataType* get_reusable_buffer(size_t index, size_t dim) {
        if (index < block_data.size() && block_data[index] != nullptr) {
            return block_data[index];
        }
        return nullptr;
    }

    ~BlockData() {
        for (auto ptr : block_data) {
            if (ptr) delete[] ptr;
        }
    }

    int get_index_by_id(unsigned id) {
        for (unsigned i = 0; i < id_list.size(); i++) {
            if (id_list[i] == id) {
                return i;
            }
        }
        return -1;
    }

    BlockData(const BlockData&) = delete;
    BlockData& operator=(const BlockData&) = delete;

    BlockData(BlockData&&) = default;
    BlockData& operator=(BlockData&&) = default;
};


typedef std::lock_guard<std::mutex> LockGuard;

static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    int left=0, right=K-1;
    if(addr[left].distance>nn.distance){
        memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if(addr[right].distance<nn.distance){
        addr[K] = nn;
        return K;
    }
    while(left<right-1){
        int mid=(left+right)/2;
        if(addr[mid].distance>nn.distance)right=mid;
        else left=mid;
    }
    //check equal ID

    while (left > 0){
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }

    if(addr[left].id == nn.id||addr[right].id==nn.id) {

        return K+1;
    }
    memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));

    addr[right]=nn;

    return right;
}

struct MemoryPool {
    std::vector<std::vector<unsigned>> neighbors_pool;
    std::unordered_map<unsigned, size_t> id_to_index;
};

class NSG: public ProximityGraph {
public:
    explicit NSG(const size_t dimension, const size_t n, ProximityGraph *initializer);

    virtual ~NSG();

    void init_files();
    void clear_file_cache();

    virtual void build(size_t n, const float* data, unsigned range, unsigned l, unsigned maxc, char* knn_graph_path, bool flag_nn_graph = false);
    void build_memory_graph_starling(size_t n, const float* data, unsigned range, unsigned l, unsigned maxc);
    virtual void Save(const char *filename);
    void Save_navigation_graph(const char *filename, std::unordered_map<unsigned, unsigned> new_old_node_map);

    // data and offset load
    virtual void Load(const char *filename);
    void Load_offsets_NO();
    void Load_offset_block();
    void Load_navigation_graph(char* filename);
    void Load_offsets();
    void Load_memory_layers(unsigned layer);
    void Load_memory_layers_NO(unsigned layer);
    // knn graph load
    void Load_nn_graph(const char *filename);

    // block data load starling
    BlockData<float>* load_block_by_node_id(unsigned id);
    void load_block_return(unsigned id, BlockData<float>* block);
    void load_block_return_buffer(unsigned id, std::vector<char>& buffer);

    // NSG build functions
    void get_neighbors(const float *query, unsigned l, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset);
    void get_neighbors(const float *query, unsigned l, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset);
    void init_graph(unsigned l);
    virtual void Link(unsigned range, unsigned l, unsigned maxc, SimpleNeighbor *cut_graph_);
    void Link_top(unsigned range, unsigned l, unsigned maxc, SimpleNeighbor *cut_graph_);
    void get_neighbors_any_final_grph(const float *query, unsigned l, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &retset,
    std::vector<Neighbor> &fullset, std::vector<unsigned> &flag_ids);
    void get_neighbors_any_final_grph(const float *query, unsigned l, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset);
    void sync_prune(unsigned q, std::vector<Neighbor> &pool, unsigned range, unsigned maxc, boost::dynamic_bitset<> &flags, SimpleNeighbor *cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void tree_grow(unsigned l);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, unsigned l);


    // search functions
    void Search(const float *query, const float *x, size_t K, unsigned l, unsigned *indices);
    // starling
    void Search_IO_starling_PQ_pruning(const float *query, size_t K, unsigned l, unsigned *indices, uint8_t* query_pq_vec);
    void Search_navigation_graph(const float *query, unsigned l, std::vector<Neighbor>& retset, boost::dynamic_bitset<>& flags);
    void Search_starling(const float *query, size_t K, unsigned l, unsigned *indices, uint8_t* query_pq_vec);

    //
    void Search_block_aware_multi_layer_memory(const uint8_t *pq_query, unsigned l, std::vector<Neighbor>& retset,
        boost::dynamic_bitset<>& flags);
    void Search_block_aware_multi_layer_memory_NO(const uint8_t *pq_query, unsigned l, std::vector<Neighbor>& retset,
        boost::dynamic_bitset<>& flags);
    // boost::dynamic_bitset<>& flags);
    void Search_one_layer_memory(const uint8_t* query_pq, unsigned l, std::vector<Neighbor>& retset,
        boost::dynamic_bitset<>& flags,unsigned layer);
    void Search_one_layer_memory_NO(const uint8_t* query_pq, unsigned l, std::vector<Neighbor>& retset,
        boost::dynamic_bitset<>& flags,unsigned layer);

    void Search_one_layer_disk_BA(unsigned l, unsigned layer, std::vector<Neighbor>& retset, boost::dynamic_bitset<>& flags,
        std::unordered_map<unsigned, unsigned>& result);

    void Search_reranking(const float *query, unsigned K, std::vector<Neighbor>& retset, float*& this_vector_data);

    void load_block_buffer(unsigned offset, char* buffer);

    void Search_BA(const float *query,  const uint8_t *query_pq, size_t K, unsigned l, unsigned *indices);

    void load_raw_vector_block(std::vector<std::tuple<long long, unsigned, unsigned>> offset_inner_offset, float* vector_data);

    void block_shuffling(const std::string& algorithm_name, unsigned BLOCK_SIZE, char* data_layout_file_name, char* offset_file_name, const float* data);

    void init_final_graph(std::vector<std::vector<unsigned>> knn_graph);
    std::vector<std::vector<unsigned>> get_final_graph();
    unsigned get_ep() const {
        return ep;
    };
    void load_vector_data(const float * vectors) {
        vector_data = vectors;
    }

    // for diskANN
    std::vector<unsigned> select_numbers(unsigned n, unsigned r, unsigned k);
    void save_diskANN(char* filename, char* offset_filename);

    // search diskANN
    void Search_diskANN_PQ(const float *query, size_t K, unsigned l, unsigned *indices, uint8_t* query_pq_vec);
    void load_block_return_disk_ANN(unsigned id, BlockData<float>* block);
    unsigned diskANN_adaptive_num_per_block;

    std::string flag_layer = "";
    boost::dynamic_bitset<> flags_in_out;
    std::unordered_map<unsigned, unsigned> last_layer_map;
    std::unordered_map<unsigned, unsigned> reverse_layer_map;
    unsigned last_ep;
    const float* last_vector_data;
    // multi layer search
    unsigned memory_layer;
    unsigned total_layer;
    std::string algorithm_name;
    boost::unordered_map<unsigned, std::pair<long long, unsigned>> ids_raw_offset;

    // data layout
    DataLayout* layout{};
    // pq related
    PQ* pq;

    // offset file path
    std::string index_offset_file_path;

    std::string data_layout_file_path; //
    std::string memory_data_layout_file_path; //
    std::string data_layout_offset_file_path; //
    std::string raw_data_file_path; //
    std::string raw_data_offset_file_path; //

    std::vector<std::vector<unsigned>> final_graph;

    // for starling navigation graph
    std::map<unsigned, std::vector<unsigned>> navigation_graph;
    std::map<unsigned, const float*> navigation_data;
    unsigned navigation_ep{};

    // for metric
    // count the IO times
    unsigned IO_counter = 0;

    std::vector<int> visit_count{};
    std::unordered_map<unsigned, unsigned> block_size_map;

    bool debug_flag = false;
    std::vector<unsigned> count_loaded_block;
    std::vector<BlockData<float>*> loaded_block_float;
    std::queue<unsigned> delete_queue;
    unsigned queue_length = 2000; // default

    //
    // MemoryPool memory_pool;

    unsigned buffer_size = 4 * 1024; // default
    unsigned max_neighbor = 50; // default
    unsigned num_node_per_block = 40; // default
    unsigned node_size;
    unsigned raw_vector_per_block;

    unsigned cluster_i;

    std::vector<std::vector<Neighbor>> pools;
    std::string flag_iopg;

    // Vamana
    float alpha_vamana = 1.1; // default


protected:
    std::vector<uint8_t*> memory_pq_vectors;

    std::vector<std::vector<std::vector<unsigned>>> multi_layer_graphs;
    std::vector<std::unordered_map<unsigned, std::vector<unsigned>>> multi_layer_graphs_NO;
    std::vector<std::vector<std::vector<unsigned>>> multi_layer_offsets;

    std::vector<std::vector<unsigned>> multi_layer_node_ids;

    ProximityGraph *initializer = nullptr;;
    char* opt_graph_ = nullptr;

    boost::unordered_map<unsigned, long long> ids_block_offset;
    boost::unordered_map<unsigned, unsigned> ids_block_id;

    std::map<unsigned, boost::unordered_map<unsigned, long long>> multi_layer_ids_block_offset;
    std::map<unsigned, boost::unordered_map<unsigned, unsigned>> multi_layer_ids_block_id;
    std::vector<unsigned> multi_layer_start_node;
    std::vector<unsigned> multi_layer_start_node_offset;
    std::vector<unsigned> multi_layer_start_offset;
    unsigned* number_of_nodes_each_layer;

    int fd_data_layout = -1;
    char* mapped_data_layout = static_cast<char *>(MAP_FAILED);
    size_t file_size_data_layout = 0;

    int fd_raw_vector = -1;
    char* mapped_raw_vector = static_cast<char *>(MAP_FAILED);
    size_t file_size_raw_vector = 0;

    unsigned ep{};
    unsigned width{};
};


#endif //NSG_H
