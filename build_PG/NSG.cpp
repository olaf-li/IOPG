//
// Created by cshlli on 2024/12/21.
//

#include "NSG.h"

#include <iostream>
#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <utility>
#include <bits/fs_fwd.h>
#include <boost/xpressive/regex_primitives.hpp>
#include <xmmintrin.h>   // for _mm_prefetch
#include <cstdlib>       // for posix_memalign, free
#include "BAMG.h"

NSG::NSG(const size_t dimension, const size_t n, ProximityGraph *initializer)
    : ProximityGraph(dimension, n), initializer{initializer} {
}

NSG::~NSG() {
    if (initializer != nullptr) {
        delete initializer;
        initializer = nullptr;
    }
    if (opt_graph_ != nullptr) {
        delete opt_graph_;
        opt_graph_ = nullptr;
    }

    if (mapped_data_layout != MAP_FAILED) {
        munmap(mapped_data_layout, file_size_data_layout);
    }
    if (fd_data_layout != -1) {
        close(fd_data_layout);
    }
}

void NSG::init_files() {
    using namespace std::chrono;
    auto s = high_resolution_clock::now();

    if (algorithm_name == "BA") {
        fd_raw_vector = open(raw_data_file_path.c_str(), O_RDONLY | O_DIRECT);
        if (fd_raw_vector == -1) {
            throw std::runtime_error("Could not open raw data file: " + raw_data_file_path);
        }
        struct stat st;
        if (fstat(fd_raw_vector, &st) == -1) {
            close(fd_raw_vector);
            throw std::runtime_error("Could not stat raw data file: " + raw_data_file_path);
        }
        file_size_raw_vector = st.st_size;
    }

    fd_data_layout = open(data_layout_file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_data_layout == -1) {
        throw std::runtime_error("Failed to open data layout file: " + data_layout_file_path);
    }
    struct stat sb;
    if (fstat(fd_data_layout, &sb) == -1) {
        close(fd_data_layout);
        throw std::runtime_error("Failed to get file status for data layout file");
    }
    file_size_data_layout = sb.st_size;

    auto e = high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    IO_time_disk += diff.count();
}

void NSG::clear_file_cache() {

    posix_fadvise(fd_data_layout, 0, file_size_data_layout, POSIX_FADV_DONTNEED);
    if (algorithm_name == "BA") {
        posix_fadvise(fd_raw_vector, 0, file_size_raw_vector, POSIX_FADV_DONTNEED);
    }
}

void NSG::get_neighbors(const float *query, unsigned l, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset) {
    retset.resize(l + 1);
    std::vector<unsigned> init_ids(l);

    boost::dynamic_bitset<> flags{n_, 0};
    l = 0;
    for (unsigned i = 0; i < init_ids.size() && i < final_graph[ep].size(); i++) {
        init_ids[i] = final_graph[ep][i];
        flags[init_ids[i]] = true;
        l++;
    }


    while (l < init_ids.size()) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        init_ids[l] = id;
        l++;
        flags[id] = true;
    }

    l = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= n_) continue;
        float dist =euclideanDistance(vector_data + dimension_ * (size_t)id, query, dimension_);
        retset[i] = Neighbor(id, dist, true);
        l++;
    }

    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int)l) {
        int nk = l;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < final_graph[n].size(); ++m) {
                unsigned id = final_graph[n][m];
                if (flags[id]) continue;
                flags[id] = true;

                float dist = euclideanDistance(query, vector_data + dimension_ * (size_t)id, dimension_);

                Neighbor nn(id, dist, true);
                fullset.push_back(nn);
                if (dist >= retset[l - 1].distance) continue;
                int r = InsertIntoPool(retset.data(), l, nn);

                if (l + 1 < retset.size()) ++l;
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }

}

void NSG::get_neighbors(const float *query, unsigned l, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &retset,
    std::vector<Neighbor> &fullset) {
    retset.resize(l + 1);
    std::vector<unsigned> init_ids(l);

    l = 0;
    for (unsigned i = 0; i < init_ids.size() && i < final_graph[ep].size(); i++) {
        init_ids[i] = final_graph[ep][i];
        flags[init_ids[i]] = true;
        l++;
    }

    while (l < init_ids.size()) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        init_ids[l] = id;
        l++;
        flags[id] = true;
    }

    l = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= n_) continue;
        float dist =euclideanDistance(vector_data + dimension_ * (size_t)id, query, dimension_);
        retset[i] = Neighbor(id, dist, true);
        fullset.push_back(retset[i]);
        l++;
    }

    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int)l) {
        int nk = l;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < final_graph[n].size(); ++m) {
                unsigned id = final_graph[n][m];
                if (flags[id]) continue;
                flags[id] = true;

                float dist = euclideanDistance(query, vector_data + dimension_ * (size_t)id, dimension_);
                Neighbor nn(id, dist, true);
                fullset.push_back(nn);
                if (dist >= retset[l - 1].distance) continue;
                int r = InsertIntoPool(retset.data(), l, nn);

                if (l + 1 < retset.size()) ++l;
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}

void NSG::get_neighbors_any_final_grph(const float *query, unsigned l, boost::dynamic_bitset<> &flags, std::vector<Neighbor> &retset,
    std::vector<Neighbor> &fullset, std::vector<unsigned> &flag_ids) {
    unsigned number = final_graph.size();

    retset.resize(l + 1);
    std::vector<unsigned> init_ids(l);
    l = 0;
    for (unsigned i = 0; i < init_ids.size() && i < final_graph[last_ep].size(); i++) {
        init_ids[i] = final_graph[last_ep][i];
        flags[init_ids[i]] = true;
        flag_ids.push_back(init_ids[i]);
        l++;
    }
    while (l < init_ids.size()) {
        unsigned id = rand() % number;
        if (flags[id]) continue;
        init_ids[l] = id;
        l++;
        flags[id] = true;
        flag_ids.push_back(id);
    }
    l = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= number) continue;
        float dist =euclideanDistance(last_vector_data + dimension_ * (size_t)id, query, dimension_);
        retset[i] = Neighbor(id, dist, true);
        fullset.push_back(retset[i]);
        l++;
    }
    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int)l) {
        int nk = l;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;
            for (unsigned m = 0; m < final_graph[n].size(); ++m) {
                unsigned id = final_graph[n][m];
                if (flags[id]) continue;
                flags[id] = true;
                flag_ids.push_back(id);
                float dist = euclideanDistance(query, last_vector_data + dimension_ * (size_t)id, dimension_);
                Neighbor nn(id, dist, true);
                fullset.push_back(nn);
                if (dist >= retset[l - 1].distance) continue;
                int r = InsertIntoPool(retset.data(), l, nn);
                if (l + 1 < retset.size()) ++l;
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}

void NSG::get_neighbors_any_final_grph(const float *query, unsigned l, std::vector<Neighbor> &retset, std::vector<Neighbor> &fullset) {
    unsigned number = final_graph.size();
    retset.resize(l + 1);
    std::vector<unsigned> init_ids(l);
    boost::dynamic_bitset<> flags{final_graph.size(), 0};
    l = 0;
    for (unsigned i = 0; i < init_ids.size() && i < final_graph[last_ep].size(); i++) {
        init_ids[i] = final_graph[last_ep][i];
        flags[init_ids[i]] = true;
        l++;
    }
    while (l < init_ids.size()) {
        unsigned id = rand() % number;
        if (flags[id]) continue;
        init_ids[l] = id;
        l++;
        flags[id] = true;
    }
    l = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= number) continue;
        float dist =euclideanDistance(last_vector_data + dimension_ * (size_t)id, query, dimension_);
        retset[i] = Neighbor(id, dist, true);
        fullset.push_back(retset[i]);
        l++;
    }
    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int)l) {
        int nk = l;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < final_graph[n].size(); ++m) {
                unsigned id = final_graph[n][m];
                if (flags[id]) continue;
                flags[id] = true;

                float dist = euclideanDistance(query, last_vector_data + dimension_ * (size_t)id, dimension_);
                Neighbor nn(id, dist, true);
                fullset.push_back(nn);
                if (dist >= retset[l - 1].distance) continue;
                int r = InsertIntoPool(retset.data(), l, nn);

                if (l + 1 < retset.size()) ++l;
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}

void NSG::init_graph(unsigned l) {
    std::cout << n_ << std::endl;
    float *center = new float[dimension_];
    for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
    for (unsigned i = 0; i < n_; i++) {
        for (unsigned j = 0; j < dimension_; j++) {
            center[j] += vector_data[i * dimension_ + j];
        }
    }
    for (unsigned j = 0; j < dimension_; j++) {
        center[j] /= n_;
    }
    std::vector<Neighbor> tmp, pool;
    ep = rand() % n_;
    if (flag_layer == "top") {
        get_neighbors_any_final_grph(center,400, tmp, pool);
        for (unsigned i = 0; i < tmp.size(); i++) {
            if (flags_in_out[last_layer_map[tmp[i].id]] == true) {
                ep = reverse_layer_map[last_layer_map[tmp[i].id]];
                break;
            }
        }
    }
    else {
        get_neighbors(center, l, tmp, pool);
        ep = tmp[0].id;
    }
    delete center;
}

void NSG::Link(unsigned range, unsigned l, unsigned maxc, SimpleNeighbor *cut_graph_) {

    std::cout << "graph link" << std::endl;
    std::vector<std::mutex> locks(n_);
    if (flag_iopg == "iopg") {
        std::cout << flag_iopg << std::endl;
        pools.resize(n_);
    }

#pragma omp parallel
    {
        std::vector<Neighbor> pool, tmp;
        boost::dynamic_bitset<> flags{n_, 0};
#pragma omp for schedule(dynamic, 64)
        for (unsigned n = 0; n < n_; ++n) {

            if (n % 1000000 == 0) std::cout << " " << n << std::endl;

            pool.clear();
            tmp.clear();
            flags.reset();

            if (flag_layer == "top") {
                unsigned temp_l = l;
                unsigned iter = 0;
                std::vector<Neighbor> pool_pre, tmp_pre;
                std::vector<unsigned> flag_ids;
                boost::dynamic_bitset<> flags_pre{final_graph.size(), 0};
                while (true){
                    get_neighbors_any_final_grph(vector_data + dimension_ * n, temp_l, flags_pre, tmp, pool_pre, flag_ids);
                    for (unsigned i = 0; i < pool_pre.size(); i++) {
                        if (flags_in_out[last_layer_map[pool_pre[i].id]] == true) {
                            pool_pre[i].id = reverse_layer_map[last_layer_map[pool_pre[i].id]];
                            pool.push_back(pool_pre[i]);
                        }
                    }
                    pool_pre.clear();
                    flags.reset();
                    flag_ids.clear();
                    tmp_pre.clear();
                    iter++;
                    if (pool.size() >= maxc || iter >= 2) {
                        for (auto flag_id : flag_ids) {
                            unsigned raw_id = last_layer_map[flag_id];
                            if (flags_in_out[raw_id] == true) {
                                flags[reverse_layer_map[raw_id]] = true;
                            }
                        }
                        break;
                    }
                    else {
                        temp_l = temp_l + l;
                        pool.clear();
                    }
                }
            }
            else {
                get_neighbors(vector_data + dimension_ * n, l, flags, tmp, pool);
            }
            if (flag_iopg == "iopg") {
                pools[n] = pool;
            }
            sync_prune(n, pool, range, maxc, flags, cut_graph_);
        }
    }

    std::cout << "insert node: " << std::endl;
#pragma omp parallel
{
#pragma omp for schedule(dynamic, 64)
    for (unsigned n = 0; n < n_; ++n) {
        if (n % 100000 == 0) {
            std::cout << n << std::endl;
        }
        InterInsert(n, range, locks, cut_graph_);
    }
}
    std::cout << "end insert node" << std::endl;
}

void NSG::Link_top(unsigned range, unsigned l, unsigned maxc, SimpleNeighbor *cut_graph_) {

    std::cout << "graph link" << std::endl;
    std::vector<std::mutex> locks(n_);
        std::vector<Neighbor> pool, tmp;
        boost::dynamic_bitset<> flags{n_, 0};
        for (unsigned n = 0; n < n_; ++n) {
            pool.clear();
            tmp.clear();
            flags.reset();

            if (flag_layer == "top") {
                unsigned temp_l = l;
                unsigned iter = 0;
                std::vector<Neighbor> pool_pre, tmp_pre;
                std::vector<unsigned> flag_ids;
                boost::dynamic_bitset<> flags_pre{final_graph.size(), 0};
                while (true){
                    get_neighbors_any_final_grph(vector_data + dimension_ * n, temp_l, flags_pre, tmp, pool_pre, flag_ids);
                    for (unsigned i = 0; i < pool_pre.size(); i++) {
                        if (flags_in_out[last_layer_map[pool_pre[i].id]] == true) {
                            pool_pre[i].id = reverse_layer_map[last_layer_map[pool_pre[i].id]];
                            pool.push_back(pool_pre[i]);
                        }
                    }
                    pool_pre.clear();
                    flags.reset();
                    flag_ids.clear();
                    tmp_pre.clear();
                    iter++;
                    if (pool.size() >= maxc || iter >= 2) {
                        for (auto flag_id : flag_ids) {
                            unsigned raw_id = last_layer_map[flag_id];
                            if (flags_in_out[raw_id] == true) {
                                flags[reverse_layer_map[raw_id]] = true;
                            }
                        }
                        break;
                    }
                    else {
                        temp_l = temp_l + l;
                        pool.clear();
                    }
                }
            }
            else {
                get_neighbors(vector_data + dimension_ * n, l, flags, tmp, pool);
            }
            sync_prune(n, pool, range, maxc, flags, cut_graph_);
        }
    std::cout << "insert node" << std::endl;

#pragma omp for schedule(dynamic, 64)
    for (unsigned n = 0; n < n_; ++n) {
        InterInsert(n, range, locks, cut_graph_);
    }
    std::cout << "end insert node" << std::endl;
}

void NSG::InterInsert(unsigned n, unsigned range, std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
            float djk = euclideanDistance(vector_data + dimension_ * (size_t)result[t].id, vector_data +
                dimension_ * (size_t)p.id, dimension_);
          if (djk * alpha_vamana < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}

void NSG::sync_prune(unsigned q, std::vector<Neighbor> &pool, unsigned range, unsigned maxc, boost::dynamic_bitset<>
    &flags, SimpleNeighbor *cut_graph_) {
    width = range;
    unsigned start = 0;
    if (flag_layer != "top") {
        for (unsigned nn = 0; nn < final_graph[q].size(); nn++) {
            unsigned id = final_graph[q][nn];
            if (flags[id]) continue;
            float dist = euclideanDistance(vector_data + dimension_ * (size_t)q, vector_data + dimension_ *
                (size_t)id, dimension_);
            pool.push_back(Neighbor(id, dist, true));
        }
    }

    std::sort(pool.begin(), pool.end());
    std::vector<Neighbor> result;
    if (pool[start].id == q) start++;
    result.push_back(pool[start]);

    while (result.size() < range && (++start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
            if (p.id == result[t].id) {
                occlude = true;
                break;
            }
            float djk = euclideanDistance(vector_data + dimension_ * (size_t)result[t].id, vector_data +
                dimension_ * (size_t)p.id, dimension_);
            if (djk * alpha_vamana < p.distance /* dik */) {
                occlude = true;
                break;
            }
        }
        if (!occlude) result.push_back(p);
    }

    SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
    for (size_t t = 0; t < result.size(); t++) {
        des_pool[t].id = result[t].id;
        des_pool[t].distance = result[t].distance;
    }
    if (result.size() < range) {
        des_pool[result.size()].distance = -1;
    }
}

void NSG::build(size_t n, const float* data, unsigned range, unsigned l, unsigned maxc, char* knn_graph_path, bool flag_nn_graph) {
    if (!flag_nn_graph) {
        if (final_graph.size() == 0) {
            std::cout<< "knn graph path: " << knn_graph_path << std::endl;
            Load_nn_graph(knn_graph_path);
            std::cout << "load nn graph" << std::endl;
        }
    }

    vector_data = data;
    init_graph(l);

    std::cout << "init the graph" << std::endl;
    SimpleNeighbor* cut_graph = new SimpleNeighbor[n_ * (size_t)range];
    if(flag_layer == "top") {
        Link_top(range, l, maxc, cut_graph);
    }
    else {
        Link(range, l, maxc, cut_graph);
    }
    final_graph.resize(n_);
    for (size_t i = 0; i < n_; i++) {
        SimpleNeighbor* pool = cut_graph + i * (size_t)range;
        unsigned pool_size = 0;
        for (unsigned j = 0; j < range; j++) {
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

void NSG::build_memory_graph_starling(size_t n, const float *data, unsigned range, unsigned l, unsigned maxc) {
    vector_data = data;
    init_graph(l);
    std::cout << "init the graph" << std::endl;
    SimpleNeighbor* cut_graph = new SimpleNeighbor[n_ * (size_t)range];
    Link(range, l, maxc, cut_graph);
    final_graph.resize(n_);
    for (size_t i = 0; i < n_; i++) {
        SimpleNeighbor* pool = cut_graph + i * (size_t)range;
        unsigned pool_size = 0;
        for (unsigned j = 0; j < range; j++) {
            if (pool[j].distance == -1) break;
            pool_size = j;
        }
        pool_size++;
        final_graph[i].resize(pool_size);
        for (unsigned j = 0; j < pool_size; j++) {
            final_graph[i][j] = pool[j].id;
        }
    }
    tree_grow(l);
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

void NSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
    unsigned tmp = root;
    std::stack<unsigned> s;
    s.push(root);
    if (!flag[root]) cnt++;
    flag[root] = true;
    while (!s.empty()) {
        unsigned next = n_ + 1;
        for (unsigned i = 0; i < final_graph[tmp].size(); i++) {
            if (flag[final_graph[tmp][i]] == false) {
                next = final_graph[tmp][i];
                break;
            }
        }
        if (next == (n_ + 1)) {
            s.pop();
            if (s.empty()) break;
            tmp = s.top();
            continue;
        }
        tmp = next;
        flag[tmp] = true;
        s.push(tmp);
        cnt++;
    }
}

void NSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root, unsigned l) {
    unsigned id = n_;
    for (unsigned i = 0; i < n_; i++) {
        if (flag[i] == false) {
            id = i;
            break;
        }
    }
    if (id == n_) return;

    std::vector<Neighbor> tmp, pool;
    get_neighbors(vector_data + dimension_ * id, l, tmp, pool);
    std::sort(pool.begin(), pool.end());

    unsigned found = 0;
    for (unsigned i = 0; i < pool.size(); i++) {
        if (flag[pool[i].id]) {
            root = pool[i].id;
            found = 1;
            break;
        }
    }
    if (found == 0) {
        while (true) {
            unsigned rid = rand() % n_;
            if (flag[rid]) {
                root = rid;
                break;
            }
        }
    }
    final_graph[root].push_back(id);
}

void NSG::tree_grow(unsigned l) {
    unsigned root = ep;
    boost::dynamic_bitset<> flags{n_, 0};
    unsigned unlinked_cnt = 0;
    while (unlinked_cnt < n_) {
        DFS(flags, root, unlinked_cnt);
        if (unlinked_cnt >= n_) break;
        findroot(flags, root, l);
    }
    for (size_t i = 0; i < n_; ++i) {
        if (final_graph[i].size() > width) {
            width = final_graph[i].size();
        }
    }
}

void NSG::Save(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    assert(final_graph.size() == n_);

    out.write((char *)&width, sizeof(unsigned));
    out.write((char *)&ep, sizeof(unsigned));
    for (unsigned i = 0; i < n_; i++) {
        unsigned GK = (unsigned)final_graph[i].size();
        out.write((char *)&GK, sizeof(unsigned));
        out.write((char *)final_graph[i].data(), GK * sizeof(unsigned));
    }
    out.close();
    std::cout << "Saved file: " << filename << std::endl << final_graph.size() << std::endl;
}

void NSG::Save_navigation_graph(const char *filename, std::unordered_map<unsigned, unsigned> new_old_node_map) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    assert(flag_layer == "top");

    out.write((char *)&n_, sizeof(unsigned));
    out.write((char *)&ep, sizeof(unsigned));
    for (unsigned i = 0; i < n_; i++) {
        out.write((char *)&new_old_node_map[i], sizeof(unsigned));
        out.write((char *)(vector_data + i * dimension_), dimension_ * sizeof(float));
        unsigned GK = (unsigned)final_graph[i].size();
        out.write((char *)&GK, sizeof(unsigned));
        for (unsigned j = 0; j < GK; j++) {
            out.write((char *)&new_old_node_map[final_graph[i][j]], sizeof(unsigned));
        }
    }
    out.close();
    std::cout << "Saved file: " << filename << std::endl;
    std::cout << "memory size: " << final_graph.size() << std::endl;
}

void NSG::Load(const char *filename) {
    std::cout<<"Loading： " << filename <<std::endl;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can not open: " << filename << std::endl;
        exit(1);
    }

    in.read((char *)&width, sizeof(unsigned));
    in.read((char *)&ep, sizeof(unsigned));
    unsigned cc = 0;
    while (!in.eof()) {
        unsigned k;
        in.read((char *)&k, sizeof(unsigned));
        if (in.eof()) break;
        cc += k;
        std::vector<unsigned> tmp(k);
        in.read((char *)tmp.data(), k * sizeof(unsigned));
        final_graph.push_back(tmp);
    }
    std::cout<< "edges:" << cc <<std::endl;
    cc /= n_;
    std::cout<< "average degree:" << cc <<std::endl;
}

void NSG::Load_offset_block() {
    std::ifstream in(data_layout_offset_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can not open: " << data_layout_offset_file_path << std::endl;
        exit(1);
    }

    unsigned n_blocks = 0;
    while (!in.eof()) {
        unsigned block_id;
        in.read((char *)&block_id, sizeof(unsigned));
        long long offset;
        in.read((char *)&offset, sizeof(long long));
        if (in.eof()) break;
        unsigned node_size;
        in.read((char *)&node_size, sizeof(unsigned));
        std::vector<unsigned> tmp(node_size);
        in.read((char *)tmp.data(), node_size * sizeof(unsigned));
        for (unsigned i = 0; i < node_size; i++) {
            ids_block_offset.emplace(tmp[i], offset);
            ids_block_id.emplace(tmp[i], block_id);
        }
        n_blocks++;
    }
    assert(ids_block_offset.size() == n_);
}

void NSG::Load_navigation_graph(char* filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can not open: " << filename << std::endl;
        exit(1);
    }

    unsigned graph_size;
    in.read((char *)&graph_size, sizeof(unsigned));
    unsigned start_point;
    in.read((char *)&start_point, sizeof(unsigned));
    navigation_ep = start_point;
    for (unsigned i = 0; i < graph_size; i++) {
        unsigned id;
        in.read((char *)&id, sizeof(unsigned));
        auto vec = new float[dimension_];
        in.read((char *)vec, dimension_ * sizeof(float));
        unsigned neighbor_size;
        in.read((char *)&neighbor_size, sizeof(unsigned));
        std::vector<unsigned> tmp(neighbor_size);
        for (unsigned j = 0; j < neighbor_size; j++) {
            in.read((char *)&tmp[j], sizeof(unsigned));
        }
        navigation_data.emplace(id, vec);
        navigation_graph.emplace(id, tmp);
    }
}

void NSG::Load_offsets() {
    std::ifstream in(data_layout_offset_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can not open: " << data_layout_offset_file_path << std::endl;
        exit(1);
    }

    unsigned number_of_layers;
    in.read((char *)&number_of_layers, sizeof(unsigned));
    unsigned* temp_number_of_nodes = new unsigned[number_of_layers];
    for (unsigned i = 0; i < number_of_layers; i++) {
        in.read((char *)&temp_number_of_nodes[i], sizeof(unsigned));
    }
    number_of_nodes_each_layer = temp_number_of_nodes;
    total_layer = number_of_layers;

    while (!in.eof()) {
        unsigned layer;
        in.read((char *)&layer, sizeof(unsigned));
        unsigned number_of_nodes;
        in.read((char *)&number_of_nodes, sizeof(unsigned));
        unsigned start_node;
        in.read((char *)&start_node, sizeof(unsigned));

        boost::unordered_map<unsigned, long long> temp_ids_block_offset;
        boost::unordered_map<unsigned, unsigned> temp_ids_block_id;
        unsigned count_node = 0;
        bool flag = true;
        while (count_node < number_of_nodes) {
            long long offset;
            in.read((char *)&offset, sizeof(long long));
            if (in.eof()) {
                flag = false;
                break;
            }
            unsigned node_size;
            in.read((char *)&node_size, sizeof(unsigned));

            unsigned tmp;
            unsigned tmp_inner_offset;
            count_node += node_size;
            for (unsigned j = 0; j < node_size; j++) {
                in.read((char *)&tmp, sizeof(unsigned));
                in.read((char *)&tmp_inner_offset, sizeof(unsigned));
                temp_ids_block_offset.emplace(tmp, offset);
                temp_ids_block_id.emplace(tmp, tmp_inner_offset);
            }
        }

        if (flag) {
            multi_layer_ids_block_offset.emplace(layer, temp_ids_block_offset);
            multi_layer_start_node.push_back(start_node);
            multi_layer_ids_block_id.emplace(layer, temp_ids_block_id);
        }
    }

    std::ifstream raw_in(raw_data_offset_file_path, std::ios::binary);
    if (!raw_in.is_open()) {
        std::cerr << "can not open: " << raw_data_offset_file_path << std::endl;
        exit(1);
    }

    while (true) {
        unsigned id;
        long long offset;
        unsigned inner_offset;

        raw_in.read((char *)&id, sizeof(unsigned));
        raw_in.read((char *)&offset, sizeof(long long));
        raw_in.read((char *)&inner_offset, sizeof(unsigned));

        if (raw_in.eof()) break; // 正常结束
        if (!raw_in.good()) {    // 异常中断
            std::cerr << "Corrupted offset file!" << std::endl;
            exit(1);
        }

        ids_raw_offset.emplace(id, std::make_pair(offset, inner_offset));
    }
}


void NSG::Load_offsets_NO() {
    std::ifstream in(data_layout_offset_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can not open: " << data_layout_offset_file_path << std::endl;
        exit(1);
    }

    unsigned number_of_layers;
    in.read((char *)&number_of_layers, sizeof(unsigned));
    unsigned* temp_number_of_nodes = new unsigned[number_of_layers];
    for (unsigned i = 0; i < number_of_layers; i++) {
        in.read((char *)&temp_number_of_nodes[i], sizeof(unsigned));
        unsigned start_node;
        in.read((char *)&start_node, sizeof(unsigned));
        multi_layer_start_node.push_back(start_node);
    }
    number_of_nodes_each_layer = temp_number_of_nodes;
    total_layer = number_of_layers;
}

void NSG::Load_memory_layers(unsigned layer) {
    std::ifstream in(data_layout_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can not open: " << data_layout_file_path << std::endl;
        exit(1);
    }
    total_layer = multi_layer_ids_block_offset.size();

    memory_pq_vectors.resize(n_, nullptr);
    for (unsigned i = 0; i < layer; i++) {
        std::vector<std::vector<unsigned>> pg;
        std::vector<unsigned> layer_node_ids;
        pg.reserve(n_);
        for (unsigned j = 0; j < n_; j++) {
            pg.emplace_back();
        }
        boost::unordered_map<unsigned, long long> id_block_offset = multi_layer_ids_block_offset[i];
        std::set<long long> offset_set;
        for (auto & it : id_block_offset) {
            offset_set.insert(it.second);
        }
        unsigned number_of_nodes = 0;
        for (auto & it : offset_set) {
            in.seekg(it);
            unsigned node_size;
            in.read((char *)&node_size, sizeof(unsigned));
            for (unsigned j = 0; j < node_size; j++) {
                unsigned id;
                in.read((char *)&id, sizeof(unsigned));
                layer_node_ids.push_back(id);
                number_of_nodes++;
                uint8_t* pq_vec = new uint8_t[pq->M];
                in.read((char *)pq_vec, pq->M * sizeof(uint8_t));
                memory_pq_vectors[id] = pq_vec;
                unsigned in_k;
                in.read((char *)&in_k, sizeof(unsigned));
                std::vector<unsigned> tmp_in(in_k);
                in.read((char *)tmp_in.data(), in_k * sizeof(unsigned));
                for (unsigned k = 0; k < in_k; k++) {
                    pg[id].push_back(tmp_in[k]);
                }

                unsigned out_k;
                in.read((char *)&out_k, sizeof(unsigned));
                std::vector<unsigned> tmp_out(out_k);
                in.read((char *)tmp_out.data(), out_k * sizeof(unsigned));
                for (unsigned k = 0; k < out_k; k++) {
                    pg[id].push_back(tmp_out[k]);
                }
            }
        }
        multi_layer_graphs.push_back(pg);
        multi_layer_node_ids.push_back(layer_node_ids);
    }
}

void NSG::Load_memory_layers_NO(unsigned layer) {
    std::ifstream in(memory_data_layout_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "can not open: " << memory_data_layout_file_path << std::endl;
        exit(1);
    }

    total_layer = multi_layer_start_node.size();

    long long read_offset = 0;
    for (unsigned i = 0; i < layer; i++) {
        std::unordered_map<unsigned, std::vector<unsigned>> pg;
        unsigned total_nodes = number_of_nodes_each_layer[i];
        unsigned num_blocks = (total_nodes + num_node_per_block - 1) / num_node_per_block;
        for (unsigned j = 0; j < num_blocks; j++) {
            in.seekg(read_offset);
            for (unsigned k = 0; k < num_node_per_block; k++) {
                unsigned id;
                in.read((char *)&id, sizeof(unsigned));
                unsigned offset_id;
                in.read((char *)&offset_id, sizeof(unsigned));

                unsigned in_k;
                in.read((char *)&in_k, sizeof(unsigned));
                std::vector<unsigned> neighbors(max_neighbor);
                in.read((char *)neighbors.data(), max_neighbor * sizeof(unsigned));
                for (unsigned k = 0; k < in_k; k++) {
                    pg[offset_id].push_back(neighbors[k]);
                }

                for (unsigned k = in_k; k < max_neighbor; k++) {
                    if (neighbors[k] >= n_) break;
                    pg[offset_id].push_back(neighbors[k]);
                }
            }
            read_offset += 4096;
        }
        multi_layer_graphs_NO.push_back(pg);
    }
}

BlockData<float>* NSG::load_block_by_node_id(unsigned id) {
        auto* block = new BlockData<float>();
        load_block_return(id, block);
        return block;
}

void NSG::load_block_return_buffer(unsigned id, std::vector<char>& buffer) {
    auto s_IO = std::chrono::steady_clock::now();
    buffer.clear();
    const long long block_offset = ids_block_offset[id];
    const char* mapped_start = mapped_data_layout + block_offset;

    memcpy(buffer.data(), mapped_start, buffer_size);
    auto e_IO = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_IO = e_IO - s_IO;
    IO_time_disk += diff_IO.count();
    IO_count_disk++;
}

void NSG::load_block_return(unsigned id, BlockData<float>* block) {
    auto s_IO = std::chrono::steady_clock::now();

    const long long block_offset = ids_block_offset[id];
    const char* current = mapped_data_layout + block_offset;

    unsigned block_id;
    memcpy(&block_id, current, sizeof(unsigned));
    current += sizeof(unsigned);

    unsigned block_size;
    memcpy(&block_size, current, sizeof(unsigned));
    current += sizeof(unsigned);

    for (unsigned i = 0; i < block_size; i++) {
        unsigned node_id;
        memcpy(&node_id, current, sizeof(unsigned));
        current += sizeof(unsigned);

        auto* vec = new float[dimension_];
        memcpy(vec, current, dimension_ * sizeof(float));
        current += dimension_ * sizeof(float);

        unsigned k;
        memcpy(&k, current, sizeof(unsigned));
        current += sizeof(unsigned);

        std::vector<unsigned> neighbors(k);
        memcpy(neighbors.data(), current, k * sizeof(unsigned));
        current += k * sizeof(unsigned);

        block->block_data.push_back(vec);
        block->block_graph.push_back(neighbors);
        block->id_list.push_back(node_id);
    }
    auto e_IO = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_IO = e_IO - s_IO;
    IO_time_disk += diff_IO.count();
    IO_count_disk++;
}

void NSG::load_block_buffer(unsigned offset, char* buffer) {
    using namespace std::chrono;
    auto s_IO = steady_clock::now();

    ssize_t bytes_read = pread(fd_data_layout, buffer, buffer_size, offset);
    if (bytes_read != (ssize_t)4096) {
        free(buffer);
        throw std::runtime_error("buffer pread failed or incomplete read");
    }

    auto e_IO = steady_clock::now();
    std::chrono::duration<double> diff_IO = e_IO - s_IO;
    IO_time_disk += diff_IO.count();
    IO_count_disk++;
}

void NSG::init_final_graph(std::vector<std::vector<unsigned>> knn_graph) {
    this->final_graph = std::move(knn_graph);
}

std::vector<std::vector<unsigned>> NSG::get_final_graph() {
    return final_graph;
}


void NSG::Load_nn_graph(const char *filename) {
    std::ifstream in(filename, std::ios::binary);

    unsigned cc = 0;
    int i = 0;

    for (; i < n_; i++) {
        unsigned id;
        in.read((char *)(&id), sizeof(unsigned));
        in.seekg(dimension_ * sizeof(float), std::ios::cur);
        unsigned k;
        in.read((char *)(&k), sizeof(unsigned));
        cc += k;
        std::vector<unsigned> neighbors(k);
        in.read((char *)neighbors.data(), k * sizeof(unsigned));
        final_graph.push_back(neighbors);
    }

    std::cout<< "number of edge:" << cc << " " << i - 1 << std::endl;
    cc = cc / (i - 1);
    std::cout<< "average degree:" << cc << std::endl;
    in.close();
}

void NSG::Search(const float *query, const float *x, size_t K, unsigned l, unsigned *indices) {
    vector_data = x;
    std::vector<Neighbor> retset(l + 1);
    std::vector<unsigned> init_ids(l);
    boost::dynamic_bitset<> flags{n_, 0};

    unsigned tmp_l = 0;
    for (; tmp_l < l && tmp_l < final_graph[ep].size(); tmp_l++) {
        init_ids[tmp_l] = final_graph[ep][tmp_l];
        flags[init_ids[tmp_l]] = true;
    }

    while (tmp_l < l) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        float dist = euclideanDistance(vector_data + dimension_ * id, query, dimension_);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int)l) {
        int nk = l;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < final_graph[n].size(); ++m) {
                unsigned id = final_graph[n][m];
                if (flags[id]) continue;
                flags[id] = 1;
                float dist = euclideanDistance(query, vector_data + dimension_ * id, dimension_);
                if (dist >= retset[l - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), l, nn);

                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    for (size_t i = 0; i < K; i++) {
        indices[i] = retset[i].id;
    }
}

void NSG::Search_navigation_graph(const float *query, unsigned l, std::vector<Neighbor>& retset,
    boost::dynamic_bitset<>& flags) {

    std::vector<unsigned> init_ids(l);
    unsigned tmp_l = 0;
    std::vector<unsigned> keys;
    for (const auto& kv : navigation_graph) {
        keys.push_back(kv.first);
    }
    while (tmp_l < l) {
        unsigned id = keys[rand() % keys.size()];
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        float dist = euclideanDistance(navigation_data[id], query, dimension_);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + l);
    int k = 0;
    while (k < (int)l) {
        int nk = l;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < navigation_graph[n].size(); ++m) {
                unsigned id = navigation_graph[n][m];
                if (flags[id]) continue;
                flags[id] = true;
                float dist = euclideanDistance(query, navigation_data[id], dimension_);
                if (dist >= retset[l - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), l, nn);

                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}

void NSG::Search_IO_starling_PQ_pruning(const float *query, size_t K, unsigned l, unsigned *indices, uint8_t* query_pq_vec) {
    std::vector<Neighbor> retset(11);
    std::vector<unsigned> init_ids(10);
    boost::dynamic_bitset<> flags{n_, 0};

    Search_navigation_graph(query, 10, retset, flags);

    flags.reset();
    retset.resize(l + 1);
    init_ids.resize(l);
    unsigned tmp_l = 0;

    for(unsigned i = 0; i < retset.size(); i++) {
        init_ids[tmp_l] = retset[i].id;
        flags[init_ids[tmp_l]] = true;
        tmp_l++;
    }
    BlockData<float>* block = load_block_by_node_id(ep);
    unsigned block_index_ep = block->get_index_by_id(ep);
    for (; tmp_l < l && tmp_l < block->block_graph[block_index_ep].size(); tmp_l++) {
        init_ids[tmp_l] = block->block_graph[block_index_ep][tmp_l];
        flags[init_ids[tmp_l]] = true;
    }

    delete block;
    while (tmp_l < l) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }
    std::vector<Neighbor> result(l + 1);

    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        BlockData<float>* block = load_block_by_node_id(id);
        unsigned block_index_id = block->get_index_by_id(id);
        float dist = euclideanDistance(block->block_data[block_index_id], query, dimension_);
        result[i] = Neighbor(id, dist, true);
        float pq_dist = pq->pq_distance_fast(pq->pq_vector + block->id_list[block_index_id] * pq->M);
        retset[i] = Neighbor(id, pq_dist, true);
    }

    std::sort(retset.begin(), retset.begin() + l);
    std::sort(result.begin(), result.begin() + l);
    boost::dynamic_bitset<> flags_result{n_, 0};
    for (unsigned i = 0; i < retset.size(); i++) {
        result[i] = Neighbor(retset[i].id, retset[i].distance, true);
        flags_result[result[i].id] = true;
    }

    int k = 0;
    while (k < (int)l) {
        int nk = l;
        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned retset_k_id = retset[k].id;
            auto* temp_block = load_block_by_node_id(retset[k].id);

            std::vector<std::pair<float, unsigned>> dist_indices;
            unsigned index = 0;
            for(auto id: temp_block->id_list) {
                float dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);
                dist_indices.push_back(std::make_pair(dist, index));
                index++;
            }

            std::sort(dist_indices.begin(), dist_indices.end(),
                [](const std::pair<float, unsigned>& a, const std::pair<float, unsigned>& b) {
                    return a.first < b.first;
                }
            );

            std::set<unsigned> pruned_indices;
            float pruning_rate = static_cast<int>(std::ceil(1 * temp_block->id_list.size()));
            for(unsigned index = 0; index < pruning_rate; index++) {
                pruned_indices.insert(dist_indices[index].second);
            }

            pruned_indices.insert(temp_block->get_index_by_id(retset_k_id));

            for (auto index: pruned_indices) {
                unsigned this_id = temp_block->id_list[index];

                if (flags_result[this_id]) continue;

                flags_result[this_id] = true;
                float real_dist = euclideanDistance(temp_block->block_data[index], query, dimension_);

                Neighbor neighbor = Neighbor(this_id, real_dist, true);
                int r_real = InsertIntoPool(result.data(), l, neighbor);
            }

            std::vector<unsigned> retset_neighbors = temp_block->block_graph[temp_block->get_index_by_id(retset_k_id)];
            for (unsigned m = 0; m < retset_neighbors.size(); m++) {
                unsigned id = retset_neighbors[m];
                if (flags[id]) {
                    continue;
                }
                flags[id] = true;
                float pq_dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);
                if (pq_dist >= retset[l - 1].distance) {
                    continue;
                }
                Neighbor nn(id, pq_dist, true);
                int r = InsertIntoPool(retset.data(), l, nn);
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }

    for (size_t i = 0; i < K; i++) {
        indices[i] = result[i].id;
    }
}

void NSG::Search_starling(const float *query, size_t K, unsigned l, unsigned *indices, uint8_t* query_pq_vec) {
    std::vector<Neighbor> retset(l + 1);
    std::vector<Neighbor> result(l + 1);
    std::vector<unsigned> init_ids(l);
    boost::dynamic_bitset<> flags{n_, 0};
    unsigned tmp_l = 0;

    char* buffer = new(std::align_val_t(4096)) char[4096];

    long long offset = ids_block_offset[ep];
    unsigned block_size;
    unsigned neighbor_k;
    auto* vector = new float[dimension_];
    load_block_buffer(offset, buffer);
    const char* parse_ptr = buffer + sizeof(unsigned);
    block_size = *reinterpret_cast<const unsigned*> (parse_ptr);
    parse_ptr += sizeof(unsigned);

    for (unsigned node_index = 0; node_index < block_size; node_index++) {
        unsigned node_id = *reinterpret_cast<const unsigned*> (parse_ptr);
        parse_ptr += sizeof(unsigned);
        if (node_id == ep) {
            parse_ptr += dimension_ * sizeof(float);
            neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
            parse_ptr += sizeof(unsigned);
            std::vector<unsigned> neighbors(neighbor_k);
            memcpy(neighbors.data(), parse_ptr, neighbor_k * sizeof(unsigned));
            for (; tmp_l < l && tmp_l < neighbors.size(); tmp_l++) {
                init_ids[tmp_l] = neighbors[tmp_l];
                flags[init_ids[tmp_l]] = true;
            }
            break;
        }
        parse_ptr += dimension_ * sizeof(float);
        neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
        parse_ptr += sizeof(unsigned);
        parse_ptr += neighbor_k * sizeof(unsigned);
    }

    while (tmp_l < l) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];

        offset = ids_block_offset[id];
        load_block_buffer(offset, buffer);
        parse_ptr = buffer + sizeof(unsigned);
        block_size = *reinterpret_cast<const unsigned*> (parse_ptr);
        parse_ptr += sizeof(unsigned);

        for (unsigned node_index = 0; node_index < block_size; node_index++) {
            unsigned node_id = *reinterpret_cast<const unsigned*> (parse_ptr);
            parse_ptr += sizeof(unsigned);
            if (node_id == id) {
                memcpy(vector, parse_ptr, dimension_ * sizeof(float));
                float real_dist = euclideanDistance(vector, query, dimension_);
                result[i] = Neighbor(id, real_dist, true);
                break;
            }
            parse_ptr += dimension_ * sizeof(float);
            neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
            parse_ptr += sizeof(unsigned);
            parse_ptr += neighbor_k * sizeof(unsigned);
        }

        uint8_t* this_pq_vec = pq->pq_vector + id * pq->M;
        float dist = pq->pq_distance_fast(this_pq_vec);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + l);
    std::sort(result.begin(), result.begin() + l);

    int k = 0;
    while (k < (int)l) {
        int nk = l;

        if (retset[k].flag) {
            retset[k].flag = false;
            offset = ids_block_offset[retset[k].id];
            load_block_buffer(offset, buffer);
            parse_ptr = buffer + sizeof(unsigned);
            block_size = *reinterpret_cast<const unsigned*> (parse_ptr);
            parse_ptr += sizeof(unsigned);

            {
            std::vector<std::pair<float, unsigned>> dist_indices;
            for (unsigned node_index = 0; node_index < block_size; node_index++) {
                unsigned node_id = *reinterpret_cast<const unsigned*> (parse_ptr);
                parse_ptr += sizeof(unsigned);

                float dist = pq->pq_distance_fast(pq->pq_vector + node_id * pq->M);
                dist_indices.push_back(std::make_pair(dist, node_id));

                parse_ptr += dimension_ * sizeof(float);
                neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
                parse_ptr += sizeof(unsigned);
                parse_ptr += neighbor_k * sizeof(unsigned);
            }

            std::sort(dist_indices.begin(), dist_indices.end(),
                [](const std::pair<float, unsigned>& a, const std::pair<float, unsigned>& b) {
                    return a.first < b.first;
                }
            );

            std::set<unsigned> pruned_nodes;
            float pruning_rate = static_cast<int>(std::ceil(1 * block_size));

            for(unsigned index = 0; index < pruning_rate; index++) {
                pruned_nodes.insert(dist_indices[index].second);
            }

            parse_ptr = buffer + sizeof(unsigned) + sizeof(unsigned);
            for (unsigned block_i = 0; block_i < block_size; block_i++) {
                unsigned node_id = *reinterpret_cast<const unsigned*> (parse_ptr);
                parse_ptr += sizeof(unsigned);
                if (flags[node_id]) {
                    parse_ptr += dimension_ * sizeof(float);
                    neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
                    parse_ptr += sizeof(unsigned);
                    parse_ptr += neighbor_k * sizeof(unsigned);
                    continue;
                }
                if (pruned_nodes.find(node_id) != pruned_nodes.end()) {
                    flags[node_id] = true;
                    memcpy(vector, parse_ptr, dimension_ * sizeof(float));
                    parse_ptr += dimension_ * sizeof(float);
                    float real_dist = euclideanDistance(query, vector, dimension_);
                    if (real_dist < result[l - 1].distance) {
                        Neighbor real_nn(node_id , real_dist, true);
                        InsertIntoPool(result.data(), l, real_nn);
                    }
                    else {
                        neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
                        parse_ptr += sizeof(unsigned);
                        parse_ptr += neighbor_k * sizeof(unsigned);
                        continue;
                    }
                    neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
                    parse_ptr += sizeof(unsigned);
                    std::vector<unsigned> neighbors(neighbor_k);
                    memcpy(neighbors.data(), parse_ptr, neighbor_k * sizeof(unsigned));
                    parse_ptr += neighbor_k * sizeof(unsigned);
                    for (unsigned m = 0; m < neighbors.size(); m++) {
                        unsigned neighbor_id = neighbors[m];
                        uint8_t* this_pq_vector = pq->pq_vector + neighbor_id * pq->M;
                        float dist = pq->pq_distance_fast(this_pq_vector);
                        if (dist >= retset[l - 1].distance) {
                            continue;
                        }
                        Neighbor nn(neighbor_id, dist, true);
                        int r = InsertIntoPool(retset.data(), l, nn);
                        if (r < nk) nk = r;
                    }
                }
                else {
                    parse_ptr += dimension_ * sizeof(float);
                    neighbor_k = *reinterpret_cast<const unsigned*> (parse_ptr);
                    parse_ptr += sizeof(unsigned);
                    parse_ptr += neighbor_k * sizeof(unsigned);
                }
            }
            }

        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    for (size_t i = 0; i < K; i++) {
        indices[i] = result[i].id;
    }
}

void NSG::Search_one_layer_memory(const uint8_t* query_pq, unsigned l, std::vector<Neighbor>& retset,
    boost::dynamic_bitset<>& flags,unsigned layer) {

    const auto& pg = multi_layer_graphs[layer];
    const auto& start_node = multi_layer_start_node[layer];
    const auto& node_ids = multi_layer_node_ids[layer];

    for(auto nn: retset) {
        flags[nn.id] = true;
    }

    unsigned tmp_l = retset.size();
    std::vector<unsigned> init_ids(l);
    unsigned ori_size = retset.size();
    retset.resize((l + 1));
    init_ids[tmp_l] = start_node;
    tmp_l++;
    for (unsigned i = 0; i < ori_size && tmp_l < l; i++) {
        for (unsigned j = 0; j < pg[retset[i].id].size() && tmp_l < l; j++) {
            if (flags[pg[retset[i].id][j]]) {
                continue;
            }
            init_ids[tmp_l] = pg[retset[i].id][j];
            flags[init_ids[tmp_l]] = true;
            tmp_l++;
        }
    }

    while (tmp_l < l) {
        unsigned id = rand() % node_ids.size();
        if (flags[node_ids[id]]) continue;
        flags[node_ids[id]] = true;
        init_ids[tmp_l] = node_ids[id];
        tmp_l++;
    }

    for (unsigned i = ori_size; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        float dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);
        retset[i] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + l);

    int k = 0;
    while (k < (int)l) {
        int nk = l;
        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;
            for (unsigned m = 0; m < pg[n].size(); ++m) {
                unsigned id = pg[n][m];
                if (flags[id]) continue;
                flags[id] = 1;
                float dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);
                if (dist >= retset[l - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), l, nn);
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}


void NSG::Search_one_layer_memory_NO(const uint8_t* query_pq, unsigned l, std::vector<Neighbor>& retset,
    boost::dynamic_bitset<>& flags,unsigned layer) {

    auto& pg = multi_layer_graphs_NO[layer];
    const auto& start_node = multi_layer_start_node[layer];

    for(auto nn: retset) {
        flags[nn.id] = true;
    }

    unsigned tmp_l = retset.size();
    std::vector<unsigned> init_ids(l);
    unsigned ori_size = retset.size();
    retset.resize((l + 1));
    if (!flags[start_node]) {
        init_ids[tmp_l] = start_node;
        tmp_l++;
    }

    for (unsigned i = 0; i < ori_size && tmp_l < l; i++) {
        for (unsigned j = 0; j < pg[retset[i].id].size() && tmp_l < l; j++) {
            if (flags[pg[retset[i].id][j]]) {
                continue;
            }
            init_ids[tmp_l] = pg[retset[i].id][j];
            flags[init_ids[tmp_l]] = true;
            tmp_l++;
        }
    }

    for (unsigned i = ori_size; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        float dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + tmp_l);

    int k = 0;
    while (k < (int)l) {
        int nk = l;
        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;
            for (unsigned m = 0; m < pg[n].size(); ++m) {
                unsigned id = pg[n][m];
                if (flags[id]) continue;
                flags[id] = 1;
                float dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);
                if (dist >= retset[l - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), l, nn);
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}


void NSG::Search_block_aware_multi_layer_memory(const uint8_t *pq_query, unsigned l, std::vector<Neighbor>& retset,
    boost::dynamic_bitset<>& flags) {
    for (unsigned layer = 0; layer < memory_layer; layer++) {
        unsigned this_l = 5 + layer * 2;
        if (layer == total_layer - 1) {
            this_l = l;
        }
        flags.reset();
        Search_one_layer_memory(pq_query, this_l, retset, flags, layer);
    }
}

void NSG::Search_block_aware_multi_layer_memory_NO(const uint8_t *pq_query, unsigned l, std::vector<Neighbor>& retset,
    boost::dynamic_bitset<>& flags) {
    for (unsigned layer = 0; layer < memory_layer; layer++) {
        unsigned this_l = 5 + layer * 2;
        if (layer == total_layer - 1) {
            this_l = l;
        }
        flags.reset();
        Search_one_layer_memory_NO(pq_query, this_l, retset, flags, layer);
    }
}

void NSG::Search_BA(const float *query, const uint8_t *query_pq, size_t K, unsigned l, unsigned *indices) {
    std::vector<Neighbor> retset(0);
    boost::dynamic_bitset<> flags{n_, 0};
    std::unordered_map<unsigned, unsigned> result;

    auto s_memory = std::chrono::steady_clock::now();
    Search_block_aware_multi_layer_memory_NO(query_pq, l, retset, flags);
    auto e_memory = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_memory = e_memory - s_memory;
    search_time_memory += diff_memory.count();
    auto s_disk = std::chrono::steady_clock::now();
    for (unsigned layer = total_layer - 1; layer < total_layer; layer++) {
        unsigned this_l = 5;
        if (layer == total_layer - 1) {
            this_l = l;
        }
        Search_one_layer_disk_BA(this_l, layer, retset, flags, result);
    }

    auto e_disk = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_disk = e_disk - s_disk;
    search_time_disk += diff_disk.count();

    auto* this_vector_data = new float[(l + 1) * dimension_];

    if (l != K) {
        Search_reranking(query, K, retset, this_vector_data);
    }

    for (size_t i = 0; i < K; i++) {
        indices[i] = result[retset[i].id];
    }
    delete[] this_vector_data;
}

void NSG::Search_reranking(const float *query, unsigned K, std::vector<Neighbor>& retset, float*& this_vector_data) {
    auto s_reranking = std::chrono::steady_clock::now();
    std::vector<std::tuple<long long, unsigned, unsigned>> offset_inner_offset;
    offset_inner_offset.reserve(retset.size());
    unsigned num_vector_block_per_block = (num_node_per_block + raw_vector_per_block - 1) / raw_vector_per_block;

    unsigned vector_i = 0;
    for(const auto& neighbor : retset) {
        unsigned id = neighbor.id;
        unsigned block_offset = id / num_node_per_block;
        unsigned base_offset = block_offset * num_vector_block_per_block + ((id - block_offset * num_node_per_block) / raw_vector_per_block);
        unsigned temp_id = id - block_offset * num_node_per_block;
        unsigned inner_offset = temp_id - temp_id / raw_vector_per_block * raw_vector_per_block;
        offset_inner_offset.emplace_back(base_offset * 4096, inner_offset, vector_i);
        vector_i++;
    }

    std::sort(offset_inner_offset.begin(), offset_inner_offset.end(),
        [](const std::tuple<long long, unsigned, unsigned>& a,
                  const std::tuple<long long, unsigned, unsigned>& b) {
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        return std::get<1>(a) < std::get<1>(b);
    });

    load_raw_vector_block(offset_inner_offset, this_vector_data);

    for (unsigned i = 0; i < retset.size(); i++) {
        float dist = euclideanDistance(query, this_vector_data + i * dimension_, dimension_);
        retset[i].distance = dist;
    }
    std::partial_sort(retset.begin(), retset.begin() + K, retset.end());
    auto e_reranking = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_reranking = e_reranking - s_reranking;
    search_time_reranking += diff_reranking.count();
}

void NSG::Search_one_layer_disk_BA(unsigned l, unsigned layer, std::vector<Neighbor>& retset,
    boost::dynamic_bitset<>& flags, std::unordered_map<unsigned, unsigned>& result) {
    flags.reset();
    for (auto& nn: retset) {
        flags[nn.id] = true;
    }

    boost::dynamic_bitset<> flags_extend{n_, 0};

    char* buffer = new(std::align_val_t(4096)) char[4096];

    uint8_t* vec = nullptr;
    if (posix_memalign((void**)&vec, 64, pq->M) != 0) {
        free(buffer);
        throw std::bad_alloc();
    }

    std::queue<std::pair<unsigned, int>> queue;
    unsigned tmp_l = retset.size();
    retset.resize(l + 1);

    const char* parse_ptr;
    float dist;
    unsigned in_k;
    unsigned block_offset;
    unsigned inner_offset;
    unsigned old_id;
    std::vector<unsigned> neighbors(max_neighbor);

    std::queue<unsigned> init_id_queue;
    unsigned i = 0;
    while (tmp_l < l) {
        init_id_queue.push(retset[i].id);
        block_offset = retset[i].id / num_node_per_block;
        load_block_buffer(block_offset * 4096, buffer);
        while (!init_id_queue.empty()) {
            unsigned id = init_id_queue.front();
            init_id_queue.pop();
            inner_offset = (id - block_offset * num_node_per_block) * node_size;
            parse_ptr = buffer + inner_offset;
            _mm_prefetch(parse_ptr, _MM_HINT_T0);
            parse_ptr += 8;
            in_k = *reinterpret_cast<const unsigned*> (parse_ptr);
            parse_ptr += sizeof(unsigned);
            memcpy(neighbors.data(), parse_ptr, max_neighbor * sizeof(unsigned));
            for (unsigned j = 0; j < in_k; j++) {
                if (flags[neighbors[j]]) continue;
                init_id_queue.push(neighbors[j]);
            }
            for (unsigned j = 0; j < max_neighbor; j++) {
                if (neighbors[j] >= n_) break;
                if (flags[neighbors[j]]) continue;
                uint8_t* this_pq_vec = pq->pq_vector + neighbors[j] * pq->M;
                dist = pq->pq_distance_fast(this_pq_vec);
                retset[tmp_l] = Neighbor(neighbors[j], dist, true);
                flags[neighbors[j]] = true;
                tmp_l++;
                if (tmp_l > l) {
                    break;
                }
            }
            if (tmp_l > l) {
                break;
            }
        }
        i++;
    }

    std::sort(retset.begin(), retset.begin() + l);

    int k = 0;
    while (k < (int)l) {
        int nk = l;
        if (flags_extend[retset[k].id] == false) {
            block_offset = retset[k].id / num_node_per_block;
            load_block_buffer(block_offset * 4096, buffer);

            unsigned max_depth = 4;
            std::queue<std::pair<unsigned, int>> empty;
            queue.swap(empty);
            queue.push({retset[k].id, 0});

            while (!queue.empty()) {
                auto [current_id, current_depth] = queue.front();
                queue.pop();
                if (max_depth >= 0 && current_depth > max_depth) {
                    continue;
                }
                flags_extend[current_id] = true;

                inner_offset = (current_id - block_offset * num_node_per_block) * node_size;
                parse_ptr = buffer + inner_offset;
                _mm_prefetch(parse_ptr, _MM_HINT_T0);
                old_id = *reinterpret_cast<const unsigned*> (parse_ptr);
                result.emplace(current_id, old_id);
                parse_ptr += 8;
                in_k = *reinterpret_cast<const unsigned*> (parse_ptr);
                parse_ptr += sizeof(unsigned);
                memcpy(neighbors.data(), parse_ptr, max_neighbor * sizeof(unsigned));

                for (unsigned j = 0; j < in_k; j++) {
                    unsigned neighbor_id = neighbors[j];
                    if (flags_extend[neighbor_id] == true) continue;
                    uint8_t* this_pq_vec = pq->pq_vector + neighbors[j] * pq->M;
                    dist = pq->pq_distance_fast(this_pq_vec);

                    if (dist >= retset[l - 1].distance * 1.1) {
                        continue;
                    }
                    queue.push({neighbor_id, current_depth + 1});

                    if (dist >= retset[l - 1].distance) {
                        continue;
                    }
                    Neighbor nn(neighbor_id, dist, true);
                    int r = InsertIntoPool(retset.data(), l, nn);
                    if (r < nk) nk = r;
                }

                for (unsigned j = in_k; j < max_neighbor; j++) {
                    unsigned neighbor_id = neighbors[j];
                    if (neighbor_id >= n_) break;
                    if (flags[neighbor_id]) continue;
                    flags[neighbor_id] = true;

                    uint8_t* this_pq_vec = pq->pq_vector + neighbors[j] * pq->M;
                    dist = pq->pq_distance_fast(this_pq_vec);
                    if (dist >= retset[l - 1].distance) {
                        continue;
                    }
                    Neighbor nn(neighbor_id, dist, true);
                    int r = InsertIntoPool(retset.data(), l, nn);
                    if (r < nk) nk = r;
                }
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    free(buffer);
    free(vec);
}

void NSG::load_raw_vector_block(std::vector<std::tuple<long long, unsigned, unsigned>> offset_inner_offset, float* vector_data) {
    auto s_IO = std::chrono::steady_clock::now();

    const unsigned vector_length = dimension_ * sizeof(float);

    char* buffer = new(std::align_val_t(4096)) char[4096];
    long long last_base_offset = -1;
    for (const auto& [base_offset, inner_offset, target_idx] : offset_inner_offset) {
        if (base_offset != last_base_offset) {
            ssize_t bytes_read = pread(fd_raw_vector, buffer, buffer_size, base_offset);
            IO_count_disk++;

            if (bytes_read != (ssize_t)4096) {
                free(buffer);
                throw std::runtime_error("raw vector pread failed or incomplete read");
            }
        }

        char* src =  buffer + inner_offset * vector_length;
        float* dst = vector_data + dimension_ * target_idx;
        std::memcpy(dst, src, vector_length);

        last_base_offset = base_offset;
    }

    auto e_IO = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_IO = e_IO - s_IO;
    IO_time_raw_vector += diff_IO.count();
}

void NSG::block_shuffling(const std::string& algorithm_name, unsigned BLOCK_SIZE, char* data_layout_file_name,
    char* offset_file_name, const float* data) {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "test " << std::endl;

    layout = new DataLayout();
    if (algorithm_name == "BNP") {
        std::cout << "block shuffling --- BNP" << std::endl;
        layout->BNP(final_graph, BLOCK_SIZE, dimension_);
    }
    else if (algorithm_name == "BS_swap") {
        std::cout << "block shuffling --- BS_swap" << std::endl;
        layout->BS_swap(final_graph, BLOCK_SIZE, dimension_);
    }
    if (dimension_ < 450) {
        layout->overlap_rate = layout->or_calculator_graph(final_graph);
        std::cout << "overlap ratio: " << layout->overlap_rate << std::endl;
        layout->out_degree = layout->out_degree_graph(final_graph);
        std::cout << "average out degree: " << layout->out_degree << std::endl;
        float in_degree = layout->in_degree_graph(final_graph);
        std::cout << "average in degree: " << in_degree << std::endl;
    }
    vector_data = data;
    if (algorithm_name == "sub_graph" || algorithm_name == "sub_graph_no_overlap") {
        layout->Save_data_layout_dynamic_block(data_layout_file_name, offset_file_name, vector_data, dimension_, final_graph);
    }
    else {
        layout->Save_data_layout(data_layout_file_name, offset_file_name, vector_data, dimension_, final_graph);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration2 = end - start;
    std::cout << "block shuffling time: " << duration2.count() << " ms" << std::endl;

}

int out_degree_for_node(int u, const std::vector<std::vector<unsigned>>& graph, const std::vector<int>& subgraph) {
    int out_degree = 0;
    for (unsigned neighbor: graph[u]) {
        if (std::find(subgraph.begin(), subgraph.end(), neighbor) != subgraph.end()) {
            out_degree += 1;
        }
    }
    return out_degree;
}

int real_out_degree_for_node(int u, const std::vector<std::vector<unsigned>>& graph, const std::vector<int>& subgraph) {
    int out_degree = 0;
    for (unsigned neighbor: graph[u]) {
        if (std::find(subgraph.begin(), subgraph.end(), neighbor) != subgraph.end()) {
            out_degree += 1;
        }
    }
    for (unsigned node: subgraph) {
        if (node != u) {
            if (std::find(graph[node].begin(), graph[node].end(), node) != graph[node].end()) {
                out_degree = out_degree - 1;
            }
        }
    }
    return out_degree;
}

void NSG::load_block_return_disk_ANN(unsigned id, BlockData<float>* block) {
    using namespace std::chrono;
    auto s_IO = steady_clock::now();

    unsigned num_per_block = 4096 / node_size;
    long long block_offset = id / num_per_block * 4096;
    char* buffer = new(std::align_val_t(4096)) char[4096];
    ssize_t bytes_read = pread(fd_data_layout, buffer, 4096, block_offset);

    if (bytes_read != (ssize_t)4096) {
        free(buffer);
        std::cout << "id: " << id << " block offset " << block_offset << " " << node_size << " " << bytes_read << std::endl;
        throw std::runtime_error("pread failed or incomplete read");
    }

    const char* current = buffer;

    for (unsigned i = 0; i < num_per_block; i++) {
        unsigned node_id;
        memcpy(&node_id, current, sizeof(unsigned));
        current += sizeof(unsigned);
        auto* vec = new float[dimension_];
        memcpy(vec, current, dimension_ * sizeof(float));
        current += dimension_ * sizeof(float);

        std::vector<unsigned> neighbors(max_neighbor);
        memcpy(neighbors.data(), current, max_neighbor * sizeof(unsigned));
        current += max_neighbor * sizeof(unsigned);

        block->block_data.push_back(vec);
        block->block_graph.push_back(neighbors);
        block->id_list.push_back(node_id);
    }

    auto e_IO = steady_clock::now();
    std::chrono::duration<double> diff_IO = e_IO - s_IO;
    IO_time_disk += diff_IO.count();
    IO_count_disk++;
}

void NSG::Search_diskANN_PQ(const float *query, size_t K, unsigned l, unsigned *indices, uint8_t* query_pq_vec) {
    std::vector<Neighbor> retset(l + 1);
    std::vector<Neighbor> result(l + 1);
    std::vector<unsigned> init_ids(l);
    boost::dynamic_bitset<> flags{n_, 0};
    unsigned tmp_l = 0;

    auto block = new BlockData<float>();
    load_block_return_disk_ANN(ep, block);

    unsigned block_index_ep = block->get_index_by_id(ep);
    for (; tmp_l < l && tmp_l < block->block_graph[block_index_ep].size(); tmp_l++) {
        unsigned id = block->block_graph[block_index_ep][tmp_l];
        if (id >= n_) {
            break;
        }
        init_ids[tmp_l] = id;
        flags[init_ids[tmp_l]] = true;
    }
    while (tmp_l < l) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        auto block = new BlockData<float>();
        load_block_return_disk_ANN(id, block);
        unsigned block_index_id = block->get_index_by_id(id);
        float dist = euclideanDistance(block->block_data[block_index_id], query, dimension_);
        result[i] = Neighbor(id, dist, true);
        float pq_dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);

        retset[i] = Neighbor(id, pq_dist, true);
    }
    std::sort(retset.begin(), retset.begin() + l);
    std::sort(result.begin(), result.begin() + l);

    int k = 0;
    while (k < (int)l) {
        int nk = l;
        if (retset[k].flag) {
            retset[k].flag = false;
            auto temp_block = new BlockData<float>();
            load_block_return_disk_ANN(retset[k].id, temp_block);
            unsigned block_inner_index = temp_block->get_index_by_id(retset[k].id);
            std::vector<unsigned> retset_neighbors = temp_block->block_graph[block_inner_index];
            float real_dist = euclideanDistance(temp_block->block_data[block_inner_index], query, dimension_);
            Neighbor neighbor = Neighbor(retset[k].id, real_dist, true);
            int r = InsertIntoPool(result.data(), l, neighbor);

            for (unsigned m = 0; m < retset_neighbors.size(); m++) {
                unsigned id = retset_neighbors[m];
                if (id >= n_) {
                    break;
                }
                if (flags[id]) {
                    continue;
                }
                flags[id] = true;
                float pq_dist = pq->pq_distance_fast(pq->pq_vector + id * pq->M);
                if (pq_dist >= retset[l - 1].distance) {
                    continue;
                }
                Neighbor nn(id, pq_dist, true);

                int r = InsertIntoPool(retset.data(), l, nn);

                if (r < nk) nk = r;
            }
            while (delete_queue.size() + l > queue_length) {
                unsigned block_id = delete_queue.front();
                delete_queue.pop();
                if (loaded_block_float[block_id] != nullptr && count_loaded_block[block_id] <= 0) {
                    delete loaded_block_float[block_id];
                    loaded_block_float[block_id] = nullptr;
                }
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    for (size_t i = 0; i < K; i++) {
        indices[i] = result[i].id;
    }
}

void NSG::save_diskANN(char* filename, char* offset_filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    assert(final_graph.size() == n_);
    unsigned max_num_neighbor = 0;

    for (unsigned i = 0; i < n_; ++i) {
        if (final_graph[i].size() > max_num_neighbor)
            max_num_neighbor = final_graph[i].size();
    }

    for (unsigned i = 0; i < n_; ++i) {
        out.write((char*)&i, sizeof(unsigned));
        out.write((char*)(vector_data + i * dimension_), dimension_ * sizeof(float));
        unsigned GK = (unsigned)final_graph[i].size();
        out.write((char *)&GK, sizeof(unsigned));
        out.write((char *)final_graph[i].data(), GK * sizeof(unsigned));
    }

    out.close();
    std::cout << "Saved file: " << filename << std::endl << final_graph.size() << std::endl;
}

std::vector<unsigned> NSG::select_numbers(unsigned n, unsigned r, unsigned k) {
    if (n <= 0 || r <= 0) return {};
    const bool exclude_k = (k >= 0 && k < n);
    const int available = exclude_k ? (n - 1) : n;
    if (r > available) return {};
    std::vector<unsigned> result;
    result.reserve(r);
    std::unordered_set<unsigned> selected;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, available - 1);
    while (selected.size() < r) {
        int x = distrib(gen);
        int num = exclude_k ? (x >= k ? x + 1 : x) : x;
        if (selected.insert(num).second) {
            result.push_back(num);
        }
    }
    return result;
}