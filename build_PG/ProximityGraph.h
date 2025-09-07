//
// Created by cshlli on 2024/12/12.
//

#ifndef PROXIMITYGRAPH_H
#define PROXIMITYGRAPH_H
#include <boost/dynamic_bitset/dynamic_bitset.hpp>


#ifndef DC_COUNT
#define DC_COUNT
inline int DC_count = 0;
inline float DC_time = 0;
#endif

class ProximityGraph {
public:
    ProximityGraph()= default;
    explicit ProximityGraph(const size_t dimension, const size_t n);
    virtual ~ProximityGraph();

protected:
    unsigned dimension_{};
    unsigned n_;
    const float *vector_data = nullptr;
    bool has_built{};

};

float euclideanDistance(const float* vec1, const float* vec2, unsigned n);
float euclideanDistance(const uint8_t* vec1, const uint8_t* vec2, unsigned n);

#endif //PROXIMITYGRAPH_H
