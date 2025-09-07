cmake_minimum_required(VERSION 3.26.5)
project(BAMG)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-mavx2 -mfma)

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

# openMP
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_executable(
        BAMG
        main.cpp
        build_PG/ProximityGraph.cpp
        build_PG/ProximityGraph.h
        build_PG/NSG.cpp
        build_PG/NSG.h
        data_layout/DataLayout.cpp
        data_layout/DataLayout.h
        build_PG/BAMG.cpp
        build_PG/BAMG.h
        data_layout/PQ.h
        data_layout/PQ.cpp
)

#target_link_libraries(BAMG ${OpenCV_LIBS})
#target_link_libraries(BAMG PRIVATE METIS::METIS)
#target_include_directories(BAMG PRIVATE ${METIS_INCLUDE_DIRS})