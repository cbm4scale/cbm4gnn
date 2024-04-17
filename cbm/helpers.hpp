#include <vector>
#include <algorithm>
#include <queue>
#include "mkl_spblas.h"
#include "mkl.h"

class Edge {
public:
    MKL_INT u, v, weight;

    Edge(const MKL_INT u, const MKL_INT v, const MKL_INT weight) 
    {
        this->u = u;
        this->v = v;
        this->weight = weight;
    }
};

class DisjointSet {
public:
    std::vector<MKL_INT> parent;

    DisjointSet(const MKL_INT n) {
        parent.resize(n);
        for (MKL_INT i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find(const MKL_INT x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void merge(const MKL_INT x, const MKL_INT y) {
        parent[find(x)] = find(y);
    }

    bool connected(const MKL_INT x, const MKL_INT y) {
        return find(x) == find(y);
    }
};

inline void check_alloc(void* ptr) {
    if (ptr == NULL) {
            std::cerr << "Memory allocation failed." << std::endl;
            exit(-1);
            // Handle allocation failure
        } else {
            // Allocation was successful, use mkl_array as needed
            std::cout << "Memory allocation successful." << std::endl;
    }
}

inline void check_status(sparse_status_t s) {
    if (s == SPARSE_STATUS_SUCCESS) {
        return;
    }
    else if (s == SPARSE_STATUS_NOT_INITIALIZED) {
        printf("MKL: Not Initialized\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_ALLOC_FAILED) {
        printf("MKL: Not Alloc'ed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INVALID_VALUE) {
        printf("MKL: Invalid Value\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_EXECUTION_FAILED) {
        printf("MKL: Execution Failed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INTERNAL_ERROR) {
        printf("MKL: Internal Error\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_NOT_SUPPORTED) {
        printf("MKL: Not Supported\n");
        exit(-1);
    }
}