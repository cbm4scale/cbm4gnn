#include <omp.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <torch/extension.h>

#define CHECK_DTYPE(x, dtype) \
    TORCH_CHECK(x.scalar_type() == dtype,  \
    "\"" #x "\" is not a tensor of type \"" #dtype "\"")

static inline void check_alloc(void* ptr) {
    if (ptr == NULL) {
            std::cerr << "Memory allocation failed." << std::endl;
            exit(-1);
            // Handle allocation failure
        } else {
            // Allocation was successful, use mkl_array as needed
            std::cout << "Memory allocation successful." << std::endl;
    }
}

double average_clustering_coefficient(const torch::Tensor& src_row_idx, 
                                      const torch::Tensor& src_col_idx, 
                                      const int num_nodes) {
    
    CHECK_DTYPE(src_row_idx, torch::kInt32);
    CHECK_DTYPE(src_col_idx, torch::kInt32);
    
    int num_edges = src_row_idx.size(0);
    int *src_nodes = src_row_idx.data_ptr<int>();
    int *dst_nodes = src_col_idx.data_ptr<int>();

    std::unordered_map<int, std::unordered_set<int>> adjacencyList;
    for (int i = 0; i < num_edges; ++i) {
        int u = src_nodes[i];
        int v = dst_nodes[i];
        // Add edges to the adjacency list (undirected graph)
        adjacencyList[u].insert(v);
        adjacencyList[v].insert(u);
    }

    double totalClusteringCoefficient = 0.0;

    // Step 2: Calculate the clustering coefficient for each node
    #pragma omp parallel for reduction(+:totalClusteringCoefficient)
    for (int node = 0; node < num_nodes; ++node) {
        const auto& neighbors = adjacencyList[node];
        int degree = neighbors.size();

        // Skip nodes with fewer than 2 neighbors as they cannot form a triangle
        if (degree < 2) continue;

        // Count the number of edges between neighbors (triangles)
        int triangles = 0;
        for (const auto& neighbor1 : neighbors) {
            for (const auto& neighbor2 : neighbors) {
                if (neighbor1 != neighbor2 && adjacencyList[neighbor1].count(neighbor2)) {
                    triangles++;
                }
            }
        }

        // Each triangle is counted twice, so divide by 2
        double clusteringCoefficient = static_cast<double>(triangles) / (degree * (degree - 1));
        totalClusteringCoefficient += clusteringCoefficient;
    }

    // Step 3: Calculate the average clustering coefficient
    double averageClustering = totalClusteringCoefficient / num_nodes;
    return averageClustering;
}



double jaccard_similarity(const std::unordered_set<int>& setA, 
                          const std::unordered_set<int>& setB) {
    int intersection_size = 0;
    for (const int& neighbor : setA) {
        if (setB.find(neighbor) != setB.end()) {
            intersection_size++;
        }
    }
    int union_size = setA.size() + setB.size() - intersection_size;

    return static_cast<double>(intersection_size) / union_size;
}

double average_jaccard_similarity(const torch::Tensor& src_row_idx, 
                                  const torch::Tensor& src_col_idx, 
                                  const int num_nodes) {
    
    CHECK_DTYPE(src_row_idx, torch::kInt32);
    CHECK_DTYPE(src_col_idx, torch::kInt32);
    
    int num_edges = src_row_idx.size(0);
    int *src_nodes = src_row_idx.data_ptr<int>();
    int *dst_nodes = src_col_idx.data_ptr<int>();

    std::unordered_map<int, std::unordered_set<int>> neighbors;

    for (int i = 0; i < num_edges; ++i) {
        int u = src_nodes[i];
        int v = dst_nodes[i];
        neighbors[u].insert(v);
        neighbors[v].insert(u);
    }

    int count = 0;
    double total_similarity = 0.0;

    #pragma omp parallel for reduction(+:count,total_similarity)  
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = i + 1; j < num_nodes; ++j) {
            double similarity = jaccard_similarity(neighbors[i], neighbors[j]);
            total_similarity += similarity;
            count++;
        }
    }

    return total_similarity / count;
}


// Function to compute the top-1 Jaccard similarity
double average_top_1_jaccard_similarity(const torch::Tensor& src_row_idx, 
                                        const torch::Tensor& src_col_idx, 
                                        const int num_nodes) {
    
    CHECK_DTYPE(src_row_idx, torch::kInt32);
    CHECK_DTYPE(src_col_idx, torch::kInt32);
    
    int num_edges = src_row_idx.size(0);
    int *src_nodes = src_row_idx.data_ptr<int>();
    int *dst_nodes = src_col_idx.data_ptr<int>();
    // Create an adjacency list representation

    std::vector<std::unordered_set<int>> neighbors(num_nodes);

    // Fill the adjacency list (considering undirected edges)
    for (int i = 0; i < num_edges; ++i) {
        neighbors[src_nodes[i]].insert(dst_nodes[i]);
        neighbors[dst_nodes[i]].insert(src_nodes[i]); // Add the reverse edge for undirected graph
    }

    double total_similarity = 0.0;
    #pragma omp parallel for reduction(+:total_similarity)
    // Compute Jaccard similarities
    for (int i = 0; i < num_nodes; ++i) {
        double max_sim = 0.0;
    
        for (int j = i + 1; j < num_nodes; ++j) {
            int intersection_size = 0;
            for (const auto& neighbor : neighbors[i]) {
                if (neighbors[j].find(neighbor) != neighbors[j].end()) {
                    intersection_size++;
                }
            }
            
            // Calculate union
            int union_size = neighbors[i].size() + neighbors[j].size() - intersection_size;

            // Calculate Jaccard similarity
            double cur_sim = (union_size != 0) ? (static_cast<double>(intersection_size) / union_size) : 0;

            // Update max similarity and index
            if (cur_sim > max_sim) {
                max_sim = cur_sim;
            }
        }

        total_similarity += max_sim; // Accumulate the maximum similarity
    }

    // Calculate average top-1 Jaccard similarity
    return (num_nodes != 0) ? (total_similarity / num_nodes) : 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_clustering_coefficient", &average_clustering_coefficient);
    m.def("avg_jaccard_similarity", &average_jaccard_similarity);
    m.def("custom_jaccard_similarity", &average_top_1_jaccard_similarity);
}
