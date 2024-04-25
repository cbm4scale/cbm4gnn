#include <torch/extension.h>
#include <omp.h>
#include "helpers.hpp"
#include <immintrin.h>

/*TODO: 
 * 1. improve intel-mkl error handling. (done?)
 * 2. variable renaming:
 *      - different variables w/ same value (e.g. src_n_rows and csr_n_rows)
 * 3. remove unnecessary code:
 *      - check unnecessary spmm and update kernel (done!)
 *      - clean helper file and included headers (done!)
 * 4. add modification for +CBM
 *      - add flag in constructor
 *      - remove edge if it has negative deltas
 * 5. check if update works for int64
 * 6. include a testing file to verify that the class works
 * 7- alpha does not behave as expected, correct this ASAP
*/

// reads dataset in coo (int32_t) format and converts it to
// cbm format, with matrix of deltas in csr (int32_t) format
// matrix of dependencies (edges) is also represented in csr.

// note: the format of these matrices can be changed on python's
// side. e.g. cbm_matrix(edge_index, valuesl, ... , deltas_format = 'coo')

std::vector<torch::Tensor> cbm_init_v1_(const torch::Tensor& src_row,
                                        const torch::Tensor& src_col,
                                        const torch::Tensor& src_val,
                                        const MKL_INT src_n_rows,
                                        const MKL_INT src_n_cols,
                                        const MKL_INT alpha) {
    
    sparse_matrix_t m_coo;
    check_status(mkl_sparse_s_create_coo(&m_coo,
                                         SPARSE_INDEX_BASE_ZERO,
                                         src_n_rows,
                                         src_n_cols,
                                         src_row.size(0),
                                         src_row.data_ptr<MKL_INT>(),
                                         src_col.data_ptr<MKL_INT>(),
                                         src_val.data_ptr<float>()));
    
    sparse_matrix_t m_csr;
    check_status(mkl_sparse_convert_csr(m_coo, SPARSE_OPERATION_NON_TRANSPOSE, &m_csr));
    
    sparse_index_base_t csr_idx_t;
    MKL_INT csr_n_rows;
    MKL_INT csr_n_cols;
    MKL_INT *csr_col; 
    MKL_INT *csr_row_b; 
    MKL_INT *csr_row_e;
    float *csr_val;
    
    check_status(mkl_sparse_s_export_csr(m_csr, 
                                         &csr_idx_t, 
                                         &csr_n_rows, 
                                         &csr_n_cols, 
                                         &csr_row_b, 
                                         &csr_row_e, 
                                         &csr_col, 
                                         &csr_val));

    // compute syrk of dataset
    sparse_matrix_t m_syrk;
    check_status(mkl_sparse_syrk(SPARSE_OPERATION_NON_TRANSPOSE, m_csr, &m_syrk));
    
    // extract syrk in in csr format
    sparse_index_base_t syrk_idx_t;
    MKL_INT syrk_n_rows;
    MKL_INT syrk_n_cols;
    MKL_INT *syrk_col; 
    MKL_INT *syrk_row_b; 
    MKL_INT *syrk_row_e;
    float *syrk_val;

    check_status(mkl_sparse_s_export_csr(m_syrk, 
                                         &syrk_idx_t, 
                                         &syrk_n_rows, 
                                         &syrk_n_cols, 
                                         &syrk_row_b, 
                                         &syrk_row_e, 
                                         &syrk_col, 
                                         &syrk_val));

    //create a vector of nnz for debugging purposes
    std::vector<MKL_INT>  nnz_vector;
    
    // declare distance graph G
    std::vector<Edge> G;

    // populate distance graph G
    for (int row = 0; row < syrk_n_rows; row++) {
        int s = syrk_row_b[row]; 
        int e = syrk_row_e[row];
        int nnz_r1 = csr_row_e[row] - csr_row_b[row];

        // again debugging purposes...
        nnz_vector.push_back(nnz_r1);    

        // add virtual edges
        G.push_back(Edge(csr_n_rows, row, nnz_r1));

        for (int i = s+1; i < e; i++) {
            int col = syrk_col[i];
            int val = syrk_val[i];
            int nnz_r2 = csr_row_e[col] - csr_row_b[col];
            int h = nnz_r1 + nnz_r2 - (2 * val);

            //add edge if suitable
            if  (h < (nnz_r1 - alpha) || h < (nnz_r2 - alpha))
                G.push_back(Edge(row, col, h));
        }
    }

    mkl_sparse_destroy(m_coo);
    mkl_sparse_destroy(m_syrk);
    
    sort(G.begin(), G.end(), [](Edge& a, Edge& b) {
        if (a.weight != b.weight) 
            return a.weight < b.weight;
        else
            return a.u > b.u;
    });

    for (unsigned int i=0; i < G.size()-1; i++) {
        
        if (G[i].weight > G[i+1].weight) {
            printf("1-not sorted\n");
        }

        if (G[i].weight == G[i+1].weight) {
            if (G[i].u < G[i+1].u) {
                printf("2-not sorted\n");
            }
        }
    }

    // initialize disjoint set
    DisjointSet dsu(src_n_rows + 1);

    // initizalize mst adjacency lists
    std::vector<std::vector<MKL_INT>> mst(src_n_rows + 1);

    MKL_INT mst_weight = 0;

    for (Edge& edge : G) {
        MKL_INT u = edge.u;
        MKL_INT v = edge.v;
        MKL_INT w = edge.weight;

        if (!dsu.connected(u, v)) {
            dsu.merge(u, v);
            mst[edge.u].push_back(edge.v);
            mst[edge.v].push_back(edge.u);
            
            mst_weight += edge.weight;
        }
    }

    //printf("elements in mst_root: %ld\n", mst[src_n_rows].size());

    // delete later 
    //std::cout << "nnz: " << src_row.size(0) << " mst_weight: " << mst_weight << " ratio: " << (float) src_row.size(0) / (float) mst_weight << std::endl;

    long int delta_len = 0;
    std::vector<float> delta_val;
    std::vector<MKL_INT> delta_col;
    std::vector<MKL_INT> delta_row;

    std::vector<std::vector<MKL_INT>> rooted_tree(src_n_rows + 1);
    
    // declare bfs queue
    std::queue<MKL_INT> fifo;
    
    // push root vertex to stack
    fifo.push(csr_n_rows);
    
    // declare and initialize list of open nodes
    std::vector<bool> opened_node(src_n_rows + 1, false);
    opened_node[src_n_rows] = true;

    while (!fifo.empty()) 
    {
        MKL_INT u = fifo.front();
        fifo.pop();
        
        for (MKL_INT v : mst[u]) 
        {
            if (!opened_node[v]) 
            {
                fifo.push(v);
                opened_node[v] = true;
                rooted_tree[u].push_back(v);

                MKL_INT u_ptr_cur = csr_row_b[u];
                MKL_INT u_ptr_max = csr_row_e[u];
                MKL_INT v_ptr_cur = csr_row_b[v];
                MKL_INT v_ptr_max = csr_row_e[v];

                // parent node is virtual node 
                if (u == csr_n_rows) {
                    while(v_ptr_cur < v_ptr_max) {        
                        //store deltas in coo format
                        delta_col.push_back(csr_col[v_ptr_cur]);
                        delta_row.push_back(v);
                        delta_val.push_back(1);
                        delta_len++;
                        v_ptr_cur++;
                    }        
                }

                //parent node is NOT virtual node 
                else {
                    // again debugging...
                    MKL_INT deltas_count = 0;

                    while (u_ptr_cur < u_ptr_max && v_ptr_cur < v_ptr_max) {
                            
                        //column index matched
                        if (csr_col[u_ptr_cur] == csr_col[v_ptr_cur]) {
                            u_ptr_cur++; v_ptr_cur++;
                        }

                        // case 1 - column index do not match:
                        // column index of u is larger than v's 
                        else if (csr_col[u_ptr_cur] > csr_col[v_ptr_cur]) {
                            //store deltas in coo format
                            delta_col.push_back(csr_col[v_ptr_cur]);
                            delta_row.push_back(v);
                            delta_val.push_back(1);
                            delta_len++;
                            v_ptr_cur++;

                            deltas_count++;
                        }

                        // case 2 - column index do not match:
                        // column index of v is larger than u's 
                        else{
                            //store deltas in coo format
                            delta_col.push_back(csr_col[u_ptr_cur]);
                            delta_row.push_back(v);
                            delta_val.push_back(-1);
                            delta_len++;
                            u_ptr_cur++;

                            deltas_count++;
                        }
                    }

                    while(u_ptr_cur < u_ptr_max) {
                        delta_col.push_back(csr_col[u_ptr_cur]);
                        delta_row.push_back(v);
                        delta_val.push_back(-1);
                        delta_len++;
                        u_ptr_cur++;

                        deltas_count++;
                    }

                    while(v_ptr_cur < v_ptr_max){
                        //store deltas in coo format
                        delta_col.push_back(csr_col[v_ptr_cur]);
                        delta_row.push_back(v);
                        delta_val.push_back(1);
                        delta_len++;
                        v_ptr_cur++;

                        deltas_count++;
                    }
                
                    // check if alpha is correctly implemented
                    //if (deltas_count >= (nnz_vector[v] - alpha))
                    //    printf("deltas_count: %ld, nnz: %ld, alpha: %ld, (%ld -> %ld)\n", deltas_count, nnz_vector[v], alpha, u, v);
                
                }
            }
        }
    }

    mkl_sparse_destroy(m_csr);

    std::vector<MKL_INT> edges_row;
    std::vector<MKL_INT> edges_col;

    edges_row.push_back(0);

    for (MKL_INT u = 0; u < src_n_rows + 1; u++) {

        // compute columns per row
        edges_row.push_back(edges_row[u] + rooted_tree[u].size());
        
        // assign column indices
        edges_col.insert(edges_col.end(), rooted_tree[u].begin(), rooted_tree[u].end());
    }

    std::vector<torch::Tensor> result;

    result.push_back(torch::from_blob((void*)edges_row.data(), 
                                      {edges_row.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)edges_col.data(), 
                                      {edges_col.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)delta_row.data(), 
                                      {delta_row.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)delta_col.data(), 
                                      {delta_col.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)delta_val.data(), 
                                      {delta_val.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(torch::kCPU))
                                      .clone());

    return result;
}

std::vector<torch::Tensor> cbm_init_v2_(const torch::Tensor& src_row,
                                        const torch::Tensor& src_col,
                                        const torch::Tensor& src_val,
                                        const MKL_INT src_n_rows,
                                        const MKL_INT src_n_cols,
                                        const MKL_INT alpha) {
    
    sparse_matrix_t m_coo;
    check_status(mkl_sparse_s_create_coo(&m_coo,
                                         SPARSE_INDEX_BASE_ZERO,
                                         src_n_rows,
                                         src_n_cols,
                                         src_row.size(0),
                                         src_row.data_ptr<MKL_INT>(),
                                         src_col.data_ptr<MKL_INT>(),
                                         src_val.data_ptr<float>()));
    
    sparse_matrix_t m_csr;
    check_status(mkl_sparse_convert_csr(m_coo, SPARSE_OPERATION_NON_TRANSPOSE, &m_csr));
    
    sparse_index_base_t csr_idx_t;
    MKL_INT csr_n_rows;
    MKL_INT csr_n_cols;
    MKL_INT *csr_col; 
    MKL_INT *csr_row_b; 
    MKL_INT *csr_row_e;
    float *csr_val;
    
    check_status(mkl_sparse_s_export_csr(m_csr, 
                                         &csr_idx_t, 
                                         &csr_n_rows, 
                                         &csr_n_cols, 
                                         &csr_row_b, 
                                         &csr_row_e, 
                                         &csr_col, 
                                         &csr_val));

    // compute syrk of dataset
    sparse_matrix_t m_syrk;
    check_status(mkl_sparse_syrk(SPARSE_OPERATION_NON_TRANSPOSE, m_csr, &m_syrk));
    
    // extract syrk in in csr format
    sparse_index_base_t syrk_idx_t;
    MKL_INT syrk_n_rows;
    MKL_INT syrk_n_cols;
    MKL_INT *syrk_col; 
    MKL_INT *syrk_row_b; 
    MKL_INT *syrk_row_e;
    float *syrk_val;

    check_status(mkl_sparse_s_export_csr(m_syrk, 
                                         &syrk_idx_t, 
                                         &syrk_n_rows, 
                                         &syrk_n_cols, 
                                         &syrk_row_b, 
                                         &syrk_row_e, 
                                         &syrk_col, 
                                         &syrk_val));

    //create a vector of nnz for debugging purposes
    std::vector<MKL_INT>  nnz_vector;
    
    // declare distance graph G
    std::vector<Edge> G;

    // populate distance graph G
    for (int row = 0; row < syrk_n_rows; row++) {
        int s = syrk_row_b[row]; 
        int e = syrk_row_e[row];
        int nnz_r1 = csr_row_e[row] - csr_row_b[row];

        // again debugging purposes...
        nnz_vector.push_back(nnz_r1);    

        // add virtual edges
        G.push_back(Edge(csr_n_rows, row, nnz_r1));

        for (int i = s+1; i < e; i++) {
            int col = syrk_col[i];
            int val = syrk_val[i];
            int nnz_r2 = csr_row_e[col] - csr_row_b[col];
            int h = nnz_r1 + nnz_r2 - (2 * val);

            //add edge if suitable
            if  (h < (nnz_r1 - alpha) || h < (nnz_r2 - alpha))
                G.push_back(Edge(row, col, h));
        }
    }

    mkl_sparse_destroy(m_coo);
    mkl_sparse_destroy(m_syrk);
    
    sort(G.begin(), G.end(), [](Edge& a, Edge& b) {
        if (a.weight != b.weight) 
            return a.weight < b.weight;
        else
            return a.u > b.u;
    });

    for (unsigned int i=0; i < G.size()-1; i++) {
        
        if (G[i].weight > G[i+1].weight) {
            printf("1-not sorted\n");
        }

        if (G[i].weight == G[i+1].weight) {
            if (G[i].u < G[i+1].u) {
                printf("2-not sorted\n");
            }
        }
    }

    // initialize disjoint set
    DisjointSet dsu(src_n_rows + 1);

    // initizalize mst adjacency lists
    std::vector<std::vector<MKL_INT>> mst(src_n_rows + 1);

    MKL_INT mst_weight = 0;

    for (Edge& edge : G) {
        MKL_INT u = edge.u;
        MKL_INT v = edge.v;
        MKL_INT w = edge.weight;

        if (!dsu.connected(u, v)) {
            dsu.merge(u, v);
            mst[edge.u].push_back(edge.v);
            mst[edge.v].push_back(edge.u);
            
            mst_weight += edge.weight;
        }
    }

    //printf("elements in mst_root: %ld\n", mst[src_n_rows].size());

    // delete later
    //std::cout << "nnz: " << src_row.size(0) << " mst_weight: " << mst_weight << " ratio: " << (float) src_row.size(0) / (float) mst_weight << std::endl;

    std::vector<float> root_delta_val;
    std::vector<MKL_INT> root_delta_col;
    std::vector<MKL_INT> root_delta_row;

    std::vector<float> child_delta_val;
    std::vector<MKL_INT> child_delta_col;
    std::vector<MKL_INT> child_delta_row;


    std::vector<std::vector<MKL_INT>> rooted_tree(src_n_rows + 1);
    
    // declare bfs queue
    std::queue<MKL_INT> fifo;
    
    // push root vertex to stack
    fifo.push(csr_n_rows);
    
    // declare and initialize list of open nodes
    std::vector<bool> opened_node(src_n_rows + 1, false);
    opened_node[src_n_rows] = true;

    while (!fifo.empty()) 
    {
        MKL_INT u = fifo.front();
        fifo.pop();
        
        for (MKL_INT v : mst[u]) 
        {
            if (!opened_node[v]) 
            {
                fifo.push(v);
                opened_node[v] = true;

                MKL_INT u_ptr_min = csr_row_b[u];
                MKL_INT u_ptr_max = csr_row_e[u];
                MKL_INT u_ptr_cur = u_ptr_min;
                MKL_INT v_ptr_min = csr_row_b[v];
                MKL_INT v_ptr_max = csr_row_e[v];
                MKL_INT v_ptr_cur = v_ptr_min;

                // to check existance of negative deltas
                bool has_negative_deltas = false;

                // parent node is virtual node 
                if (u == csr_n_rows) {
                    while(v_ptr_cur < v_ptr_max) {        
                        //store deltas in coo format
                        root_delta_col.push_back(csr_col[v_ptr_cur]);
                        root_delta_row.push_back(v);
                        root_delta_val.push_back(1);
                        v_ptr_cur++;
                    }
                    rooted_tree[u].push_back(v);        
                }

                //parent node is NOT virtual node 
                else {
                    // again debugging...
                    MKL_INT deltas_count = 0;
                    MKL_INT deltas_added = 0;
                    while (u_ptr_cur < u_ptr_max && v_ptr_cur < v_ptr_max) {
                            
                        //column index matched
                        if (csr_col[u_ptr_cur] == csr_col[v_ptr_cur]) {
                            u_ptr_cur++; v_ptr_cur++;
                        }

                        // case 1 - column index do not match:
                        // column index of u is larger than v's 
                        else if (csr_col[u_ptr_cur] > csr_col[v_ptr_cur]) {
                            //store deltas in coo format
                            child_delta_col.push_back(csr_col[v_ptr_cur]);
                            child_delta_row.push_back(v);
                            child_delta_val.push_back(1);
                            v_ptr_cur++;

                            deltas_added++;
                        }

                        // case 2 - column index do not match:
                        // column index of v is larger than u's 
                        else{
                            //store deltas in coo format
                            child_delta_col.push_back(csr_col[u_ptr_cur]);
                            child_delta_row.push_back(v);
                            child_delta_val.push_back(-1);
                            u_ptr_cur++;

                            has_negative_deltas = true;
                            
                            deltas_added++;
                        }
                    }

                    while(u_ptr_cur < u_ptr_max) {
                        child_delta_col.push_back(csr_col[u_ptr_cur]);
                        child_delta_row.push_back(v);
                        child_delta_val.push_back(-1);
                        u_ptr_cur++;

                        has_negative_deltas = true;

                        deltas_added++;
                    }

                    while(v_ptr_cur < v_ptr_max){
                        //store deltas in coo format
                        child_delta_col.push_back(csr_col[v_ptr_cur]);
                        child_delta_row.push_back(v);
                        child_delta_val.push_back(1);
                        v_ptr_cur++;

                        deltas_added++;
                    }

                    // if it contains negative deltas use virtual node instead
                    if (has_negative_deltas) {
                        // undoes deltas
                        while (deltas_added--)
                        {
                            child_delta_row.pop_back();
                            child_delta_col.pop_back();
                            child_delta_val.pop_back();
                        }
                        
                        // link row to virtual node
                        v_ptr_cur = v_ptr_min;
                        while(v_ptr_cur < v_ptr_max) {        
                            root_delta_col.push_back(csr_col[v_ptr_cur]);
                            root_delta_row.push_back(v);
                            root_delta_val.push_back(1);
                            v_ptr_cur++;
                        }
                        // node v depends on virtual node
                        rooted_tree[src_n_rows].push_back(v);
                    }

                    else {
                        rooted_tree[u].push_back(v);
                    }               
                }
            }
        }
    }

    //printf("root_outdegree: %d | number of nodes: %d\n", rooted_tree[src_n_rows].size(), src_n_rows);
    mkl_sparse_destroy(m_csr);

    // prepare tree of dependencies in csr format
    std::vector<MKL_INT> edges_row;
    std::vector<MKL_INT> edges_col;

    edges_row.push_back(0);

    for (MKL_INT u = 0; u < src_n_rows + 1; u++) {

        // compute columns per row
        edges_row.push_back(edges_row[u] + rooted_tree[u].size());
        
        // assign column indices
        edges_col.insert(edges_col.end(), rooted_tree[u].begin(), rooted_tree[u].end());
    }

    // prepare matrix of deltas in coo format
    root_delta_row.insert(root_delta_row.end(), child_delta_row.begin(), child_delta_row.end());
    root_delta_col.insert(root_delta_col.end(), child_delta_col.begin(), child_delta_col.end());
    root_delta_val.insert(root_delta_val.end(), child_delta_val.begin(), child_delta_val.end());

    std::vector<torch::Tensor> result;

    result.push_back(torch::from_blob((void*)edges_row.data(), 
                                      {edges_row.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)edges_col.data(), 
                                      {edges_col.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)root_delta_row.data(), 
                                      {root_delta_row.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)root_delta_col.data(), 
                                      {root_delta_col.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kInt32)
                                          .device(torch::kCPU))
                                      .clone());

    result.push_back(torch::from_blob((void*)root_delta_val.data(), 
                                      {root_delta_val.size()}, 
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(torch::kCPU))
                                      .clone());

    return result;
}

// naming convention threading[seq, omp] + data_type[s, d, ...] + 
// sequence_of_operations[update, spmm] + sparse_format[coo, csr] +
// mkl_interface[int32, int64] 

static inline void seq_s_spmm_update_csr_int32_(const at::Tensor& lhs_row_b, 
                                                const at::Tensor& lhs_row_e, 
                                                const at::Tensor& lhs_col, 
                                                const at::Tensor& lhs_val,
                                                const at::Tensor& rhs, 
                                                const at::Tensor& edges_src,
                                                const at::Tensor& edges_dst,
                                                at::Tensor& dst) {

    TORCH_CHECK(lhs_row_b.scalar_type() == torch::kInt32, "lhs_row_b (NOT torch::kInt32)");
    TORCH_CHECK(lhs_row_e.scalar_type() == torch::kInt32, "lhs_row_e (NOT torch::kInt32)");
    TORCH_CHECK(lhs_col.scalar_type() == torch::kInt32, "lhs_col (NOT torch::kInt32)");
    TORCH_CHECK(lhs_val.scalar_type() == torch::kFloat32, "lhs_val (NOT torch::kFloat32)");
    TORCH_CHECK(rhs.scalar_type() == torch::kFloat32, "rhs tensor (NOT torch::kFloat32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");
    TORCH_CHECK(edges_src.scalar_type() == torch::kInt32, "edges_src (NOT torch::kInt32)");
    TORCH_CHECK(edges_dst.scalar_type() == torch::kInt32, "edge_dst (NOT torch::kInt32)");

    int max_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

    MKL_INT lhs_n_rows = dst.size(0);
    MKL_INT lhs_n_cols = rhs.size(1);
    MKL_INT *lhs_row_b_ptr = lhs_row_b.data_ptr<MKL_INT>();
    MKL_INT *lhs_row_e_ptr = lhs_row_e.data_ptr<MKL_INT>();
    MKL_INT *lhs_col_ptr = lhs_col.data_ptr<MKL_INT>();
    float *lhs_val_ptr = lhs_val.data_ptr<float>();

    sparse_matrix_t m_csr; 
    check_status(mkl_sparse_s_create_csr(&m_csr, 
                                         SPARSE_INDEX_BASE_ZERO,
                                         lhs_n_rows, 
                                         lhs_n_cols,
                                         lhs_row_b_ptr, 
                                         lhs_row_e_ptr,
                                         lhs_col_ptr,
                                         lhs_val_ptr));

    float *rhs_ptr = rhs.data_ptr<float>();
    float *dst_ptr = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_ptr, 
                                 lhs_n_cols,
                                 lhs_n_cols, 
                                 0.0f, 
                                 dst_ptr,
                                 lhs_n_cols));

    // spmm done; start update

    MKL_INT dst_n_rows = dst.size(0);
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT *edges_src_ptr = edges_src.data_ptr<MKL_INT>();
    MKL_INT *edges_dst_ptr = edges_dst.data_ptr<MKL_INT>();

    std::stack<MKL_INT> lifo;
    
    for (MKL_INT i = edges_src_ptr[dst_n_rows]; i < edges_src_ptr[dst_n_rows + 1]; i++) {
        lifo.push(edges_dst_ptr[i]);
    }

    while (!lifo.empty()) {
        // get index of src vertex
        MKL_INT u = lifo.top();
        MKL_INT u_ptr_cur = edges_src_ptr[u];
        MKL_INT u_ptr_max = edges_src_ptr[u + 1];

        lifo.pop();

        for (;u_ptr_cur < u_ptr_max; u_ptr_cur++) {
            // get index of dst vertex
            MKL_INT v = edges_dst_ptr[u_ptr_cur];

            // update destination matrix            
            float *x = dst_ptr + (u * dst_n_cols);
            float *y = dst_ptr + (v * dst_n_cols);
            cblas_saxpy(dst_n_cols, 1.0f, x, 1, y, 1);

            lifo.push(v);
        }
    }
    mkl_set_num_threads(max_threads);
}

static inline void seq_s_update_csr_int32_(const at::Tensor& edges_src,
                                           const at::Tensor& edges_dst,
                                           at::Tensor& dst) {

    TORCH_CHECK(edges_src.scalar_type() == torch::kInt32, "edges_src (NOT torch::kInt32)");
    TORCH_CHECK(edges_dst.scalar_type() == torch::kInt32, "edge_dst (NOT torch::kInt32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");

    int max_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

    MKL_INT dst_n_rows = dst.size(0);
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT *edges_src_ptr = edges_src.data_ptr<MKL_INT>();
    MKL_INT *edges_dst_ptr = edges_dst.data_ptr<MKL_INT>();
    float *dst_ptr = dst.data_ptr<float>();

    std::stack<MKL_INT> lifo;
    
    for (MKL_INT i = edges_src_ptr[dst_n_rows]; i < edges_src_ptr[dst_n_rows + 1]; i++) {
        lifo.push(edges_dst_ptr[i]);
    }

    while (!lifo.empty()) {
        // get index of src vertex
        MKL_INT u = lifo.top();
        MKL_INT u_ptr_cur = edges_src_ptr[u];
        MKL_INT u_ptr_max = edges_src_ptr[u + 1];

        lifo.pop();

        for (;u_ptr_cur < u_ptr_max; u_ptr_cur++) {
            // get index of dst vertex
            MKL_INT v = edges_dst_ptr[u_ptr_cur];

            // update destination matrix            
            float *x = dst_ptr + (u * dst_n_cols);
            float *y = dst_ptr + (v * dst_n_cols);
            cblas_saxpy(dst_n_cols, 1.0f, x, 1, y, 1);

            lifo.push(v);
        }
    }
    mkl_set_num_threads(max_threads);
}


// use this function for benchmarking against intel-mkl csr-s-spmm
static inline void seq_s_spmm_csr_int32_(const at::Tensor& lhs_row_b, 
                                         const at::Tensor& lhs_row_e, 
                                         const at::Tensor& lhs_col, 
                                         const at::Tensor& lhs_val,
                                         const at::Tensor& rhs, 
                                         at::Tensor& dst) {

    TORCH_CHECK(lhs_row_b.scalar_type() == torch::kInt32, "lhs_row_b (NOT torch::kInt32)");
    TORCH_CHECK(lhs_row_e.scalar_type() == torch::kInt32, "lhs_row_e (NOT torch::kInt32)");
    TORCH_CHECK(lhs_col.scalar_type() == torch::kInt32, "lhs_col (NOT torch::kInt32)");
    TORCH_CHECK(lhs_val.scalar_type() == torch::kFloat32, "lhs_val (NOT torch::kFloat32)");
    TORCH_CHECK(rhs.scalar_type() == torch::kFloat32, "rhs tensor (NOT torch::kFloat32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");
    
    int max_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

    MKL_INT lhs_n_rows = dst.size(0);
    MKL_INT lhs_n_cols = rhs.size(1);
    MKL_INT *lhs_row_b_ptr = lhs_row_b.data_ptr<MKL_INT>();
    MKL_INT *lhs_row_e_ptr = lhs_row_e.data_ptr<MKL_INT>();
    MKL_INT *lhs_col_ptr = lhs_col.data_ptr<MKL_INT>();
    float *lhs_val_ptr = lhs_val.data_ptr<float>();

    sparse_matrix_t m_csr; 
    check_status(mkl_sparse_s_create_csr(&m_csr, 
                                         SPARSE_INDEX_BASE_ZERO,
                                         lhs_n_rows, 
                                         lhs_n_cols,
                                         lhs_row_b_ptr, 
                                         lhs_row_e_ptr,
                                         lhs_col_ptr,
                                         lhs_val_ptr));

    float *rhs_ptr = rhs.data_ptr<float>();
    float *dst_ptr = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_ptr, 
                                 lhs_n_cols,
                                 lhs_n_cols, 
                                 0.0f, 
                                 dst_ptr,
                                 lhs_n_cols));

    mkl_set_num_threads(16);
}

static inline void omp_s_spmm_update_csr_int32_(const at::Tensor& lhs_row_b, 
                                                const at::Tensor& lhs_row_e, 
                                                const at::Tensor& lhs_col, 
                                                const at::Tensor& lhs_val,
                                                const at::Tensor& rhs, 
                                                const at::Tensor& edges_src,
                                                const at::Tensor& edges_dst,
                                                at::Tensor& dst) {

    TORCH_CHECK(lhs_row_b.scalar_type() == torch::kInt32, "lhs_row_b (NOT torch::kInt32)");
    TORCH_CHECK(lhs_row_e.scalar_type() == torch::kInt32, "lhs_row_e (NOT torch::kInt32)");
    TORCH_CHECK(lhs_col.scalar_type() == torch::kInt32, "lhs_col (NOT torch::kInt32)");
    TORCH_CHECK(lhs_val.scalar_type() == torch::kFloat32, "lhs_val (NOT torch::kFloat32)");
    TORCH_CHECK(rhs.scalar_type() == torch::kFloat32, "rhs tensor (NOT torch::kFloat32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");
    TORCH_CHECK(edges_src.scalar_type() == torch::kInt32, "edges_src (NOT torch::kInt32)");
    TORCH_CHECK(edges_dst.scalar_type() == torch::kInt32, "edge_dst (NOT torch::kInt32)");

    MKL_INT lhs_n_rows = dst.size(0);
    MKL_INT lhs_n_cols = rhs.size(1);
    MKL_INT *lhs_row_b_ptr = lhs_row_b.data_ptr<MKL_INT>();
    MKL_INT *lhs_row_e_ptr = lhs_row_e.data_ptr<MKL_INT>();
    MKL_INT *lhs_col_ptr = lhs_col.data_ptr<MKL_INT>();
    float *lhs_val_ptr = lhs_val.data_ptr<float>();

    sparse_matrix_t m_csr; 
    check_status( mkl_sparse_s_create_csr(&m_csr, 
                                          SPARSE_INDEX_BASE_ZERO,
                                          lhs_n_rows, 
                                          lhs_n_cols,
                                          lhs_row_b_ptr, 
                                          lhs_row_e_ptr,
                                          lhs_col_ptr,
                                          lhs_val_ptr));

    float *rhs_ptr = rhs.data_ptr<float>();
    float *dst_ptr = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_ptr, 
                                 lhs_n_cols,
                                 lhs_n_cols, 
                                 0.0f, 
                                 dst_ptr,
                                 lhs_n_cols));

    MKL_INT dst_n_rows = dst.size(0);
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT *edges_src_ptr = edges_src.data_ptr<MKL_INT>();
    MKL_INT *edges_dst_ptr = edges_dst.data_ptr<MKL_INT>();

    std::stack<MKL_INT> lifo;

    #pragma omp parallel for schedule(dynamic, 10) private(lifo)
    for (MKL_INT i = edges_src_ptr[dst_n_rows]; i < edges_src_ptr[dst_n_rows + 1]; i++) {
        lifo.push(edges_dst_ptr[i]);
    
        while (!lifo.empty()) {
            // get index of src vertex
            MKL_INT u = lifo.top();
            MKL_INT u_ptr_cur = edges_src_ptr[u];
            MKL_INT u_ptr_max = edges_src_ptr[u + 1];

            lifo.pop();

            for (;u_ptr_cur < u_ptr_max; u_ptr_cur++) {
                // get index of dst vertex
                MKL_INT v = edges_dst_ptr[u_ptr_cur];

                // update destination matrix            
                float *x = dst_ptr + (u * dst_n_cols);
                float *y = dst_ptr + (v * dst_n_cols);
                
                cblas_saxpy(dst_n_cols, 1.0f, x, 1, y, 1);

                lifo.push(v);
            }
        }
    }
}
/*
static inline void omp_s_update_csr_int32_(const at::Tensor& edges_src,
                                           const at::Tensor& edges_dst,
                                           at::Tensor& dst) {

    TORCH_CHECK(edges_src.scalar_type() == torch::kInt32, "edges_src (NOT torch::kInt32)");
    TORCH_CHECK(edges_dst.scalar_type() == torch::kInt32, "edge_dst (NOT torch::kInt32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");

    MKL_INT dst_n_rows = dst.size(0);
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT *edges_src_ptr = edges_src.data_ptr<MKL_INT>();
    MKL_INT *edges_dst_ptr = edges_dst.data_ptr<MKL_INT>();
    float *dst_ptr = dst.data_ptr<float>();
    
    //printf("descendents of virtual: %d\n", edges_src_ptr[dst_n_rows + 1] - edges_src_ptr[dst_n_rows]);

    #pragma omp parallel
    {
        //printf("number of threads: %d\n", omp_get_num_threads());

        #pragma omp for schedule(dynamic)
        for (MKL_INT i = edges_src_ptr[dst_n_rows]; i < edges_src_ptr[dst_n_rows + 1]; i++) {
            update_task(edges_dst_ptr[i], 
                        dst_n_rows, 
                        dst_n_cols, 
                        edges_src_ptr, 
                        edges_dst_ptr, 
                        dst_ptr);
        }
    }
}
*/
static inline void omp_s_update_csr_int32_(const at::Tensor& edges_src,
                                           const at::Tensor& edges_dst,
                                           at::Tensor& dst) {

    TORCH_CHECK(edges_src.scalar_type() == torch::kInt32, "edges_src (NOT torch::kInt32)");
    TORCH_CHECK(edges_dst.scalar_type() == torch::kInt32, "edge_dst (NOT torch::kInt32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");

    MKL_INT dst_n_rows = dst.size(0);
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT *edges_src_ptr = edges_src.data_ptr<MKL_INT>();
    MKL_INT *edges_dst_ptr = edges_dst.data_ptr<MKL_INT>();
    float *dst_ptr = dst.data_ptr<float>();

    #pragma omp parallel
    {
        std::stack<MKL_INT> lifo;

        #pragma omp for schedule(dynamic, 50)
        for (MKL_INT i = edges_src_ptr[dst_n_rows]; i < edges_src_ptr[dst_n_rows + 1]; i++) {
            lifo.push(edges_dst_ptr[i]);
        
            while (!lifo.empty()) {
                // get index of src vertex
                MKL_INT u = lifo.top();
                MKL_INT u_ptr_cur = edges_src_ptr[u];
                MKL_INT u_ptr_max = edges_src_ptr[u + 1];

                lifo.pop();

                for (;u_ptr_cur < u_ptr_max; u_ptr_cur++) {
                    // get index of dst vertex
                    MKL_INT v = edges_dst_ptr[u_ptr_cur];

                    // update destination matrix            
                    float *x = dst_ptr + (u * dst_n_cols);
                    float *y = dst_ptr + (v * dst_n_cols);
                    
                    MKL_INT i;
                    for (i = 0; i < dst_n_cols - 31; i += 32) {
                        __m256 ymm0 = _mm256_loadu_ps(x + i);
                        __m256 ymm1 = _mm256_loadu_ps(y + i);
                        __m256 ymm2 = _mm256_loadu_ps(x + i + 8);
                        __m256 ymm3 = _mm256_loadu_ps(y + i + 8);
                        __m256 ymm4 = _mm256_loadu_ps(x + i + 16);
                        __m256 ymm5 = _mm256_loadu_ps(y + i + 16);
                        __m256 ymm6 = _mm256_loadu_ps(x + i + 24);
                        __m256 ymm7 = _mm256_loadu_ps(y + i + 24);

                        ymm1 = _mm256_add_ps(ymm0, ymm1);
                        ymm3 = _mm256_add_ps(ymm2, ymm3);
                        ymm5 = _mm256_add_ps(ymm4, ymm5);
                        ymm7 = _mm256_add_ps(ymm6, ymm7);

                        _mm256_storeu_ps(y + i, ymm1);
                        _mm256_storeu_ps(y + i + 8, ymm3);
                        _mm256_storeu_ps(y + i + 16, ymm5);
                        _mm256_storeu_ps(y + i + 24, ymm7);
                    }

                    if(i < dst_n_cols - 15) {
                        __m256 ymm0 = _mm256_loadu_ps(x + i);
                        __m256 ymm1 = _mm256_loadu_ps(y + i);
                        __m256 ymm2 = _mm256_loadu_ps(x + i + 8);
                        __m256 ymm3 = _mm256_loadu_ps(y + i + 8);
                        ymm1 = _mm256_add_ps(ymm0, ymm1);
                        ymm3 = _mm256_add_ps(ymm2, ymm3);

                        _mm256_storeu_ps(y + i, ymm1);
                        _mm256_storeu_ps(y + i + 8, ymm3);
                        
                        i += 15;
                    }

                    if (i < dst_n_cols - 7) {
                        __m256 ymm0 = _mm256_loadu_ps(x + i);
                        __m256 ymm1 = _mm256_loadu_ps(y + i);
                        ymm1 = _mm256_add_ps(ymm0, ymm1);

                        _mm256_storeu_ps(y + i, ymm1);

                        i+=8;
                    }
                    
                    for (; i < dst_n_cols; i++) {
                        y[i] += x[i];
                    }

                    lifo.push(v);
                }
            }
        }
    }
/*
    #pragma omp parallel
    {
        std::stack<MKL_INT> lifo;

        #pragma omp for schedule(dynamic, 50)
        for (MKL_INT i = edges_src_ptr[dst_n_rows]; i < edges_src_ptr[dst_n_rows + 1]; i++) {
            lifo.push(edges_dst_ptr[i]);
        
            while (!lifo.empty()) {
                // get index of src vertex
                MKL_INT u = lifo.top();
                MKL_INT u_ptr_cur = edges_src_ptr[u];
                MKL_INT u_ptr_max = edges_src_ptr[u + 1];

                lifo.pop();

                for (;u_ptr_cur < u_ptr_max; u_ptr_cur++) {
                    // get index of dst vertex
                    MKL_INT v = edges_dst_ptr[u_ptr_cur];

                    // update destination matrix            
                    float *x = dst_ptr + (u * dst_n_cols);
                    float *y = dst_ptr + (v * dst_n_cols);
                    
                    cblas_saxpy(dst_n_cols, 1.0f, x, 1, y, 1);

                    lifo.push(v);
                }
            }
        }
    }
*/
}

static inline void omp_s_update_avx_int32_(const at::Tensor& edges_src,
                                           const at::Tensor& edges_dst,
                                           at::Tensor& dst) {

    TORCH_CHECK(edges_src.scalar_type() == torch::kInt32, "edges_src (NOT torch::kInt32)");
    TORCH_CHECK(edges_dst.scalar_type() == torch::kInt32, "edge_dst (NOT torch::kInt32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");

    MKL_INT dst_n_rows = dst.size(0);
    MKL_INT dst_n_cols = dst.size(1);
    MKL_INT *edges_src_ptr = edges_src.data_ptr<MKL_INT>();
    MKL_INT *edges_dst_ptr = edges_dst.data_ptr<MKL_INT>();
    float *dst_ptr = dst.data_ptr<float>();

    #pragma omp parallel
    {
        std::stack<MKL_INT> lifo;

        #pragma omp for schedule(dynamic, 50)
        for (MKL_INT i = edges_src_ptr[dst_n_rows]; i < edges_src_ptr[dst_n_rows + 1]; i++) {
            lifo.push(edges_dst_ptr[i]);
        
            while (!lifo.empty()) {
                // get index of src vertex
                MKL_INT u = lifo.top();
                MKL_INT u_ptr_cur = edges_src_ptr[u];
                MKL_INT u_ptr_max = edges_src_ptr[u + 1];

                lifo.pop();

                for (;u_ptr_cur < u_ptr_max; u_ptr_cur++) {
                    // get index of dst vertex
                    MKL_INT v = edges_dst_ptr[u_ptr_cur];

                    // update destination matrix            
                    float *x = dst_ptr + (u * dst_n_cols);
                    float *y = dst_ptr + (v * dst_n_cols);
                    
                    MKL_INT i;
                    for (i = 0; i < dst_n_cols - 31; i += 32) {
                        __m256 ymm0 = _mm256_loadu_ps(x + i);
                        __m256 ymm1 = _mm256_loadu_ps(y + i);
                        __m256 ymm2 = _mm256_loadu_ps(x + i + 8);
                        __m256 ymm3 = _mm256_loadu_ps(y + i + 8);
                        __m256 ymm4 = _mm256_loadu_ps(x + i + 16);
                        __m256 ymm5 = _mm256_loadu_ps(y + i + 16);
                        __m256 ymm6 = _mm256_loadu_ps(x + i + 24);
                        __m256 ymm7 = _mm256_loadu_ps(y + i + 24);

                        ymm1 = _mm256_add_ps(ymm0, ymm1);
                        ymm3 = _mm256_add_ps(ymm2, ymm3);
                        ymm5 = _mm256_add_ps(ymm4, ymm5);
                        ymm7 = _mm256_add_ps(ymm6, ymm7);

                        _mm256_storeu_ps(y + i, ymm1);
                        _mm256_storeu_ps(y + i + 8, ymm3);
                        _mm256_storeu_ps(y + i + 16, ymm5);
                        _mm256_storeu_ps(y + i + 24, ymm7);
                    }

                    if(i < dst_n_cols - 15) {
                        __m256 ymm0 = _mm256_loadu_ps(x + i);
                        __m256 ymm1 = _mm256_loadu_ps(y + i);
                        __m256 ymm2 = _mm256_loadu_ps(x + i + 8);
                        __m256 ymm3 = _mm256_loadu_ps(y + i + 8);
                        ymm1 = _mm256_add_ps(ymm0, ymm1);
                        ymm3 = _mm256_add_ps(ymm2, ymm3);

                        _mm256_storeu_ps(y + i, ymm1);
                        _mm256_storeu_ps(y + i + 8, ymm3);
                        
                        i += 15;
                    }

                    if (i < dst_n_cols - 7) {
                        __m256 ymm0 = _mm256_loadu_ps(x + i);
                        __m256 ymm1 = _mm256_loadu_ps(y + i);
                        ymm1 = _mm256_add_ps(ymm0, ymm1);

                        _mm256_storeu_ps(y + i, ymm1);

                        i+=8;
                    }
                    
                    for (; i < dst_n_cols; i++) {
                        y[i] += x[i];
                    }

                    lifo.push(v);
                }
            }
        }
    }
}

static inline void omp_s_spmm_csr_int32_(const at::Tensor& lhs_row_b, 
                                         const at::Tensor& lhs_row_e, 
                                         const at::Tensor& lhs_col, 
                                         const at::Tensor& lhs_val,
                                         const at::Tensor& rhs, 
                                         at::Tensor& dst) {

    TORCH_CHECK(lhs_row_b.scalar_type() == torch::kInt32, "lhs_row_b (NOT torch::kInt32)");
    TORCH_CHECK(lhs_row_e.scalar_type() == torch::kInt32, "lhs_row_e (NOT torch::kInt32)");
    TORCH_CHECK(lhs_col.scalar_type() == torch::kInt32, "lhs_col (NOT torch::kInt32)");
    TORCH_CHECK(lhs_val.scalar_type() == torch::kFloat32, "lhs_val (NOT torch::kFloat32)");
    TORCH_CHECK(rhs.scalar_type() == torch::kFloat32, "rhs tensor (NOT torch::kFloat32)");
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32, "dst tensor (NOT torch::kFloat32)");
    
    MKL_INT lhs_n_rows = dst.size(0);
    MKL_INT lhs_n_cols = rhs.size(1);
    MKL_INT *lhs_row_b_ptr = lhs_row_b.data_ptr<MKL_INT>();
    MKL_INT *lhs_row_e_ptr = lhs_row_e.data_ptr<MKL_INT>();
    MKL_INT *lhs_col_ptr = lhs_col.data_ptr<MKL_INT>();
    float *lhs_val_ptr = lhs_val.data_ptr<float>();

    sparse_matrix_t m_csr; 
    check_status( mkl_sparse_s_create_csr(&m_csr, 
                                          SPARSE_INDEX_BASE_ZERO,
                                          lhs_n_rows, 
                                          lhs_n_cols,
                                          lhs_row_b_ptr, 
                                          lhs_row_e_ptr,
                                          lhs_col_ptr,
                                          lhs_val_ptr));

    float *rhs_ptr = rhs.data_ptr<float>();
    float *dst_ptr = dst.data_ptr<float>();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    check_status(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                                 1.0f, 
                                 m_csr,
                                 descr,
                                 SPARSE_LAYOUT_ROW_MAJOR, 
                                 rhs_ptr, 
                                 lhs_n_cols,
                                 lhs_n_cols, 
                                 0.0f, 
                                 dst_ptr,
                                 lhs_n_cols));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("from_coo_int32_t_v1_", &cbm_init_v1_);
    m.def("from_coo_int32_t_v2_", &cbm_init_v2_);
    m.def("seq_s_spmm_update_csr_int32",&seq_s_spmm_update_csr_int32_);
    m.def("seq_s_update_csr_int32",&seq_s_update_csr_int32_);
    m.def("seq_s_spmm_csr_int32",&seq_s_spmm_csr_int32_);
    m.def("omp_s_spmm_update_csr_int32",&omp_s_spmm_update_csr_int32_);
    m.def("omp_s_update_avx_int32",&omp_s_update_avx_int32_);
    m.def("omp_s_update_csr_int32",&omp_s_update_csr_int32_);
    m.def("omp_s_spmm_csr_int32",&omp_s_spmm_csr_int32_);
}
