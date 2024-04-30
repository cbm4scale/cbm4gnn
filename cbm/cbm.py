import torch
import warnings
import itertools
from collections import deque
from cbm import cbm_mkl_cpp as cbm_

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state", category=UserWarning)

class cbm_matrix:
    
    def __init__(self, edge_index, values, rows=None, cols=None, alpha=0, style=False, deltas_format="csr"):

        # TODO: check if rows matches expect value (edge_index.max() + 1)            

        print("input_rows: " , edge_index[0].max() + 1, " input_cols: ", edge_index[0].max() + 1)


        # call c++ constructor 
        if (style) :
            data = cbm_.from_coo_int32_t_v2_(edge_index[0],                # row_indices
                                             edge_index[1],                # col_indices
                                             values,                       # values (required for syrk)
                                             edge_index[0].max() + 1,      # number of rows
                                             edge_index[1].max() + 1,      # number of cols
                                             alpha)                        # tuning parameter
        
        else:
            data = cbm_.from_coo_int32_t_v1_(edge_index[0],                # row_indices
                                             edge_index[1],                # col_indices
                                             values,                       # values (required for syrk)
                                             edge_index[0].max() + 1,      # number of rows
                                             edge_index[1].max() + 1,      # number of cols
                                             alpha)                        # tuning parameter
        
        # unpack resulting data
        self.mst_row, self.mst_col, row, col, val = data

        #TODO: Improve later! Treat unintended behaviour w/ exceptions.

        if (deltas_format == "csr"):
            #store matrix of deltas as a csr tensor  
            self.deltas = self.to_sparse_csr_tensor(row, col, val)

        elif (deltas_format == "coo"):
            #to store it as a coo tensor do:             
            self.deltas  = self.to_sparse_coo_tensor(row, col, val)
        
        else:
            print("format requested not supported... rolled back to csr.")
            self.deltas  = self.to_sparse_csr_tensor(row, col, val)

    # check if cbm format is correct by converting it to csr format (for now)
    def check_format(self, edge_index):

        n_rows = edge_index[0].max().item() + 1
        
        # create list of sets
        s = [set() for n in range(n_rows + 1)]

        d_crows = self.deltas.crow_indices().to(torch.int32)
        d_cols = self.deltas.col_indices().to(torch.int32)
        d_vals = self.deltas.values().to(torch.int32)

        visited = set()        
        queue = deque([n_rows])
        
        while queue:
            u = queue.popleft()

            if u not in visited:
                u_bot = self.mst_row[u]
                u_top = self.mst_row[u + 1]
                visited.add(u)

                for v in self.mst_col[u_bot : u_top]:
                    queue.append(v)
                    v_bot = d_crows[v]
                    v_top = d_crows[v + 1]
                    s[v].update(s[u])

                    for col_index, value in zip(d_cols[v_bot : v_top],d_vals[v_bot : v_top]):
                        if value.item() == 1:
                            s[v].add(col_index.item())
                        elif value.item() == -1:
                            s[v].remove(col_index.item())
                        else:
                            raise ValueError(f"Failed with the edge index: {edge_index}")

        # convert edge_index into list of adjacencies
        a = [set() for n in range(n_rows)]

        for row_index, col_index in zip(edge_index[0], edge_index[1]):
            a[row_index.item()].add(col_index.item())


    # convert coo matrix to pytorch coo tensor.  
    def to_sparse_coo_tensor(self, row, col, val, data_type=torch.float32):
        max_size = max(row.max(), col.max()) + 1
        edge_index = torch.stack((row, col)) # <-- is this required?
        coo_tensor = torch.sparse_coo_tensor(edge_index, val, size=(max_size, max_size)).to(data_type).coalesce()
        
        # TODO: validate tensor (values) type
        #torch.testing.assert_close(coo_tensor.values().dtype, data_type)
        print("type of coo values: ", coo_tensor.values().dtype, ", type expected: ", data_type)

        return coo_tensor

    # convert coo matrix to pytorch csr tensor.
    def to_sparse_csr_tensor(self, row, col, val, data_type=torch.float32):
        
        coo_tensor = self.to_sparse_coo_tensor(row, col , val, data_type)
        csr_tensor = coo_tensor.to_sparse_csr()

        # TODO: validate tensor (values) type
        #torch.testing.assert_dtype(csr_tensor.values(), data_type)
        print("type of csr values: ", csr_tensor.values().dtype, ", type expected: ", data_type)

        return csr_tensor
    
    # Sequential matmul methods below:
    def seq_mkl_csr_spmm_update(self, x, y):
        row_ptr_s = self.deltas.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = self.deltas.crow_indices()[1: ].to(torch.int32)
        col_ptr = self.deltas.col_indices().to(torch.int32)
        val_ptr = self.deltas.values().to(torch.float32)

        cbm_.seq_s_spmm_update_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr,
            x, self.mst_row.to(torch.int32), 
            self.mst_col.to(torch.int32), y)

    def seq_mkl_csr_spmm(self, x, y):
        row_ptr_s = self.deltas.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = self.deltas.crow_indices()[1: ].to(torch.int32)
        col_ptr = self.deltas.col_indices().to(torch.int32)
        val_ptr = self.deltas.values().to(torch.float32)

        cbm_.seq_s_spmm_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)
        
    def seq_mkl_csr_update(self, y):
        cbm_.seq_s_update_csr_int32(
            self.mst_row.to(torch.int32), self.mst_col.to(torch.int32), y)

    # Parallel matmul methods below:
    def omp_mkl_csr_spmm_update(self, x, y):
        row_ptr_s = self.deltas.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = self.deltas.crow_indices()[1: ].to(torch.int32)
        col_ptr = self.deltas.col_indices().to(torch.int32)
        val_ptr = self.deltas.values().to(torch.float32)

        cbm_.omp_s_spmm_update_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, 
            x, self.mst_row.to(torch.int32), 
            self.mst_col.to(torch.int32), y)

    def omp_mkl_csr_spmm(self, x, y):
        row_ptr_s = self.deltas.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = self.deltas.crow_indices()[1: ].to(torch.int32)
        col_ptr = self.deltas.col_indices().to(torch.int32)
        val_ptr = self.deltas.values().to(torch.float32)

        cbm_.omp_s_spmm_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)
        
    def omp_mkl_csr_update(self, y):
        cbm_.omp_s_update_csr_int32(
            self.mst_row.to(torch.int32), self.mst_col.to(torch.int32), y)
        
    def omp_avx_csr_update(self, y):
        cbm_.omp_s_update_avx_int32(
            self.mst_row.to(torch.int32), self.mst_col.to(torch.int32), y)

    def omp_mkl_csr_spmm_update_v2(self, x, y):
        row_ptr_s = self.deltas.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = self.deltas.crow_indices()[1: ].to(torch.int32)
        col_ptr = self.deltas.col_indices().to(torch.int32)
        val_ptr = self.deltas.values().to(torch.float32)

        cbm_.omp_s_spmm_update_csr_int32_v2(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, 
            x, self.mst_row.to(torch.int32), 
            self.mst_col.to(torch.int32), y)
        
    # functions mixed with pytorch kernels
    
    def seq_torch_csr_matmul(self, x):
        # assumes deltas is stored as csr
        y = self.deltas @ x
        self.seq_mkl_csr_update(y)
        return y 
    
    def omp_torch_csr_matmul(self, x):
        # assumes deltas is stored as csr
        y = self.deltas @ x
        self.omp_mkl_csr_update(y)
        return y
