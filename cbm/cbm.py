import torch
import warnings

from cbm import cbm_mkl_cpp as cbm_

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state", category=UserWarning)

class cbm_matrix:
    
    def __init__(self, edge_index, values, rows=None, cols=None, alpha=0, deltas_format="csr"):

        # TODO: check if rows matches expect value (edge_index.max() + 1)            

        # call c++ constructor 
        data = cbm_.from_coo_int32_t_(edge_index[0],                # row_indices
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
            
    # convert coo matrix to pytorch coo tensor.  
    def to_sparse_coo_tensor(self, row, col, val, data_type=torch.float32):
        
        edge_index = torch.stack((row, col)) # <-- is this required?
        coo_tensor = torch.sparse_coo_tensor(edge_index, val).to(data_type).coalesce()
        
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
