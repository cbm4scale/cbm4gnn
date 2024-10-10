import torch
from collections import deque
from cbm import cbm_mkl_cpp as cbm_cpp

#TODO: remove edge_values

class cbm4mm:

    def __init__(self, edge_index, edge_values, alpha=0):
        
        # get number of rows in input dataset 
        num_rows = max(edge_index[0].max(), edge_index[1].max()) + 1

        # represent input dataset in CBM
        cbm_data = cbm_cpp.init(edge_index[0],  # row indices
                                edge_index[1],  # column indices
                                edge_values,    # value of nnz's
                                num_rows,       # number of rows
                                alpha)          # prunning param

        # unpack resulting data
        delta_edge_index = torch.stack([cbm_data[0], cbm_data[1]])
        delta_values = cbm_data[2]
        self.mca_branches = cbm_data[3]
        self.mca_row_idx = cbm_data[4]
        self.mca_col_idx = cbm_data[5] 

        # convert matrix of deltas to COO tensor (torch.float32)
        coo_tensor = torch.sparse_coo_tensor(delta_edge_index, 
                                             delta_values, 
                                             (num_rows, num_rows))
        
        # convert matrix of deltas to CSR tensor (torch.float32)
        self.deltas = coo_tensor.to(torch.float32).coalesce().to_sparse_csr()


    def matmul(self, x, y):
        """
        Matrix multiplication with CBM format:

        Computes the product between the matrix of delta (self.deltas) and a 
        dense real-valued matrix. The result of this product is stored in 
        another dense real-valued matrix y. Matrix y is subsequently updated
        according to the compression tree (self.mca_row_ptr and self.mca_col_idx) that 
        was obtained during the construction of the CBM format.
        
        Note: -This method wraps C++ code and resorts to Intel MKL sparse BLAS.
              -Use OpenMP environment variables to control parallelism.

        Args:
            x (pytorch.Tensor): right-hand side operand matrix.
            y (pytorch.Tensor): result matrix.
        """


        cbm_cpp.s_spmm_update_csr_int32(
            self.deltas.crow_indices()[:-1].to(torch.int32),
            self.deltas.crow_indices()[1: ].to(torch.int32), 
            self.deltas.col_indices().to(torch.int32), 
            self.deltas.values().to(torch.float32), 
            x, 
            self.mca_branches.to(torch.int32), 
            self.mca_row_idx.to(torch.int32), 
            self.mca_col_idx.to(torch.int32), 
            y)
        
    def update(self, y):
        """
        Helper / Debugging / Benchmarking method:

        Computes the update stage of CBM format, according to the compression 
        tree (self.mca_row_ptr and self.mca_col_idx) that was obtained during the 
        construction of the format.

        Note: -Use OpenMP environment variables to control parallelism.
        
        Args:
            y (pytorch.Tensor): matrix to be updated.     
        """
        
        cbm_cpp.s_update_csr_int32(
            self.mca_branches.to(torch.int32), 
            self.mca_row_idx.to(torch.int32), 
            self.mca_col_idx.to(torch.int32), 
            y)
            

    def deltas_spmm(self, x, y):
        """
        Helper / Debugging / Benchmarking method:

        Computes the product between the matrix of deltas (self.deltas) and a 
        dense real-valued matrix. The result of this product is stored in 
        another dense real-valued matrix y. 
        
        Note: -This method wraps C++ code and resorts to Intel MKL sparse BLAS.
              -Use OpenMP environment variables to control parallelism.
        
        Args:
            x (pytorch.Tensor): right-hand side operand matrix.
            y (pytorch.Tensor): result matrix.      
        """

        cbm_cpp.s_spmm_csr_int32(
            self.deltas.crow_indices()[:-1].to(torch.int32), 
            self.deltas.crow_indices()[1: ].to(torch.int32), 
            self.deltas.col_indices().to(torch.int32), 
            self.deltas.values().to(torch.float32), 
            x, y)


    # check if cbm format is correct by converting it to csr format (for now)
    
    # TODO: adapt to new representation
    '''
    def check_format(self, edge_index):

        n_rows = edge_index[0].max().item() + 1
        
        # create list of sets
        s = [set() for n in range(n_rows + 1)]

        d_row_ptr = self.deltas.crow_indices().to(torch.int32)
        d_col_idx = self.deltas.col_indices().to(torch.int32)
        d_values = self.deltas.values().to(torch.float32)

        visited = set()        
        queue = deque([n_rows])
        
        while queue:
            u = queue.popleft()

            if u not in visited:
                u_bot = self.mca_row_ptr[u]
                u_top = self.mca_row_ptr[u + 1]
                visited.add(u)

                for v in self.mca_col_idx[u_bot : u_top]:
                    queue.append(v)
                    v_bot = d_row_ptr[v]
                    v_top = d_row_ptr[v + 1]
                    s[v].update(s[u])

                    for col_index, value in zip(d_col_idx[v_bot : v_top],d_values[v_bot : v_top]):
                        if value.item() > 0:
                            s[v].add(col_index.item())
                        elif value.item() < 0:
                            s[v].remove(col_index.item())
                        else:
                            raise ValueError(f"Failed with the edge index: {edge_index}")

        # convert edge_index into list of adjacencies
        a = [set() for n in range(n_rows)]

        for row_index, col_index in zip(edge_index[0], edge_index[1]):
            a[row_index.item()].add(col_index.item())

        for ss, sa in zip(s, a):
            if(not ss == sa):
                raise ValueError(f"Format is incorrect!")
        
        print("CBM format is equivalent to CSR...")
    '''