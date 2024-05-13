from torch import int32, float32, sparse_coo_tensor, ones, empty

from cbm import cbm_mkl_cpp as cbm_
from gnns.message_passing.base_message_passing import MessagePassing
from gnns.utility import normalize_edge_index


def seq_mkl_csr_spmm(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(int32)
    row_ptr_e = a.crow_indices()[1:].to(int32)
    col_ptr = a.col_indices().to(int32)
    val_ptr = a.values().to(float32)
    cbm_.seq_s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)


def omp_mkl_csr_spmm(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(int32)
    row_ptr_e = a.crow_indices()[1:].to(int32)
    col_ptr = a.col_indices().to(int32)
    val_ptr = a.values().to(float32)
    cbm_.omp_s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)


class MKLSequentialCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(MKLSequentialCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
        self.cached = cached
        self.cached_weight = None
        self.cached_a_t = None
        self.cached_empty_tensor = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
            size = size if size is not None else (x.size(0), x.size(0))
        else:
            raise ValueError("x must be in kwargs")

        if self.cached:
            if self.cached_weight is None:
                if self.normalize:
                    self.cached_weight = normalize_edge_index(edge_index)
                else:
                    self.cached_weight = ones(edge_index.size(1), dtype=x.dtype)
            edge_weight = self.cached_weight

            if self.cached_a_t is None:
                a = sparse_coo_tensor(edge_index, edge_weight, size=size, device=x.device)
                self.cached_a_t = a.t().to_sparse_csr()
            if self.cached_empty_tensor is None:
                self.cached_empty_tensor = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)
            a_t = self.cached_a_t
            out = self.cached_empty_tensor
        else:
            if self.normalize:
                edge_weight = normalize_edge_index(edge_index)
            else:
                edge_weight = ones(edge_index.size(1), dtype=x.dtype)

            a = sparse_coo_tensor(edge_index, edge_weight, size=size, device=x.device)
            a_t = a.t().to_sparse_csr()
            out = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)

        seq_mkl_csr_spmm(a_t, x, out)
        return out


class MKLParallelCSRSparseMatrixMessagePassing(MKLSequentialCSRSparseMatrixMessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(MKLSequentialCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
        self.cached = cached
        self.cached_weight = None
        self.cached_a_t = None
        self.cached_empty_tensor = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
            size = size if size is not None else (x.size(0), x.size(0))
        else:
            raise ValueError("x must be in kwargs")

        if self.cached:
            if self.cached_weight is None:
                if self.normalize:
                    self.cached_weight = normalize_edge_index(edge_index)
                else:
                    self.cached_weight = ones(edge_index.size(1), dtype=x.dtype)
            edge_weight = self.cached_weight

            if self.cached_a_t is None:
                a = sparse_coo_tensor(edge_index, edge_weight, size=size, device=x.device)
                self.cached_a_t = a.t().to_sparse_csr()
            if self.cached_empty_tensor is None:
                self.cached_empty_tensor = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)
            a_t = self.cached_a_t
            out = self.cached_empty_tensor
        else:
            if self.normalize:
                edge_weight = normalize_edge_index(edge_index)
            else:
                edge_weight = ones(edge_index.size(1), dtype=x.dtype)

            a = sparse_coo_tensor(edge_index, edge_weight, size=size, device=x.device)
            a_t = a.t().to_sparse_csr()
            out = empty(size=(edge_index.max().item() + 1, x.size(1)), dtype=x.dtype, device=x.device)

        omp_mkl_csr_spmm(a_t, x, out)
        return out
