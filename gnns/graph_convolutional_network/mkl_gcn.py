from torch import int32, float32, ones_like, sparse_coo_tensor, empty
from torch.nn import Module, Linear, Parameter

from cbm import cbm_mkl_cpp as mkl
from gnns.utility import normalize_edge_index


def mkl_matmul(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(int32)
    row_ptr_e = a.crow_indices()[1:].to(int32)
    col_ptr = a.col_indices().to(int32)
    val_ptr = a.values().to(float32)
    mkl.s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)


class MKLCSRSparseMatrixGCN(Module):
    def __init__(self, in_channels, out_channels, cached: bool = True):
        super(MKLCSRSparseMatrixGCN, self).__init__()
        self.cached = cached
        self.cached_weight = None
        self.cached_a = None
        self.cached_empty_tensor = None
        self.linear = Linear(in_channels, out_channels, bias=False)
        self.linear.weight = Parameter(ones_like(self.linear.weight))

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.linear.weight = Parameter(ones_like(self.linear.weight))
        self.cached_weight = None
        self.cached_a = None
        self.cached_empty_tensor = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
            size = size if size is not None else (x.size(0), x.size(0))
        else:
            raise ValueError("x must be in kwargs")

        
        x = self.linear(x)

        if self.cached:
            if self.cached_weight is None:
                self.cached_weight = normalize_edge_index(edge_index)
            edge_weight = self.cached_weight

            if self.cached_a is None:
                a = sparse_coo_tensor(edge_index, edge_weight, size=size, device=x.device)
                self.cached_a = a.to_sparse_csr()
            if self.cached_empty_tensor is None:
                self.cached_empty_tensor = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)
            a = self.cached_a
            out = self.cached_empty_tensor
        #else:
        #    edge_weight = normalize_edge_index(edge_index)
        #    a = sparse_coo_tensor(edge_index, edge_weight, size=size, device=x.device)
        #    a = a.to_sparse_csr()
        #    out = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)

        mkl_matmul(a, x, out)
        return out
