from torch import stack, int32, ones, empty

from cbm.cbm import cbm_matrix
from gnns.message_passing.base_message_passing import MessagePassing


class CBMSequentialMKLCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, alpha: int = 2, cached: bool = True):
        super(CBMSequentialMKLCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.alpha = alpha
        self.cached_a_t = None
        self.cached_empty_tensor = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
        else:
            raise ValueError("x must be in kwargs")

        if self.cached:
            if self.cached_a_t is None:
                edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
                self.cached_a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), normalized=True, alpha=self.alpha)
            if self.cached_empty_tensor is None:
                self.cached_empty_tensor = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)
            a_t = self.cached_a_t
            out = self.cached_empty_tensor
        else:
            edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
            a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), normalized=True, alpha=self.alpha)
            out = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)

        a_t.seq_mkl_csr_fused_spmm_update(x, out)
        return out


class CBMParallelMKLCSRSparseMatrixMessagePassing(CBMSequentialMKLCSRSparseMatrixMessagePassing):
    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
        else:
            raise ValueError("x must be in kwargs")

        if self.cached:
            if self.cached_a_t is None:
                edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
                self.cached_a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), normalized=True, alpha=self.alpha)
            if self.cached_empty_tensor is None:
                self.cached_empty_tensor = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)
            a_t = self.cached_a_t
            out = self.cached_empty_tensor
        else:
            edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
            a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), normalized=True, alpha=self.alpha)
            out = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)

        a_t.omp_mkl_csr_fused_spmm_update(x, out)
        return out


class CBMSequentialTorchCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, alpha: int = 2, cached: bool = True):
        super(CBMSequentialTorchCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.alpha = alpha
        self.cached_a_t = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
        else:
            raise ValueError("x must be in kwargs")

        if self.cached:
            if self.cached_a_t is None:
                edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
                self.cached_a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), alpha=self.alpha)
            a_t = self.cached_a_t
        else:
            edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
            a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), alpha=self.alpha)

        out = a_t.seq_torch_csr_matmul(x)
        return out


class CBMParallelTorchCSRSparseMatrixMessagePassing(CBMSequentialTorchCSRSparseMatrixMessagePassing):
    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
        else:
            raise ValueError("x must be in kwargs")

        if self.cached:
            if self.cached_a_t is None:
                edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
                self.cached_a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), alpha=self.alpha)
            a_t = self.cached_a_t
        else:
            edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
            a_t = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=x.dtype), alpha=self.alpha)

        out = a_t.omp_torch_csr_matmul(x)
        return out