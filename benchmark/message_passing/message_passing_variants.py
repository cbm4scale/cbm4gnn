from torch import tensor, cat, empty, ones, int32, zeros, sparse_coo_tensor
from torch_scatter import scatter, segment_coo, segment_csr

from benchmark.message_passing.base_message_passing import MessagePassing
from benchmark.mkl_csr_spmm import mkl_single_csr_spmm


class NativePytorchScatterAddMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(NativePytorchScatterAddMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_index = None

    def aggregate(self, inputs, index, dim_size):
        num_features = inputs.shape[1]
        if self.cached:
            if self.cached_index is None:
                self.cached_index = index.view(-1, 1).expand(-1, num_features) if inputs.dim() > 1 else index
            index = self.cached_index
        else:
            index = index.view(-1, 1).expand(-1, num_features) if inputs.dim() > 1 else index
        return zeros(size=(dim_size, num_features),
                     device=inputs.device,
                     dtype=inputs.dtype,
                     ).scatter_add_(self.node_dim, index, inputs)


class NativePytorchCOOSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(NativePytorchCOOSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_a_t = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
            size = size if size is not None else (x.size(0), x.size(0))
        else:
            raise ValueError("x must be in kwargs")
        if self.cached:
            if self.cached_a_t is None:
                self.cached_a_t = sparse_coo_tensor(edge_index, ones(edge_index.size(1), dtype=x.dtype), size=size, device=x.device).t()
            a_t = self.cached_a_t
        else:
            a_t = sparse_coo_tensor(edge_index, ones(edge_index.size(1), dtype=x.dtype), size=size, device=x.device).t()
        return a_t @ x


class NativePytorchCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(NativePytorchCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_a_t = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
            size = size if size is not None else (x.size(0), x.size(0))
        else:
            raise ValueError("x must be in kwargs")
        if self.cached:
            if self.cached_a_t is None:
                a = sparse_coo_tensor(edge_index, ones(edge_index.size(1), dtype=x.dtype), size=size, device=x.device)
                self.cached_a_t = a.to_sparse_csr().t()
            a_t = self.cached_a_t
        else:
            a = sparse_coo_tensor(edge_index, ones(edge_index.size(1), dtype=x.dtype), size=size, device=x.device)
            a_t = a.to_sparse_csr().t()
        return a_t @ x


class TorchScatterCOOScatterAddMessagePassing(MessagePassing):
    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="add")


class TorchScatterGatherCOOSegmentCOO(MessagePassing):
    def aggregate(self, inputs, index, dim_size):
        return segment_coo(inputs, index, dim_size=dim_size, reduce="add")


class TorchScatterGatherCSRSegmentCSR(MessagePassing):
    def aggregate(self, inputs, index, dim_size):
        index_ptr = cat([tensor([0]), index.bincount().cumsum(0)], dim=0)
        return segment_csr(inputs, index_ptr, reduce="add")


class MKLCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(MKLCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_values = None

    def aggregate(self, inputs, index, dim_size):
        if self.edge_index is None:
            raise ValueError("edge_index must be provided to forward before calling aggregate.")

        if self.cached:
            if self.cached_values is None:
                self.cached_values = ones(self.edge_index.size(1), dtype=inputs.dtype, device=inputs.device)
            values = self.cached_values
        else:
            values = ones(self.edge_index.size(1), dtype=inputs.dtype, device=inputs.device)

        out = empty(dim_size, inputs.size(1), dtype=inputs.dtype, device=inputs.device)
        mkl_single_csr_spmm(self.edge_index.to(int32), values, inputs, out)
        return out
