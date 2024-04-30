from torch import ones, cat, tensor
from torch_scatter import scatter, segment_coo, segment_csr
from torch_sparse import SparseTensor

from gnns.message_passing.base_message_passing import MessagePassing


class TorchScatterCOOScatterAddMessagePassing(MessagePassing):
    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="add")


class TorchScatterGatherCOOSegmentCOO(MessagePassing):
    def aggregate(self, inputs, index, dim_size):
        return segment_coo(inputs, index, dim_size=dim_size, reduce="add")


class TorchSparseCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(TorchSparseCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
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
                self.cached_a_t = SparseTensor(row=edge_index[1], col=edge_index[0], value=ones(edge_index.size(1), dtype=x.dtype), sparse_sizes=size).to_torch_sparse_csr_tensor()
            a_t = self.cached_a_t
        else:
            a_t = SparseTensor(row=edge_index[1], col=edge_index[0], value=ones(edge_index.size(1), dtype=x.dtype), sparse_sizes=size).to_torch_sparse_csr_tensor()
        return a_t @ x


class TorchScatterGatherCSRSegmentCSR(MessagePassing):
    def aggregate(self, inputs, index, dim_size):
        index_ptr = cat([tensor([0]), index.bincount().cumsum(0)], dim=0)
        return segment_csr(inputs, index_ptr, reduce="add")
