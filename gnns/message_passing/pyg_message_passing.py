from torch import ones, cat, tensor, float32
from torch_scatter import scatter, segment_coo, segment_csr
from torch_sparse import SparseTensor

from gnns.message_passing.base_message_passing import MessagePassing
from gnns.utilization import normalize_edge_index


class TorchScatterCOOScatterAddMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(TorchScatterCOOScatterAddMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
        self.cached = cached
        self.cached_edge = [None, None]  # (edge_index, edge_weight)

    def forward(self, edge_index, size=None, **kwargs):
        if self.cached:
            if self.cached_edge[0] is None:
                self.cached_edge[0] = edge_index
                if self.normalize:
                    self.cached_edge[1] = normalize_edge_index(edge_index)
                else:
                    self.cached_edge[1] = ones(edge_index.size(1), dtype=float32, device=edge_index.device)
            edge_index, edge_weight = self.cached_edge
        else:
            if self.normalize:
                edge_weight = normalize_edge_index(edge_index)
            else:
                edge_weight = ones(edge_index.size(1), dtype=float32, device=edge_index.device)
        return super().forward(edge_index, size=size, edge_weight=edge_weight, **kwargs)

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="add")


class TorchScatterGatherCOOSegmentCOOMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(TorchScatterGatherCOOSegmentCOOMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
        self.cached = cached
        self.cached_edge = [None, None]  # (edge_index, edge_weight)

    def forward(self, edge_index, size=None, **kwargs):
        if self.cached:
            if self.cached_edge[0] is None:
                self.cached_edge[0] = edge_index
                if self.normalize:
                    self.cached_edge[1] = normalize_edge_index(edge_index)
                else:
                    self.cached_edge[1] = ones(edge_index.size(1), dtype=float32, device=edge_index.device)
            edge_index, edge_weight = self.cached_edge
        else:
            if self.normalize:
                edge_weight = normalize_edge_index(edge_index)
            else:
                edge_weight = ones(edge_index.size(1), dtype=float32, device=edge_index.device)
        return super().forward(edge_index, size=size, edge_weight=edge_weight, **kwargs)

    def aggregate(self, inputs, index, dim_size):
        return segment_coo(inputs, index, dim_size=dim_size, reduce="add")


class TorchSparseCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(TorchSparseCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
        self.cached = cached
        self.cached_weight = None
        self.cached_a_t = None

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
                self.cached_a_t = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_weight, sparse_sizes=size).to_torch_sparse_csr_tensor()
            a_t = self.cached_a_t
        else:
            if self.normalize:
                edge_weight = normalize_edge_index(edge_index)
            else:
                edge_weight = ones(edge_index.size(1), dtype=x.dtype)
            a_t = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_weight, sparse_sizes=size).to_torch_sparse_csr_tensor()
        return a_t @ x
