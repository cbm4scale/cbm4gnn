from torch import zeros, sparse_coo_tensor, ones, float32

from gnns.utilization import normalize_edge_index, normalize_torch_adj
from gnns.message_passing.base_message_passing import MessagePassing


class NativePytorchScatterAddMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(NativePytorchScatterAddMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
        self.cached = cached
        self.cached_index = None  # (index)
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
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(NativePytorchCOOSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
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
                a = normalize_torch_adj(a) if self.normalize else a
                self.cached_a_t = a.t()
            a_t = self.cached_a_t
        else:
            a = sparse_coo_tensor(edge_index, ones(edge_index.size(1), dtype=x.dtype), size=size, device=x.device)
            a_t = normalize_torch_adj(a).t() if self.normalize else a.t()
        return a_t @ x


class NativePytorchCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, normalize: bool = False, cached: bool = True):
        super(NativePytorchCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.normalize = normalize
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
                a = normalize_torch_adj(a) if self.normalize else a
                self.cached_a_t = a.to_sparse_csr().t()
            a_t = self.cached_a_t
        else:
            a = sparse_coo_tensor(edge_index, ones(edge_index.size(1), dtype=x.dtype), size=size, device=x.device)
            a_t = normalize_torch_adj(a.to_sparse_csr()).t() if self.normalize else a.to_sparse_csr().t()
        return a_t @ x
