from torch import zeros, ones, sparse_coo_tensor

from benchmark_message_passing.message_passing.base_message_passing import MessagePassing


class NativePytorchIndexSelectScatterAddMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(NativePytorchIndexSelectScatterAddMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_index = None

    def collect(self, inputs, index, dim):
        return inputs.index_select(dim, index)

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


class NativePytorchGatherScatterAddMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(NativePytorchGatherScatterAddMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_collect_index = None
        self.cached_aggregate_index = None

    def collect(self, inputs, index, dim):
        if self.cached:
            if self.cached_collect_index is None:
                self.cached_collect_index = index.view(-1, 1).expand(-1, inputs.size(1))
            index = self.cached_collect_index
        else:
            index = index.view(-1, 1).expand(-1, inputs.size(1))
        return inputs.gather(self.node_dim, index)

    def aggregate(self, inputs, index, dim_size):
        num_features = inputs.shape[1]
        if self.cached:
            if self.cached_aggregate_index is None:
                self.cached_aggregate_index = index.view(-1, 1).expand(-1, num_features) if inputs.dim() > 1 else index
            index = self.cached_aggregate_index
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
