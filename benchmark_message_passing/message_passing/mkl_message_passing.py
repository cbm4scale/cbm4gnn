from torch import ones, int32, empty

from benchmark_message_passing.message_passing.mkl_csr_spmm import mkl_single_csr_spmm
from benchmark_message_passing.message_passing.base_message_passing import MessagePassing


class MKLCSRSparseMatrixMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(MKLCSRSparseMatrixMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_values = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
        else:
            raise ValueError("x must be in kwargs")
        if self.cached:
            if self.cached_values is None:
                self.cached_values = ones(edge_index.size(1), dtype=x.dtype, device=x.device)
            values = self.cached_values
        else:
            values = ones(edge_index.size(1), dtype=x.dtype, device=x.device)
        result = empty(edge_index.max() + 1, *x.size()[1:], dtype=x.dtype, device=x.device)
        mkl_single_csr_spmm(edge_index.to(int32), values, x, result)
        return result
