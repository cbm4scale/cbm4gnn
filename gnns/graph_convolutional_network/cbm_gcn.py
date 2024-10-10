from torch import stack, int32, ones, ones_like, empty
from torch.nn import Module, Linear, Parameter

from cbm.cbm4gcn import cbm4gcn


class CBMSparseMatrixGCN(Module):
    def __init__(self, in_channels, out_channels, cached: bool = True, alpha: int = 0):
        super(CBMSparseMatrixGCN, self).__init__()
        self.alpha = alpha
        self.cached = cached
        self.cached_a = None
        self.cached_empty_tensor = None
        self.linear = Linear(in_channels, out_channels, bias=False)
        self.linear.weight = Parameter(ones_like(self.linear.weight))


    def reset_parameters(self):
        self.linear.reset_parameters()
        self.linear.weight = Parameter(ones_like(self.linear.weight))
        self.cached_a = None
        self.cached_empty_tensor = None

    def forward(self, edge_index, size=None, **kwargs):
        if "x" in kwargs:
            x = kwargs["x"]
        else:
            raise ValueError("x must be in kwargs")

        x = self.linear(x)

        if self.cached:
            if self.cached_a is None:
                self.cached_a = cbm4gcn(edge_index.to(int32), ones(edge_index.size(1), dtype=x.dtype), alpha=self.alpha)
            if self.cached_empty_tensor is None:
                self.cached_empty_tensor = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)
            a_t = self.cached_a
            out = self.cached_empty_tensor
        #else:
        #    a_t = cbm4gcn(edge_index.to(int32), ones(edge_index.size(1), dtype=x.dtype), alpha=self.alpha)
        #    out = empty(size=(edge_index.max().item() + 1, x.size(1),), dtype=x.dtype, device=x.device)

        a_t.matmul(x, out)
        return out