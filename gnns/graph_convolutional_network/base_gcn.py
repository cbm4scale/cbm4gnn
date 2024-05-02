from torch import empty, zeros
from torch.nn import Module, Parameter
from torch_geometric.nn import Linear


class BaseGCN(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

    def reset_parameters(self):
        # Reset parameters is common to all GCN types
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index):
        x = self.lin(x)
        x = self.message_passing(edge_index, x=x)
        x = x + self.bias
        return x

    def message_passing(self, edge_index, x):
        raise NotImplementedError("Subclasses should implement this!")
