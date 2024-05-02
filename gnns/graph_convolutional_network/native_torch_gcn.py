from torch.nn.init import constant
from torch_geometric.nn import Linear

from gnns.message_passing import (NativePytorchScatterAddMessagePassing, NativePytorchCOOSparseMatrixMessagePassing,
                                  NativePytorchCSRSparseMatrixMessagePassing, )


# Subclasses for each specific type of message passing
class NativePytorchScatterAddGCN(NativePytorchScatterAddMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(NativePytorchScatterAddGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=True, weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = super().forward(edge_index, x=x)
        x = self.lin(x)
        return x


class NativePytorchCOOSparseMatrixGCN(NativePytorchCOOSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(NativePytorchCOOSparseMatrixGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=True, weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = super().forward(edge_index, x=x)
        x = self.lin(x)
        return x


class NativePytorchCSRSparseMatrixGCN(NativePytorchCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(NativePytorchCSRSparseMatrixGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=True, weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = super().forward(edge_index, x=x)
        x = self.lin(x)
        return x
