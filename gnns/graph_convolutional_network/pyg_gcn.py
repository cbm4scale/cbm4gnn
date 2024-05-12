from torch_geometric.nn import Linear

from gnns.message_passing import (TorchScatterCOOScatterAddMessagePassing,
                                  TorchScatterGatherCOOSegmentCOOMessagePassing,
                                  TorchSparseCSRSparseMatrixMessagePassing)


class TorchScatterCOOScatterAddGCN(TorchScatterCOOScatterAddMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(TorchScatterCOOScatterAddGCN, self).__init__(normalize=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin(x)
        x = super().forward(edge_index, x=x)
        return x


class TorchScatterGatherCOOSegmentCOOGCN(TorchScatterGatherCOOSegmentCOOMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(TorchScatterGatherCOOSegmentCOOGCN, self).__init__(normalize=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin(x)
        x = super().forward(edge_index, x=x)
        return x


class TorchSparseCSRSparseMatrixGCN(TorchSparseCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(TorchSparseCSRSparseMatrixGCN, self).__init__(normalize=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin(x)
        x = super().forward(edge_index, x=x)
        return x
