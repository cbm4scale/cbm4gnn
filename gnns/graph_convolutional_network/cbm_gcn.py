from torch_geometric.nn import Linear

from gnns.message_passing import (CBMSequentialMKLCSRSparseMatrixMessagePassing,
                                  CBMParallelMKLCSRSparseMatrixMessagePassing,
                                  CBMSequentialTorchCSRSparseMatrixMessagePassing,
                                  CBMParallelTorchCSRSparseMatrixMessagePassing,
                                  )


class CBMSequentialMKLCSRSparseMatrixGCN(CBMSequentialMKLCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CBMSequentialMKLCSRSparseMatrixGCN, self).__init__()
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


class CBMParallelMKLCSRSparseMatrixGCN(CBMParallelMKLCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CBMParallelMKLCSRSparseMatrixGCN, self).__init__()
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


class CBMSequentialTorchCSRSparseMatrixGCN(CBMSequentialTorchCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CBMSequentialTorchCSRSparseMatrixGCN, self).__init__()
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


class CBMParallelTorchCSRSparseMatrixGCN(CBMParallelTorchCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CBMParallelTorchCSRSparseMatrixGCN, self).__init__()
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