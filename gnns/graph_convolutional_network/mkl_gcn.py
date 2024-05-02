from torch_geometric.nn import Linear

from gnns.message_passing import MKLSequentialCSRSparseMatrixMessagePassing, MKLParallelCSRSparseMatrixMessagePassing


class MKLSequentialCSRSparseMatrixGCN(MKLSequentialCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MKLSequentialCSRSparseMatrixGCN, self).__init__()
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


class MKLParallelCSRSparseMatrixGCN(MKLParallelCSRSparseMatrixMessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MKLParallelCSRSparseMatrixGCN, self).__init__()
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