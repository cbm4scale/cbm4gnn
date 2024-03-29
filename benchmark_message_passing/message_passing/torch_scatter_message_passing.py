from torch_scatter import gather_coo, gather_csr, scatter, segment_coo

from benchmark_message_passing.message_passing.base_message_passing import MessagePassing, coo_index_to_csr_indexptr


class TorchScatterGatherCOOScatterAddMessagePassing(MessagePassing):
    def collect(self, inputs, index, dim):
        return gather_coo(inputs, index)

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="add")


class TorchScatterGatherCSRScatterAddMessagePassing(MessagePassing):
    def __init__(self, flow: str = "source_to_target", node_dim: int = 0, cached: bool = True):
        super(TorchScatterGatherCSRScatterAddMessagePassing, self).__init__(flow, node_dim)
        self.cached = cached
        self.cached_index = None

    def collect(self, inputs, index, dim):
        if self.cached:
            if self.cached_index is None:
                self.cached_index = coo_index_to_csr_indexptr(index).to(inputs.device)
            index = self.cached_index
        else:
            index = coo_index_to_csr_indexptr(index).to(inputs.device)
        return gather_csr(inputs, index)

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='add')

