from unittest import TestCase, main

from torch import randint, int64, float32, tensor, testing, rand
from torch_geometric.nn import GCNConv
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from gnns.message_passing import (NativePytorchScatterAddMessagePassing, NativePytorchCOOSparseMatrixMessagePassing,
                                  NativePytorchCSRSparseMatrixMessagePassing, TorchScatterCOOScatterAddMessagePassing,
                                  TorchScatterGatherCOOSegmentCOO, TorchSparseCSRSparseMatrixMessagePassing,
                                  MKLSequentialCSRSparseMatrixMessagePassing, MKLParallelCSRSparseMatrixMessagePassing,
                                  CBMSequentialMKLCSRSparseMatrixMessagePassing, CBMParallelMKLCSRSparseMatrixMessagePassing,
                                  CBMSequentialTorchCSRSparseMatrixMessagePassing, CBMParallelTorchCSRSparseMatrixMessagePassing)

from gnns.graph_convolutional_network import (NativePytorchScatterAddGCN, NativePytorchCOOSparseMatrixGCN,
                                              NativePytorchCSRSparseMatrixGCN, TorchScatterCOOScatterAddGCN,
                                              TorchScatterGatherCOOSegmentCOOGCN, TorchSparseCSRSparseMatrixGCN,
                                              MKLSequentialCSRSparseMatrixGCN, MKLParallelCSRSparseMatrixGCN,
                                              CBMSequentialMKLCSRSparseMatrixGCN, CBMParallelMKLCSRSparseMatrixGCN,
                                              CBMSequentialTorchCSRSparseMatrixGCN, CBMParallelTorchCSRSparseMatrixGCN)


def generate_erdos_renyi_graph(num_nodes, p_edges, num_features):
    """
    Generate a random graph with an Erdos-Renyi model

    :param num_nodes: Number of nodes
    :param p_edges: Probability of an edge between two nodes
    :param num_features: Length of the feature vector
    :return: Tuple of edge_index, features
    """
    edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=False)
    edge_index, _ = remove_self_loops(edge_index)
    features = randint(0, 10, (num_nodes, num_features), dtype=float32)
    return edge_index, features


class TestAllNativeMessagePassing(TestCase):
    def setUp(self):
        num_nodes = 25
        p_edges = 0.75
        num_features = 10
        self.edge_index, self.x = generate_erdos_renyi_graph(num_nodes, p_edges, num_features)

    def test_native_pytorch(self):
        # Cycle graph to test the message passing
        edge_index = tensor([[0, 1,], [1, 0]], dtype=int64)
        x = rand((2, 10), dtype=float32)
        message_passing = NativePytorchScatterAddMessagePassing()
        y = message_passing.forward(edge_index, x=x)
        y = message_passing.forward(edge_index, x=y)
        testing.assert_allclose(y, x, atol=1e-4, rtol=1e-4)

    def test_all_message_passing_against_native_pytorch(self):
        native_pytorch_scatter_add_mp = NativePytorchScatterAddMessagePassing()
        native_pytorch_coo_mp = NativePytorchCOOSparseMatrixMessagePassing()
        native_pytorch_csr_mp = NativePytorchCSRSparseMatrixMessagePassing()
        torch_scatter_coo_mp = TorchScatterCOOScatterAddMessagePassing()
        torch_scatter_coo_segment_coo_mp = TorchScatterGatherCOOSegmentCOO()
        torch_sparse_csr_mp = TorchSparseCSRSparseMatrixMessagePassing()
        mkl_sequential_csr_mp = MKLSequentialCSRSparseMatrixMessagePassing()
        mkl_parallel_csr_mp = MKLParallelCSRSparseMatrixMessagePassing()
        cbm_sequential_mkl_mp = CBMSequentialMKLCSRSparseMatrixMessagePassing()
        cbm_parallel_mkl_mp = CBMParallelMKLCSRSparseMatrixMessagePassing()
        cbm_sequential_torch_csr_mp = CBMSequentialTorchCSRSparseMatrixMessagePassing()
        cbm_parallel_torch_csr_mp = CBMParallelTorchCSRSparseMatrixMessagePassing()

        native_pytorch_scatter_add_y = native_pytorch_scatter_add_mp.forward(self.edge_index, x=self.x)
        native_pytorch_coo_y = native_pytorch_coo_mp.forward(self.edge_index, x=self.x)
        native_pytorch_csr_y = native_pytorch_csr_mp.forward(self.edge_index, x=self.x)
        torch_scatter_coo_y = torch_scatter_coo_mp.forward(self.edge_index, x=self.x)
        torch_scatter_coo_segment_coo_y = torch_scatter_coo_segment_coo_mp.forward(self.edge_index, x=self.x)
        torch_sparse_csr_y = torch_sparse_csr_mp.forward(self.edge_index, x=self.x)
        mkl_sequential_csr_y = mkl_sequential_csr_mp.forward(self.edge_index, x=self.x)
        mkl_parallel_csr_y = mkl_parallel_csr_mp.forward(self.edge_index, x=self.x)
        cbm_sequential_mkl_y = cbm_sequential_mkl_mp.forward(self.edge_index, x=self.x)
        cbm_parallel_mkl_y = cbm_parallel_mkl_mp.forward(self.edge_index, x=self.x)
        cbm_sequential_torch_csr_y = cbm_sequential_torch_csr_mp.forward(self.edge_index, x=self.x)
        cbm_parallel_torch_csr_y = cbm_parallel_torch_csr_mp.forward(self.edge_index, x=self.x)

        testing.assert_allclose(native_pytorch_scatter_add_y, native_pytorch_coo_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, native_pytorch_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, torch_scatter_coo_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, torch_scatter_coo_segment_coo_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, torch_sparse_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, mkl_sequential_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, mkl_parallel_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, cbm_sequential_mkl_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, cbm_parallel_mkl_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, cbm_sequential_torch_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_scatter_add_y, cbm_parallel_torch_csr_y, atol=1e-4, rtol=1e-4)


class TestGCN(TestCase):
    def setUp(self):
        num_nodes = 25
        p_edges = 0.75
        self.num_features = 10
        self.out_channels = 1
        self.edge_index, self.x = generate_erdos_renyi_graph(num_nodes, p_edges, self.num_features)

    def test_native_pytorch_against_native_pyg(self):
        native_pytorch_gcn = NativePytorchScatterAddGCN(in_channels=self.num_features, out_channels=self.out_channels)
        native_pyg_gcn_layer = GCNConv(in_channels=self.num_features, out_channels=self.out_channels, normalize=False)

        native_pyg_gcn_layer.lin.weight.data = native_pytorch_gcn.lin.weight.data
        native_pyg_gcn_layer.bias.data = native_pytorch_gcn.lin.bias.data

        native_pytorch_y = native_pytorch_gcn(self.x, self.edge_index)
        native_pyg_y = native_pyg_gcn_layer(self.x, self.edge_index)

        testing.assert_allclose(native_pytorch_y, native_pyg_y, atol=1e-4, rtol=1e-4)

        native_pytorch_y.sum().backward()
        native_pyg_y.sum().backward()

        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, native_pyg_gcn_layer.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.bias.grad, native_pyg_gcn_layer.bias.grad, atol=1e-4, rtol=1e-4)

    def sync_weights_and_biases(self, gcn1, gcn2):
        gcn2.lin.weight.data = gcn1.lin.weight.data
        gcn2.lin.bias.data = gcn1.lin.bias.data

    def test_all_gcns_against_native_pytorch(self):
        native_pytorch_gcn = NativePytorchScatterAddGCN(in_channels=self.num_features, out_channels=self.out_channels)
        native_pytorch_coo_gcn = NativePytorchCOOSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        native_pytorch_csr_gcn = NativePytorchCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        torch_scatter_coo_gcn = TorchScatterCOOScatterAddGCN(in_channels=self.num_features, out_channels=self.out_channels)
        torch_scatter_gather_coo_segment_coo_gcn = TorchScatterGatherCOOSegmentCOOGCN(in_channels=self.num_features, out_channels=self.out_channels)
        torch_sparse_csr_gcn = TorchSparseCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        mkl_sequential_csr_gcn = MKLSequentialCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        mkl_parallel_csr_gcn = MKLParallelCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        cbm_sequential_mkl_gcn = CBMSequentialMKLCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        cbm_parallel_mkl_gcn = CBMParallelMKLCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        cbm_sequential_torch_csr_gcn = CBMSequentialTorchCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)
        cbm_parallel_torch_csr_gcn = CBMParallelTorchCSRSparseMatrixGCN(in_channels=self.num_features, out_channels=self.out_channels)

        self.sync_weights_and_biases(native_pytorch_gcn, native_pytorch_coo_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, native_pytorch_csr_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, torch_scatter_coo_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, torch_scatter_gather_coo_segment_coo_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, torch_sparse_csr_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, mkl_sequential_csr_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, mkl_parallel_csr_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, cbm_sequential_mkl_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, cbm_parallel_mkl_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, cbm_sequential_torch_csr_gcn)
        self.sync_weights_and_biases(native_pytorch_gcn, cbm_parallel_torch_csr_gcn)

        native_pytorch_y = native_pytorch_gcn(self.x, self.edge_index)
        native_pytorch_coo_y = native_pytorch_coo_gcn(self.x, self.edge_index)
        native_pytorch_csr_y = native_pytorch_csr_gcn(self.x, self.edge_index)
        torch_scatter_coo_y = torch_scatter_coo_gcn(self.x, self.edge_index)
        torch_scatter_gather_coo_segment_coo_y = torch_scatter_gather_coo_segment_coo_gcn(self.x, self.edge_index)
        torch_sparse_csr_y = torch_sparse_csr_gcn(self.x, self.edge_index)
        mkl_sequential_csr_y = mkl_sequential_csr_gcn(self.x, self.edge_index)
        mkl_parallel_csr_y = mkl_parallel_csr_gcn(self.x, self.edge_index)
        cbm_sequential_mkl_y = cbm_sequential_mkl_gcn(self.x, self.edge_index)
        cbm_parallel_mkl_y = cbm_parallel_mkl_gcn(self.x, self.edge_index)
        cbm_sequential_torch_csr_y = cbm_sequential_torch_csr_gcn(self.x, self.edge_index)
        cbm_parallel_torch_csr_y = cbm_parallel_torch_csr_gcn(self.x, self.edge_index)

        testing.assert_allclose(native_pytorch_y, native_pytorch_coo_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, native_pytorch_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, torch_scatter_coo_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, torch_scatter_gather_coo_segment_coo_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, torch_sparse_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, mkl_sequential_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, mkl_parallel_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, cbm_sequential_mkl_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, cbm_parallel_mkl_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, cbm_sequential_torch_csr_y, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_y, cbm_parallel_torch_csr_y, atol=1e-4, rtol=1e-4)

        native_pytorch_y.sum().backward()
        native_pytorch_coo_y.sum().backward()
        native_pytorch_csr_y.sum().backward()
        torch_scatter_coo_y.sum().backward()
        torch_scatter_gather_coo_segment_coo_y.sum().backward()
        torch_sparse_csr_y.sum().backward()
        mkl_sequential_csr_y.sum().backward()
        mkl_parallel_csr_y.sum().backward()
        cbm_sequential_mkl_y.sum().backward()
        cbm_parallel_mkl_y.sum().backward()
        cbm_sequential_torch_csr_y.sum().backward()
        cbm_parallel_torch_csr_y.sum().backward()

        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, native_pytorch_coo_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, native_pytorch_csr_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, torch_scatter_coo_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, torch_scatter_gather_coo_segment_coo_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, torch_sparse_csr_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, mkl_sequential_csr_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, mkl_parallel_csr_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, cbm_sequential_mkl_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, cbm_parallel_mkl_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, cbm_sequential_torch_csr_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)
        testing.assert_allclose(native_pytorch_gcn.lin.weight.grad, cbm_parallel_torch_csr_gcn.lin.weight.grad, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    main()
