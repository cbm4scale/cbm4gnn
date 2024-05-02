from unittest import TestCase, main

from torch import randint, int64, float32, tensor, testing, rand
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from gnns.message_passing import (NativePytorchScatterAddMessagePassing, NativePytorchCOOSparseMatrixMessagePassing,
                                  NativePytorchCSRSparseMatrixMessagePassing, TorchScatterCOOScatterAddMessagePassing,
                                  TorchScatterGatherCOOSegmentCOO, TorchSparseCSRSparseMatrixMessagePassing,
                                  MKLSequentialCSRSparseMatrixMessagePassing, MKLParallelCSRSparseMatrixMessagePassing,
                                  CBMSequentialMKLCSRSparseMatrixMessagePassing, CBMParallelMKLCSRSparseMatrixMessagePassing,
                                  CBMSequentialTorchCSRSparseMatrixMessagePassing, CBMParallelTorchCSRSparseMatrixMessagePassing)


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


class TestAllMessagePassing(TestCase):
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


if __name__ == '__main__':
    main()
