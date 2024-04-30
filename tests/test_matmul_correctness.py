import os
from unittest import TestCase, main

from torch import float32, randint, ones, int32, sparse_coo_tensor, testing
from torch_geometric.datasets import SuiteSparseMatrixCollection
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from cbm.cbm import cbm_matrix


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
    return edge_index.to(int32), features


class TestMatMulCbmAgainstTorchSparse(TestCase):
    def setUp(self):
        self.num_features = 10

    def test_sequential_random_generate_erdos_renyi_graph(self):
        num_nodes = 25
        p_edges = 0.5

        edge_index, features = generate_erdos_renyi_graph(num_nodes, p_edges, self.num_features)
        values = ones(edge_index.size(1), dtype=float32)

        adjacency_as_torch_sparse = sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        torch_sparse_result = adjacency_as_torch_sparse @ features

        cbm_m = cbm_matrix(edge_index, values, alpha=3)
        cbm_result = cbm_m.seq_torch_csr_matmul(features)

        testing.assert_close(torch_sparse_result, cbm_result, atol=1e-4, rtol=1e-4)

    def test_parallel_random_generate_erdos_renyi_graph(self):
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['GOMP_CPU_AFFINITY'] = '0-8'

        num_nodes = 25
        p_edges = 0.5

        edge_index, features = generate_erdos_renyi_graph(num_nodes, p_edges, self.num_features)
        values = ones(edge_index.size(1), dtype=float32)

        adjacency_as_torch_sparse = sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        torch_sparse_result = adjacency_as_torch_sparse @ features

        cbm_m = cbm_matrix(edge_index, values, alpha=3)
        cbm_result = cbm_m.omp_torch_csr_matmul(features)

        testing.assert_close(torch_sparse_result, cbm_result, atol=1e-4, rtol=1e-4)

    def test_sequential_suite_sparse_graph(self):
        for group, name in [("SNAP", "ca-AstroPh"), ("SNAP", "ca-HepPh")]:
            edge_index = SuiteSparseMatrixCollection(root="/tmp", name=name, group=group).data.edge_index.to(int32)
            values = ones(edge_index.size(1), dtype=float32)
            num_nodes = edge_index.max() + 1
            features = randint(0, 10, (num_nodes, self.num_features), dtype=float32)

            adjacency_as_torch_sparse = sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
            torch_sparse_result = adjacency_as_torch_sparse @ features

            cbm_m = cbm_matrix(edge_index, values, alpha=3)
            cbm_result = cbm_m.seq_torch_csr_matmul(features)

            testing.assert_close(torch_sparse_result, cbm_result, atol=1e-4, rtol=1e-4)

    def test_parallel_suite_sparse_graph(self):
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['GOMP_CPU_AFFINITY'] = '0-8'

        for group, name in [("SNAP", "ca-AstroPh"), ("SNAP", "ca-HepPh")]:
            edge_index = SuiteSparseMatrixCollection(root="/tmp", name=name, group=group).data.edge_index.to(int32)
            values = ones(edge_index.size(1), dtype=float32)
            num_nodes = edge_index.max() + 1
            features = randint(0, 10, (num_nodes, self.num_features), dtype=float32)

            adjacency_as_torch_sparse = sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
            torch_sparse_result = adjacency_as_torch_sparse @ features

            cbm_m = cbm_matrix(edge_index, values, alpha=3)
            cbm_result = cbm_m.omp_torch_csr_matmul(features)

            testing.assert_close(torch_sparse_result, cbm_result, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    main()
