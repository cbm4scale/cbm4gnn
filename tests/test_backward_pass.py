from unittest import TestCase, main

from torch import float32, randint, ones, int32, sparse_coo_tensor, testing
from torch_geometric.datasets import SuiteSparseMatrixCollection
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from cbm.cbm import cbm_matrix


def generate_erdos_renyi_graph(num_nodes, p_edges, num_features, requires_grad=True):
    """
    Generate a random graph with an Erdos-Renyi model

    :param num_nodes: Number of nodes
    :param p_edges: Probability of an edge between two nodes
    :param num_features: Length of the feature vector
    :param requires_grad: Whether the features require gradients
    :return: Tuple of edge_index, features
    """
    edge_index = erdos_renyi_graph(num_nodes, p_edges, directed=False)
    edge_index, _ = remove_self_loops(edge_index)
    features = randint(0, 10, (num_nodes, num_features), dtype=float32, requires_grad=requires_grad)
    return edge_index.to(int32), features


class TestMatMulBackwardPassCbmAgainstTorchSparse(TestCase):
    def setUp(self):
        self.num_features = 10

    def test_random_generate_erdos_renyi_graph(self):
        num_nodes = 25
        p_edges = 0.5

        edge_index, features = generate_erdos_renyi_graph(num_nodes, p_edges, self.num_features)
        values = ones(edge_index.size(1), dtype=float32, requires_grad=True)

        adjacency_as_torch_sparse = sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        torch_sparse_result = adjacency_as_torch_sparse @ features
        torch_sparse_result.sum().backward()
        torch_feature_grad = features.grad.clone()

        cbm_m = cbm_matrix(edge_index, values, alpha=3)
        cbm_result = cbm_m.seq_torch_csr_matmul(features)
        cbm_result.sum().backward()
        cbm_feature_grad = features.grad.clone()

        testing.assert_close(torch_feature_grad.shape, cbm_feature_grad.shape, atol=1, rtol=1)

    def test_suite_sparse_graph(self):
        for group, name in [("SNAP", "ca-AstroPh"), ("SNAP", "ca-HepPh")]:
            edge_index = SuiteSparseMatrixCollection(root="/tmp", name=name, group=group).data.edge_index.to(int32)
            values = ones(edge_index.size(1), dtype=float32, requires_grad=True)
            num_nodes = edge_index.max() + 1
            features = randint(0, 10, (num_nodes, self.num_features), dtype=float32, requires_grad=True)

            adjacency_as_torch_sparse = sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
            torch_sparse_result = adjacency_as_torch_sparse @ features
            torch_sparse_result.sum().backward()
            torch_feature_grad = features.grad.clone()

            cbm_m = cbm_matrix(edge_index, values, alpha=3)
            cbm_result = cbm_m.seq_torch_csr_matmul(features)
            cbm_result.sum().backward()
            cbm_feature_grad = features.grad.clone()

            testing.assert_close(torch_feature_grad.shape, cbm_feature_grad.shape, atol=1, rtol=1)


if __name__ == '__main__':
    main()
