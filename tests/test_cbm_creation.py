from unittest import TestCase, main

from torch import float32, randint, ones, int32
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


class TestCreationCBM(TestCase):
    def setUp(self):
        self.num_nodes = 25
        self.p_edges = 0.5
        self.num_features = 10

        self.alpha_values = [1, 2, 3, 4]

    def test_random_generate_erdos_renyi_graph(self):
        edge_index, _ = generate_erdos_renyi_graph(self.num_nodes, self.p_edges, self.num_features)
        values = ones(edge_index.size(1), dtype=float32)
        # Check cbm creation with different alpha values
        for alpha in self.alpha_values:
            cbm_m = cbm_matrix(edge_index, values, alpha=alpha)
            self.assertIsNotNone(cbm_m)

    def test_suite_sparse_graph(self):
        for group, name in [("SNAP", "ca-AstroPh"), ("SNAP", "ca-HepPh")]:
            edge_index = SuiteSparseMatrixCollection(root="/tmp", name=name, group=group).data.edge_index.to(int32)
            values = ones(edge_index.size(1), dtype=float32)
            # Check cbm creation with different alpha values
            for alpha in self.alpha_values:
                cbm_m = cbm_matrix(edge_index, values, alpha=alpha)
                self.assertIsNotNone(cbm_m)


if __name__ == '__main__':
    main()
