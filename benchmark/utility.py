from torch import sparse_coo_tensor, float32, ones
from torch_geometric.datasets import Planetoid, Amazon, SuiteSparseMatrixCollection


def underline(text, flag=True):
    return f"\033[4m{text}\033[0m" if flag else text


def bold(text, flag=True):
    return f"\033[1m{text}\033[0m" if flag else text


def download_only_datasets(list_of_datasets_groups_and_names):
    [Planetoid(root="../data", name=name)
     if group == "Planetoid" else
     Amazon(root="../data", name=name)
     if group == "Amazon" else
     SuiteSparseMatrixCollection(root="../data", name=name, group=group)
     for group, name in list_of_datasets_groups_and_names]


def download_and_return_datasets_as_dict(list_of_datasets_groups_and_names):
    return {
        name:
            Planetoid(root="../data", name=name).data
            if group == "Planetoid" else
            Amazon(root="../data", name=name).data
            if group == "Amazon" else
            SuiteSparseMatrixCollection(root="../data", name=name, group=group).data
        for group, name in list_of_datasets_groups_and_names
    }


def calculate_compression_ratio(edge_index, c_matrix):
    a_matrix = sparse_coo_tensor(edge_index,
                                 ones(edge_index.size(1), dtype=float32),
                                 (edge_index.max() + 1, edge_index.max() + 1),
                                 dtype=float32).to_sparse_csr()
    # Calculate total number of elements in a_matrix representation
    a_matrix_total = a_matrix.crow_indices().numel() + a_matrix.col_indices().numel() + a_matrix.values().numel()

    # Calculate total number of elements in c_matrix representation
    c_matrix_total = (
            c_matrix.deltas.col_indices().numel() +
            c_matrix.deltas.crow_indices().numel() +
            c_matrix.deltas.values().numel() +
            c_matrix.mst_row.numel() +
            c_matrix.mst_col.numel()
    )

    # Compute compression ratio
    compression_ratio = a_matrix_total / c_matrix_total
    return compression_ratio
