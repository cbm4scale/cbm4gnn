from torch import sparse_coo_tensor, float32, ones
from torch_geometric.data import Batch
from torch_geometric.datasets import Planetoid, Amazon, SuiteSparseMatrixCollection, Reddit, TUDataset, PPI, Coauthor, JODIEDataset


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
    datasets_dict = {}
    for group, name in list_of_datasets_groups_and_names:
        if group == "Planetoid":
            datasets_dict[name] = Planetoid(root="../data", name=name).data
        elif group == "Coauthor":
            datasets_dict[name] = Coauthor(root="../data", name=name).data
        elif group == "Amazon":
            datasets_dict[name] = Amazon(root="../data", name=name).data
        elif group == "Reddit":
            datasets_dict[name] = Reddit(root="../data").data
        elif group == "TUDataset":
            datasets_dict[name] = Batch.from_data_list(TUDataset(root="../data", name=name))
        elif group == "PPI":
            datasets_dict[name] = Batch.from_data_list(PPI(root="../data"))
        elif group == "JODIEDataset":
            datasets_dict[name] = JODIEDataset(root="../data", name=name).data
        else:
            datasets_dict[name] = SuiteSparseMatrixCollection(root="../data", name=name, group=group).data
    return datasets_dict


def calculate_compression_ratio(edge_index, c_matrix):
    a_matrix = sparse_coo_tensor(edge_index,
                                 ones(edge_index.size(1), dtype=float32),
                                 (edge_index.max() + 1, edge_index.max() + 1),
                                 dtype=float32).to_sparse_csr()
    # Calculate total number of elements in a_matrix representation
    a_matrix_total = a_matrix.crow_indices().numel() + a_matrix.col_indices().numel() + a_matrix.values().numel()

    # Calculate total number of elements in c_matrix representation
    c_matrix_total = (
            c_matrix.deltas.crow_indices().numel() +
            c_matrix.deltas.col_indices().numel() +
            c_matrix.deltas.values().numel() +
            c_matrix.mca_branches.numel() +
            c_matrix.mca_row_idx.numel() +
            c_matrix.mca_col_idx.numel()
    )

    # Compute compression ratio
    compression_ratio = a_matrix_total / c_matrix_total
    return compression_ratio

def calculate_compression_ratio_dad(edge_index, c_matrix):
    a_matrix = sparse_coo_tensor(edge_index,
                                 ones(edge_index.size(1), dtype=float32),
                                 (edge_index.max() + 1, edge_index.max() + 1),
                                 dtype=float32).to_sparse_csr()
    # Calculate total number of elements in a_matrix representation
    a_matrix_total = a_matrix.crow_indices().numel() + a_matrix.col_indices().numel() + a_matrix.values().numel()

    # Calculate total number of elements in c_matrix representation
    c_matrix_total = (
            c_matrix.deltas.crow_indices().numel() +
            c_matrix.deltas.col_indices().numel() +
            c_matrix.deltas.values().numel() +
            c_matrix.mca_branches.numel() +
            c_matrix.mca_row_idx.numel() +
            c_matrix.mca_col_idx.numel() +
            c_matrix.D.numel()
    )

    # Compute compression ratio
    compression_ratio = a_matrix_total / c_matrix_total
    return compression_ratio
