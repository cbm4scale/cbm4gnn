from torch import Tensor


def check_edge_index_contiguous(edge_index: Tensor) -> None:
    is_contiguous = edge_index.unique().size(0) == edge_index.max() + 1
    if not is_contiguous:
        raise ValueError("Edge index is not contiguous")


def check_edge_index_coalesced(edge_index: Tensor) -> None:
    is_coalesced = edge_index.shape[1] == edge_index.unique(dim=1).shape[1]
    if not is_coalesced:
        raise ValueError("Edge index is not coalesced")


def check_edge_index_indices(edge_index: Tensor) -> None:
    if edge_index.min() != 0:
        raise ValueError("Edge index should start from 0")


def check_edge_index_shape(edge_index: Tensor) -> None:
    if edge_index.shape[0] != 2:
        raise ValueError("Edge index should have shape (2, num_edges)")


def check_edge_index(edge_index: Tensor) -> None:
    check_edge_index_shape(edge_index)
    check_edge_index_indices(edge_index)
    check_edge_index_contiguous(edge_index)
    check_edge_index_coalesced(edge_index)