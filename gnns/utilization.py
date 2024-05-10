from torch import diag, float32, Tensor, zeros, ones


def normalize_torch_adj(a: Tensor) -> Tensor:
    deg = a.sum(dim=0, keepdim=True)
    deg_inv_sqrt = deg.to_dense().pow_(-0.5)
    deg_inv_sqrt = diag(deg_inv_sqrt.squeeze())
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return (deg_inv_sqrt @ a @ deg_inv_sqrt).to_sparse_csr()


def degree(index: Tensor, num_nodes: int = None, dtype=None) -> Tensor:
    if num_nodes is None:
        num_nodes = index.max().item() + 1
    if dtype is None:
        dtype = index.dtype
    out = zeros((num_nodes,), dtype=dtype, device=index.device)
    one = ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def normalize_edge_index(edge_index: Tensor, num_nodes: int = None) -> Tensor:
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes, float32)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_weight
