import time
import itertools

import torch
from prettytable import PrettyTable
from scipy.sparse import csr_matrix

from torch_geometric.datasets import Planetoid, SuiteSparseMatrixCollection


from cbm.cbm import cbm_matrix
from cbm import cbm_mkl_cpp as cbm_
from cbm.utilization import check_edge_index

short_rows = [
    ("SNAP", "ca-HepTh"),
    ("SNAP", "ca-HepPh"),
    ("SNAP", "cit-HepTh"),
    ("SNAP", "ca-AstroPh"),
    ("SNAP", "web-Stanford"),
    ("SNAP", "web-NotreDame"),
]


def omp_mkl_csr_spmm(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
    row_ptr_e = a.crow_indices()[1:].to(torch.int32)
    col_ptr = a.col_indices().to(torch.int32)
    val_ptr = a.values().to(torch.float32)
    cbm_.omp_s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)


def read_suite_sparse_dataset(dataset):
    group, name = dataset
    edge_index = SuiteSparseMatrixCollection(root="../data", name=name, group=group).data.edge_index
    return edge_index


def read_planetoid_dataset(name):
    edge_index = Planetoid(root="./", name=name).data.edge_index
    return edge_index


def underline(text, flag=True):
    return f"\033[4m{text}\033[0m" if flag else text


def bold(text, flag=True):
    return f"\033[1m{text}\033[0m" if flag else text


@torch.no_grad()
def correctness(dataset):
    group, name = dataset
    if group == "Planetoid":
        edge_index = read_planetoid_dataset(name)
    else:
        edge_index = read_suite_sparse_dataset(dataset)

    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(torch.long)
    col = torch.from_numpy(mat.tocoo().col).to(torch.long)
    edge_index = torch.stack([row, col]).to(torch.long)
    check_edge_index(edge_index)

    values = torch.ones(row.size(0), dtype=torch.float32)
    a = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    c = cbm_matrix(edge_index.to(torch.int32), values, alpha=3)
    c.check_format(edge_index)

    dim_size = rowptr.size(0) - 1

    for size in sizes:
        x = torch.randn((mat.shape[0], size))
        x = x.squeeze(-1) if size == 1 else x

        out0 = a @ x

        out1 = torch.empty(dim_size, size, dtype=x.dtype)
        omp_mkl_csr_spmm(a, x, out1)

        out2 = torch.empty(dim_size, size, dtype=x.dtype)
        c.omp_mkl_csr_spmm_update(x, out2)

        out3 = c.omp_torch_csr_matmul(x)

        torch.testing.assert_close(out0, out1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out2, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out3, atol=1e-2, rtol=1e-2)


def timing(dataset):
    group, name = dataset
    if group == "Planetoid":
        edge_index = read_planetoid_dataset(name)
    else:
        edge_index = read_suite_sparse_dataset(dataset)

    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)

    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    a = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    c = cbm_matrix(edge_index.to(torch.int32), values, alpha=3)
    dim_size = rowptr.size(0) - 1
    avg_row_len = edge_index.size(1) / dim_size

    torch_spmm_time_list, mkl_spmm_time_list, cbm_mkl_spmm_time_list, cbm_torch_csr_matmul_time_list = [], [], [], []

    for size in sizes:
        x = torch.randn((mat.shape[0], size))
        x = x.squeeze(-1) if size == 1 else x

        start_time = time.perf_counter()
        for _ in range(iters):
            y0 = a @ x
        torch_spmm_time = (time.perf_counter() - start_time)

        time.sleep(1)

        y = torch.empty(dim_size, size, dtype=x.dtype)
        start_time = time.perf_counter()
        for _ in range(iters):
            y1 = c.omp_torch_csr_matmul(x)
        cbm_torch_csr_matmul_time = (time.perf_counter() - start_time)

        time.sleep(1)

        y2 = torch.empty(dim_size, size, dtype=x.dtype)
        start_time = time.perf_counter()
        for _ in range(iters):
            omp_mkl_csr_spmm(a, x, y2)
        mkl_spmm_time = (time.perf_counter() - start_time)

        time.sleep(1)

        y3 = torch.empty(dim_size, size, dtype=x.dtype)
        start_time = time.perf_counter()
        for _ in range(iters):
            c.omp_mkl_csr_spmm_update(x, y3)
        cbm_mkl_spmm_time = (time.perf_counter() - start_time)

        torch_spmm_time_list += [torch_spmm_time]
        cbm_torch_csr_matmul_time_list += [cbm_torch_csr_matmul_time]
        mkl_spmm_time_list += [mkl_spmm_time]
        cbm_mkl_spmm_time_list += [cbm_mkl_spmm_time]

        torch.testing.assert_close(y0, y1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(y0, y2, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(y0, y3, atol=1e-2, rtol=1e-2)
        del x

    del rowptr, mat, values, edge_index

    ts = torch.tensor([torch_spmm_time_list, mkl_spmm_time_list, cbm_mkl_spmm_time_list, cbm_torch_csr_matmul_time_list])
    winner = torch.zeros_like(ts, dtype=torch.bool)
    winner[ts.argmin(dim=0), torch.arange(len(sizes))] = 1
    winner = winner.tolist()

    table = PrettyTable()
    name = f"{group}/{name}"
    table.title = f"{name} (avg row length: {avg_row_len:.2f})"
    header = [""] + [f"{size:>5}" for size in sizes]
    table.field_names = header
    methods = ["torch.spmm", "mkl_csr_spmm", "cbm_mkl_csr_spmm", "cbm_torch_csr_matmul"]
    time_data = [torch_spmm_time_list, mkl_spmm_time_list, cbm_mkl_spmm_time_list, cbm_torch_csr_matmul_time_list]
    for method, times, wins in zip(methods, time_data, winner):
        row = [method, ] + [f"{underline(t, w)}" for t, w in zip(times, wins)]
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    iters = 10
    sizes = [50, 100, 500, 1000, 2000, 4000]

    for _ in range(10):  # Warmup.
        torch.randn(100, 100).sum()
    for dataset in itertools.chain(short_rows):
        group, name = dataset
        if group == "Planetoid":
            mat = read_planetoid_dataset(name)
        else:
            mat = read_suite_sparse_dataset(dataset)
        correctness(dataset)
        timing(dataset)