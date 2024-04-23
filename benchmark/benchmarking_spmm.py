# This script is based on the original script from torch_scatter library by rusty1s:
# https://github.com/rusty1s/pytorch_scatter/blob/c095c62e4334fcd05e4ac3c4bb09d285960d6be6/benchmark/scatter_segment.py

import time
import os.path as osp
import itertools

import argparse
import wget
import torch
from prettytable import PrettyTable
from scipy.sparse import csr_matrix
from torch import zeros

from torch_geometric.datasets import Planetoid, SuiteSparseMatrixCollection

from torch_scatter import scatter, segment_coo, segment_csr

from cbm.cbm import cbm_matrix
import cbm.cbm_mkl_cpp as cbm_

short_rows = [
    ("Planetoid", "Cora"),
    ("Planetoid", "Citeseer"),
    ("Planetoid", "Pubmed"),
    ("SNAP", "web-Stanford"),
    ("SNAP", "ca-HepTh"),
    ("SNAP", "ca-AstroPh"),
]


def seq_mkl_csr_spmm(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
    row_ptr_e = a.crow_indices()[1:].to(torch.int32)
    col_ptr = a.col_indices().to(torch.int32)
    val_ptr = a.values().to(torch.float32)
    cbm_.seq_s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)

def omp_mkl_csr_spmm(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
    row_ptr_e = a.crow_indices()[1:].to(torch.int32)
    col_ptr = a.col_indices().to(torch.int32)
    val_ptr = a.values().to(torch.float32)
    cbm_.omp_s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)



def download_suite_sparse_dataset(dataset):
    url = "https://sparse.tamu.edu/mat/{}/{}.mat"
    group, name = dataset
    if not osp.exists(f"{name}.mat"):
        print(f"Downloading {group}/{name}:")
        wget.download(url.format(group, name))
        print("")


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
    rowptr = torch.from_numpy(mat.indptr).to(device_name, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(device_name, torch.long)
    col = torch.from_numpy(mat.tocoo().col).to(device_name, torch.long)
    edge_index = torch.stack([row, col]).to(device_name, torch.long)
    values = torch.ones(row.size(0), dtype=torch.float32, device=device_name)
    a = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    c = cbm_matrix(edge_index.to(torch.int32), values, alpha=3)

    dim_size = rowptr.size(0) - 1

    for size in sizes:
        x = torch.randn((mat.shape[0], size), device=device_name)
        x = x.squeeze(-1) if size == 1 else x

        x_j = x.index_select(0, col)

        out0 = a @ x
        out1 = scatter(x_j, row, dim=0, dim_size=dim_size, reduce="add")
        out2 = segment_coo(x_j, row, dim_size=dim_size, reduce="add")
        out3 = segment_csr(x_j, rowptr, reduce="add")

        out4 = x.new_zeros(dim_size, *x.size()[1:])
        row_tmp = row.view(-1, 1).expand_as(x_j) if x.dim() > 1 else row
        out4.scatter_reduce_(0, row_tmp, x_j, "sum", include_self=False)

        out5 = torch.empty(dim_size, *x.size()[1:], dtype=x.dtype, device=x.device)
        func = omp_mkl_csr_spmm if is_parallel else seq_mkl_csr_spmm
        func(a, x, out5)

        out6 = torch.empty(dim_size, *x.size()[1:], dtype=x.dtype, device=x.device)
        func = c.omp_mkl_csr_spmm_update if is_parallel else c.seq_mkl_csr_spmm_update
        func(x, out6)

        func = c.omp_torch_csr_matmul if is_parallel else c.seq_torch_csr_matmul
        out7 = func(x)

        torch.testing.assert_close(out0, out1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out2, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out3, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out4, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out5, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out6, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out7, atol=1e-2, rtol=1e-2)


def time_func(func, x):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()

    with torch.no_grad():
        for _ in range(iters):
            func(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t


def timing(dataset):
    group, name = dataset
    if group == "Planetoid":
        edge_index = read_planetoid_dataset(name)
    else:
        edge_index = read_suite_sparse_dataset(dataset)

    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(device_name, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(device_name, torch.long)
    col = torch.from_numpy(mat.tocoo().col).to(device_name, torch.long)
    values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device_name)
    a = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    c = cbm_matrix(edge_index.to(torch.int32), values, alpha=3)
    dim_size = rowptr.size(0) - 1
    avg_row_len = edge_index.size(1) / dim_size

    def torch_spmm(x):
        return a @ x

    def torch_scatter_row(x):
        x_j = x.index_select(0, col)
        row_tmp = row.view(-1, 1).expand_as(x_j) if x.dim() > 1 else row
        return zeros(dim_size, *x.size()[1:], dtype=x.dtype, device=device_name).scatter_add_(0, row_tmp, x_j)

    def pyg_scatter_row(x):
        x_j = x.index_select(0, col)
        return scatter(x_j, row, dim=0, dim_size=dim_size, reduce="sum")

    def pyg_segment_coo(x):
        x_j = x.index_select(0, col)
        return segment_coo(x_j, row, reduce="sum")

    def pyg_segment_csr(x):
        x_j = x.index_select(0, col)
        return segment_csr(x_j, rowptr, reduce="sum")

    def mkl_spmm(x):
        out5 = torch.empty(dim_size, *x.size()[1:], dtype=x.dtype, device=x.device)
        func = omp_mkl_csr_spmm if is_parallel else seq_mkl_csr_spmm
        func(a, x, out5)
        return out5

    def cbm_mkl_spmm(x):
        out6 = torch.empty(dim_size, *x.size()[1:], dtype=x.dtype, device=x.device)
        func = c.omp_mkl_csr_spmm_update if is_parallel else c.seq_mkl_csr_spmm_update
        func(x, out6)
        return out6

    def cbm_torch_csr_matmul(x):
        func = c.omp_torch_csr_matmul if is_parallel else c.seq_torch_csr_matmul
        return func(x)


    t0, t1, t2, t3, t4, t5, t6, t7 = [], [], [], [], [], [], [], []

    for size in sizes:
        x = torch.randn((mat.shape[0], size), device=device_name)
        x = x.squeeze(-1) if size == 1 else x

        t0 += [time_func(torch_spmm, x)]
        t1 += [time_func(torch_scatter_row, x)]
        t2 += [time_func(pyg_scatter_row, x)]
        t3 += [time_func(pyg_segment_coo, x)]
        t4 += [time_func(pyg_segment_csr, x)]
        t5 += [time_func(mkl_spmm, x)]
        t6 += [time_func(cbm_mkl_spmm, x)]
        t7 += [time_func(cbm_torch_csr_matmul, x)]

        del x

    del row, rowptr, mat, values, edge_index
    torch.cuda.empty_cache() if torch.cuda.is_available() and device_name == "cuda" else None

    ts = torch.tensor([t0, t1, t2, t3, t4, t5, t6, t7])
    winner = torch.zeros_like(ts, dtype=torch.bool)
    winner[ts.argmin(dim=0), torch.arange(len(sizes))] = 1
    winner = winner.tolist()

    table = PrettyTable()
    name = f"{group}/{name}"
    table.title = f"{name} (avg row length: {avg_row_len:.2f})"
    header = [""] + [f"{size:>5}" for size in sizes]
    table.field_names = header
    methods = ["torch.spmm",
               "torch.scatter_add",
               "pyg_scatter",
               "pyg_segment (COO)",
               "pyg_segment (CSR)",
               "mkl_csr_spmm",
               "cbm_mkl_csr_spmm",
               "cbm_torch_csr_matmul"]
    time_data = [t0, t1, t2, t3, t4, t5, t6, t7]
    for method, times, wins in zip(methods, time_data, winner):
        row = [method, ] + [f"{underline(t, w)}" for t, w in zip(times, wins)]
        table.add_row(row)
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_parallel", action="store_true")
    args = parser.parse_args()

    device_name = args.device
    is_parallel = args.use_parallel

    if not is_parallel:
        torch.set_num_threads(1)

    iters = 10
    sizes = [50, 100, 500, 1000, 2000, 4000]

    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=device_name).sum()
    for dataset in itertools.chain(short_rows):
        group, name = dataset
        if group == "Planetoid":
            mat = read_planetoid_dataset(name)
        else:
            mat = download_suite_sparse_dataset(dataset)
        correctness(dataset)
        timing(dataset)