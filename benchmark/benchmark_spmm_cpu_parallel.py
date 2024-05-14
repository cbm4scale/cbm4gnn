import time
from datetime import datetime
from os import makedirs
from os.path import exists

import torch
from prettytable import PrettyTable
from scipy.sparse import csr_matrix

from cbm.cbm import cbm_matrix
from cbm import cbm_mkl_cpp as cbm_
from cbm.utility import check_edge_index
from benchmark.utility import underline, bold, download_and_return_datasets_as_dict, calculate_compression_ratio
from logger import setup_logger

iters = 50
sizes = [128, 256, 512, 1024, 2048]
datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "ca-HepTh"),
    ("SNAP", "cit-HepPh"),
    ("SNAP", "cit-HepTh"),
    ("SNAP", "ca-AstroPh"),
    ("SNAP", "web-Stanford"),
    ("SNAP", "web-NotreDame"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
]


alpha_per_dataset = {
    "ca-HepPh": 3,
    "ca-HepTh": 3,
    "cit-HepPh": 3,
    "cit-HepTh": 3,
    "ca-AstroPh": 3,
    "web-Stanford": 3,
    "web-NotreDame": 3,
    "Cora": 3,
    "PubMed": 3,
    "coPapersDBLP": 3,
    "coPapersCiteseer": 3,
}

def omp_mkl_csr_spmm(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
    row_ptr_e = a.crow_indices()[1:].to(torch.int32)
    col_ptr = a.col_indices().to(torch.int32)
    val_ptr = a.values().to(torch.float32)
    cbm_.omp_s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)


@torch.no_grad()
def correctness(edge_index, alpha):
    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(torch.long)
    col = torch.from_numpy(mat.tocoo().col).to(torch.long)
    edge_index = torch.stack([row, col]).to(torch.long)
    check_edge_index(edge_index)

    values = torch.ones(row.size(0), dtype=torch.float32)
    a = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    c = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha)
    c.check_format(edge_index)

    logger.info(f"Compression ratio: {calculate_compression_ratio(edge_index, c):.2f}")
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


def timing(edge_index, name, alpha):
    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)

    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    a = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    c = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha)
    dim_size = rowptr.size(0) - 1
    avg_row_len = edge_index.size(1) / dim_size

    torch_csr_spmm_time_list, mkl_spmm_time_list, cbm_mkl_spmm_time_list, cbm_torch_csr_spmm_time_list = [], [], [], []

    for size in sizes:
        x = torch.randn((mat.shape[0], size))
        x = x.squeeze(-1) if size == 1 else x

        y0 = a @ x
        start_time = time.perf_counter()
        for _ in range(iters):
            y0 = a @ x
        torch_spmm_time = (time.perf_counter() - start_time) / iters

        time.sleep(1)

        y = torch.empty(dim_size, size, dtype=x.dtype)
        y1 = c.omp_torch_csr_matmul(x)
        start_time = time.perf_counter()
        for _ in range(iters):
            y1 = c.omp_torch_csr_matmul(x)
        cbm_torch_csr_matmul_time = (time.perf_counter() - start_time) / iters

        time.sleep(1)

        y2 = torch.empty(dim_size, size, dtype=x.dtype)
        omp_mkl_csr_spmm(a, x, y2)
        start_time = time.perf_counter()
        for _ in range(iters):
            omp_mkl_csr_spmm(a, x, y2)
        mkl_spmm_time = (time.perf_counter() - start_time) / iters

        time.sleep(1)

        y3 = torch.empty(dim_size, size, dtype=x.dtype)
        c.omp_mkl_csr_spmm_update(x, y3)
        start_time = time.perf_counter()
        for _ in range(iters):
            c.omp_mkl_csr_spmm_update(x, y3)
        cbm_mkl_spmm_time = (time.perf_counter() - start_time) / iters

        torch_csr_spmm_time_list += [torch_spmm_time]
        cbm_torch_csr_spmm_time_list += [cbm_torch_csr_matmul_time]
        mkl_spmm_time_list += [mkl_spmm_time]
        cbm_mkl_spmm_time_list += [cbm_mkl_spmm_time]

        torch.testing.assert_close(y0, y1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(y0, y2, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(y0, y3, atol=1e-2, rtol=1e-2)
        del x

    del rowptr, mat, values

    torch_time = torch.tensor([torch_csr_spmm_time_list, cbm_torch_csr_spmm_time_list])
    mkl_time = torch.tensor([mkl_spmm_time_list, cbm_mkl_spmm_time_list])

    torch_winner = torch.zeros_like(torch_time, dtype=torch.bool)
    mkl_winner = torch.zeros_like(mkl_time, dtype=torch.bool)

    torch_winner[torch_time.argmin(dim=0), torch.arange(len(sizes))] = 1
    mkl_winner[mkl_time.argmin(dim=0), torch.arange(len(sizes))] = 1

    winner = torch.cat([torch_winner, mkl_winner], dim=0).tolist()

    table = PrettyTable()
    table.title = f"{name} alpha: {alpha} (avg row length: {avg_row_len:.2f}, num_nodes: {a.size(0)}, num_edges: {edge_index.size(1)})"
    header = [""] + [f"{size:>5}" for size in sizes]
    table.field_names = header
    methods = ["torch_csr_spmm", "cbm_torch_csr_spmm", "mkl_csr_spmm", "cbm_mkl_csr_spmm"]

    compute_improvement_percentage = lambda reference, value: ((reference - value) / reference) * 100

    cbm_torch_csr_spmm_time_list = [f"{cbm_t} ({compute_improvement_percentage(torch_t, cbm_t):.1f}%)"
                                    for cbm_t, torch_t in zip(cbm_torch_csr_spmm_time_list, torch_csr_spmm_time_list)]
    cbm_mkl_spmm_time_list = [f"{cbm_t} ({compute_improvement_percentage(mkl_t, cbm_t):.1f}%)"
                              for cbm_t, mkl_t in zip(cbm_mkl_spmm_time_list, mkl_spmm_time_list)]

    time_data = [torch_csr_spmm_time_list, cbm_torch_csr_spmm_time_list, mkl_spmm_time_list, cbm_mkl_spmm_time_list]
    for method, times, wins in zip(methods, time_data, winner):
        row = [method, ] + [f"{underline(bold(t, w), w)}" for t, w in zip(times, wins)]
        table.add_row(row)
    logger.info(table)


if __name__ == "__main__":
    log_path = f"./logs/"
    if not exists(log_path):
        makedirs(log_path)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    logger = setup_logger(filename=f"{log_path}/spmm_cpu_parallel-{current_time}.log", verbose=True)
    name_edge_index_dict = download_and_return_datasets_as_dict(datasets)
    for name_i, data_i in name_edge_index_dict.items():
        correctness(data_i.edge_index, alpha_per_dataset[name_i])
        timing(data_i.edge_index, name_i, alpha_per_dataset[name_i])