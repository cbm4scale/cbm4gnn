import time
from datetime import datetime
from os import makedirs
from os.path import exists

import numpy as np
import torch
from prettytable import PrettyTable
from scipy.sparse import csr_matrix

from cbm.cbm import cbm_matrix
from cbm import cbm_mkl_cpp as cbm_
from benchmark.utility import underline, bold, download_and_return_datasets_as_dict
from logger import setup_logger

iters = 50
sizes = [512, 1024]
alphas = [0, 1, 2, 4, 8, 16, 32, 64, 128]
datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "cit-HepTh"),
    ("SNAP", "ca-AstroPh"),
    ("SNAP", "web-Stanford"),
    ("SNAP", "web-NotreDame"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
]


def omp_mkl_csr_spmm(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
    row_ptr_e = a.crow_indices()[1:].to(torch.int32)
    col_ptr = a.col_indices().to(torch.int32)
    val_ptr = a.values().to(torch.float32)
    cbm_.omp_s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)

def timing(edge_index, name):
    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)

    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    a = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    dim_size = rowptr.size(0) - 1
    avg_row_len = edge_index.size(1) / dim_size

    for size in sizes:
        torch_csr_spmm_time_list, mkl_spmm_time_list, cbm_mkl_spmm_time_list, cbm_torch_csr_spmm_time_list = [], [], [], []
        for alpha_i in alphas:
            c = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha_i)

            x = torch.randn((mat.shape[0], size))
            x = x.squeeze(-1) if size == 1 else x

            t_list = []
            y0 = a @ x
            for _ in range(iters):
                start_time = time.perf_counter()
                y0 = a @ x
                t_list += [time.perf_counter() - start_time]
            torch_spmm_time = np.mean(t_list)

            time.sleep(1)

            t_list = []
            y = torch.empty(dim_size, size, dtype=x.dtype)
            y1 = c.omp_torch_csr_matmul(x)
            for _ in range(iters):
                start_time = time.perf_counter()
                y1 = c.omp_torch_csr_matmul(x)
                t_list += [time.perf_counter() - start_time]
            cbm_torch_csr_matmul_time = np.mean(t_list)

            time.sleep(1)

            t_list = []
            y2 = torch.empty(dim_size, size, dtype=x.dtype)
            omp_mkl_csr_spmm(a, x, y2)
            for _ in range(iters):
                start_time = time.perf_counter()
                omp_mkl_csr_spmm(a, x, y2)
                t_list += [time.perf_counter() - start_time]
            mkl_spmm_time = np.mean(t_list)

            time.sleep(1)

            t_list = []
            y3 = torch.empty(dim_size, size, dtype=x.dtype)
            c.omp_mkl_csr_spmm_update(x, y3)
            for _ in range(iters):
                start_time = time.perf_counter()
                c.omp_mkl_csr_spmm_update(x, y3)
                t_list += [time.perf_counter() - start_time]
            cbm_mkl_spmm_time = np.mean(t_list)

            torch_csr_spmm_time_list += [torch_spmm_time]
            cbm_torch_csr_spmm_time_list += [cbm_torch_csr_matmul_time]
            mkl_spmm_time_list += [mkl_spmm_time]
            cbm_mkl_spmm_time_list += [cbm_mkl_spmm_time]

            torch.testing.assert_close(y0, y1, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(y0, y2, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(y0, y3, atol=1e-2, rtol=1e-2)
            del x

        del rowptr, mat, values

        time_tensor = torch.tensor([torch_csr_spmm_time_list,
                                   cbm_torch_csr_spmm_time_list,
                                   mkl_spmm_time_list,
                                   cbm_mkl_spmm_time_list])

        winner = torch.zeros_like(time_tensor, dtype=torch.bool)
        winner[torch.arange(len(time_tensor)), time_tensor.argmin(dim=1)] = 1
        winner[0] = 0
        winner[2] = 0
        winner = winner.tolist()

        table = PrettyTable()
        table.title = f"{name} / size: {size} (avg row length: {avg_row_len:.2f}, num_nodes: {a.size(0)}, num_edges: {edge_index.size(1)})"
        header = [""] + [f"{alpha:>5}" for alpha in alphas]
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
        timing(data_i.edge_index, name_i)