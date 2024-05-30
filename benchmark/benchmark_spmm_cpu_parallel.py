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
from gnns.utility import normalize_torch_adj
from logger import setup_logger

iters = 50
sizes = [100, 500, 1000]
datasets = [
    ("SNAP", "ca-HepPh"),
    # ("SNAP", "ca-HepTh"),
    # ("SNAP", "cit-HepPh"),
    # ("SNAP", "cit-HepTh"),
    ("SNAP", "ca-AstroPh"),
    # ("SNAP", "web-Stanford"),
    # ("SNAP", "web-NotreDame"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
]


alpha_per_dataset = {
    "ca-HepPh": 64,
    # "ca-HepTh": 64,
    # "cit-HepPh": 8,
    # "cit-HepTh": 8,
    "ca-AstroPh": 8,
    # "web-Stanford": 16,
    # "web-NotreDame": 32,
    "Cora": 16,
    "PubMed": 16,
    "coPapersDBLP": 16,
    "coPapersCiteseer": 16,
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
    a_un_normalized = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    a_normalized = normalize_torch_adj(a_un_normalized)


    c_un_normalized = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha)
    c_normalized = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha, normalized=True)

    c_un_normalized.check_format(edge_index)
    c_normalized.check_format(edge_index)

    logger.info(f"Compression ratio: {calculate_compression_ratio(edge_index, c_un_normalized):.2f}")
    dim_size = rowptr.size(0) - 1

    for size in sizes:
        x = torch.randn((mat.shape[0], size))
        x = x.squeeze(-1) if size == 1 else x

        out0 = a_un_normalized @ x

        out1 = torch.empty(dim_size, size, dtype=x.dtype)
        omp_mkl_csr_spmm(a_un_normalized, x, out1)

        out2 = torch.empty(dim_size, size, dtype=x.dtype)
        c_un_normalized.omp_mkl_csr_spmm_update(x, out2)

        # out3 = c.omp_torch_csr_matmul(x)

        out4 = a_normalized @ x

        out5 = torch.empty(dim_size, size, dtype=x.dtype)
        c_normalized.omp_mkl_csr_fused_spmm_update(x, out5)

        torch.testing.assert_close(out0, out1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out0, out2, atol=1e-2, rtol=1e-2)
        # torch.testing.assert_close(out0, out3, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out4, out5, atol=1e-2, rtol=1e-2)


def timing(edge_index, name, alpha):
    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)

    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    a_un_normalized = torch.sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
    a_normalized = normalize_torch_adj(a_un_normalized)

    c_un_normalized = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha)
    c_normalized = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha, normalized=True)

    dim_size = rowptr.size(0) - 1
    avg_row_len = edge_index.size(1) / dim_size

    # torch_csr_spmm_time_list, mkl_spmm_time_list, cbm_mkl_spmm_time_list, cbm_torch_csr_spmm_time_list = [], [], [], []
    un_normalized_csr_spmm_time_list, un_normalized_cbm_spmm_time_list, normalized_csr_spmm_time_list, normalized_cbm_spmm_time_list = [], [], [], []

    for size in sizes:
        x = torch.randn((mat.shape[0], size))
        x = x.squeeze(-1) if size == 1 else x

        y0 = torch.empty(dim_size, size, dtype=x.dtype)
        omp_mkl_csr_spmm(a_un_normalized, x, y0)
        start_time = time.perf_counter()
        for _ in range(iters):
            omp_mkl_csr_spmm(a_un_normalized, x, y0)
        csr_un_normalized_spmm_time = (time.perf_counter() - start_time) / iters

        time.sleep(1)

        y1 = torch.empty(dim_size, size, dtype=x.dtype)
        c_un_normalized.omp_mkl_csr_spmm_update(x, y1)
        start_time = time.perf_counter()
        for _ in range(iters):
            c_un_normalized.omp_mkl_csr_spmm_update(x, y1)
        cbm_un_normalized_spmm_time = (time.perf_counter() - start_time) / iters

        time.sleep(1)

        y2 = torch.empty(dim_size, size, dtype=x.dtype)
        omp_mkl_csr_spmm(a_normalized, x, y2)
        start_time = time.perf_counter()
        for _ in range(iters):
            omp_mkl_csr_spmm(a_normalized, x, y2)
        csr_normalized_spmm_time = (time.perf_counter() - start_time) / iters

        time.sleep(1)

        y3 = torch.empty(dim_size, size, dtype=x.dtype)
        c_normalized.omp_mkl_csr_fused_spmm_update(x, y3)
        start_time = time.perf_counter()
        for _ in range(iters):
            c_normalized.omp_mkl_csr_fused_spmm_update(x, y3)
        cbm_normalized_spmm_time = (time.perf_counter() - start_time) / iters

        un_normalized_csr_spmm_time_list += [csr_un_normalized_spmm_time]
        un_normalized_cbm_spmm_time_list += [cbm_un_normalized_spmm_time]
        normalized_csr_spmm_time_list += [csr_normalized_spmm_time]
        normalized_cbm_spmm_time_list += [cbm_normalized_spmm_time]

        torch.testing.assert_close(y0, y1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(y2, y3, atol=1e-2, rtol=1e-2)
        del x

    del rowptr, mat, values

    # torch_time = torch.tensor([torch_csr_spmm_time_list, cbm_torch_csr_spmm_time_list])
    un_normalized_time = torch.tensor([un_normalized_csr_spmm_time_list, un_normalized_cbm_spmm_time_list])
    normalized_time = torch.tensor([normalized_csr_spmm_time_list, normalized_cbm_spmm_time_list])

    # torch_winner = torch.zeros_like(torch_time, dtype=torch.bool)
    un_normalized_winner = torch.zeros_like(un_normalized_time, dtype=torch.bool)
    normalized_winner = torch.zeros_like(normalized_time, dtype=torch.bool)

    # torch_winner[torch_time.argmin(dim=0), torch.arange(len(sizes))] = 1
    un_normalized_winner[un_normalized_time.argmin(dim=0), torch.arange(len(sizes))] = 1
    normalized_winner[normalized_time.argmin(dim=0), torch.arange(len(sizes))] = 1

    # winner = torch.cat([torch_winner, mkl_winner], dim=0).tolist()
    winner = torch.cat([un_normalized_winner, normalized_winner], dim=0).tolist()

    table = PrettyTable()
    table.title = f"{name} alpha: {alpha} (avg row length: {avg_row_len:.2f}, num_nodes: {a_un_normalized.size(0)}, num_edges: {edge_index.size(1)})"
    header = [""] + [f"{size:>5}" for size in sizes]
    table.field_names = header
    methods = ["csr_spmm", "cbm_spmm", "csr_spmm (norm)", "cbm_spmm (norm)"]

    compute_improvement_percentage = lambda reference, value: ((reference - value) / reference) * 100

    un_normalized_cbm_spmm_time_list = [f"{cbm_t} ({compute_improvement_percentage(csr_t, cbm_t):.1f}%)"
                                        for cbm_t, csr_t in zip(un_normalized_cbm_spmm_time_list, un_normalized_csr_spmm_time_list)]
    normalized_cbm_spmm_time_list = [f"{cbm_t} ({compute_improvement_percentage(csr_t, cbm_t):.1f}%)"
                                     for cbm_t, csr_t in zip(normalized_cbm_spmm_time_list, normalized_csr_spmm_time_list)]

    time_data = [un_normalized_csr_spmm_time_list, un_normalized_cbm_spmm_time_list, normalized_csr_spmm_time_list, normalized_cbm_spmm_time_list]
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