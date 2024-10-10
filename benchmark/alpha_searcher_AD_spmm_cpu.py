import time
from datetime import datetime
from os import makedirs
from os.path import exists

import numpy as np
import torch
from prettytable import PrettyTable
from scipy.sparse import csr_matrix

from benchmark.utility import bold, download_and_return_datasets_as_dict, calculate_compression_ratio
from cbm.cbm4ad import cbm4ad
from cbm import cbm_mkl_cpp as mkl
from logger import setup_logger
from gnns.utility import normalize_edge_index
from cbm.utility import check_edge_index

iters = 250
sizes = [500]
alphas = [0, 1, 2, 4, 8, 16, 32]
datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "ca-AstroPh"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("TUDataset", "COLLAB"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
]


def mkl_matmul(a, x, y):
    row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
    row_ptr_e = a.crow_indices()[1:].to(torch.int32)
    col_ptr = a.col_indices().to(torch.int32)
    val_ptr = a.values().to(torch.float32)
    mkl.s_spmm_csr_int32(row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)

@torch.no_grad()
def correctness(edge_index):
    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(torch.long)
    col = torch.from_numpy(mat.tocoo().col).to(torch.long)
    edge_index = torch.stack([row, col]).to(torch.long)
    values = torch.ones(row.size(0), dtype=torch.float32)
    
    dim_size = rowptr.size(0) - 1

    # get out-degrees and scale columns
    out_degree = ((rowptr[1:] - rowptr[:-1])**(-1/2)).view(-1)
    scaled_values = values * out_degree[col]

    a_csr = torch.sparse_coo_tensor(edge_index, scaled_values, mat.shape).to_sparse_csr()
    
    for alpha in alphas:
        
        a_cbm = cbm4ad(edge_index.to(torch.int32), values, alpha=alpha)

        for size in sizes:
            x = torch.randn((mat.shape[0], size), dtype=torch.float32)
            x = x.squeeze(-1) if size == 1 else x
            a_csr_x = a_csr @ x
            
            a_cbm_x = torch.empty(dim_size, size, dtype=x.dtype)
            a_cbm.matmul(x, a_cbm_x)

            torch.testing.assert_close(a_csr_x, a_cbm_x, atol=1e-4, rtol=1e-4)

def timing(edge_index, name):
    mat = csr_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
    rowptr = torch.from_numpy(mat.indptr).to(torch.long)
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    dim_size = rowptr.size(0) - 1
    avg_row_len = edge_index.size(1) / dim_size

    # get out-degrees and scale columns
    out_degree = rowptr[1:] - rowptr[:-1]
    scaled_values = values * out_degree[edge_index[1]]

    a = torch.sparse_coo_tensor(edge_index, scaled_values, mat.shape).to_sparse_csr()

    for size in sizes:
        x = torch.randn((mat.shape[0], size))
        x = x.squeeze(-1) if size == 1 else x

        compression_ratio_list = []
        csr_time_list, cbm_time_list = [], []
        #csr_time_std_list, cbm_time_std_list = [], []

        # Measure CSR with MKL
        y2 = torch.empty(dim_size, size, dtype=x.dtype)
        mkl_matmul(a, x, y2)
        
        start_time = time.time()
        for _ in range(iters):
            mkl_matmul(a, x, y2)
        stop_time = time.time()
        csr_time = (stop_time - start_time) / iters
        csr_time_list += [csr_time] * len(alphas)

        # Measure CBM with different alphas values
        for alpha_i in alphas:
            c = cbm4ad(edge_index.to(torch.int32), values, alpha=alpha_i)
            compression_ratio = calculate_compression_ratio(edge_index, c)
            compression_ratio_list += [compression_ratio]

            y3 = torch.empty(dim_size, size, dtype=x.dtype)
            c.matmul(x, y3)
            
            start_time = time.time()
            for _ in range(iters):
                c.matmul(x, y3)
            stop_time = time.time()
            cbm_time = (stop_time - start_time) / iters
            cbm_time_list += [cbm_time]

        # Construct table to display results
        table = PrettyTable()
        table.title = f"{name} / size: {size} (avg row length: {avg_row_len:.2f}, num_nodes: {a.size(0)}, num_edges: {edge_index.size(1)})"
        header = [""] + [f"{alpha:>5}" for alpha in alphas]
        table.field_names = header
        methods = ["csr", "cbm"]
        table.add_row([bold("Compression Ratio")] + [bold(f"{c:.2f}") for c in compression_ratio_list])

        compute_speed_up_factor = lambda reference, value: reference / value
        cbm_time_list = [f"{cbm_t} ({compute_speed_up_factor(csr_t, cbm_t):.2f}x)"
                         for cbm_t, csr_t in zip(cbm_time_list, csr_time_list)]
        time_data = [csr_time_list, cbm_time_list]
        for method, times in zip(methods, time_data):
            row = [method, ] + [f"{t}" for t in times]
            table.add_row(row)
        logger.info(table)


if __name__ == "__main__":
    log_path = f"./logs/"
    if not exists(log_path):
        makedirs(log_path)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    logger = setup_logger(filename=f"{log_path}/spmm_cpu-{current_time}.log", verbose=True)
    name_edge_index_dict = download_and_return_datasets_as_dict(datasets)
    for name_i, data_i in name_edge_index_dict.items():
        correctness(data_i.edge_index)
        timing(data_i.edge_index, name_i)