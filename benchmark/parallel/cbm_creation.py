import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["GOMP_CPU_AFFINITY"] = "0-15"

import time
import torch
from prettytable import PrettyTable
import numpy as np

from benchmark.utility import download_and_return_datasets_as_dict, calculate_compression_ratio
from cbm.cbm import cbm_matrix

iters = 5
alphas = [0, 1, 2, 4, 8, 16, 32, 64]
datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "ca-AstroPh"),
    # ("Planetoid", "Cora"),
    # ("Planetoid", "PubMed"),
    # ("DIMACS10", "coPapersDBLP"),
    # ("DIMACS10", "coPapersCiteseer"),
]


def benchmark_datasets(name_edge_index_dict):
    headers = ["Dataset/Alpha"] + [f"Alpha {alpha}" for alpha in alphas]

    runtime_table = PrettyTable()
    runtime_table.title = "CBM Matrix Creation Time Benchmark Across Datasets"
    runtime_table.field_names = headers

    compression_table = PrettyTable()
    compression_table.title = "CBM Matrix Compression Ratio Benchmark Across Datasets"
    compression_table.field_names = headers

    for name, data in name_edge_index_dict.items():
        runtime_row = [name]
        compression_row = [name]

        edge_index = data.edge_index
        values = torch.ones(edge_index.size(1), dtype=torch.float32)
        for alpha_i in alphas:
            t_list = []
            for _ in range(iters):
                start_time = time.perf_counter()
                c = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha_i)
                t_list.append(time.perf_counter() - start_time)

            cbm_creation_time = np.mean(t_list)
            cbm_creation_time_std = np.std(t_list)
            compression_ratio = calculate_compression_ratio(edge_index, c)

            runtime_row.append(f"{cbm_creation_time:.5f} ± {cbm_creation_time_std:.3f}")
            compression_row.append(f"{compression_ratio:.5f}")

        runtime_table.add_row(runtime_row)
        compression_table.add_row(compression_row)

    print(runtime_table)
    print(compression_table)


if __name__ == "__main__":
    name_edge_index_dict = download_and_return_datasets_as_dict(datasets)
    benchmark_datasets(name_edge_index_dict)
