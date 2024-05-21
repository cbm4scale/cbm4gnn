import time
import torch
from prettytable import PrettyTable
import numpy as np

from benchmark.utility import download_and_return_datasets_as_dict
from cbm.cbm import cbm_matrix

iters = 5
alphas = [0, 1, 2, 4, 8, 16, 32, 64]
datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "ca-AstroPh"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
]


def benchmark_all_datasets(name_edge_index_dict):
    table = PrettyTable()
    table.title = "CBM Matrix Creation Benchmark Across Datasets"
    headers = ["Dataset/Alpha"] + [f"Alpha {alpha}" for alpha in alphas]
    table.field_names = headers

    for name, data in name_edge_index_dict.items():
        row = [name]
        edge_index = data.edge_index
        values = torch.ones(edge_index.size(1), dtype=torch.float32)
        for alpha_i in alphas:
            t_list = []
            for _ in range(iters):
                start_time = time.perf_counter()
                c = cbm_matrix(edge_index.to(torch.int32), values, alpha=alpha_i)
                t_list.append(time.perf_counter() - start_time)
            cbm_creation_time = np.median(t_list)
            cbm_creation_time_std = np.std(t_list)
            row.append(f"{cbm_creation_time:.5f} ± {cbm_creation_time_std:.3f}")
        table.add_row(row)

    print(table)


if __name__ == "__main__":
    name_edge_index_dict = download_and_return_datasets_as_dict(datasets)
    benchmark_all_datasets(name_edge_index_dict)
