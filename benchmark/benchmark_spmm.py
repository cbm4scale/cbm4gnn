import time
import argparse
from os.path import exists

import scipy
import wget
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.io import loadmat
import torch

import cbm.cbm_mkl_cpp as cbm_
from torch_geometric.datasets import SNAPDataset
from torch_geometric.utils import to_edge_index

from cbm.cbm import cbm_matrix


def seq_mkl_csr_spmm(a, x, y):
        row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = a.crow_indices()[1: ].to(torch.int32)
        col_ptr = a.col_indices().to(torch.int32)
        val_ptr = a.values().to(torch.float32)

        cbm_.seq_s_spmm_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)

def omp_mkl_csr_spmm(a, x, y):
        row_ptr_s = a.crow_indices()[:-1].to(torch.int32)
        row_ptr_e = a.crow_indices()[1: ].to(torch.int32)
        col_ptr = a.col_indices().to(torch.int32)
        val_ptr = a.values().to(torch.float32)

        cbm_.omp_s_spmm_csr_int32(
            row_ptr_s, row_ptr_e, col_ptr, val_ptr, x, y)


def download_texas_A_and_M_university(dataset):
    url = "https://sparse.tamu.edu/mat/{}/{}.mat"
    group, name = dataset
    if not exists(f"{name}.mat"):
        print(f"Downloading {group}/{name}:")
        wget.download(url.format(group, name))
        print("")

def read_texas_A_and_M_university(dataset):
    group, name = dataset
    return loadmat(f"{name}.mat")["Problem"][0][0][2].tocsr()


if __name__ == "__main__":
    # Setting up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--features", type=int, default=64)
    args = parser.parse_args()

    # Set number of threads if sequential execution is specified
    if args.sequential:
        torch.set_num_threads(1)

    n_iterations = args.iterations
    n_features = args.features

    execution_mode = "seq" if args.sequential else "omp"
    print(f"Execution mode: {execution_mode}")

    # dataset = Planetoid(root="./", name="Cora")
    dataset = SNAPDataset(root="./", name='soc-ca-grqc')
    #dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    # edge_index, *_ = remove_isolated_nodes(to_undirected(dataset[0].edge_index))
    edge_index = dataset[0].edge_index if dataset[0].edge_index is not None else to_edge_index(dataset[0].adj_t)

    n_rows = edge_index[0].max() + 1
    n_cols = edge_index[1].max() + 1
    values = torch.ones(edge_index.size(1), dtype=torch.float32)

    torch_sparse_coo = torch.sparse_coo_tensor(edge_index, 
                                               values, 
                                               (n_rows, n_cols), 
                                               dtype=torch.float32)
    
    # Initialize matrices
    a_matrix = torch_sparse_coo.coalesce().to_sparse_csr()
    c_matrix = cbm_matrix(edge_index.to(torch.int32), values, alpha=2)
    embeddings = torch.randn((n_cols, n_features), dtype=torch.float32)

    # Timing variables
    timings = {}

    start_time = time.time()
    with torch.no_grad():
    
        for _ in range(n_iterations):
            y1 = a_matrix @ embeddings
    
    end_time = time.time()
    timings[f"{execution_mode}-torch-csr"] = end_time - start_time

    start_time = time.time()
    with torch.no_grad():
        update_func = getattr(c_matrix, f"{execution_mode}_torch_csr_matmul")
    
        for _ in range(n_iterations):
            y2 = update_func(embeddings)
    
    end_time = time.time()
    timings[f"{execution_mode}-torch-cbm"] = end_time - start_time

    start_time = time.time()
    with torch.no_grad():
        y3 = torch.empty((n_rows, embeddings.shape[1]), dtype=torch.float32)
        mkl_csr_spmm = locals()[f"{execution_mode}_mkl_csr_spmm"]
        
        for _ in range(n_iterations):
            mkl_csr_spmm(a_matrix, embeddings, y3)
    
    end_time = time.time()
    timings[f"{execution_mode}-mkl-csr"] = end_time - start_time

    start_time = time.time()
    with torch.no_grad():
        y4 = torch.empty((n_rows, embeddings.shape[1]), dtype=torch.float32)
        update_func = getattr(c_matrix, f"{execution_mode}_mkl_csr_spmm_update")
        
        for _ in range(n_iterations):
            update_func(embeddings, y4)

    end_time = time.time()
    timings[f"{execution_mode}-mkl-cbm"] = end_time - start_time

    # Print all timings
    for operation, time_taken in timings.items():
        print(f"{operation}: {time_taken:.3f} seconds")

    torch.testing.assert_close(y1, y3, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y1, y4, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)

