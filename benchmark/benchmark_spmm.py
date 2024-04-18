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

    execution_mode = "seq" if args.sequential else "omp"
    print(f"Execution mode: {execution_mode}")
    dataset = PygNodePropPredDataset(name="ogbn-proteins")
    # edge_index, *_ = remove_isolated_nodes(to_undirected(dataset[0].edge_index))
    edge_index = dataset[0].edge_index if dataset[0].edge_index is not None else to_edge_index(dataset[0].adj_t)
    torch_sparse_coo = torch.sparse_coo_tensor(edge_index,
                                               torch.ones(edge_index.size(1)),
                                               (edge_index.size(1), edge_index.size(1)),
                                               )
    mat = scipy.sparse.csr_matrix((torch_sparse_coo.coalesce().values().numpy(),
                                   torch_sparse_coo.coalesce().indices().numpy()),
                                  shape=(edge_index.size(1), edge_index.size(1)))

    # Prepare matrix formats
    row = torch.from_numpy(mat.tocoo().row).to(torch.int32)
    col = torch.from_numpy(mat.tocoo().col).to(torch.int32)
    edge_index = torch.stack([row, col])
    values = torch.ones(row.size(0), dtype=torch.float32)

    # Initialize matrices
    csr_matrix = torch.sparse_coo_tensor(edge_index.to(torch.int32), values, mat.shape).to_sparse_csr()
    cbm_matrix = cbm_matrix(edge_index, values, rows=mat.shape[0], cols=mat.shape[1], alpha=2)
    embeddings = torch.randn((mat.shape[1], args.features), dtype=torch.float32)

    # Timing variables
    timings = {}

    start_time = time.time()
    with torch.no_grad():
        for _ in range(args.features):
            y1 = csr_matrix @ embeddings
    end_time = time.time()
    timings[f"{execution_mode}-torch-csr"] = end_time - start_time

    start_time = time.time()
    with torch.no_grad():
        update_func = getattr(cbm_matrix, f"{execution_mode}_torch_csr_matmul")
        for _ in range(args.features):
            y2 = update_func(embeddings)
    end_time = time.time()
    timings[f"{execution_mode}-torch-cbm"] = end_time - start_time

    start_time = time.time()
    with torch.no_grad():
        y3 = torch.empty((mat.shape[0], embeddings.shape[1]), dtype=torch.float32)
        mkl_csr_spmm = locals()[f"{execution_mode}_mkl_csr_spmm"]
        for _ in range(args.features):
            mkl_csr_spmm(csr_matrix, embeddings, y3)
    end_time = time.time()
    timings[f"{execution_mode}-mkl-csr"] = end_time - start_time

    start_time = time.time()
    with torch.no_grad():
        y4 = torch.empty((mat.shape[0], embeddings.shape[1]), dtype=torch.float32)
        update_func = getattr(cbm_matrix, f"{execution_mode}_mkl_csr_spmm_update")
        for _ in range(args.features):
            update_func(embeddings, y4)
    end_time = time.time()
    timings["seq-mkl-cbm"] = end_time - start_time

    torch.testing.assert_close(y1, y2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y1, y3, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y1, y4, atol=1e-4, rtol=1e-4)

    # Print all timings
    for operation, time_taken in timings.items():
        print(f"{operation}: {time_taken:.3f} seconds")
