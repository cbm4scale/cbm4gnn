from typing import Tuple
from os.path import exists
from time import perf_counter

from wget import download
from scipy.io import loadmat
from prettytable import PrettyTable

from torch import (arange, bool, cuda, device, Tensor, from_numpy, int64, stack, ones, testing, no_grad, tensor,
                   zeros_like, float32, rand)

from benchmark_message_passing.message_passing import (NativePytorchIndexSelectScatterAddMessagePassing,
                                                       NativePytorchGatherScatterAddMessagePassing,
                                                       NativePytorchCOOSparseMatrixMessagePassing,
                                                       NativePytorchCSRSparseMatrixMessagePassing,
                                                       TorchScatterGatherCOOScatterAddMessagePassing,
                                                       TorchScatterGatherCSRScatterAddMessagePassing,
                                                       MKLCSRSparseMatrixMessagePassing,
                                                       )


def underline(text, flag=True):
    return f'\033[4m{text}\033[0m' if flag else text


def bold(text, flag=True):
    return f'\033[1m{text}\033[0m' if flag else text


def download_from_tamu_sparse_matrix(dataset: Tuple[str, str]) -> None:
    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'
    group, name = dataset
    if not exists(f'{name}.mat'):
        print(f'Downloading {group}/{name}:')
        download(url.format(group, name))
        print('')


def load_matrix(name: str) -> Tensor:
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    row = from_numpy(mat.tocoo().row).to(device, int64)
    col = from_numpy(mat.tocoo().col).to(device, int64)
    edge_index = stack([row, col], dim=0)
    return edge_index


def time_func(message_passing_cls, edge_index, num_nodes, size, iters=5, warmup=3):
    if cuda.is_available():
        cuda.synchronize()

    message_passing = message_passing_cls()
    with no_grad():
        for _ in range(warmup):
            x = rand((num_nodes, size), device=device)
            message_passing(x=x, edge_index=edge_index)
        t_total = 0
        for _ in range(iters):
            x = rand((num_nodes, size), device=device)
            t_start = perf_counter()
            message_passing(x=x, edge_index=edge_index)
            t_total += (perf_counter() - t_start)
    if cuda.is_available():
        cuda.synchronize()
    return t_total / iters


if __name__ == '__main__':
    # MKL doesn't support cuda so don't run this on cuda to avoid errors and have a fair comparison
    device = device("cpu")

    datasets = [
        ("DIMACS10", "citationCiteseer"),
        ("SNAP", "web-Stanford"),
        ("Janna", "StocF-1465"),
        ("GHS_psdef", "ldoor"),
    ]
    sizes = [2, 10, 50, 100, 500, 1000, 1500]

    # Prepare data in edge_index format (COO format)
    [download_from_tamu_sparse_matrix(dataset) for dataset in datasets]
    edge_index_dict = {name: load_matrix(name) for _, name in datasets}

    message_passing_classes = [NativePytorchIndexSelectScatterAddMessagePassing,
                               NativePytorchGatherScatterAddMessagePassing,
                               NativePytorchCOOSparseMatrixMessagePassing,
                               NativePytorchCSRSparseMatrixMessagePassing,
                               TorchScatterGatherCOOScatterAddMessagePassing,
                               TorchScatterGatherCSRScatterAddMessagePassing,
                               MKLCSRSparseMatrixMessagePassing,
                               ]
    message_passing_classes_names = {NativePytorchIndexSelectScatterAddMessagePassing: "Native Pytorch Index-Select and Scatter Add",
                                     NativePytorchGatherScatterAddMessagePassing: "Native Pytorch Gather and Scatter Add",
                                     NativePytorchCOOSparseMatrixMessagePassing: "Native Pytorch COO Sparse Matrix",
                                     NativePytorchCSRSparseMatrixMessagePassing: "Native Pytorch CSR Sparse Matrix",
                                     TorchScatterGatherCOOScatterAddMessagePassing: "Torch_Scatter COO Scatter Add",
                                     TorchScatterGatherCSRScatterAddMessagePassing: "Torch_Scatter CSR Scatter Add",
                                     MKLCSRSparseMatrixMessagePassing: "MKL CSR Sparse Matrix",
                                     }

    # Correctness check
    edge_index = [*edge_index_dict.values()][0]
    num_nodes = edge_index.max() + 1
    edge_index = edge_index.to(device)
    output_per_message_passing = {}
    for cls in message_passing_classes:
        message_passing = cls()
        x = ones((num_nodes, 1), device=device, dtype=float32)
        output_per_message_passing[cls.__name__] = message_passing.forward(edge_index, x=x)

    outputs_0 = output_per_message_passing[NativePytorchIndexSelectScatterAddMessagePassing.__name__]
    for cls in message_passing_classes[1:]:
        outputs_i = output_per_message_passing[cls.__name__]
        msg = f"Failed on {cls.__name__},\n{outputs_0} != \n{outputs_i}"
        testing.assert_close(outputs_0, outputs_i, rtol=1e-5, atol=1e-5, msg=msg)
    print("All tests passed! All message passing classes have the same output.")

    # Timing
    for name, edge_index in edge_index_dict.items():
        num_nodes = edge_index.max() + 1
        ts = [[] for _ in range(len(message_passing_classes))]
        for size in sizes:
            for i, cls in enumerate(message_passing_classes):
                ts[i] += [time_func(cls, edge_index, num_nodes, size), ]

        ts = tensor(ts)
        winner = zeros_like(ts, dtype=bool)
        winner[ts.argmin(dim=0), arange(len(sizes))] = 1
        winner = winner.tolist()

        table = PrettyTable(align="l")
        table.field_names = [bold("SIZES"), ] + [bold(f"{size}") for size in sizes]
        for i, cls in enumerate(message_passing_classes):
            table.add_row([bold(message_passing_classes_names[cls])] + [underline(f'{t:.5f}', f) for t, f in zip(ts[i], winner[i])])
        print(f'{bold(name.upper())}:')
        print(table)