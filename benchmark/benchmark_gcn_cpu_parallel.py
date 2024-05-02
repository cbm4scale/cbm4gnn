from time import perf_counter

from prettytable import PrettyTable

from torch import arange, bool, device, no_grad, tensor, zeros_like, rand
from torch_geometric.datasets import SuiteSparseMatrixCollection

from gnns.graph_convolutional_network import (NativePytorchScatterAddGCN,
                                              NativePytorchCOOSparseMatrixGCN,
                                              NativePytorchCSRSparseMatrixGCN,
                                              TorchScatterCOOScatterAddGCN,
                                              TorchScatterGatherCOOSegmentCOOGCN,
                                              TorchScatterGatherCSRSegmentCSRGCN,
                                              TorchSparseCSRSparseMatrixGCN,
                                              MKLParallelCSRSparseMatrixGCN,
                                              CBMParallelMKLCSRSparseMatrixGCN,
                                              CBMParallelTorchCSRSparseMatrixGCN,
                                              )


def underline(text, flag=True):
    return f"\033[4m{text}\033[0m" if flag else text


def bold(text, flag=True):
    return f"\033[1m{text}\033[0m" if flag else text


def time_func(gcn_cls, edge_index, num_nodes, size, iters=5, warmup=3):
    gcn_model = gcn_cls(size, 1)
    with no_grad():
        for _ in range(warmup):
            x = rand((num_nodes, size), device=device)
            gcn_model(x=x, edge_index=edge_index)
        t_total = 0
        for _ in range(iters):
            x = rand((num_nodes, size), device=device)
            t_start = perf_counter()
            gcn_model(x=x, edge_index=edge_index)
            t_total += (perf_counter() - t_start)
    return t_total / iters


if __name__ == "__main__":
    # MKL doesn't support cuda so don't run this on cuda to avoid errors and have a fair comparison
    device = device("cpu")

    datasets = [
        ("SNAP", "ca-AstroPh"),
    ]
    sizes = [2, 10, 50, 100, 500, 1000, 1500, 2000]

    [SuiteSparseMatrixCollection(root="../data", name=name, group=group) for group, name in datasets]
    edge_index_dict = {name: SuiteSparseMatrixCollection(root="../data", name=name, group=group).data.edge_index
                       for group, name in datasets}

    gcn_classes = (NativePytorchScatterAddGCN,
                   NativePytorchCOOSparseMatrixGCN,
                   NativePytorchCSRSparseMatrixGCN,
                   TorchScatterCOOScatterAddGCN,
                   TorchScatterGatherCOOSegmentCOOGCN,
                   TorchScatterGatherCSRSegmentCSRGCN,
                   TorchSparseCSRSparseMatrixGCN,
                   MKLParallelCSRSparseMatrixGCN,
                   CBMParallelMKLCSRSparseMatrixGCN,
                   CBMParallelTorchCSRSparseMatrixGCN,
                   )
    gcn_classes_names = {NativePytorchScatterAddGCN: "Native Pytorch Scatter Add",
                         NativePytorchCOOSparseMatrixGCN: "Native Pytorch COO Sparse",
                         NativePytorchCSRSparseMatrixGCN: "Native Pytorch CSR Sparse",
                         TorchScatterCOOScatterAddGCN: "Torch Scatter COO Scatter Add",
                         TorchScatterGatherCOOSegmentCOOGCN: "Torch Scatter Gather COO Segment COO",
                         TorchScatterGatherCSRSegmentCSRGCN: "Torch Scatter Gather CSR Segment CSR",
                         TorchSparseCSRSparseMatrixGCN: "Torch Sparse CSR Sparse",
                         MKLParallelCSRSparseMatrixGCN: "MKL CSR Sparse",
                         CBMParallelMKLCSRSparseMatrixGCN: "CBM MKL CSR Sparse",
                         CBMParallelTorchCSRSparseMatrixGCN: "CBM Native Torch CSR Sparse",
                         }

    # Correctness check
    edge_index = [*edge_index_dict.values()][0]
    num_nodes = edge_index.max() + 1
    edge_index = edge_index.to(device)

    # Timing
    for name, edge_index in edge_index_dict.items():
        num_nodes = edge_index.max() + 1
        ts = [[] for _ in range(len(gcn_classes))]
        for size in sizes:
            for i, cls in enumerate(gcn_classes):
                ts[i] += [time_func(cls, edge_index, num_nodes, size), ]

        ts = tensor(ts)
        winner = zeros_like(ts, dtype=bool)
        winner[ts.argmin(dim=0), arange(len(sizes))] = 1
        winner = winner.tolist()

        table = PrettyTable(align="l")
        table.field_names = [bold("SIZES"), ] + [bold(f"{size}") for size in sizes]
        for i, cls in enumerate(gcn_classes):
            table.add_row([bold(gcn_classes_names[cls])] +
                          [underline(f"{t:.5f}", f) for t, f in zip(ts[i], winner[i])])
        print(f"{bold(name.upper())}:")
        print(table)
