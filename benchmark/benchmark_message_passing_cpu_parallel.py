from time import perf_counter

from prettytable import PrettyTable

from torch import (arange, bool, cuda, device, ones, testing, no_grad, tensor, zeros_like, float32, rand)
from torch_geometric.datasets import SuiteSparseMatrixCollection

from gnns.message_passing import (NativePytorchScatterAddMessagePassing,
                                  NativePytorchCOOSparseMatrixMessagePassing,
                                  NativePytorchCSRSparseMatrixMessagePassing,
                                  TorchScatterCOOScatterAddMessagePassing,
                                  TorchScatterGatherCOOSegmentCOO,
                                  TorchScatterGatherCSRSegmentCSR,
                                  TorchSparseCSRSparseMatrixMessagePassing,
                                  MKLParallelCSRSparseMatrixMessagePassing,
                                  CBMParallelMKLCSRSparseMatrixMessagePassing,
                                  CBMParallelTorchCSRSparseMatrixMessagePassing,
                                  )


def underline(text, flag=True):
    return f"\033[4m{text}\033[0m" if flag else text


def bold(text, flag=True):
    return f"\033[1m{text}\033[0m" if flag else text


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

    message_passing_classes = (NativePytorchScatterAddMessagePassing,
                               NativePytorchCOOSparseMatrixMessagePassing,
                               NativePytorchCSRSparseMatrixMessagePassing,
                               TorchScatterCOOScatterAddMessagePassing,
                               TorchScatterGatherCOOSegmentCOO,
                               TorchScatterGatherCSRSegmentCSR,
                               TorchSparseCSRSparseMatrixMessagePassing,
                               MKLParallelCSRSparseMatrixMessagePassing,
                               CBMParallelMKLCSRSparseMatrixMessagePassing,
                               CBMParallelTorchCSRSparseMatrixMessagePassing,
                               )
    message_passing_classes_names = {NativePytorchScatterAddMessagePassing: "Native Pytorch Scatter Add",
                                     NativePytorchCOOSparseMatrixMessagePassing: "Native Pytorch COO Sparse Matrix",
                                     NativePytorchCSRSparseMatrixMessagePassing: "Native Pytorch CSR Sparse Matrix",
                                     TorchScatterCOOScatterAddMessagePassing: "Torch Scatter COO Scatter Add",
                                     TorchScatterGatherCOOSegmentCOO: "Torch Scatter Gather COO Segment COO",
                                     TorchScatterGatherCSRSegmentCSR: "Torch Scatter Gather CSR Segment CSR",
                                     TorchSparseCSRSparseMatrixMessagePassing: "Torch Sparse CSR Sparse Matrix",
                                     MKLParallelCSRSparseMatrixMessagePassing: "MKL CSR Sparse Matrix",
                                     CBMParallelMKLCSRSparseMatrixMessagePassing: "CBM MKL CSR Sparse Matrix",
                                     CBMParallelTorchCSRSparseMatrixMessagePassing: "CBM Native Torch CSR Sparse Matrix",
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

    outputs_0 = output_per_message_passing[NativePytorchScatterAddMessagePassing.__name__]
    for cls in message_passing_classes:
        outputs_i = output_per_message_passing[cls.__name__]
        msg = f"Failed on {cls.__name__},\n{outputs_0} != \n{outputs_i}"
        testing.assert_close(outputs_0, outputs_i, rtol=1e-5, atol=1e-5, msg=msg)
    print("Message passing classes have the same output.")

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
            table.add_row([bold(message_passing_classes_names[cls])] +
                          [underline(f"{t:.5f}", f) for t, f in zip(ts[i], winner[i])])
        print(f"{bold(name.upper())}:")
        print(table)
