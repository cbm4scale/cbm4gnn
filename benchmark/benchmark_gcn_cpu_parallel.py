from datetime import datetime
from os import makedirs
from os.path import exists
from time import perf_counter, sleep

from prettytable import PrettyTable

from torch import arange, bool, device, float32, no_grad, tensor, zeros_like, rand, stack, ones, int32
from torch.nn import Module
from torch_geometric.nn import GCNConv

from cbm.cbm import cbm_matrix
from gnns.utility import add_self_loops
from gnns.graph_convolutional_network import (NativePytorchScatterAddGCN,
                                              NativePytorchCOOSparseMatrixGCN,
                                              NativePytorchCSRSparseMatrixGCN,
                                              MKLParallelCSRSparseMatrixGCN,
                                              CBMParallelMKLCSRSparseMatrixGCN,
                                              CBMParallelTorchCSRSparseMatrixGCN)
from benchmark.utility import underline, bold, download_and_return_datasets_as_dict
from logger import setup_logger

number_of_layers = 2
hidden_channels = [128, 256, 512, 1024, 2048]
datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "ca-HepTh"),
    ("SNAP", "cit-HepPh"),
    ("SNAP", "cit-HepTh"),
    ("SNAP", "ca-AstroPh"),
    ("SNAP", "web-Stanford"),
    ("SNAP", "web-NotreDame"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
]


alpha_per_dataset = {
    "ca-HepPh": 64,
    "ca-HepTh": 64,
    "cit-HepPh": 8,
    "cit-HepTh": 8,
    "ca-AstroPh": 8,
    "web-Stanford": 16,
    "web-NotreDame": 32,
    "Cora": 16,
    "PubMed": 16,
    "coPapersDBLP": 16,
    "coPapersCiteseer": 16,
}


def create_layer(cls, in_channels, out_channels):
    if cls is GCNConv:
        return cls(in_channels, out_channels, normalize=True, add_self_loops=False, bias=False)
    if cls in (CBMParallelMKLCSRSparseMatrixGCN, CBMParallelTorchCSRSparseMatrixGCN):
        layer = cls(in_channels, out_channels)
        # store it in the layer to avoid creating it multiple times for each layer
        layer.cached_a_t = c
        return layer
    return cls(in_channels, out_channels)


def create_model(cls, in_channels, hidden_channels, out_channels, num_layers):
    assert num_layers >= 2, "Number of layers must be at least 2"

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.first_layer = create_layer(cls, in_channels, hidden_channels)
            self.layers = [create_layer(cls, hidden_channels, hidden_channels) for _ in range(num_layers - 2)]
            self.last_layer = create_layer(cls, hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.first_layer(x, edge_index)
            x = x.relu()
            for layer in self.layers:
                x = layer(x, edge_index)
                x = x.relu()
            x = self.last_layer(x, edge_index)
            return x

    return Model()


def time_func(gcn_cls, edge_index, x, hidden_size, num_layers=2, iters=10, warmup=3):
    in_channels = x.size(1)
    gcn_model = create_model(gcn_cls, in_channels, hidden_size, hidden_size, num_layers)
    with no_grad():
        for _ in range(warmup):
            gcn_model(x=x, edge_index=edge_index)
        t_total = 0
        for _ in range(iters):
            t_start = perf_counter()
            gcn_model(x=x, edge_index=edge_index)
            t_total += (perf_counter() - t_start)
    return t_total / iters


if __name__ == "__main__":
    # MKL doesn't support cuda so don't run this on cuda to avoid errors and have a fair comparison
    device = device("cpu")

    log_path = f"./logs/"
    if not exists(log_path):
        makedirs(log_path)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    logger = setup_logger(filename=f"{log_path}/gcn_cpu_parallel-{current_time}.log", verbose=True)

    data_dict = download_and_return_datasets_as_dict(datasets)
    gcn_classes = (
        NativePytorchScatterAddGCN,
        NativePytorchCOOSparseMatrixGCN,
        NativePytorchCSRSparseMatrixGCN,
        GCNConv,
        MKLParallelCSRSparseMatrixGCN,
        CBMParallelMKLCSRSparseMatrixGCN,
        CBMParallelTorchCSRSparseMatrixGCN,
    )
    gcn_classes_names = {
        NativePytorchScatterAddGCN: "Native Pytorch Scatter Add",
        NativePytorchCOOSparseMatrixGCN: "Native Pytorch COO Sparse",
        NativePytorchCSRSparseMatrixGCN: "Native Pytorch CSR Sparse",
        GCNConv: "Torch Geometric GCN Conv",
        MKLParallelCSRSparseMatrixGCN: "MKL CSR Sparse",
        CBMParallelMKLCSRSparseMatrixGCN: "CBM MKL CSR Sparse",
        CBMParallelTorchCSRSparseMatrixGCN: "CBM Native Torch CSR Sparse",
    }

    for name_i, data_i in data_dict.items():
        edge_index = add_self_loops(data_i.edge_index)
        edge_index_t = stack([edge_index[1], edge_index[0]], dim=0).to(int32)
        alpha = alpha_per_dataset[name_i]
        c = cbm_matrix(edge_index_t, ones(edge_index.size(1), dtype=float32), normalized=True, alpha=alpha)
        x = rand(data_i.num_nodes, 1)
        in_size = x.size(1)
        ts = [[] for _ in range(len(gcn_classes))]
        for hidden_channel in hidden_channels:
            for i, cls in enumerate(gcn_classes):
                sleep(0.5)
                ts[i] += [time_func(cls, edge_index, x, hidden_channel, number_of_layers), ]

        ts = tensor(ts)
        winner = zeros_like(ts, dtype=bool)
        winner[ts.argmin(dim=0), arange(len(hidden_channels))] = 1
        winner = winner.tolist()

        table = PrettyTable(align="l")
        table.title = f"{name_i.upper()} alpha: {alpha} size: {in_size} (avg row length: {data_i.edge_index.size(1) / data_i.num_nodes:.2f}, num_nodes: {data_i.num_nodes}, num_edges: {data_i.edge_index.size(1)})"
        table.field_names = [bold(f"Input Size: {in_size} / Embedding Sizes"), ] + [bold(f"{hidden_channel}") for
                                                                                    hidden_channel in hidden_channels]
        for i, cls in enumerate(gcn_classes):
            table.add_row([bold(gcn_classes_names[cls])] +
                          [underline(f"{t:.5f}", f) for t, f in zip(ts[i], winner[i])])
        logger.info(f"{bold(name_i.upper())}:")
        logger.info(table)
