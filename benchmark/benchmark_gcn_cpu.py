from datetime import datetime
from os import makedirs
from os.path import exists
from time import time, sleep

import numpy as np
from prettytable import PrettyTable

from torch import device, float32, no_grad, rand, ones, int32, testing, set_num_threads
from torch.nn import Module
from torch_geometric.nn import GCNConv

from cbm.cbm4gcn import cbm4gcn
from gnns.graph_convolutional_network import (CBMSparseMatrixGCN,
                                              MKLCSRSparseMatrixGCN)
from benchmark.utility import bold, download_and_return_datasets_as_dict
from logger import setup_logger

num_iter = 250
number_of_layers = 2
hidden_channels = [500, ]

datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "ca-AstroPh"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("TUDataset", "COLLAB"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
]


alpha_per_dataset = {
    "ca-HepPh": 1,
    "ca-AstroPh": 8,
    "Cora": 32,
    "PubMed": 32,
    "COLLAB": 16,
    "coPapersDBLP": 16,
    "coPapersCiteseer": 32,
}


def create_layer(cls, in_channels, out_channels):
    if cls is GCNConv:
        return cls(in_channels, out_channels, normalize=True, add_self_loops=False, bias=False)
    if cls in (CBMSparseMatrixGCN, ):
        layer = cls(in_channels, out_channels)
        # store it in the layer to avoid creating it multiple times for each layer
        layer.cached_a = c
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
            x = self.first_layer(x=x, edge_index=edge_index)
            x = x.relu()
            for layer in self.layers:
                x = layer(x=x, edge_index=edge_index)
                x = x.relu()
            x = self.last_layer(x=x, edge_index=edge_index)
            return x

    return Model()


def time_func(gcn_cls, edge_index, x, hidden_size, num_layers=2, iters=num_iter, warmup=3):
    in_channels = x.size(1)
    gcn_model = create_model(gcn_cls, in_channels, hidden_size, hidden_size, num_layers)
    with no_grad():
        for _ in range(warmup):
            gcn_model(x=x, edge_index=edge_index)

        time_list = []
        for _ in range(iters):
            t_start = time()
            y = gcn_model(x=x, edge_index=edge_index)
            t_end = time()
            time_list += [t_end - t_start]
    return np.mean(time_list), y


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
        MKLCSRSparseMatrixGCN,
        CBMSparseMatrixGCN,
    )
    gcn_classes_names = {
        MKLCSRSparseMatrixGCN: "MKL CSR Sparse",
        CBMSparseMatrixGCN: "CBM CSR Sparse",
    }

    for name_i, data_i in data_dict.items():
        print(name_i + "...")
        edge_index = data_i.edge_index
        alpha = alpha_per_dataset[name_i]
        c = cbm4gcn(edge_index.to(int32), ones(edge_index.size(1), dtype=float32), alpha=alpha)
        x = rand(data_i.num_nodes, 500)
        in_size = x.size(1)
        timing_per_gcn_type = [[] for _ in range(len(gcn_classes))]
        for hidden_channel in hidden_channels:
            model_out = []
            for i, cls in enumerate(gcn_classes):
                sleep(0.5)
                time_per_gcn_i, y_per_gcn = time_func(cls, edge_index, x, hidden_channel, number_of_layers)
                timing_per_gcn_type[i] += [time_per_gcn_i]
                model_out.append(y_per_gcn) 

            
            # compare models produced by each approach
            for out_idx_1 in range(len(model_out)):
                for out_idx_2 in range(len(model_out)):
                    if out_idx_1 != out_idx_2:
                        testing.assert_close(model_out[out_idx_1], model_out[out_idx_2], atol=1e-2, rtol=1e-2)
            

        table = PrettyTable(align="l")
        table.title = f"{name_i.upper()} alpha: {alpha} size: {in_size} (avg row length: {data_i.edge_index.size(1) / data_i.num_nodes:.2f}, num_nodes: {data_i.num_nodes}, num_edges: {data_i.edge_index.size(1)})"
        table.field_names = [bold(f"Input Size: {in_size} / Embedding Sizes"), ] + [bold(f"{hidden_channel}") for hidden_channel in hidden_channels]
        for i, cls in enumerate(gcn_classes):
            table.add_row([bold(gcn_classes_names[cls])] + timing_per_gcn_type[i])
        logger.info(f"{bold(name_i.upper())}:")
        logger.info(table)
