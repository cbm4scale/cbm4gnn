from os import makedirs
from os.path import exists

import torch
from prettytable import PrettyTable
from metrics import cbm_metrics_cpp
from benchmark.utility import download_and_return_datasets_as_dict

datasets = [
    ("SNAP", "ca-HepPh"),
    ("SNAP", "ca-AstroPh"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("TUDataset", "COLLAB"),
    #("DIMACS10", "coPapersDBLP"),
    #("DIMACS10", "coPapersCiteseer"),
]

compression_ratio = {
    "ca-HepPh" : "2.72319",
    "ca-AstroPh": "1.72238",
    "Cora" : "1.04414",
    "PubMed" : "1.0438",
    "COLLAB" : "10.99672",
    "coPapersDBLP" : "5.96545",
    "coPapersCiteseer" : "9.87388",    
}

def run_metrics(edge_index):

    num_nodes = edge_index.max() + 1


    r1 = cbm_metrics_cpp.avg_clustering_coefficient(edge_index[0].to(torch.int32), 
                                                    edge_index[1].to(torch.int32),
                                                    num_nodes)
    
    r2 = cbm_metrics_cpp.avg_jaccard_similarity(edge_index[0].to(torch.int32), 
                                        edge_index[1].to(torch.int32),
                                        num_nodes)
    
    
    r3 = cbm_metrics_cpp.custom_jaccard_similarity(edge_index[0].to(torch.int32), 
                                           edge_index[1].to(torch.int32),
                                           num_nodes)
    return r1,r2,r3
    

if __name__ == "__main__":
    # Initialize the table
    table = PrettyTable()

    # Define the columns
    table.field_names = ["Dataset",
                         "Compression Ratio", 
                         "Avg. Clustering Coefficient",
                         "Avg. Jaccard Similarity",
                         "Custom Jaccard Similarity"]

    name_edge_index_dict = download_and_return_datasets_as_dict(datasets)
    for name_i, data_i in name_edge_index_dict.items():
        print("processing " + name_i + "...")
        r = run_metrics(data_i.edge_index)
        table.add_row([name_i, compression_ratio[name_i], f"{r[0]:.4f}", f"{r[1]:.4f}", f"{r[2]:.4f}"])

    print(table)