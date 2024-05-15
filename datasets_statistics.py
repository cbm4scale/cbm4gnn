from datetime import datetime
from os import makedirs
from os.path import exists

from prettytable import PrettyTable
from scipy.sparse import csr_matrix
from torch import ones, from_numpy, long, sparse_coo_tensor, float32

from gnns.utility import add_self_loops
from benchmark.utility import download_and_return_datasets_as_dict
from logger import setup_logger

datasets = [
    ("SNAP", "ca-HepPh"),
    # ("SNAP", "ca-HepTh"),
    # ("SNAP", "cit-HepPh"),
    ("SNAP", "cit-HepTh"),
    ("SNAP", "ca-AstroPh"),
    ("SNAP", "web-Stanford"),
    ("SNAP", "web-NotreDame"),
    ("Planetoid", "Cora"),
    ("Planetoid", "PubMed"),
    ("DIMACS10", "coPapersDBLP"),
    ("DIMACS10", "coPapersCiteseer"),
    # ("TKK", "s4dkt3m2"),
    # ("TKK", "g3rmt3m3"),
    # ("FIDAP", "ex26"),
]

dataset_types = {
    "ca-HepPh": "collaboration network",
    # "ca-HepTh": "collaboration network",
    # "cit-HepPh": "collaboration network",
    "cit-HepTh": "collaboration network",
    "ca-AstroPh": "collaboration network",
    "web-Stanford": "social network",
    "web-NotreDame": "web network",
    "Cora": "citation network",
    "PubMed": "citation network",
    "coPapersDBLP": "citation network",
    "coPapersCiteseer": "citation network",
    # "s4dkt3m2": "",
    # "g3rmt3m3": "",
    # "ex26": "",
}

# ca-HepPh: Ryan A. Rossi and Nesreen K. Ahmed. The Network Data Repository with Interactive Graph Analytics and Visualization. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
# ca-HepTh: Ryan A. Rossi and Nesreen K. Ahmed. The Network Data Repository with Interactive Graph Analytics and Visualization. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
# cit-HepPh: Ryan A. Rossi and Nesreen K. Ahmed. The Network Data Repository with Interactive Graph Analytics and Visualization. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.
# cit-HepTh: J. Leskovec, J. Kleinberg and C. Faloutsos. Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2005.
# ca-AstroPh: J. Leskovec, J. Kleinberg and C. Faloutsos. Graph Evolution: Densification and Shrinking Diameters. ACM Transactions on Knowledge Discovery from Data (ACM TKDD), 1(1), 2007.
# web-Stanford: J. Leskovec, K. Lang, A. Dasgupta, M. Mahoney. Community Structure in Large Networks: Natural Cluster Sizes and the Absence of Large Well-Defined Clusters. Internet Mathematics 6(1) 29--123, 2009.
# web-NotreDame: R. Albert, H. Jeong, A.-L. Barabasi. Diameter of the World-Wide Web. Nature, 1999.
# Cora: Andrew McCallum et al. in Automating the Construction of Internet Portals with Machine Learning
# PubMed: Prithviraj Sen et al. in Collective Classification in Network Data
# coPapersDBLP: R. Geisberger, P. Sanders, and D. Schultes. Better approximation of betweenness centrality. In 10th Workshop on Algorithm Engineering and Experimentation, 2008. SIAM.
# coPapersCiteseer: R. Geisberger, P. Sanders, and D. Schultes. Better approximation of betweenness centrality. In 10th Workshop on Algorithm Engineering and Experimentation, 2008. SIAM.

if __name__ == "__main__":
    # MKL doesn't support cuda so don't run this on cuda to avoid errors and have a fair comparison

    log_path = f"./statistics_logs/"
    if not exists(log_path):
        makedirs(log_path)
    current_time = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    logger = setup_logger(filename=f"{log_path}/dataset-{current_time}.log", verbose=True)

    data_dict = download_and_return_datasets_as_dict(datasets)

    table = PrettyTable()
    table.field_names = ["Dataset", "Type", "Number of Nodes", "Number of Edges", "Average Row Length"]
    for name_i, data_i in data_dict.items():
        edge_index = add_self_loops(data_i.edge_index)
        mat = csr_matrix((ones(edge_index.size(1)), edge_index), shape=(edge_index.max() + 1, edge_index.max() + 1))
        rowptr = from_numpy(mat.indptr).to(long)

        values = ones(edge_index.size(1), dtype=float32)
        a = sparse_coo_tensor(edge_index, values, mat.shape).to_sparse_csr()
        dim_size = rowptr.size(0) - 1
        avg_row_len = edge_index.size(1) / dim_size

        table.add_row([name_i] + [dataset_types[name_i]] + [data_i.num_nodes, data_i.edge_index.size(1), avg_row_len])

    logger.info(table)
