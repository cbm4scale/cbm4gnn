import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Sequential runtime data
    runtime_data = {
        "ca-HepPh": {
            "PyTorch COO SpMM": 0.15424,
            "PyTorch CSR SpMM": 0.15390,
            "MKL CSR SpMM": 0.09196,
            "CBM MKL CSR SpMM (ours)": 0.07092,
            "CBM PyTorch CSR SpMM (ours)": 0.08441
        },
        # "ca-HepTh": {
        #     "PyTorch COO SpMM": 0.06449,
        #     "PyTorch CSR SpMM": 0.06436,
        #     "MKL CSR SpMM": 0.05030,
        #     "CBM MKL CSR SpMM (ours)": 0.05194,
        #     "CBM PyTorch CSR SpMM (ours)": 0.07015
        # },
        # "cit-HepPh": {
        #     "PyTorch COO SpMM": 0.43073,
        #     "PyTorch CSR SpMM": 0.43193,
        #     "MKL CSR SpMM": 0.28065,
        #     "CBM MKL CSR SpMM (ours)": 0.28021,
        #     "CBM PyTorch CSR SpMM (ours)": 0.35109
        # },
        "cit-HepTh": {
            "PyTorch COO SpMM": 0.33650,
            "PyTorch CSR SpMM": 0.33639,
            "MKL CSR SpMM": 0.21624,
            "CBM MKL CSR SpMM (ours)": 0.21657,
            "CBM PyTorch CSR SpMM (ours)": 0.27312
        },
        "ca-AstroPh": {
            "PyTorch COO SpMM": 0.28791,
            "PyTorch CSR SpMM": 0.28892,
            "MKL CSR SpMM": 0.17861,
            "CBM MKL CSR SpMM (ours)": 0.14728,
            "CBM PyTorch CSR SpMM (ours)": 0.18590
        },
        "web-Stanford": {
            "PyTorch COO SpMM": 3.43959,
            "PyTorch CSR SpMM": 3.41346,
            "MKL CSR SpMM": 2.26319,
            "CBM MKL CSR SpMM (ours)": 1.77595,
            "CBM PyTorch CSR SpMM (ours)": 2.13718
        },
        "web-NoteDame": {
            "PyTorch COO SpMM": 2.79660,
            "PyTorch CSR SpMM": 2.79634,
            "MKL CSR SpMM": 1.64382,
            "CBM MKL CSR SpMM (ours)": 1.87871,
            "CBM PyTorch CSR SpMM (ours)": 2.09499
        },
        "Cora": {
            "PyTorch COO SpMM": 0.01468,
            "PyTorch CSR SpMM": 0.01483,
            "MKL CSR SpMM": 0.01066,
            "CBM MKL CSR SpMM (ours)": 0.01138,
            "CBM PyTorch CSR SpMM (ours)": 0.01210
        },
        "PubMed": {
            "PyTorch COO SpMM": 0.13111,
            "PyTorch CSR SpMM": 0.13112,
            "MKL CSR SpMM": 0.10312,
            "CBM MKL CSR SpMM (ours)": 0.11085,
            "CBM PyTorch CSR SpMM (ours)": 0.12136
        },
        "coPapersDBLP": {
            "PyTorch COO SpMM": 14.44788,
            "PyTorch CSR SpMM": 14.11325,
            "MKL CSR SpMM": 6.13422,
            "CBM MKL CSR SpMM (ours)": 3.45567,
            "CBM PyTorch CSR SpMM (ours)": 3.95343
        },
        "oPapersCiteseer": {
            "PyTorch COO SpMM": 12.41265,
            "PyTorch CSR SpMM": 12.41086,
            "MKL CSR SpMM": 5.32524,
            "CBM MKL CSR SpMM (ours)": 2.51003,
            "CBM PyTorch CSR SpMM (ours)": 2.85606
        }
    }

    # # Parallel runtime data
    # runtime_data = {
    #     "ca-HepPh": {
    #         "Native Pytorch COO Sparse": 0.03653,
    #         "Native Pytorch CSR Sparse": 0.03885,
    #         "MKL CSR Sparse": 0.01300,
    #         "CBM MKL CSR Sparse": 0.01152,
    #         "CBM Native Torch CSR Sparse": 0.01970
    #     },
    #     "cit-HepPh": {
    #         "Native Pytorch COO Sparse": 0.01794,
    #         "Native Pytorch CSR Sparse": 0.01941,
    #         "MKL CSR Sparse": 0.00771,
    #         "CBM MKL CSR Sparse": 0.00848,
    #         "CBM Native Torch CSR Sparse": 0.01217
    #     },
    #     "cit-HepPh": {
    #         "Native Pytorch COO Sparse": 0.09177,
    #         "Native Pytorch CSR Sparse": 0.09177,
    #         "MKL CSR Sparse": 0.04799,
    #         "CBM MKL CSR Sparse": 0.04974,
    #         "CBM Native Torch CSR Sparse": 0.05947
    #     },
    #     "cit-HepTh": {
    #         "Native Pytorch COO Sparse": 0.07371,
    #         "Native Pytorch CSR Sparse": 0.07371,
    #         "MKL CSR Sparse": 0.03305,
    #         "CBM MKL CSR Sparse": 0.03490,
    #         "CBM Native Torch CSR Sparse": 0.04593
    #     },
    #     "ca-AstroPh": {
    #         "Native Pytorch COO Sparse": 0.06327,
    #         "Native Pytorch CSR Sparse": 0.06290,
    #         "MKL CSR Sparse": 0.02584,
    #         "CBM MKL CSR Sparse": 0.02501,
    #         "CBM Native Torch CSR Sparse": 0.03409
    #     },
    #     "web-Stanford": {
    #         "Native Pytorch COO Sparse": 0.65733,
    #         "Native Pytorch CSR Sparse": 0.65523,
    #         "MKL CSR Sparse": 0.33806,
    #         "CBM MKL CSR Sparse": 0.29746,
    #         "CBM Native Torch CSR Sparse": 0.39165
    #     },
    #     "web-NoteDame": {
    #         "Native Pytorch COO Sparse": 0.49924,
    #         "Native Pytorch CSR Sparse": 0.49747,
    #         "MKL CSR Sparse": 0.28226,
    #         "CBM MKL CSR Sparse": 0.30426,
    #         "CBM Native Torch CSR Sparse": 0.37971
    #     },
    #     "Cora": {
    #         "Native Pytorch COO Sparse": 0.00782,
    #         "Native Pytorch CSR Sparse": 0.00782,
    #         "MKL CSR Sparse": 0.00217,
    #         "CBM MKL CSR Sparse": 0.00259,
    #         "CBM Native Torch CSR Sparse": 0.00291
    #     },
    #     "PubMed": {
    #         "Native Pytorch COO Sparse": 0.03086,
    #         "Native Pytorch CSR Sparse": 0.03215,
    #         "MKL CSR Sparse": 0.01595,
    #         "CBM MKL CSR Sparse": 0.01734,
    #         "CBM Native Torch CSR Sparse": 0.02201
    #     },
    #     "coPapersDBLP": {
    #         "Native Pytorch COO Sparse": 3.46036,
    #         "Native Pytorch CSR Sparse": 3.45750,
    #         "MKL CSR Sparse": 1.17130,
    #         "CBM MKL CSR Sparse": 0.59917,
    #         "CBM Native Torch CSR Sparse": 0.70153
    #     },
    #     "oPapersCiteseer": {
    #         "Native Pytorch COO Sparse": 3.23957,
    #         "Native Pytorch CSR Sparse": 3.21811,
    #         "MKL CSR Sparse": 1.21605,
    #         "CBM MKL CSR Sparse": 0.40522,
    #         "CBM Native Torch CSR Sparse": 0.55151
    #     }
    # }

    # Convert nested dictionary to DataFrame suitable for Seaborn
    data = []
    for dataset, methods in runtime_data.items():
        for method, runtime in methods.items():
            data.append({"Dataset": dataset, "Method": method, "Runtime": runtime})

    df = pd.DataFrame(data)

    # Styling with LaTeX and cmbright
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

    # Set style
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))
    barplot = sns.barplot(data=df, x='Dataset', y='Runtime', hue='Method', dodge=True)
    ax.set_ylabel(r"\textbf{Runtime Logarithm (s)}", fontsize=18)
    ax.set_xlabel(r"\textbf{Dataset Name}", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    plt.xticks(rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Method', title_fontsize='16', fontsize='14', loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    fig.savefig(f"gcn_sequential_runtime_comparison.pdf", bbox_inches="tight")
    # fig.savefig(f"gcn_parallel_runtime_comparison.pdf", bbox_inches="tight")
    plt.close()