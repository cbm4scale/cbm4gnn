import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

    # Fake data, replace with real data
    figure_name = "fake_data"
    data = pd.DataFrame({
        "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
        "Sequential (Left)": [10, 20, 30, 40, 50, 60, 70, 80],
        "Parallel 16-core (Left)": [15, 25, 35, 45, 55, 65, 75, 85],
        "Compression Ratio (Right)": [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
    })

    # figure_name = "ca-AstroPh"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [26.4, 27.6, 28.1, 28.0, 26.7, 22.6, 14.4, 3.9],
    #     "Parallel 16-core (Left)": [7.8, 10.5, 10.2, 10.1, 11.3, 7.3, 4.6, 2.3],
    #     "Compression Ratio (Right)": [1.70, 1.68, 1.65, 1.61, 1.54, 1.40, 1.21, 1.02]
    # })
    #
    # figure_name = "ca-HepPh"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [39.5, 42.3, 43.6, 44.4, 43.9, 42.8, 37.7, 29.6],
    #     "Parallel 16-core (Left)": [18.0, 19.5, 18.8, 10.7, 12.6, 10.4, 23.5, 26.6],
    #     "Compression Ratio (Right)": [2.65, 2.56, 2.48, 2.39, 2.32, 2.21, 1.89, 1.56]
    # })
    #
    # figure_name = "cit-HepTh"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [-0.1, 3.7, 4.9, 5.5, 4.7, 2.6, 0.7, -0.3],
    #     "Parallel 16-core (Left)": [-9.3, -2.1, 0.0, 3.8, 5.3, 3.3, 1.0, -1.2],
    #     "Compression Ratio (Right)": [1.23, 1.20, 1.17, 1.12, 1.05, 0.99, 0.95, 0.93]
    # })
    #
    # figure_name = "Cora"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [-81.8, -15.9, -1.4, -0.8, -1.1, -1.0, -1.0, -0.9],
    #     "Parallel 16-core (Left)": [-168.0, -60.7, -35.3, -26.2, -19.7, -19.1, -19.4, -19.2],
    #     "Compression Ratio (Right)": [0.96, 0.89, 0.84, 0.82, 0.81, 0.81, 0.81, 0.81]
    # })
    #
    # figure_name = "coPapersDBLP"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [59.0, 59.2, 59.2, 59.3, 59.2, 58.1, 53.8, 42.9],
    #     "Parallel 16-core (Left)": [61.4, 61.6, 61.7, 61.9, 62.3, 62.7, 62.6, 59.1],
    #     "Compression Ratio (Right)": [5.92, 5.91, 5.88, 5.81, 5.57, 4.89, 3.61, 2.25]
    # })
    #
    # figure_name = "coPapersCiteseer"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [71.1, 71.1, 71.3, 71.2, 71.2, 70.4, 67.2, 59.1],
    #     "Parallel 16-core (Left)": [77.4, 77.6, 77.6, 77.8, 78.0, 78.5, 78.9, 77.3],
    #     "Compression Ratio (Right)": [9.80, 9.77, 9.73, 9.59, 9.12, 7.85, 5.58, 3.38]
    # })
    #
    # figure_name = "PubMed"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [-20.2, -3.3, -0.7, -0.2, -0.2, -0.3, -0.4, -0.4],
    #     "Parallel 16-core (Left)": [-57.4, -10.5, -3.7, -1.9, -2.1, -1.8, -2.3, -2.0],
    #     "Compression Ratio (Right)": [1.02, 0.90, 0.86, 0.85, 0.84, 0.84, 0.84, 0.83]
    # })
    #
    # figure_name = "
    #
    # figure_name = "web-Stanford"
    # data = pd.DataFrame({
    #     "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
    #     "Sequential (Left)": [25.5, 31.8, 30.3, 30.4, 27.7, 22.7, 12.2, 2.0],
    #     "Parallel 16-core (Left)": [-30.4, -21.4, 1.4, -5.1, 3.5, 16.3, 9.7, 2.0],
    #     "Compression Ratio (Right)": [2.25, 2.15, 2.02, 1.78, 1.51, 1.24, 1.04, 0.91]
    # })

    data_melted = data.melt(id_vars="Alpha", value_vars=["Sequential (Left)", "Parallel 16-core (Left)"], var_name="Type", value_name="Speed-up Improvement Percentage (%)")

    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(x="Alpha", y="Speed-up Improvement Percentage (%)", hue="Type", data=data_melted, dodge=True, ax=ax1)

    ax2 = ax1.twinx()
    lineplot = sns.lineplot(x="Alpha", y="Compression Ratio (Right)", data=data, marker="o", ax=ax2, color="r", label="Compression Ratio (Right)")

    ax1.set_ylabel(r"Relative Speed-up Improvement Percentage (\%)", fontsize=18)
    ax2.set_ylabel(r"\textbf{Compression Ratio}", fontsize=18)
    ax1.set_xlabel(r"\textbf{Alpha}", fontsize=18)
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)

    # ax1.set_ylabel(r"", fontsize=18)
    # ax2.set_ylabel(r"", fontsize=18)
    ax1.set_xlabel(r"", fontsize=18)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=15)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    plt.show()
    fig.savefig(f"alpha_searcher_{figure_name}.pdf", bbox_inches="tight")
    plt.close()