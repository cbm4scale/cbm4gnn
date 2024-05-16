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
        "Parallel (Left)": [15, 25, 35, 45, 55, 65, 75, 85],
        "Compression Ratio (Right)": [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
    })

    data_melted = data.melt(id_vars="Alpha", value_vars=["Sequential (Left)", "Parallel (Left)"], var_name="Type", value_name="Speed-up Improvement Percentage (%)")

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

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=15)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    plt.show()
    fig.savefig(f"alpha_searcher_{figure_name}.pdf", bbox_inches="tight")
    plt.close()
