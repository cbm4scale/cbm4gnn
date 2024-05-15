import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Fake data, replace with real data
    data = pd.DataFrame({
        "Alpha": ["0", "1", "2", "4", "8", "16", "32", "64"],
        "Sequential (Left)": [10, 20, 30, 40, 50, 60, 70, 80],
        "Parallel (Left)": [15, 25, 35, 45, 55, 65, 75, 85],
        "Compression Ratio (Right)": [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
    })

    data_melted = data.melt(id_vars="Alpha", value_vars=["Sequential (Left)", "Parallel (Left)"], var_name="Type", value_name="Improvement Percentage (%)")

    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(x="Alpha", y="Improvement Percentage (%)", hue="Type", data=data_melted, dodge=True, ax=ax1)

    bars = barplot.patches
    half = int(len(bars) / 2)
    for i in range(half):
        bars[i].set_hatch("//")
        bars[i + half].set_facecolor(bars[i + half].get_facecolor())

    ax2 = ax1.twinx()
    lineplot = sns.lineplot(x="Alpha", y="Compression Ratio (Right)", data=data, marker="o", ax=ax2, color="r", label="Compression Ratio (Right)")

    ax2.set_ylabel("Compression Ratio", fontsize=18, fontweight="bold")
    ax1.set_xlabel("Alpha", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Improvement Percentage (%)", fontsize=18, fontweight="bold")
    ax1.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    plt.show()
