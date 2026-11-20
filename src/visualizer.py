from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ResultVisualizer:
    def __init__(self, output_dir="results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def plot_condition_comparison(self, summary_df: pd.DataFrame, filename="condition_comparison.png"):
        plt.figure(figsize=(8, 4.5))
        ax = sns.barplot(data=summary_df, x="condition", y="accuracy", hue="model_strategy", palette="viridis")
        ax.set_ylim(0, 1.05)
        ax.set_title("Literary Irony Comprehension Across Translation Conditions")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.show()
