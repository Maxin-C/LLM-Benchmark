# =========================
# Global Plot Style (Nature-like)
# =========================

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# =========================
# EASE Benchmark Palette
# =========================

EASE_COLORS = {
    "purple": "#7B61C9",
    "green": "#7CC67C",
    "cyan": "#5EC7C2",
    "blue": "#5B8FF9",
    "orange": "#FF9F5A",
    "red": "#FF6B6B",
    "gray": "#6B7280"
}

MODEL_COLORS = [
    "#7B61C9",
    "#7CC67C",
    "#5EC7C2",
    "#5B8FF9",
    "#FF9F5A",
    "#FF6B6B"
]

# =========================
# Better Save Function
# =========================

def save_figure(fig, output_path):
    fig.savefig(output_path + ".png", bbox_inches='tight')
    fig.savefig(output_path + ".pdf", bbox_inches='tight')
    fig.savefig(output_path + ".svg", bbox_inches='tight')

# =========================
# Improved Confidence Interval Plot
# =========================

def plot_confidence_intervals(bootstrap_results, output_dir):

    models = list(bootstrap_results.keys())

    means = [bootstrap_results[m]['mean'] for m in models]
    ci_lower = [bootstrap_results[m]['ci_lower'] for m in models]
    ci_upper = [bootstrap_results[m]['ci_upper'] for m in models]

    sorted_idx = np.argsort(means)[::-1]

    models = [models[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    ci_lower = [ci_lower[i] for i in sorted_idx]
    ci_upper = [ci_upper[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4.8))

    y = np.arange(len(models))

    for i in range(len(models)):
        ax.plot(
            [ci_lower[i], ci_upper[i]],
            [y[i], y[i]],
            lw=5,
            color=MODEL_COLORS[i],
            solid_capstyle='round'
        )

        ax.scatter(
            means[i],
            y[i],
            s=120,
            color=MODEL_COLORS[i],
            edgecolor='black',
            linewidth=0.8,
            zorder=3
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [m.replace('qwen3-', 'Qwen3-').upper() for m in models]
    )

    ax.set_xlabel("Overall Score")
    ax.set_title("Bootstrap 95% Confidence Intervals")

    ax.grid(axis='x', linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    save_figure(
        fig,
        os.path.join(output_dir, "confidence_intervals")
    )

    plt.close()

# =========================
# Improved Significance Matrix
# =========================

def plot_significance_matrix(comparisons, models, output_dir):

    n = len(models)

    matrix = np.zeros((n, n))

    for comp in comparisons:
        i = models.index(comp['model1'])
        j = models.index(comp['model2'])

        value = -np.log10(comp['p_value_corrected'] + 1e-10)

        matrix[i, j] = value
        matrix[j, i] = value

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.imshow(
        matrix,
        cmap='RdYlBu_r'
    )

    for i in range(n):
        for j in range(n):

            if i == j:
                txt = "-"
            else:
                txt = "✓" if matrix[i, j] > 1.3 else "×"

            ax.text(
                j,
                i,
                txt,
                ha='center',
                va='center',
                fontsize=16,
                fontweight='bold'
            )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    ax.set_xticklabels(
        [m.replace('qwen3-', 'Qwen3-').upper() for m in models],
        rotation=45,
        ha='right'
    )

    ax.set_yticklabels(
        [m.replace('qwen3-', 'Qwen3-').upper() for m in models]
    )

    ax.set_title("Pairwise Statistical Significance")

    cbar = plt.colorbar(im)
    cbar.set_label("-log10(p-value)")

    plt.tight_layout()

    save_figure(
        fig,
        os.path.join(output_dir, "significance_matrix")
    )

    plt.close()

# =========================
# Improved Ranking Stability
# =========================

def plot_ranking_stability(ranking_stats, output_dir):

    models = list(ranking_stats.keys())

    sorted_models = sorted(
        models,
        key=lambda m: ranking_stats[m]['mean_rank']
    )

    means = [ranking_stats[m]['mean_rank'] for m in sorted_models]
    stds = [ranking_stats[m]['rank_std'] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y = np.arange(len(sorted_models))

    bars = ax.barh(
        y,
        means,
        color=MODEL_COLORS[:len(models)],
        edgecolor='black',
        linewidth=0.6,
        xerr=stds,
        capsize=4
    )

    ax.set_yticks(y)

    ax.set_yticklabels([
        m.replace('qwen3-', 'Qwen3-').upper()
        for m in sorted_models
    ])

    ax.invert_yaxis()

    ax.set_xlabel("Average Rank")
    ax.set_title("Ranking Stability")

    ax.grid(axis='x', linestyle='--', alpha=0.25)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, val in enumerate(means):
        ax.text(
            val + 0.08,
            i,
            f"{val:.2f}",
            va='center',
            fontsize=10
        )

    plt.tight_layout()

    save_figure(
        fig,
        os.path.join(output_dir, "ranking_stability")
    )

    plt.close()

# =========================
# Effect Size Visualization
# =========================

def plot_effect_sizes(effect_sizes, output_dir):

    labels = [
        f"{e['model1']} vs\n{e['model2']}"
        for e in effect_sizes
    ]

    values = [e['cohens_d'] for e in effect_sizes]

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = [
        EASE_COLORS["red"] if abs(v) > 0.8
        else EASE_COLORS["orange"] if abs(v) > 0.5
        else EASE_COLORS["blue"]
        for v in values
    ]

    bars = ax.bar(
        np.arange(len(values)),
        values,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )

    ax.axhline(0, color='black', linewidth=1)

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect Size Analysis")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    save_figure(
        fig,
        os.path.join(output_dir, "effect_sizes")
    )

    plt.close()
