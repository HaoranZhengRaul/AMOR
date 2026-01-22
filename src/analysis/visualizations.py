"""
Visualization utilities for AMOR v2 paper figures.

Generates publication-ready figures for:
- Architecture diagrams
- Entropy validation histograms
- SSM horizon curves
- Efficiency-accuracy tradeoffs
- Gate firing patterns
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path


def setup_plot_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_entropy_histogram(entropy_at_retrieval, entropy_at_local, save_path=None):
    """
    Figure 2: Entropy validation histogram.
    Shows clear separation between entropy at retrieval vs local positions.

    Args:
        entropy_at_retrieval: list/array of entropy values at retrieval positions
        entropy_at_local: list/array of entropy values at local positions
        save_path: path to save figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Compute statistics
    local_mean = np.mean(entropy_at_local)
    retrieval_mean = np.mean(entropy_at_retrieval)
    gap = retrieval_mean - local_mean

    # Adaptive bins based on data range (works for both normalized and raw entropy)
    all_values = np.concatenate([entropy_at_retrieval, entropy_at_local])
    min_val, max_val = np.min(all_values), np.max(all_values)
    bins = np.linspace(min_val - 0.1, max_val + 0.1, 40)

    ax.hist(entropy_at_local, bins=bins, alpha=0.7, label='Local positions',
            color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.hist(entropy_at_retrieval, bins=bins, alpha=0.7, label='Retrieval positions',
            color='#e74c3c', edgecolor='black', linewidth=0.5)

    # Add mean lines
    ax.axvline(local_mean, color='#27ae60', linestyle='--', linewidth=2,
               label=f'Local mean: {local_mean:.2f}')
    ax.axvline(retrieval_mean, color='#c0392b', linestyle='--', linewidth=2,
               label=f'Retrieval mean: {retrieval_mean:.2f}')

    ax.set_title(f'Entropy Distribution (Gap = {gap:.2f})')
    ax.set_xlabel('Entropy (nats)')
    ax.set_ylabel('Count')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_ssm_horizon_curve(noise_lengths, retrieval_accs, save_path=None):
    """
    Figure 3: SSM horizon curve.
    Shows how retrieval accuracy degrades with noise length.

    Args:
        noise_lengths: list of (min, max) tuples or midpoints
        retrieval_accs: corresponding retrieval accuracies
        save_path: path to save figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Convert noise ranges to midpoints if needed
    if isinstance(noise_lengths[0], tuple):
        x = [(n[0] + n[1]) / 2 for n in noise_lengths]
    else:
        x = noise_lengths

    ax.plot(x, retrieval_accs, 'o-', linewidth=2, markersize=10,
            color='#3498db', markeredgecolor='black')

    # Add horizontal reference line
    ax.axhline(0.5, color='#95a5a6', linestyle=':', linewidth=1.5,
               label='50% accuracy')

    # Mark "horizon" region
    ax.fill_between([min(x), 50], [0, 0], [1, 1], alpha=0.1, color='green',
                    label='Within SSM horizon')
    ax.fill_between([50, max(x)], [0, 0], [1, 1], alpha=0.1, color='red',
                    label='Beyond SSM horizon')

    ax.set_title('SSM State Decay vs Noise Length')
    ax.set_xlabel('Average Noise Length (tokens)')
    ax.set_ylabel('Oracle Retrieval Accuracy')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_efficiency_accuracy_tradeoff(models, save_path=None):
    """
    Figure 4: Efficiency-accuracy tradeoff.
    Shows Pareto frontier with different models.

    Args:
        models: dict of {name: {'flops_saved': float, 'accuracy': float}}
        save_path: path to save figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colors: SSM blue, Full Attention orange, Oracle green, Entropy purple
    colors = {
        'SSM Only': '#3498db',       # Blue
        'Full Attention': '#e67e22', # Orange
        'AMOR Oracle': '#27ae60',    # Green
        'AMOR Entropy': '#9b59b6',   # Purple
    }

    # Markers: Oracle circle, Entropy star
    markers = {
        'SSM Only': 's',             # Square
        'Full Attention': '^',       # Triangle
        'AMOR Oracle': 'o',          # Circle
        'AMOR Entropy': '*',         # Star
    }

    marker_sizes = {
        'SSM Only': 200,
        'Full Attention': 200,
        'AMOR Oracle': 250,
        'AMOR Entropy': 400,  # Star needs to be bigger to look similar
    }

    for name, data in models.items():
        color = colors.get(name, '#34495e')
        marker = markers.get(name, 'o')
        size = marker_sizes.get(name, 200)
        ax.scatter(data['flops_saved'] * 100, data['accuracy'] * 100,
                   s=size, c=color, marker=marker, label=name,
                   edgecolors='black', linewidths=1.5, zorder=5)

    ax.set_xlabel('FLOPs Saved (%)')
    ax.set_ylabel('Retrieval Accuracy (%)')
    ax.set_title('Efficiency-Accuracy Tradeoff')
    ax.set_xlim(-5, 105)
    ax.set_ylim(60, 103)  # Extra room at top for the star
    ax.legend(loc='lower left', framealpha=0.9, markerscale=0.6, fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_gate_visualization(tokens, gate_pattern, special_tokens, save_path=None):
    """
    Figure 5: Gate firing visualization (optional appendix figure).
    Shows example sequence with gate firing pattern.

    Args:
        tokens: list of token indices
        gate_pattern: binary gate decisions for each position
        special_tokens: dict mapping special token names to indices
        save_path: path to save figure
    """
    setup_plot_style()

    seq_len = min(len(tokens), 100)  # Limit display length
    tokens = tokens[:seq_len]
    gate_pattern = gate_pattern[:seq_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4),
                                    gridspec_kw={'height_ratios': [1, 3]})

    # Gate pattern
    gate_colors = ['#2ecc71' if g == 0 else '#e74c3c' for g in gate_pattern]
    ax1.bar(range(seq_len), [1]*seq_len, color=gate_colors, edgecolor='none')
    ax1.set_xlim(-0.5, seq_len - 0.5)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Gate')
    ax1.set_yticks([])
    ax1.set_title('Gate Firing Pattern (Red = Fire, Green = Skip)')

    # Token visualization
    colors = []
    for t in tokens:
        if t == special_tokens.get('MARKER') or t == special_tokens.get('STORE'):
            colors.append('#f39c12')  # Yellow for store markers
        elif t == special_tokens.get('RETRIEVE') or t == special_tokens.get('QUERY'):
            colors.append('#9b59b6')  # Purple for query markers
        else:
            colors.append('#3498db')  # Blue for regular tokens

    ax2.bar(range(seq_len), [1]*seq_len, color=colors, edgecolor='black', linewidth=0.3)
    ax2.set_xlim(-0.5, seq_len - 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Token')
    ax2.set_yticks([])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#f39c12', label='Store/Marker'),
        mpatches.Patch(facecolor='#9b59b6', label='Query/Retrieve'),
        mpatches.Patch(facecolor='#3498db', label='Regular token'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_architecture_diagram(save_path=None):
    """
    Figure 1: Architecture diagram.
    Shows System 1/2 flow with entropy gate - circuit style with both paths.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(22, 11))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Colors
    BLUE = '#3498db'      # SSM
    CORAL = '#e74c3c'     # AMOR Gate (prominent red)
    GREEN = '#27ae60'     # Fast path
    PURPLE = '#9b59b6'    # Ghost KV
    ORANGE = '#e67e22'    # Attention path - BRIGHT ORANGE
    GRAY = '#bdc3c7'      # Input/Output
    DARK_GRAY = '#7f8c8d' # Darker gray for text
    LIGHT_GREEN = '#eafaf1'   # Fast path background
    LIGHT_ORANGE = '#fef5e7'  # High entropy background - LIGHT ORANGE

    # Background regions - more room, separated from AMOR box
    region_width = 7.0
    region_height = 3.5

    # Fast path region (top) - more space from AMOR box
    fast_region = mpatches.FancyBboxPatch(
        (13.0, 6.5), region_width, region_height,
        boxstyle="round,pad=0.15",
        facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=2.5, alpha=0.6
    )
    ax.add_patch(fast_region)

    # Attention path region (bottom) - LIGHT ORANGE background, lower to give text room
    attn_region = mpatches.FancyBboxPatch(
        (13.0, 0.8), region_width, region_height,
        boxstyle="round,pad=0.15",
        facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=2.5, alpha=0.6
    )
    ax.add_patch(attn_region)

    # Path labels - BIGGER FONTS, black base color
    ax.text(16.5, 9.6, 'LOW ENTROPY', ha='center', fontsize=22,
            fontweight='bold', color=GREEN)
    ax.text(16.5, 8.8, '"I know this" → Use SSM output directly', ha='center',
            fontsize=18, style='italic', color='black')

    ax.text(16.5, 4.0, 'HIGH ENTROPY', ha='center', fontsize=22,
            fontweight='bold', color=ORANGE)
    ax.text(16.5, 3.2, '"I don\'t know" → Retrieve via attention', ha='center',
            fontsize=18, style='italic', color='black')

    # Helper function for boxes - matching border colors, BIGGER FONTS
    def draw_box(x, y, w, h, color, text, fontsize=18, text_color='black', border_color=None):
        if border_color is None:
            border_color = color  # Match border to fill color
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor=border_color, linewidth=3
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)

    # Main components - BIGGER
    # Input
    draw_box(1.8, 5.5, 2.2, 1.5, GRAY, 'Input', 20, 'black', DARK_GRAY)

    # SSM - full name, border matches blue
    draw_box(5.5, 5.5, 3.8, 2.2, BLUE, 'State Space\nModel (SSM)', 18, 'black')

    # AMOR Entropy Gate - BIGGER and prominent, BLACK text
    gate_x, gate_y = 10.5, 5.5
    gate_w, gate_h = 3.4, 3.0
    gate_rect = mpatches.FancyBboxPatch(
        (gate_x - gate_w/2, gate_y - gate_h/2), gate_w, gate_h,
        boxstyle="round,pad=0.1",
        facecolor=CORAL, edgecolor='#c0392b', linewidth=3.5
    )
    ax.add_patch(gate_rect)
    ax.text(gate_x, gate_y + 0.75, 'AMOR', ha='center', fontsize=22,
            fontweight='bold', color='black')
    ax.text(gate_x, gate_y, 'Entropy Gate', ha='center', fontsize=18,
            fontweight='bold', color='black')
    ax.text(gate_x, gate_y - 0.85, r'$H(x) > \theta$?', ha='center', fontsize=17,
            color='black', style='italic')

    # Ghost KV Cache - border matches purple
    draw_box(5.5, 1.6, 3.2, 1.8, PURPLE, 'Ghost KV\nCache', 18, 'black')

    # Sparse Attention - BRIGHT ORANGE, border matches orange
    draw_box(16.5, 2.0, 3.2, 1.9, ORANGE, 'Sparse\nAttention', 18, 'black')

    # Output boxes - aligned vertically at same x, with space from Sparse Attention
    output_x = 19.8
    draw_box(output_x, 7.8, 2.0, 1.3, GRAY, 'Output', 19, 'black', DARK_GRAY)
    draw_box(output_x, 2.0, 2.0, 1.3, GRAY, 'Output', 19, 'black', DARK_GRAY)

    # Arrows - carefully positioned, THICKER

    # Input -> SSM
    ax.annotate('', xy=(3.6, 5.5), xytext=(2.9, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=3.5,
                               shrinkA=0, shrinkB=0))

    # SSM -> Gate
    ax.annotate('', xy=(8.8, 5.5), xytext=(7.4, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=3.5,
                               shrinkA=0, shrinkB=0))

    # SSM -> Ghost KV (downward)
    ax.annotate('', xy=(5.5, 2.5), xytext=(5.5, 4.4),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=3.5,
                               shrinkA=0, shrinkB=0))

    # Gate -> Fast path (No - up then right to Output) - GREEN SOLID
    ax.plot([10.5, 10.5], [7.0, 7.8], color=GREEN, lw=4, solid_capstyle='round')
    ax.plot([10.5, 18.8], [7.8, 7.8], color=GREEN, lw=4, solid_capstyle='round')
    ax.annotate('', xy=(18.8, 7.8), xytext=(18.3, 7.8),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=4,
                               shrinkA=0, shrinkB=0))
    # "No" label - BIGGER
    ax.text(11.0, 7.3, 'No', fontsize=19, fontweight='bold', color=GREEN)

    # Gate -> Attention path (Yes - down then right) - ORANGE SOLID
    ax.plot([10.5, 10.5], [4.0, 2.0], color=ORANGE, lw=4, solid_capstyle='round')
    ax.plot([10.5, 14.9], [2.0, 2.0], color=ORANGE, lw=4, solid_capstyle='round')
    ax.annotate('', xy=(14.9, 2.0), xytext=(14.4, 2.0),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=4,
                               shrinkA=0, shrinkB=0))
    # "Yes" label - BIGGER, ORANGE
    ax.text(11.0, 3.2, 'Yes', fontsize=19, fontweight='bold', color=ORANGE)

    # Ghost KV -> Attention - U-shape from bottom of Ghost KV
    # Down from Ghost KV bottom, right along bottom, then up to Sparse Attention
    ghost_kv_x = 5.5
    ghost_kv_bottom = 0.7  # bottom of Ghost KV box
    bottom_y = 0.3  # how low the U goes
    attn_x = 16.5
    attn_bottom = 1.05  # bottom of Sparse Attention box

    ax.plot([ghost_kv_x, ghost_kv_x], [ghost_kv_bottom, bottom_y], color=PURPLE, lw=3.5, solid_capstyle='round')
    ax.plot([ghost_kv_x, attn_x], [bottom_y, bottom_y], color=PURPLE, lw=3.5, solid_capstyle='round')
    ax.plot([attn_x, attn_x], [bottom_y, attn_bottom], color=PURPLE, lw=3.5, solid_capstyle='round')
    ax.annotate('', xy=(attn_x, attn_bottom), xytext=(attn_x, 0.7),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=3.5,
                               shrinkA=0, shrinkB=0))

    # Attention -> Output - ORANGE
    ax.annotate('', xy=(18.8, 2.0), xytext=(18.1, 2.0),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=4,
                               shrinkA=0, shrinkB=0))

    # "Thinking Fast" / "Thinking Slow" labels after Output boxes
    ax.text(21.2, 7.8, '"Thinking\nFast"', ha='left', va='center', fontsize=17,
            fontweight='bold', color=GREEN, style='italic')
    ax.text(21.2, 2.0, '"Thinking\nSlow"', ha='left', va='center', fontsize=17,
            fontweight='bold', color=ORANGE, style='italic')

    # Title - BIGGER
    ax.set_title('AMOR: Adaptive Metacognitive Output Router',
                 fontsize=26, fontweight='bold', pad=35, color='black')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_entropy_evolution(history, save_path=None):
    """
    Figure: Entropy gap evolution during training.
    Shows how the model learns to be uncertain at retrieval positions.

    Args:
        history: list of dicts with 'epoch' and 'train_entropy_gap' keys
        save_path: path to save figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = [h['epoch'] for h in history]
    gaps = [h.get('train_entropy_gap', h.get('entropy_gap', 0)) for h in history]

    # Plot entropy gap over epochs
    ax.plot(epochs, gaps, 'o-', linewidth=2, markersize=8,
            color='#9b59b6', markeredgecolor='black')

    # Fill under curve
    ax.fill_between(epochs, 0, gaps, alpha=0.2, color='#9b59b6')

    # Annotations
    if len(gaps) > 0:
        final_gap = gaps[-1]
        ax.axhline(final_gap, color='#7f8c8d', linestyle=':', linewidth=1.5,
                   alpha=0.7)
        ax.annotate(f'Final gap: {final_gap:.2f}',
                    xy=(epochs[-1], final_gap),
                    xytext=(epochs[-1] * 0.7, final_gap * 1.1),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d'))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy Gap (retrieval - local)')
    ax.set_title('Learning to Know What You Don\'t Know')
    ax.set_ylim(0, max(gaps) * 1.2 if gaps else 1)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_ssm_horizon_with_errorbars(results, save_path=None):
    """
    Figure 3: SSM horizon curve with error bars from sweep results.

    Args:
        results: dict from noise sweep with noise_length -> {'retrieval_accuracy_mean', 'retrieval_accuracy_std'}
        save_path: path to save figure
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by noise length
    noise_lengths = sorted(results.keys())
    means = [results[n]['retrieval_accuracy_mean'] for n in noise_lengths]
    stds = [results[n]['retrieval_accuracy_std'] for n in noise_lengths]

    # Plot line with error bars
    ax.errorbar(noise_lengths, means, yerr=stds, fmt='o-',
                linewidth=2, markersize=10, capsize=5,
                color='#3498db', markeredgecolor='black', ecolor='#7f8c8d')

    # Fill error region
    ax.fill_between(noise_lengths,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='#3498db')

    # Reference line at 50%
    ax.axhline(0.5, color='#95a5a6', linestyle=':', linewidth=1.5, label='50% accuracy')

    # Find approximate horizon (where accuracy drops below 50%)
    below_50 = [n for n, m in zip(noise_lengths, means) if m < 0.5]
    if below_50:
        horizon = below_50[0]
        ax.axvline(horizon, color='#e74c3c', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Approx. horizon (~{horizon} tokens)')

    ax.set_xlabel('Noise Length (tokens)')
    ax.set_ylabel('Oracle Retrieval Accuracy')
    ax.set_title('SSM State Decay vs Noise Length')
    ax.set_ylim(0, 1)
    ax.set_xlim(min(noise_lengths) - 5, max(noise_lengths) + 5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    return fig


def generate_all_figures(output_dir='paper/figures', results_dir='experiments'):
    """Generate all paper figures from experiment results."""
    setup_plot_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Architecture diagram
    plot_architecture_diagram(f'{output_dir}/architecture.png')

    # Load results for other figures
    try:
        # Entropy histogram (need to generate from saved data)
        # This requires entropy data from training runs
        print("Entropy histogram - requires entropy data from training")

        # SSM horizon curve with error bars from multi-seed sweep
        import json
        try:
            with open(f'{results_dir}/diagnostic_needlehaystack/noise_sweep_detailed.json', 'r') as f:
                sweep_data = json.load(f)
            # Convert string keys to int
            results_for_plot = {int(k): v for k, v in sweep_data['results'].items()}
            plot_ssm_horizon_with_errorbars(results_for_plot, f'{output_dir}/ssm_horizon.png')
        except FileNotFoundError:
            # Fallback to hardcoded values without error bars
            noise_data = [(20, 50), (30, 60), (50, 150)]
            accs = [0.2727, 0.1018, 0.1243]  # Multi-seed means
            plot_ssm_horizon_curve(noise_data, accs, f'{output_dir}/ssm_horizon.png')

        # Efficiency-accuracy tradeoff
        # Note: SSM+MLP deliberately excluded - discussed in limitations instead
        models = {
            'SSM Only': {'flops_saved': 1.0, 'accuracy': 0.6835},
            'Full Attention': {'flops_saved': 0.0, 'accuracy': 0.873},
            'AMOR Oracle': {'flops_saved': 0.971, 'accuracy': 0.9963},
            'AMOR Entropy': {'flops_saved': 0.7768, 'accuracy': 1.0},
        }
        plot_efficiency_accuracy_tradeoff(models, f'{output_dir}/tradeoff.png')

        print(f"\nFigures saved to {output_dir}/")

    except Exception as e:
        print(f"Error generating figures: {e}")


if __name__ == '__main__':
    generate_all_figures()
