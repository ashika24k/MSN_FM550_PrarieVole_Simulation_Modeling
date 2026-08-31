"""
Fig3_Combined_ABCD.py

Runs the whole Figure 3 pipeline in one go and produces a single combined
figure with all four panels:

    A — dMSN single-cell I-V curve
    B — IMSN single-cell I-V curve
  C — dMSN population mean ± SEM I-V curve (from existing raw CSVs)
    D — IMSN population mean ± SEM I-V curve (from existing raw CSVs)

Panels C/D are read from per-cell CSVs in 'generated_inputs/'. If those CSVs do
not exist, run F3_generate_population_data.py first.
That step launches full NEURON population simulations and is slow.

Produces:
  Figure3_AllPanels.png
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from F3_single_cell_analysis import calculate_rectification  # noqa: E402

# ---------------------------------------------------------------------------
# Global font settings (matches the individual Figure 3 scripts)
# ---------------------------------------------------------------------------
plt.rcParams['font.family']      = 'Calibri'
plt.rcParams['font.weight']      = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size']        = 18
plt.rcParams['axes.titlesize']   = 24
plt.rcParams['axes.labelsize']   = 22
plt.rcParams['xtick.labelsize']  = 18
plt.rcParams['ytick.labelsize']  = 18

CONDITION_COLORS = {
    'Control':   '#285C81',
    '1.25x KIR': '#F57C00',
    '1.5x KIR':  '#2E7D32',
    'FM550':     '#9D2222',
}
MARKEVERY_MAP = {
    'Control':   None,
    '1.25x KIR': None,
    '1.5x KIR':  (0, 2),
    'FM550':     (1, 2),
}
CONDITION_ORDER = ['Control', '1.25x KIR', '1.5x KIR', 'FM550']

# The raw population CSVs label conditions differently; map them to the
# canonical names above so panels C/D get the same colors as A/B.
CONDITION_ALIASES = {
    'Female Vehicle (Control)': 'Control',
    'Female FM550':             'FM550',
}

CONDITION_SPECS = [
    ('Control',   dict(v_init=-82.8, kir_modification=None,                   use_female_params=False)),
    ('1.25x KIR', dict(v_init=-82.8, kir_modification={'scale_factor': 1.25}, use_female_params=False)),
    ('1.5x KIR',  dict(v_init=-82.8, kir_modification={'scale_factor': 1.5},  use_female_params=False)),
    ('FM550',     dict(v_init=-82.3, kir_modification=None,                   use_female_params=True)),
]

# Per-cell CSVs produced by F3_generate_population_data.py.
RAW_CSV = {
    'dmsn': os.path.join(HERE, 'generated_inputs', 'Figure3C_DMSN_RAW.csv'),
    'imsn': os.path.join(HERE, 'generated_inputs', 'Figure3D_IMSN_RAW.csv'),
}


# Both the single-cell and population sweeps use the same current steps
# (-0.09 to -0.01 nA in 0.01 nA increments) — tick every one, consistently.
CURRENT_STEP = 0.01
ALL_PANELS_Y_LIMITS = (-35, 0)
ALL_PANELS_Y_TICK_STEP = 5


def _style_axes(ax, panel_label, label_offset_x=-0.2):
    ax.text(label_offset_x, 1.15, panel_label, transform=ax.transAxes,
            fontsize=28, fontweight='bold', fontfamily='Calibri', va='top', ha='left')
    ax.set_xlabel('Current (nA)', fontsize=22, fontweight='bold', fontfamily='Calibri')
    ax.set_ylabel('ΔV (mV)', fontsize=22, fontweight='bold', fontfamily='Calibri')
    ax.xaxis.set_major_locator(MultipleLocator(CURRENT_STEP))
    ax.set_ylim(*ALL_PANELS_Y_LIMITS)
    ax.yaxis.set_major_locator(MultipleLocator(ALL_PANELS_Y_TICK_STEP))
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5)
    ax.tick_params(axis='x', labelrotation=45)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def plot_single_cell_panel(ax, cell_type, panel_label, cell_index=21, show_legend=False):
    """Panel A/B — single-cell I-V curve across the 4 KIR conditions."""
    print(f"\n=== [Panel {panel_label}] Single-cell I-V — {cell_type.upper()} #{cell_index} ===")
    legend_entries = []
    for cond_name, kwargs in CONDITION_SPECS:
        t0 = time.time()
        result = calculate_rectification(cell_type=cell_type, cell_index=cell_index, **kwargs)
        print(f"  ✓ '{cond_name}' done in {time.time() - t0:.1f}s")

        color = CONDITION_COLORS.get(cond_name, '#333333')
        me = MARKEVERY_MAP.get(cond_name, None)
        data = result['analysis_data']
        ax.plot(data['currents'], data['delta_v_values'],
                'o-', color=color, linewidth=2, markersize=6,
                **(dict(markevery=me) if me is not None else {}))
        legend_entries.append(plt.Line2D([0], [0], color=color, linewidth=2,
                                          marker='o', markersize=6, label=cond_name))

    _style_axes(ax, panel_label)
    if show_legend:
        ax.legend(handles=legend_entries, loc='lower right', fontsize=14,
                  prop={'family': 'Calibri', 'weight': 'bold', 'size': 14},
                  framealpha=0.95, edgecolor='gray', fancybox=False)


# Reported one-way ANOVA result across the 4 conditions (Control, 1.25x KIR,
# 1.5x KIR, FM550), shared by panels C and D.
ANOVA_ANNOTATION = 'p < 0.001'


def plot_population_panel(ax, cell_type, panel_label):
    """Panel C/D — population mean ± SEM I-V curve from existing raw CSVs."""
    csv_path = RAW_CSV[cell_type]
    print(f"\n=== [Panel {panel_label}] Population I-V — {cell_type.upper()} ===")
    print(f"  Reading {csv_path}")
    df = pd.read_csv(csv_path)
    df['condition'] = df['condition'].replace(CONDITION_ALIASES)

    current_cols = [c for c in df.columns if c.startswith('current_')]
    currents = np.array([float(c.replace('current_', '').replace('nA', ''))
                         for c in current_cols])

    conditions_in_file = df['condition'].unique().tolist()
    ordered = [c for c in CONDITION_ORDER if c in conditions_in_file] + \
              [c for c in conditions_in_file if c not in CONDITION_ORDER]

    for cond_name in ordered:
        subset = df[df['condition'] == cond_name][current_cols].values
        n = subset.shape[0]
        mean = np.mean(subset, axis=0)
        sem = np.std(subset, axis=0, ddof=1) / np.sqrt(n)
        color = CONDITION_COLORS.get(cond_name, '#333333')

        ax.errorbar(
            currents, mean, yerr=sem,
            fmt='o-', color=color, linewidth=2, markersize=5,
            capsize=4, capthick=1.5, elinewidth=1.5,
            label=f'{cond_name} (n={n})',
        )

        _style_axes(ax, panel_label)
    ax.text(0.97, 0.03, ANOVA_ANNOTATION, transform=ax.transAxes,
            fontsize=22, fontweight='bold', fontfamily='Calibri',
            va='bottom', ha='right')


if __name__ == '__main__':
    print("=" * 60)
    print("FIGURE 3 — COMBINED PANELS A, B, C, D")
    print("=" * 60)
    overall_start = time.time()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    plot_single_cell_panel(axes[0, 0], 'dmsn', 'A', show_legend=True)
    plot_single_cell_panel(axes[0, 1], 'imsn', 'B')
    plot_population_panel(axes[1, 0], 'dmsn', 'C')
    plot_population_panel(axes[1, 1], 'imsn', 'D')

    plt.tight_layout()
    fig.savefig(os.path.join(HERE, 'Figure3_AllPanels.png'), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as Figure3_AllPanels.png")

    plt.show()

    print(f"\nTotal elapsed time: {time.time() - overall_start:.1f}s")
