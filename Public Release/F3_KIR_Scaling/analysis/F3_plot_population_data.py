"""
IV_Kir_Scaling_Population_Plot.py

Reads the raw per-cell delta-V CSVs produced by IV_Kir_Scaling_Population.py
and plots mean ± SEM I-V curves with error bars (lines + caps, no shading).

Produces:
  IV_Kir_Scaling_Population_Errorbars_DMSN.pdf
  IV_Kir_Scaling_Population_Errorbars_IMSN.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style  (matches IV_Kir_Scaling.py)
# ---------------------------------------------------------------------------
# Global font settings
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
    'Female Vehicle (Control)': '#285C81',
    '1.25x KIR':                '#F57C00',
    '1.5x KIR':                 '#2E7D32',
    'Female FM550':             '#9D2222',
}

CONDITION_ORDER = [
    'Female Vehicle (Control)',
    '1.25x KIR',
    '1.5x KIR',
    'Female FM550',
]

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def plot_from_raw(cell_type, save_path=None):
    # CSVs are written by F3_generate_population_data.py.
    prefix = 'Figure3C' if cell_type == 'dmsn' else 'Figure3D'
    suffix = 'DMSN' if cell_type == 'dmsn' else 'IMSN'
    csv_path = os.path.join(HERE, 'generated_inputs', f'{prefix}_{suffix}_RAW.csv')
    df = pd.read_csv(csv_path)

    # Current values from column names
    current_cols = [c for c in df.columns if c.startswith('current_')]
    currents = np.array([float(c.replace('current_', '').replace('nA', ''))
                         for c in current_cols])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Respect display order; fall back to whatever is in the file
    conditions_in_file = df['condition'].unique().tolist()
    ordered = [c for c in CONDITION_ORDER if c in conditions_in_file] + \
              [c for c in conditions_in_file if c not in CONDITION_ORDER]

    cond_means = {}  # overall mean ΔV per condition (for percent diff printout)

    for cond_name in ordered:
        subset = df[df['condition'] == cond_name][current_cols].values
        n      = subset.shape[0]
        mean   = np.mean(subset, axis=0)
        sem    = np.std(subset, axis=0, ddof=1) / np.sqrt(n)
        color  = CONDITION_COLORS.get(cond_name, '#333333')
        cond_means[cond_name] = np.mean(mean)

        ax.errorbar(
            currents, mean,
            yerr=sem,
            fmt='o-',
            color=color,
            linewidth=2,
            markersize=5,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            label=f'{cond_name} (n={n})',
        )

    # Panel label: C for dMSN, D for IMSN
    panel_label = 'C' if cell_type == 'dmsn' else 'D'
    ax.text(-0.2, 1.15, panel_label, transform=ax.transAxes,
            fontsize=28, fontweight='bold', fontfamily='Calibri', va='top', ha='left')

    ax.set_xlabel('Current (nA)', fontsize=22, fontweight='bold', fontfamily='Calibri')
    ax.set_ylabel('ΔV (mV)',      fontsize=22, fontweight='bold', fontfamily='Calibri')
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=10,
              prop={'family': 'Calibri', 'weight': 'bold', 'size': 10},
              framealpha=0.95, edgecolor='gray', fancybox=False)

    plt.tight_layout()

    if save_path:
        # Convert .pdf to .png if needed
        png_path = save_path.replace('.pdf', '.png') if '.pdf' in save_path else save_path
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved → {png_path}')

    plt.show()

    # ------------------------------------------------------------------
    # Print overall percent difference from baseline
    # ------------------------------------------------------------------
    if len(cond_means) > 1:
        print(f'\n{cell_type.upper()} — RELATIVE CHANGES FROM BASELINE')
        print('=' * 60)
        baseline_cond = ordered[0]
        baseline_mean = cond_means[baseline_cond]
        print(f'  Baseline: {baseline_cond}  (mean ΔV = {baseline_mean:.2f} mV)')
        print('-' * 60)
        for cond_name in ordered[1:]:
            pct = ((cond_means[cond_name] - baseline_mean) / abs(baseline_mean)) * 100
            direction = '↑' if pct > 0 else '↓'
            print(f'  {cond_name:<28} {direction} {abs(pct):>6.1f}% change from baseline')
        print('=' * 60)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    for cell_type in ['dmsn', 'imsn']:
        plot_from_raw(
            cell_type,
            save_path=os.path.join(
                HERE,
                f'Figure3_IV_Kir_Scaling_Population_Errorbars_{cell_type.upper()}.png'
            )
        )
