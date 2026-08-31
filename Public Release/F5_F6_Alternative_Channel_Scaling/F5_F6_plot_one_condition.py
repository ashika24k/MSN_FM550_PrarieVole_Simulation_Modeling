"""
Figure 5 — Alternative Channel Population Graph
Extended with: larger fonts, 1-way ANOVA, Tukey HSD post-hoc, no legend

Based on: [2.2.5.2025] Graphing Pop.py
Changes:
  - Font sizes increased throughout for publication readability
  - Legend removed (x-axis labels already identify each channel)
  - Paired t-tests replaced by one-way ANOVA (scipy)
  - Tukey HSD post-hoc (statsmodels) when ANOVA is significant
  - Stars above columns show significance vs FM550 (Tukey p-adj)
  - Script divided into clearly labeled sections
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Statistics ---
from scipy import stats                                    # 1-way ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # Tukey HSD

# =============================================================================
# SECTION 2 — GLOBAL CONFIGURATION  (fonts, colors)
# =============================================================================

plt.rcParams['font.family']      = 'Calibri'
plt.rcParams['font.weight']      = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size']        = 18
plt.rcParams['axes.titlesize']   = 24
plt.rcParams['axes.labelsize']   = 22
plt.rcParams['xtick.labelsize']  = 18
plt.rcParams['ytick.labelsize']  = 18

ALPHA = 0.05   # significance threshold

# =============================================================================
# SECTION 3 — LOAD DATA
# =============================================================================

# ---- Change this line to switch between the four CSVs: ----------------------
#   'Figure5&6_Input_IR_IMSN_1.25.csv'
#   'Figure5&6_Input_IR_IMSN_1.5.csv'
#   'Figure5&6_Input_IR_DMSN_1.25.csv'
#   'Figure5&6_Input_IR_DMSN_1.5.csv'
CSV_FILE = 'Figure5&6_Input_IR_IMSN_1.25.csv'
# -----------------------------------------------------------------------------

csv_path = os.path.join(os.path.dirname(__file__), CSV_FILE)
df_raw   = pd.read_csv(csv_path)

# Column names in csv → display labels & colors
data_cols = ['FM550', 'kir', 'kaf', 'kas', 'kdr', 'naf', 'sk', 'bk']
labels    = ['FM550', 'KIR', 'KAF', 'KAS', 'KDR', 'NAF', 'SK', 'BK']
colors    = [
    '#9D2222',  # FM550  — dark red
    '#285C81',  # KIR    — dark blue
    '#296E6C',  # KAF    — dark teal
    '#2B5736',  # KAS    — dark green
    '#C9AE50',  # KDR    — gold
    '#B06F1F',  # NAF    — dark orange
    '#C0436C',  # SK     — dark pink
    '#4F2465',  # BK     — dark purple
]

# Build clean per-group arrays (drop NaN, convert to float)
data = {col: df_raw[col].dropna().astype(float).values for col in data_cols}

# =============================================================================
# SECTION 4 — STATISTICAL ANALYSIS  (1-way ANOVA + Tukey HSD)
# =============================================================================

print("=" * 65)
print("SECTION 4 — STATISTICAL ANALYSIS")
print(f"Dataset: {CSV_FILE}")
print("=" * 65)

group_arrays = [data[col] for col in data_cols]

# ---- Descriptive stats -----------------------------------------------------
print(f"\nDescriptive statistics")
print(f"{'Group':<8}  {'n':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
print("-" * 50)
for label, arr in zip(labels, group_arrays):
    print(f"{label:<8}  {len(arr):>4}  {np.mean(arr):>8.2f}  "
          f"{np.std(arr, ddof=1):>8.2f}  {np.min(arr):>8.2f}  {np.max(arr):>8.2f}")

# ---- 1-Way ANOVA -----------------------------------------------------------
print(f"\n--- One-Way ANOVA ---")
print(f"H0 : All group means are equal")
print(f"H1 : At least one group mean differs")

F_stat, p_anova = stats.f_oneway(*group_arrays)
significant = p_anova < ALPHA
p_str = f"p = {p_anova:.4g}" if p_anova >= 0.001 else "p < 0.001"

# Degrees of freedom
k          = len(group_arrays)
N_total    = sum(len(a) for a in group_arrays)
df_between = k - 1
df_within  = N_total - k

print(f"\n  F-statistic       : {F_stat:.4f}")
print(f"  df (between)      : {df_between}   (k - 1, k = {k} groups)")
print(f"  df (within/error) : {df_within}   (N - k, N = {N_total} obs.)")
print(f"  F({df_between}, {df_within}) = {F_stat:.4f}")
print(f"  p-value           : {p_anova:.4g}")
print(f"  alpha             : {ALPHA}")
print(f"  Result            : {'SIGNIFICANT — reject H0' if significant else 'NOT significant — fail to reject H0'}")

# ---- Tukey HSD post-hoc ----------------------------------------------------
tukey_df  = None
# sig_vs_fm550: dict of label -> Tukey p-adj for the FM550 vs that group pair
sig_vs_fm550 = {}

if significant:
    print(f"\n--- Tukey HSD Post-Hoc (pairwise, corrected) ---")

    all_values = np.concatenate(group_arrays)
    all_labels = np.concatenate([[lbl] * len(arr)
                                 for lbl, arr in zip(labels, group_arrays)])

    tukey_result = pairwise_tukeyhsd(endog=all_values, groups=all_labels, alpha=ALPHA)

    tukey_df = pd.DataFrame(
        data    = tukey_result._results_table.data[1:],
        columns = tukey_result._results_table.data[0],
    )
    tukey_df.columns = ['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper', 'reject']

    print(f"\n{'Pair':<25}  {'mean diff':>10}  {'p-adj':>10}  Significant")
    print("-" * 55)
    for _, row in tukey_df.iterrows():
        sig_flag = "YES *" if row['reject'] else "no"
        print(f"{row['group1']} vs {row['group2']:<15}  "
              f"{float(row['meandiff']):>10.3f}  "
              f"{float(row['p-adj']):>10.4f}  {sig_flag}")

        # Collect pairs that involve FM550 and are significant
        p_adj = float(row['p-adj'])
        if row['reject']:
            if row['group1'] == 'FM550':
                sig_vs_fm550[row['group2']] = p_adj
            elif row['group2'] == 'FM550':
                sig_vs_fm550[row['group1']] = p_adj
else:
    print("\n  Post-hoc test skipped (ANOVA not significant).")

print("=" * 65)

# =============================================================================
# SECTION 5 — SAVE STATISTICS TO CSV
# =============================================================================

def _stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

stats_rows = [{
    'Test': 'One-Way ANOVA',
    'Comparison': 'All groups',
    'H0': 'All group means are equal',
    'H1': 'At least one group mean differs',
    'df_between': df_between,
    'df_within': df_within,
    'F_statistic': round(F_stat, 4),
    'F_notation': f'F({df_between}, {df_within}) = {round(F_stat, 4)}',
    'p_value': round(p_anova, 6),
    'alpha': ALPHA,
    'result': 'SIGNIFICANT — reject H0' if significant else 'NOT significant — fail to reject H0',
    'significance': _stars(p_anova),
}]

if tukey_df is not None:
    for _, row in tukey_df.iterrows():
        p_adj = float(row['p-adj'])
        stats_rows.append({
            'Test': 'Tukey HSD',
            'Comparison': f"{row['group1']} vs {row['group2']}",
            'H0': '',
            'H1': '',
            'df_between': None,
            'df_within': df_within,
            'F_statistic': None,
            'F_notation': '',
            'p_value': round(p_adj, 6),
            'alpha': ALPHA,
            'result': 'significant' if p_adj < ALPHA else 'not significant',
            'significance': _stars(p_adj),
        })

stats_out = os.path.join(os.path.dirname(__file__), 'Fig5_significance_results_stats.csv')
pd.DataFrame(stats_rows).to_csv(stats_out, index=False)
print(f"Statistics saved to: {stats_out}")

# ---- Build export DataFrames -----------------------------------------------
desc_rows = []
for label, arr in zip(labels, group_arrays):
    desc_rows.append({
        'Group': label,
        'n':     len(arr),
        'mean':  round(float(np.mean(arr)), 4),
        'std':   round(float(np.std(arr, ddof=1)), 4),
        'min':   round(float(np.min(arr)), 4),
        'max':   round(float(np.max(arr)), 4),
    })
desc_df = pd.DataFrame(desc_rows)

anova_summary_df = pd.DataFrame([
    {'Field': 'H0',                'Value': 'All group means are equal'},
    {'Field': 'H1',                'Value': 'At least one group mean differs'},
    {'Field': 'F-statistic',       'Value': round(F_stat, 4)},
    {'Field': 'df (between)',      'Value': f'{df_between}  (k - 1, k = {k} groups)'},
    {'Field': 'df (within/error)', 'Value': f'{df_within}  (N - k, N = {N_total} obs.)'},
    {'Field': 'F notation',        'Value': f'F({df_between}, {df_within}) = {round(F_stat, 4)}'},
    {'Field': 'p-value',           'Value': round(p_anova, 6)},
    {'Field': 'alpha',             'Value': ALPHA},
    {'Field': 'Result',            'Value': 'SIGNIFICANT — reject H0' if significant else 'NOT significant — fail to reject H0'},
])

if tukey_df is not None:
    tukey_export_df = pd.DataFrame({
        'Pair':        tukey_df['group1'] + ' vs ' + tukey_df['group2'],
        'mean diff':   tukey_df['meandiff'].astype(float).round(4),
        'p-adj':       tukey_df['p-adj'].astype(float).round(6),
        'Significant': tukey_df['reject'].map({True: 'YES *', False: 'no'}),
    })
else:
    tukey_export_df = pd.DataFrame(columns=['Pair', 'mean diff', 'p-adj', 'Significant'])

xlsx_out = os.path.join(os.path.dirname(__file__), 'Fig5_results.xlsx')
with pd.ExcelWriter(xlsx_out, engine='openpyxl') as writer:
    desc_df.to_excel(writer,          sheet_name='Descriptive Stats', index=False)
    anova_summary_df.to_excel(writer, sheet_name='ANOVA Summary',     index=False)
    tukey_export_df.to_excel(writer,  sheet_name='Tukey HSD',         index=False)
print(f"XLSX saved to: {xlsx_out}")

# =============================================================================
# SECTION 6 — MAIN POPULATION STRIP-PLOT  (with ANOVA box + Tukey stars)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 7))
rng = np.random.default_rng(42)

y_vals_all = np.concatenate(list(data.values()))
y_max_all  = y_vals_all.max()
y_min_all  = y_vals_all.min()

for x_pos, (col, label, color) in enumerate(zip(data_cols, labels, colors)):
    vals   = data[col]
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))

    ax.scatter(x_pos + jitter, vals,
               color=color, alpha=1.0, s=35,
               linewidths=0, zorder=3)   # no label= so no legend entry

    ax.hlines(np.mean(vals), x_pos - 0.22, x_pos + 0.22,
              colors='black', linewidths=1.5, zorder=5)

# ---- Tukey HSD significance brackets (FM550 vs each sig. different group) --
# Use a fixed step based on data range so brackets are always readable.
data_range = y_max_all - y_min_all
label_to_x = {lbl: i for i, lbl in enumerate(labels)}
fm550_x    = label_to_x['FM550']

# Sort shortest-span first to minimise crossings
fm550_brackets = sorted(
    [(label_to_x[lbl], p_adj) for lbl, p_adj in sig_vs_fm550.items()],
    key=lambda t: abs(t[0] - fm550_x)
)

n_b          = len(fm550_brackets)
bracket_step = data_range * 0.07          # 7% of data range per bracket
tick_h       = bracket_step * 0.30
bracket_gap  = data_range * 0.04          # gap above highest data point

for bracket_idx, (other_x, p_adj) in enumerate(fm550_brackets):
    x1, x2   = sorted([fm550_x, other_x])
    y_bracket = y_max_all + bracket_gap + bracket_idx * bracket_step

    # Horizontal bar
    ax.plot([x1, x2], [y_bracket, y_bracket],
            color='black', linewidth=1.2, clip_on=False)
    # Vertical ticks
    ax.plot([x1, x1], [y_bracket - tick_h, y_bracket],
            color='black', linewidth=1.2, clip_on=False)
    ax.plot([x2, x2], [y_bracket - tick_h, y_bracket],
            color='black', linewidth=1.2, clip_on=False)
    # Stars
    ax.text((x1 + x2) / 2, y_bracket + bracket_step * 0.35, _stars(p_adj),
            ha='center', va='bottom', fontsize=16, fontweight='bold',
            fontfamily='Calibri')

# ---- Axes formatting --------------------------------------------------------
ax.set_xticks(range(len(data_cols)))
ax.set_xticklabels(labels, fontsize=18, fontweight='bold', fontfamily='Calibri')
ax.set_xlabel('Channel Type', labelpad=12,
              fontsize=22, fontweight='bold', fontfamily='Calibri')
ax.set_ylabel('% Inward Rectification', labelpad=12,
              fontsize=22, fontweight='bold', fontfamily='Calibri')
ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5)
for sp in ax.spines.values():
    sp.set_linewidth(1.5)

# Panel label
ax.text(-0.15, 1.1, 'A', transform=ax.transAxes,
        fontsize=28, fontweight='bold', fontfamily='Calibri', va='top', ha='left')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-0.55, len(data_cols) - 0.45)

# Extended y-axis limits to accommodate significance brackets
y_pad = max(data_range * 0.45, 3.0)
ax.set_ylim(bottom=y_min_all - y_pad, top=y_max_all + y_pad)

# No legend — x-axis labels already identify each channel

plt.tight_layout()
plt.show()
