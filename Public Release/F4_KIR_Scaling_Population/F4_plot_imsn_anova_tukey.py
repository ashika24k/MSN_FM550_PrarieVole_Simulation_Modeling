"""
Figure 4 — KIR Scaling Population Graph
Extended with: larger fonts, 1-way ANOVA, Tukey HSD post-hoc

Based on: [2.25.2025] Graphing.py
Changes:
  - Font sizes increased throughout for publication readability
  - Paired t-tests replaced by a single one-way ANOVA (scipy)
  - Tukey HSD post-hoc (statsmodels) added when ANOVA is significant
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
plt.rcParams['legend.fontsize']  = 16

ALPHA = 0.05   # significance threshold

# =============================================================================
# SECTION 3 — LOAD DATA
# =============================================================================

csv_path = os.path.join(os.path.dirname(__file__),
                        'Figure4_Input_IR_IMSN.csv')
df = pd.read_csv(csv_path)

# Column → display label & color
columns = ['control_ir', 'fm550_ir', 'kir125x_ir', 'kir15x_ir']
labels  = ['Control', 'FM550', '1.25\u00d7 KIR', '1.5\u00d7 KIR']   # × via unicode
colors  = ['#285C81', '#9D2222', '#F57C00', '#2E7D32']

# =============================================================================
# SECTION 4 — STATISTICAL ANALYSIS  (1-way ANOVA + Tukey HSD)
# =============================================================================

print("=" * 65)
print("SECTION 4 — STATISTICAL ANALYSIS")
print("=" * 65)

# ---- Build per-group arrays ------------------------------------------------
group_arrays = [df[col].dropna().values for col in columns]

# ---- Descriptive stats -----------------------------------------------------
print(f"\nDescriptive statistics")
print(f"{'Group':<16}  {'n':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
print("-" * 58)
for label, arr in zip(labels, group_arrays):
    print(f"{label:<16}  {len(arr):>4}  {np.mean(arr):>8.2f}  "
          f"{np.std(arr, ddof=1):>8.2f}  {np.min(arr):>8.2f}  {np.max(arr):>8.2f}")

# ---- 1-Way ANOVA -----------------------------------------------------------
print(f"\n--- One-Way ANOVA ---")
print(f"H0 : All group means are equal")
print(f"H1 : At least one group mean differs")

F_stat, p_anova = stats.f_oneway(*group_arrays)
significant = p_anova < ALPHA
p_str = f"p = {p_anova:.4g}" if p_anova >= 0.001 else "p < 0.001"

# Degrees of freedom
k        = len(group_arrays)                          # number of groups
N_total  = sum(len(a) for a in group_arrays)          # total observations
df_between = k - 1                                    # numerator df
df_within  = N_total - k                              # denominator df

print(f"\n  F-statistic       : {F_stat:.4f}")
print(f"  df (between)      : {df_between}   (k - 1, k = {k} groups)")
print(f"  df (within/error) : {df_within}   (N - k, N = {N_total} obs.)")
print(f"  F({df_between}, {df_within}) = {F_stat:.4f}")
print(f"  p-value           : {p_anova:.4g}")
print(f"  alpha             : {ALPHA}")
print(f"  Result            : {'SIGNIFICANT — reject H0' if significant else 'NOT significant — fail to reject H0'}")

# ---- Tukey HSD post-hoc ----------------------------------------------------
tukey_df = None
sig_pairs = []

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

    print(f"\n{'Pair':<38}  {'mean diff':>10}  {'p-adj':>10}  Significant")
    print("-" * 68)
    for _, row in tukey_df.iterrows():
        sig_flag = "YES *" if row['reject'] else "no"
        print(f"{row['group1']} vs {row['group2']:<28}  "
              f"{float(row['meandiff']):>10.3f}  "
              f"{float(row['p-adj']):>10.4f}  {sig_flag}")
        if row['reject']:
            sig_pairs.append((row['group1'], row['group2'], float(row['p-adj'])))
else:
    print("\n  Post-hoc test skipped (ANOVA not significant).")

print("=" * 65)

# =============================================================================
# SECTION 5 — SAVE STATISTICS RESULTS TO CSV
# =============================================================================

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
    'significance': ('***' if p_anova < 0.001 else
                     '**'  if p_anova < 0.01  else
                     '*'   if p_anova < 0.05  else 'ns'),
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
            'significance': ('***' if p_adj < 0.001 else
                             '**'  if p_adj < 0.01  else
                             '*'   if p_adj < 0.05  else 'ns'),
        })

# ---- Build ANOVA summary DataFrame (mirrors terminal output) ---------------
anova_summary_df = pd.DataFrame([
    {'Field': 'H0',               'Value': 'All group means are equal'},
    {'Field': 'H1',               'Value': 'At least one group mean differs'},
    {'Field': 'F-statistic',      'Value': round(F_stat, 4)},
    {'Field': 'df (between)',     'Value': f'{df_between}  (k - 1, k = {k} groups)'},
    {'Field': 'df (within/error)','Value': f'{df_within}  (N - k, N = {N_total} obs.)'},
    {'Field': 'F notation',       'Value': f'F({df_between}, {df_within}) = {round(F_stat, 4)}'},
    {'Field': 'p-value',          'Value': round(p_anova, 6)},
    {'Field': 'alpha',            'Value': ALPHA},
    {'Field': 'Result',           'Value': 'SIGNIFICANT — reject H0' if significant else 'NOT significant — fail to reject H0'},
])

# ---- Build descriptive stats DataFrame ------------------------------------
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

# ---- Build clean Tukey sheet matching terminal output ----------------------
if tukey_df is not None:
    tukey_export_df = pd.DataFrame({
        'Pair':        tukey_df['group1'] + ' vs ' + tukey_df['group2'],
        'mean diff':   tukey_df['meandiff'].astype(float).round(4),
        'p-adj':       tukey_df['p-adj'].astype(float).round(6),
        'Significant': tukey_df['reject'].map({True: 'YES *', False: 'no'}),
    })
else:
    tukey_export_df = pd.DataFrame(columns=['Pair', 'mean diff', 'p-adj', 'Significant'])

# ---- Single XLSX with 3 sheets ---------------------------------------------
xlsx_out_path = os.path.join(os.path.dirname(__file__), 'Figure4_Output_ANOVA_TukeyHSD_Results.xlsx')
with pd.ExcelWriter(xlsx_out_path, engine='openpyxl') as writer:
    desc_df.to_excel(writer,          sheet_name='Descriptive Stats', index=False)
    anova_summary_df.to_excel(writer, sheet_name='ANOVA Summary',     index=False)
    tukey_export_df.to_excel(writer,  sheet_name='Tukey HSD',         index=False)
print(f"XLSX saved to: {xlsx_out_path}")

# =============================================================================
# SECTION 6 — MAIN POPULATION STRIP-PLOT  (with ANOVA + Tukey annotations)
# =============================================================================

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


fig, ax = plt.subplots(figsize=(8, 8.5))
rng = np.random.default_rng(42)

y_max_all = df[columns].max().max()
y_min_all = df[columns].min().min()

for x_pos, (col, label, color) in enumerate(zip(columns, labels, colors)):
    vals   = df[col].dropna().values
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))

    ax.scatter(x_pos + jitter, vals,
               color=color, alpha=1.0, s=35,
               linewidths=0, zorder=3)

    ax.hlines(np.mean(vals), x_pos - 0.22, x_pos + 0.22,
              colors='black', linewidths=1.5, zorder=5)

# ---- Tukey HSD significance brackets (condensed, sorted by span) ----------
label_to_x = {lbl: i for i, lbl in enumerate(labels)}

# Sort shortest span first — keeps lower brackets narrow, reduces crossings
sorted_pairs = sorted(sig_pairs,
                      key=lambda t: abs(label_to_x[t[1]] - label_to_x[t[0]]))

n_b          = len(sorted_pairs)
data_range   = y_max_all - y_min_all
bracket_step = max(6, data_range * 0.07)   # tighter than before
tick_h       = bracket_step * 0.25

for bracket_idx, (g1, g2, p_adj) in enumerate(sorted_pairs):
    x1 = label_to_x[g1]
    x2 = label_to_x[g2]
    xl, xr    = sorted([x1, x2])
    y_bracket = y_max_all + bracket_step * 0.4 + bracket_idx * bracket_step

    # Horizontal bar
    ax.plot([xl, xr], [y_bracket, y_bracket],
            color='black', linewidth=1.2, clip_on=False)
    # Vertical ticks
    ax.plot([xl, xl], [y_bracket - tick_h, y_bracket],
            color='black', linewidth=1.2, clip_on=False)
    ax.plot([xr, xr], [y_bracket - tick_h, y_bracket],
            color='black', linewidth=1.2, clip_on=False)
    # Stars
    ax.text((xl + xr) / 2, y_bracket + bracket_step * 0.05,
            sig_stars(p_adj),
            ha='center', va='bottom', fontsize=16, fontweight='bold',
            fontfamily='Calibri')

# ---- Axes formatting -------------------------------------------------------
ax.set_xticks(range(len(columns)))
ax.set_xticklabels(labels, fontsize=18, fontweight='bold', fontfamily='Calibri')
ax.set_xlabel('Simulation Group', labelpad=12,
              fontsize=22, fontweight='bold', fontfamily='Calibri')
ax.set_ylabel('Input Resistance (MΩ)', labelpad=12,
              fontsize=22, fontweight='bold', fontfamily='Calibri')
ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5)
for sp in ax.spines.values():
    sp.set_linewidth(1.5)

# Panel label
ax.text(-0.2, 1.15, 'B', transform=ax.transAxes,
        fontsize=28, fontweight='bold', fontfamily='Calibri', va='top', ha='left')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-0.55, len(columns) - 0.45)

n_brackets = max(len(sorted_pairs), 1)
ax.set_ylim(bottom=None,
            top=y_max_all + bracket_step * 0.4 + n_brackets * bracket_step + bracket_step * 0.6)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 7 — PRINT STATISTICS SUMMARY TABLE TO TERMINAL
# =============================================================================

# Column widths
CW = {'comparison': 34, 'test': 14, 'df_b': 12, 'df_w': 12, 'f': 12, 'p': 12, 'sig': 6}
total_w = sum(CW.values()) + len(CW) - 1

header = (f"{'Comparison':<{CW['comparison']}} "
          f"{'Test':<{CW['test']}} "
          f"{'df(between)':>{CW['df_b']}} "
          f"{'df(within)':>{CW['df_w']}} "
          f"{'F-stat':>{CW['f']}} "
          f"{'p-value':>{CW['p']}} "
          f"{'Sig.':>{CW['sig']}}")

print("\n" + "=" * total_w)
print("STATISTICAL RESULTS SUMMARY — One-Way ANOVA + Tukey HSD")
print("=" * total_w)
print(header)
print("-" * total_w)

for row in stats_rows:
    df_b_str = str(row['df_between']) if row['df_between'] is not None else '—'
    df_w_str = str(row['df_within'])  if row['df_within']  is not None else '—'
    f_str    = f"{row['F_statistic']:.4f}" if row['F_statistic'] is not None else '—'
    p_str_r  = f"{row['p_value']:.4e}"
    print(f"{row['Comparison']:<{CW['comparison']}} "
          f"{row['Test']:<{CW['test']}} "
          f"{df_b_str:>{CW['df_b']}} "
          f"{df_w_str:>{CW['df_w']}} "
          f"{f_str:>{CW['f']}} "
          f"{p_str_r:>{CW['p']}} "
          f"{row['significance']:>{CW['sig']}}")

print("=" * total_w)
print("Sig. key:  *** p<0.001   ** p<0.01   * p<0.05   ns = not significant")
print("=" * total_w + "\n")
