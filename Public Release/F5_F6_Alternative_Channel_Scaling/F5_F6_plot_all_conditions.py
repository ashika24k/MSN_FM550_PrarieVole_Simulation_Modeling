"""
Figure 5 & 6 — Alternative Channel Population Graph (2×2 Grid)
All four conditions: DMSN 1.25, DMSN 1.5, IMSN 1.25, IMSN 1.5

Panel layout:
  A (top-left):    DMSN 1.25× KIR scaling
  B (top-right):   DMSN 1.5× KIR scaling
  C (bottom-left):  IMSN 1.25× KIR scaling
  D (bottom-right): IMSN 1.5× KIR scaling

Each panel:
  - Strip plot with jittered data points
  - Mean lines for each channel
  - One-way ANOVA + Tukey HSD post-hoc
  - Significance brackets and stars
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Statistics ---
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# =============================================================================
# SECTION 2 — GLOBAL CONFIGURATION
# =============================================================================

plt.rcParams['font.family']      = 'Calibri'
plt.rcParams['font.weight']      = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size']        = 16
plt.rcParams['axes.titlesize']   = 20
plt.rcParams['axes.labelsize']   = 18
plt.rcParams['xtick.labelsize']  = 14
plt.rcParams['ytick.labelsize']  = 14

ALPHA = 0.05

# =============================================================================
# SECTION 3 — LOAD ALL 4 CSVS
# =============================================================================

HERE = os.path.dirname(__file__)

CSV_FILES = {
    'A': ('Figure5&6_Input_IR_DMSN_1.25.csv',  'DMSN 1.25×'),
    'B': ('Figure5&6_Input_IR_DMSN_1.5.csv',   'DMSN 1.5×'),
    'C': ('Figure5&6_Input_IR_IMSN_1.25.csv',  'IMSN 1.25×'),
    'D': ('Figure5&6_Input_IR_IMSN_1.5.csv',   'IMSN 1.5×'),
}

data_all = {}
for panel, (csv_file, panel_title) in CSV_FILES.items():
    csv_path = os.path.join(HERE, csv_file)
    df = pd.read_csv(csv_path)
    
    data_cols = ['FM550', 'kir', 'kaf', 'kas', 'kdr', 'naf', 'sk', 'bk']
    data = {col: df[col].dropna().astype(float).values for col in data_cols}
    
    data_all[panel] = {
        'data': data,
        'csv_file': csv_file,
        'panel_title': panel_title,
    }

# =============================================================================
# SECTION 4 — CREATE 2×2 SUBPLOT GRID
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()  # Convert to 1D array for easier iteration

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

def _stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# Store results for export
anova_results = []
tukey_results = []

# Plot each panel
for panel_idx, (panel_letter, ax) in enumerate(zip(['A', 'B', 'C', 'D'], axes)):
    
    # Get data for this panel
    panel_data = data_all[panel_letter]
    data = panel_data['data']
    panel_title = panel_data['panel_title']
    
    # Build arrays for ANOVA
    group_arrays = [data[col] for col in data_cols]
    
    # ---- Run ANOVA ----
    F_stat, p_anova = stats.f_oneway(*group_arrays)
    significant = p_anova < ALPHA
    
    # Store ANOVA result
    anova_results.append({
        'Panel': panel_letter,
        'Condition': panel_title,
        'F_statistic': F_stat,
        'p_value': p_anova,
        'Significant': 'Yes' if significant else 'No'
    })
    
    # ---- Run Tukey HSD if significant ----
    sig_vs_fm550 = {}
    if significant:
        # Build data for Tukey HSD
        all_vals = []
        all_groups = []
        for col, label in zip(data_cols, labels):
            vals = data[col]
            all_vals.extend(vals)
            all_groups.extend([label] * len(vals))
        
        tukey_result = pairwise_tukeyhsd(all_vals, all_groups, alpha=ALPHA)
        tukey_df = pd.DataFrame(data=tukey_result.summary().data[1:], 
                                columns=tukey_result.summary().data[0])
        
        for _, row in tukey_df.iterrows():
            p_adj = float(row['p-adj'])
            group1 = row['group1']
            group2 = row['group2']
            meandiff = float(row['meandiff'])
            
            # Store ALL Tukey results (significant and non-significant)
            tukey_results.append({
                'Panel': panel_letter,
                'Condition': panel_title,
                'Group1': group1,
                'Group2': group2,
                'MeanDiff': meandiff,
                'p_adjusted': p_adj,
                'Significant': 'Yes' if p_adj < ALPHA else 'No'
            })
            
            # Only include for bracket drawing if actually significant
            if p_adj < ALPHA:
                if group1 == 'FM550':
                    sig_vs_fm550[group2] = p_adj
                elif group2 == 'FM550':
                    sig_vs_fm550[group1] = p_adj
    
    # ---- Plot data points and means ----
    y_vals_all = np.concatenate(list(data.values()))
    y_max_all = y_vals_all.max()
    y_min_all = y_vals_all.min()
    
    rng = np.random.default_rng(42)
    
    for x_pos, (col, label, color) in enumerate(zip(data_cols, labels, colors)):
        vals = data[col]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        
        ax.scatter(x_pos + jitter, vals,
                   color=color, alpha=1.0, s=30,
                   linewidths=0, zorder=3)
        
        ax.hlines(np.mean(vals), x_pos - 0.22, x_pos + 0.22,
                  colors='black', linewidths=1.5, zorder=5)
    
    # ---- Significance brackets ----
    data_range = y_max_all - y_min_all
    label_to_x = {lbl: i for i, lbl in enumerate(labels)}
    fm550_x = label_to_x['FM550']
    
    fm550_brackets = sorted(
        [(label_to_x[lbl], p_adj) for lbl, p_adj in sig_vs_fm550.items()],
        key=lambda t: abs(t[0] - fm550_x)
    )
    
    bracket_step = data_range * 0.07
    tick_h = bracket_step * 0.30
    bracket_gap = data_range * 0.04
    
    # Only draw brackets if there are significant differences
    if significant and len(fm550_brackets) > 0:
        for bracket_idx, (other_x, p_adj) in enumerate(fm550_brackets):
            x1, x2 = sorted([fm550_x, other_x])
            y_bracket = y_max_all + bracket_gap + bracket_idx * bracket_step
            
            ax.plot([x1, x2], [y_bracket, y_bracket],
                    color='black', linewidth=1.2, clip_on=False)
            ax.plot([x1, x1], [y_bracket - tick_h, y_bracket],
                    color='black', linewidth=1.2, clip_on=False)
            ax.plot([x2, x2], [y_bracket - tick_h, y_bracket],
                    color='black', linewidth=1.2, clip_on=False)
            ax.text((x1 + x2) / 2, y_bracket - 0.5, _stars(p_adj),
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    fontfamily='Calibri')
    
    # ---- Axes formatting ----
    ax.set_xticks(range(len(data_cols)))
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_xlabel('Channel Type', labelpad=10, fontsize=16, fontweight='bold')
    ax.set_ylabel('% Inward Rectification', labelpad=10, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.2, length=4)
    
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.55, len(data_cols) - 0.45)
    
    # Extended y-axis for brackets
    y_pad = max(data_range * 0.45, 3.0)
    ax.set_ylim(bottom=y_min_all - y_pad, top=y_max_all + y_pad)
    
    # Panel letter
    ax.text(-0.15, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=28, fontweight='bold', fontfamily='Calibri', va='top', ha='left')

# =============================================================================
# SECTION 5 — EXPORT RESULTS
# =============================================================================

# Export ANOVA results
anova_df = pd.DataFrame(anova_results)
anova_csv = os.path.join(HERE, 'Figure5&6_ANOVA_Results.csv')
anova_df.to_csv(anova_csv, index=False)

# Export Tukey HSD results
tukey_df_export = pd.DataFrame(tukey_results)
tukey_csv = os.path.join(HERE, 'Figure5&6_Tukey_HSD_Results.csv')
tukey_df_export.to_csv(tukey_csv, index=False)

print(f"\nResults exported:")
print(f"  {anova_csv}")
print(f"  {tukey_csv}")

# ---- Overall layout ----
plt.tight_layout(rect=[0, 0, 1, 0.99])

# ---- Save ----
out_png = os.path.join(HERE, 'Figure5&6_AllConditions_2x2Grid.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')

print(f"Figure saved as:")
print(f"  {out_png}")

plt.show()
