import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os
from scipy import stats

# ── Font / style ──────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# ── Load data ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__),
                        'Figure4_Input_IR_DMSN.csv')
df = pd.read_csv(csv_path)

# ── Column → display label & color ───────────────────────────────────────────
columns = ['control_ir', 'fm550_ir', 'kir125x_ir', 'kir15x_ir']
labels  = ['Control', 'FM550', '1.25× KIR', '1.5× KIR']
colors  = ['#285C81', '#9D2222', '#F57C00', '#2E7D32']   # blue, red, orange, green

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7.5))

rng = np.random.default_rng(42)          # reproducible jitter

for x_pos, (col, label, color) in enumerate(zip(columns, labels, colors)):
    vals = df[col].dropna().values
    n    = len(vals)

    # jittered x positions
    jitter = rng.uniform(-0.18, 0.18, size=n)
    x_jit  = x_pos + jitter

    # individual data points
    ax.scatter(x_jit, vals,
               color=color, alpha=1.0, s=28,
               linewidths=0, zorder=3, label=label)

    # mean line (thin, black)
    mean = np.mean(vals)

    ax.hlines(mean, x_pos - 0.22, x_pos + 0.22,
              colors='black', linewidths=1.2, zorder=5)

# ── Significance annotations (paired t-test vs Control) ─────────────────────
def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

control_vals = df['control_ir'].dropna().values

sig_rows = []
y_max_all  = df[columns].max().max()
y_min_all  = df[columns].min().min()
data_range = y_max_all - y_min_all
bracket_step = max(12, data_range * 0.10)  # at least 12 MΩ, or 10% of range

for bracket_idx, (x_pos, col, label) in enumerate(zip(range(1, len(columns)), columns[1:], labels[1:])):
    comp_vals = df[col].dropna().values
    t_stat, p = stats.ttest_rel(control_vals, comp_vals)
    star = sig_stars(p)
    n_pairs = min(len(control_vals), len(comp_vals))
    sig_rows.append({
        'Comparison': f'Control vs {label}',
        'Test': 'Paired t-test',
        'n_pairs': n_pairs,
        't_statistic': round(t_stat, 4),
        'p_value': round(p, 6),
        'significance': star
    })

    # Bracket height: stagger so brackets don't overlap
    y_bracket = y_max_all + 10 + bracket_idx * bracket_step
    tick_h    = bracket_step * 0.3    # height of vertical ticks

    # Horizontal bar from x=0 (Control) to x=x_pos
    ax.plot([0, x_pos], [y_bracket, y_bracket], color='black', linewidth=1.2, clip_on=False)
    # Vertical tick at Control end
    ax.plot([0, 0], [y_bracket - tick_h, y_bracket], color='black', linewidth=1.2, clip_on=False)
    # Vertical tick at comparison end
    ax.plot([x_pos, x_pos], [y_bracket - tick_h, y_bracket], color='black', linewidth=1.2, clip_on=False)
    # Label above the midpoint of the bracket
    ax.text((0 + x_pos) / 2, y_bracket + 1.5, star,
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')

# ── Save significance results to CSV & display as table ──────────────────────
sig_df = pd.DataFrame(sig_rows)
sig_out = os.path.join(os.path.dirname(__file__), 'Figure4_Output_TTest_Results.csv')
sig_df.to_csv(sig_out, index=False)
print(f"Significance results saved to: {sig_out}")

# Render as a separate matplotlib table figure
col_labels = ['Comparison', 'Test', 'n', 't-stat', 'p-value', 'Sig.']
table_data = [
    [row['Comparison'], row['Test'], row['n_pairs'],
     f"{row['t_statistic']:.4f}", f"{row['p_value']:.2e}", row['significance']]
    for row in sig_rows
]

fig_tbl, ax_tbl = plt.subplots(figsize=(8, 1.4 + 0.45 * len(table_data)))
ax_tbl.axis('off')
tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.auto_set_column_width(col=list(range(len(col_labels))))

# Style header row
for col_idx in range(len(col_labels)):
    tbl[(0, col_idx)].set_facecolor('#285C81')
    tbl[(0, col_idx)].set_text_props(color='white', fontweight='bold')

# Alternating row shading
for row_idx in range(1, len(table_data) + 1):
    bg = '#f0f4f8' if row_idx % 2 == 0 else 'white'
    for col_idx in range(len(col_labels)):
        tbl[(row_idx, col_idx)].set_facecolor(bg)

fig_tbl.suptitle('Significance Testing Results (Paired t-test vs Control)',
                 fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(range(len(columns)))
ax.set_xticklabels(labels, fontsize=13)
ax.set_xlabel('Simulation Group', fontsize=13, labelpad=12)
ax.set_ylabel('Inward Rectification (MΩ)', fontsize=13, labelpad=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(-0.55, len(columns) - 0.45)
# headroom for stacked significance brackets (3 comparisons × bracket_step each)
ax.set_ylim(bottom=None, top=y_max_all + 10 + (len(columns) - 1) * bracket_step + 14)
ax.tick_params(axis='both', which='major', labelsize=11)

# legend – nudged down from upper right to stay below significance brackets
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.82), fontsize=11,
          frameon=True, framealpha=0.9, edgecolor='gray', fancybox=True)

plt.tight_layout()
plt.show()
