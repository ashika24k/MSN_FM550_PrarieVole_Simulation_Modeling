import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from msn.cell import MSN
from msn.instrumentation import Stim
from neuron import h

# ─── Global font settings ────────────────────────────────────────────────────
plt.rcParams['font.family']  = 'Sans Serif'
plt.rcParams['font.weight']  = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize']   = 20
plt.rcParams['axes.titlesize']   = 22
plt.rcParams['xtick.labelsize']  = 18
plt.rcParams['ytick.labelsize']  = 18

TITLE_KW  = dict(fontsize=24, fontweight='bold', fontfamily='Sans Serif')
LABEL_KW  = dict(fontsize=22, fontweight='bold', fontfamily='Sans Serif')
TICK_SIZE = 18

dark_red  = "#9D2222"
dark_blue = "#285C81"

# ─── Create figure with three side-by-side subplots ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.subplots_adjust(wspace=0.40, left=0.07, right=0.97, top=0.88, bottom=0.15)

# ──────────────────────────────────────────────────────────────────────────────
# Panel A – single action-potential trace (dMSN, +0.05 nA + rheob)
# ──────────────────────────────────────────────────────────────────────────────
ax_a = axes[0]

cell_a = MSN('dmsn', 21)

stim_init = Stim(cell_a)
stim_init.set_stim(delay=10, duration=150, amplitude=0.0, tmax=200, add_rheob=False)
stim_init.run()
initial_voltage = float(np.mean(np.array(stim_init.v)[:100]))
print(f"[Panel A] Initial membrane voltage: {initial_voltage:.2f} mV")

stim_a = Stim(cell_a)
stim_a.set_stim(delay=10, duration=150, amplitude=0.05, tmax=200, add_rheob=True)
stim_a.run()

ax_a.plot(stim_a.t, stim_a.v, color=dark_red, linewidth=2.0)
ax_a.set_xlabel("Time (ms)", **LABEL_KW)
ax_a.set_ylabel("Membrane Potential (mV)", **LABEL_KW)
ax_a.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1.5, length=5)
for spine in ax_a.spines.values():
    spine.set_linewidth(1.5)

# Panel label
ax_a.text(-0.35, 1.15, 'A', transform=ax_a.transAxes,
          fontsize=28, fontweight='bold', fontfamily='Sans Serif', va='top', ha='left')

print("[Panel A] Done.")

# ──────────────────────────────────────────────────────────────────────────────
# Panel B – inward-rectification voltage traces (IMSN, negative currents)
# ──────────────────────────────────────────────────────────────────────────────
ax_b = axes[1]

cell_type_b = 'imsn'
cell_index_b = 32
v_init_b = -84.0

# Calculate holding current
stim_test = Stim(MSN(cell_type_b, cell_index_b, v_init=v_init_b))
stim_test.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
stim_test.run()
natural_eq = np.mean(np.array(stim_test.v)[-100:])
holding_current = (v_init_b - natural_eq) / 100.0

for iteration in range(10):
    cell_tmp = MSN(cell_type_b, cell_index_b, v_init=v_init_b)
    stim_tmp = Stim(cell_tmp)
    stim_tmp.set_stim(delay=0, duration=200, amplitude=holding_current, tmax=200, add_rheob=False)
    stim_tmp.run()
    natural_eq = np.mean(np.array(stim_tmp.v)[-100:])
    if abs(natural_eq - v_init_b) < 0.01:
        print(f"[Panel B] Holding current converged in {iteration + 1} iterations.")
        break
    holding_current += (v_init_b - natural_eq) / 100.0

print(f"[Panel B] Holding current: {holding_current:.4f} nA")

pre_equil = 200
delay_b   = 10
duration_b = 150
tmax_b    = pre_equil + delay_b + duration_b + 150

cell_b = MSN(cell_type_b, cell_index_b, v_init=v_init_b)

iclamp_hold = h.IClamp(cell_b.soma(0.5))
iclamp_hold.delay = 0
iclamp_hold.dur   = tmax_b
iclamp_hold.amp   = holding_current

iclamp_test = h.IClamp(cell_b.soma(0.5))
iclamp_test.delay = pre_equil + delay_b
iclamp_test.dur   = duration_b

v_rec = h.Vector(); t_rec = h.Vector()
v_rec.record(cell_b.soma(0.5)._ref_v)
t_rec.record(h._ref_t)

current_injections = []
voltage_changes    = []

for amplitude in np.arange(-0.01, -1.0, -0.10):
    iclamp_test.amp = amplitude
    h.tstop = tmax_b
    h.v_init = v_init_b
    h.finitialize(v_init_b)
    h.run()

    t_arr = np.array(t_rec.as_numpy())
    v_arr = np.array(v_rec.as_numpy())

    plot_start = np.searchsorted(t_arr, pre_equil)
    t_plot = t_arr[plot_start:] - pre_equil
    v_plot = v_arr[plot_start:]

    test_end    = delay_b + duration_b
    steady_mask = (t_arr >= (pre_equil + test_end - 50)) & (t_arr <= (pre_equil + test_end))
    v_steady    = np.mean(v_arr[steady_mask])
    delta_v     = v_steady - v_init_b

    current_injections.append(amplitude)
    voltage_changes.append(delta_v)

    ax_b.plot(t_plot, v_plot, color=dark_blue, linewidth=1.8)
    print(f"[Panel B] Current: {amplitude:.2f} nA, ΔV: {delta_v:.2f} mV")

ax_b.set_xlabel("Time (ms)", **LABEL_KW)
ax_b.set_ylabel("Membrane Potential (mV)", **LABEL_KW)
ax_b.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1.5, length=5)
for spine in ax_b.spines.values():
    spine.set_linewidth(1.5)

ax_b.text(-0.45, 1.15, 'B', transform=ax_b.transAxes,
          fontsize=28, fontweight='bold', fontfamily='Sans Serif', va='top', ha='left')

print("[Panel B] Done.")

# ──────────────────────────────────────────────────────────────────────────────
# Panel C – IV / non-linearity plot
# ──────────────────────────────────────────────────────────────────────────────
ax_c = axes[2]

ax_c.plot(current_injections, voltage_changes,
          marker='o', color=dark_blue, linestyle='-', markersize=8, linewidth=2.0,
          markerfacecolor=dark_blue, markeredgewidth=1.5)

ax_c.set_xlabel("Injected Current (nA)", **LABEL_KW)
ax_c.set_ylabel("ΔMembrane Potential (mV)", **LABEL_KW)
ax_c.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1.5, length=5)
for spine in ax_c.spines.values():
    spine.set_linewidth(1.5)

ax_c.text(-0.40, 1.15, 'C', transform=ax_c.transAxes,
          fontsize=28, fontweight='bold', fontfamily='Sans Serif', va='top', ha='left')

print("[Panel C] Done.")

# ─── Save & show ─────────────────────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Figure1_Combined_ABC.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nFigure saved as {output_path}")
