import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from msn.cell import MSN
from msn.instrumentation import Stim
from neuron import h

# ─── Global font settings ────────────────────────────────────────────────────
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.weight']      = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize']   = 22
plt.rcParams['axes.titlesize']   = 24
plt.rcParams['xtick.labelsize']  = 18
plt.rcParams['ytick.labelsize']  = 18

LABEL_KW  = dict(fontsize=22, fontweight='bold', fontfamily='sans-serif')
TITLE_KW  = dict(fontsize=24, fontweight='bold', fontfamily='sans-serif')
TICK_SIZE = 18
LEGEND_KW = dict(fontsize=16, prop={'family': 'sans-serif', 'weight': 'bold', 'size': 16})

dark_blue = "#285C81"   # Control / Vehicle
dark_red  = "#9D2222"   # FM550


# ──────────────────────────────────────────────────────────────────────────────
# Helper: find holding current to maintain v_init
# ──────────────────────────────────────────────────────────────────────────────
def find_holding_current(cell_type, cell_index, v_init, rheobase_nA):
    cell_test = MSN(cell_type, cell_index, v_init=v_init)
    if hasattr(cell_test, 'rheobase'):
        cell_test.rheobase = rheobase_nA

    stim_test = Stim(cell_test)
    stim_test.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
    stim_test.run()
    natural_eq = np.mean(np.array(stim_test.v)[-100:])
    holding_current = (v_init - natural_eq) / 100.0

    for iteration in range(10):
        cell_tmp = MSN(cell_type, cell_index, v_init=v_init)
        if hasattr(cell_tmp, 'rheobase'):
            cell_tmp.rheobase = rheobase_nA
        stim_tmp = Stim(cell_tmp)
        stim_tmp.set_stim(delay=0, duration=200, amplitude=holding_current, tmax=200, add_rheob=False)
        stim_tmp.run()
        natural_eq = np.mean(np.array(stim_tmp.v)[-100:])
        if abs(natural_eq - v_init) < 0.01:
            print(f"  Holding current converged in {iteration + 1} iterations.")
            break
        holding_current += (v_init - natural_eq) / 100.0

    print(f"  Holding current: {holding_current:.4f} nA")
    return holding_current


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run all current-step simulations and return traces + IV data
# ──────────────────────────────────────────────────────────────────────────────
def run_condition(cell_type, cell_index, v_init, rheobase_pA, scaling_factor):
    rheobase_nA = rheobase_pA / 1000.0
    holding_current = find_holding_current(cell_type, cell_index, v_init, rheobase_nA)

    pre_equil  = 200
    delay      = 10
    duration   = 150
    tmax       = pre_equil + delay + duration + 150
    neg_currents = np.arange(-0.09, 0, 0.01)

    cell_reuse = MSN(cell_type, cell_index, v_init=v_init)
    if hasattr(cell_reuse, 'rheobase'):
        cell_reuse.rheobase = rheobase_nA

    iclamp_hold       = h.IClamp(cell_reuse.soma(0.5))
    iclamp_hold.delay = 0
    iclamp_hold.dur   = tmax
    iclamp_hold.amp   = holding_current

    iclamp_test       = h.IClamp(cell_reuse.soma(0.5))
    iclamp_test.delay = pre_equil + delay
    iclamp_test.dur   = duration

    v_rec = h.Vector(); t_rec = h.Vector()
    v_rec.record(cell_reuse.soma(0.5)._ref_v)
    t_rec.record(h._ref_t)

    test_end     = pre_equil + delay + duration
    steady_start = test_end - 50

    all_traces      = []
    voltage_changes = []
    inward_rects    = []

    for i_test in neg_currents:
        iclamp_test.amp = i_test
        h.tstop  = tmax
        h.v_init = v_init
        h.finitialize(v_init)
        h.run()

        v_array = np.array(v_rec.as_numpy())
        t_array = np.array(t_rec.as_numpy())

        plot_start = np.searchsorted(t_array, pre_equil)
        all_traces.append((t_array[plot_start:] - pre_equil, v_array[plot_start:], i_test))

        ss_idx_s = np.searchsorted(t_array, steady_start)
        ss_idx_e = np.searchsorted(t_array, test_end)
        v_steady = np.mean(v_array[ss_idx_s:ss_idx_e])

        delta_v = (v_steady - v_init) * scaling_factor
        voltage_changes.append(delta_v)
        inward_rects.append(abs(delta_v / i_test))

    return {
        'currents':              neg_currents,
        'all_traces':            all_traces,
        'voltage_changes':       np.array(voltage_changes),
        'inward_rectifications': np.array(inward_rects),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Run both conditions
# ──────────────────────────────────────────────────────────────────────────────
cell_type  = 'dmsn'
cell_index = 21

print("Running Control condition...")
data_ctrl = run_condition(cell_type, cell_index,
                          v_init=-82.8,
                          rheobase_pA=112.5,
                          scaling_factor=211.7 / 32.205)

print("\nRunning FM550 condition...")
data_fm550 = run_condition(cell_type, cell_index,
                           v_init=-82.3,
                           rheobase_pA=153.0,
                           scaling_factor=163.8 / 33.216)

# ──────────────────────────────────────────────────────────────────────────────
# Build figure — three panels side by side
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.subplots_adjust(wspace=0.42, left=0.08, right=0.97, top=0.88, bottom=0.15)

sample_currents = [-0.09, -0.07, -0.05, -0.03, -0.02]

# ── Panel A: voltage trace overlay ───────────────────────────────────────────
ax_a = axes[0]
for t_arr, v_arr, i_test in data_ctrl['all_traces']:
    if min(abs(i_test - np.array(sample_currents))) < 0.005:
        ax_a.plot(t_arr, v_arr, color=dark_blue, alpha=0.85, linewidth=2.0)
for t_arr, v_arr, i_test in data_fm550['all_traces']:
    if min(abs(i_test - np.array(sample_currents))) < 0.005:
        ax_a.plot(t_arr, v_arr, color=dark_red, alpha=0.85, linewidth=2.0)

ax_a.set_xlabel("Time (ms)", **LABEL_KW)
ax_a.set_ylabel("Membrane Potential (mV)", **LABEL_KW)
ax_a.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1.5, length=5)
for sp in ax_a.spines.values(): sp.set_linewidth(1.5)
ax_a.text(-0.35, 1.15, 'A', transform=ax_a.transAxes,
          fontsize=28, fontweight='bold', fontfamily='sans-serif', va='top', ha='left')
ax_a.legend(handles=[Line2D([0], [0], color=dark_blue, lw=2.5, label='Control'),
             Line2D([0], [0], color=dark_red, lw=2.5, label='FM550')],
         loc='lower right', frameon=True, **LEGEND_KW)

# ── Panel B: I-V overlay ─────────────────────────────────────────────────────
ax_b = axes[1]
ax_b.plot(data_ctrl['currents'],  data_ctrl['voltage_changes'],
          'o-', color=dark_blue, linewidth=2.0, markersize=7, label='Control')
ax_b.plot(data_fm550['currents'], data_fm550['voltage_changes'],
          'o-', color=dark_red,  linewidth=2.0, markersize=7, label='FM550')

ax_b.set_xlabel("Injected Current (nA)", **LABEL_KW)
ax_b.set_ylabel("ΔMembrane Potential (mV)", **LABEL_KW)
ax_b.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1.5, length=5)
for sp in ax_b.spines.values(): sp.set_linewidth(1.5)
ax_b.text(-0.35, 1.15, 'B', transform=ax_b.transAxes,
          fontsize=28, fontweight='bold', fontfamily='sans-serif', va='top', ha='left')

# ── Panel C: Inward rectification overlay ────────────────────────────────────
ax_c = axes[2]
ax_c.plot(data_ctrl['currents'],  data_ctrl['inward_rectifications'],
          'o-', color=dark_blue, linewidth=2.0, markersize=7, label='Control')
ax_c.plot(data_fm550['currents'], data_fm550['inward_rectifications'],
          'o-', color=dark_red,  linewidth=2.0, markersize=7, label='FM550')

ax_c.set_xlabel("Injected Current (nA)", **LABEL_KW)
ax_c.set_ylabel("Input Resistance (MΩ)", **LABEL_KW)
ax_c.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1.5, length=5)
for sp in ax_c.spines.values(): sp.set_linewidth(1.5)
ax_c.text(-0.35, 1.15, 'C', transform=ax_c.transAxes,
          fontsize=28, fontweight='bold', fontfamily='sans-serif', va='top', ha='left')

# ── Save & show ───────────────────────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "Figure2_OverlayRow.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nFigure saved as {output_path}")
