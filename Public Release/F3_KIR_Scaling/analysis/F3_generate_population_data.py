"""
MSN Rectification Analysis — Population-Level I-V Curves

Runs the same 4 conditions from IV_Kir_Scaling.py across every cell of each
type (71 dMSNs, 34 IMSNs) and plots mean ± SEM I-V curves.

Two figures are produced:
  - IV_Kir_Scaling_Population_DMSN.pdf  (71 dMSNs)
    - IV_Kir_Scaling_Population_IMSN.pdf  (34 IMSNs)
"""

import os
import sys
import signal
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from msn.cell import MSN
from msn.instrumentation import Stim
from neuron import h

# ---------------------------------------------------------------------------
# Global style  (matches IV_Kir_Scaling.py)
# ---------------------------------------------------------------------------
plt.rcParams['font.family']     = 'Times New Roman'
plt.rcParams['font.size']       = 12
plt.rcParams['axes.titlesize']  = 14
plt.rcParams['axes.labelsize']  = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['grid.alpha']      = 0.3

CONDITION_COLORS = {
    'Female Vehicle (Control)': '#285C81',   # dark blue
    '1.25x KIR':                '#F57C00',   # dark orange
    '1.5x KIR':                 '#2E7D32',   # dark green
    'Female FM550':             '#9D2222',   # dark red
}

# Current steps used in IV_Kir_Scaling.py
NEG_CURRENTS = np.arange(-0.09, 0, 0.01)

# Total cells per type in the Lindroos dataset
N_CELLS = {'dmsn': 71, 'imsn': 34}

# ---------------------------------------------------------------------------
# Interrupt handler
# ---------------------------------------------------------------------------
interrupted = False

def _signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\n\nInterruption requested — will stop after the current cell.")

signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# KIR helpers  (identical to IV_Kir_Scaling.py)
# ---------------------------------------------------------------------------
def _should_modify_section(section, compartment):
    if compartment == 'all':
        return True
    return compartment in section.name()


def modify_kir_conductance(cell, scale_factor, compartment='all'):
    for section in cell.all:
        if _should_modify_section(section, compartment):
            if hasattr(section, 'gbar_kir'):
                for seg in section:
                    seg.gbar_kir *= scale_factor

# ---------------------------------------------------------------------------
# Single-cell simulation  (mirrors calculate_rectification in IV_Kir_Scaling.py)
# ---------------------------------------------------------------------------
def _run_single_cell(cell_type, cell_index, v_init, kir_modification, use_female_params):
    """
    Run the full holding-current + I-V sweep for one cell.

    Returns a 1-D numpy array of delta_v values (one per current step),
    or None if the cell errored out or the run was interrupted.
    """
    global interrupted
    if interrupted:
        return None

    try:
        cell = MSN(cell_type, cell_index, v_init=v_init)

        # Apply KIR modification if requested
        if kir_modification is not None and 'scale_factor' in kir_modification:
            modify_kir_conductance(
                cell,
                kir_modification['scale_factor'],
                kir_modification.get('compartment', 'all')
            )

        # ------------------------------------------------------------
        # Experimental parameters
        # NOTE: IMSN values come directly from IV_Kir_Scaling.py.
        #       dMSN values mirror IMSN; update below if you have
        #       dMSN-specific experimental calibration numbers.
        # ------------------------------------------------------------
        if use_female_params:
            rheobase_pA    = 153.0
            scaling_factor = 163.8 / 33.216   # Female FM550
        else:
            rheobase_pA    = 112.5
            scaling_factor = 211.7 / 32.205   # Female Vehicle (Control)

        rheobase_nA = rheobase_pA / 1000.0
        if hasattr(cell, 'rheobase'):
            cell.rheobase = rheobase_nA

        # ------------------------------------------------------------
        # Step 1 — find holding current (identical algorithm to IV_Kir_Scaling.py)
        # ------------------------------------------------------------
        stim = Stim(cell)
        stim.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
        stim.run()
        natural_eq = np.mean(stim.v.as_numpy()[-100:])
        holding_current = (v_init - natural_eq) / 100.0

        for _ in range(10):
            stim.set_stim(delay=0, duration=200,
                          amplitude=holding_current, tmax=200, add_rheob=False)
            stim.run()
            natural_eq = np.mean(stim.v.as_numpy()[-100:])
            if abs(natural_eq - v_init) < 0.01:
                break
            holding_current += (v_init - natural_eq) / 100.0

        # ------------------------------------------------------------
        # Step 2 — main current sweep
        # ------------------------------------------------------------
        pre_eq    = 200
        delay     = 10
        duration  = 150
        tmax      = pre_eq + delay + duration + 150
        ss_start  = pre_eq + delay + duration - 50
        ss_end    = pre_eq + delay + duration

        iclamp      = h.IClamp(cell.soma(0.5))
        iclamp.delay = 0
        iclamp.dur   = tmax

        v_rec = h.Vector()
        t_rec = h.Vector()
        v_rec.record(cell.soma(0.5)._ref_v)
        t_rec.record(h._ref_t)

        delta_v_values = []
        for i_test in NEG_CURRENTS:
            if interrupted:
                return None
            iclamp.amp = holding_current + i_test
            h.finitialize(v_init)
            h.continuerun(tmax)

            v_arr = v_rec.as_numpy()
            t_arr = t_rec.as_numpy()
            i0 = np.searchsorted(t_arr, ss_start)
            i1 = np.searchsorted(t_arr, ss_end)
            v_steady = np.mean(v_arr[i0:i1])
            delta_v_values.append((v_steady - v_init) * scaling_factor)

        return np.array(delta_v_values)

    except Exception as exc:
        print(f"\n  Error — {cell_type} #{cell_index}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Population loop
# ---------------------------------------------------------------------------
def run_population(cell_type):
    """
    Run all 4 conditions for every cell of `cell_type`.

    Returns
    -------
    dict  condition_name -> np.ndarray of shape (n_valid_cells, n_current_steps)
    """
    n_cells = N_CELLS[cell_type]

    # Condition definitions — mirror the __main__ block in IV_Kir_Scaling.py
    conditions = [
        ('Female Vehicle (Control)',
         dict(v_init=-82.8, kir_modification=None,                     use_female_params=False)),
        ('1.25x KIR',
         dict(v_init=-82.8, kir_modification={'scale_factor': 1.25},   use_female_params=False)),
        ('1.5x KIR',
         dict(v_init=-82.8, kir_modification={'scale_factor': 1.50},   use_female_params=False)),
        ('Female FM550',
         dict(v_init=-82.3, kir_modification=None,                     use_female_params=True)),
    ]

    # Accumulate per-condition cell arrays
    raw = {name: [] for name, _ in conditions}

    total   = n_cells * len(conditions)
    counter = 0
    t0      = time.time()

    print(f"\n{'='*70}")
    print(f"POPULATION ANALYSIS — {cell_type.upper()}  "
          f"({n_cells} cells × {len(conditions)} conditions = {total} runs)")
    print(f"{'='*70}")

    for cell_idx in range(n_cells):
        if interrupted:
            print(f"\nStopped at cell {cell_idx}/{n_cells}.")
            break

        for cond_name, cond_kwargs in conditions:
            counter += 1
            if interrupted:
                break

            elapsed = time.time() - t0
            eta     = (elapsed / counter) * (total - counter) if counter > 1 else 0
            print(
                f"  [{counter:>5}/{total}]  {cell_type.upper()} #{cell_idx:>3}  |  "
                f"{cond_name:<30}  ETA {eta:>6.0f}s   ",
                end="\r"
            )
            sys.stdout.flush()

            dv = _run_single_cell(cell_type=cell_type,
                                  cell_index=cell_idx,
                                  **cond_kwargs)

            if dv is not None and len(dv) == len(NEG_CURRENTS):
                raw[cond_name].append(dv)

    print(f"\n  Finished in {time.time() - t0:.1f}s.  Valid cells per condition:")
    for name, arrays in raw.items():
        print(f"    {name}: {len(arrays)}")

    # Convert lists → 2-D arrays
    return {name: np.array(arrays) for name, arrays in raw.items() if arrays}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def save_csvs(population_data, cell_type, out_dir):
    """
    Saves two CSV files per cell type:

    1. Figure3[C/D]_[DMSN/IMSN]_RAW.csv
       One row per cell per condition.  Columns:
         condition | cell_index | current_-0.09nA | current_-0.08nA | ...

    2. Figure3[C/D]_[DMSN/IMSN]_MEAN_SEM.csv
       Mean and SEM across cells for every (condition, current) pair.  Columns:
         condition | current_nA | mean_delta_v_mV | sem_delta_v_mV | n_cells
    """
    current_cols = [f'current_{c:.3f}nA' for c in NEG_CURRENTS]
    
    # Determine prefix and suffix based on cell type
    prefix = 'Figure3C' if cell_type == 'dmsn' else 'Figure3D'
    suffix = 'DMSN' if cell_type == 'dmsn' else 'IMSN'

    # ---- 1. Raw per-cell data ------------------------------------------------
    raw_rows = []
    for cond_name, matrix in population_data.items():
        for cell_idx, row in enumerate(matrix):
            raw_rows.append(
                {'condition': cond_name, 'cell_index': cell_idx}
                | dict(zip(current_cols, row))
            )
    df_raw = pd.DataFrame(raw_rows)
    raw_path = os.path.join(out_dir, f'{prefix}_{suffix}_RAW.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f'Raw data saved  → {raw_path}')

    # ---- 2. Mean ± SEM summary -----------------------------------------------
    summary_rows = []
    for cond_name, matrix in population_data.items():
        n    = matrix.shape[0]
        mean = np.mean(matrix, axis=0)
        sem  = np.std(matrix, axis=0, ddof=1) / np.sqrt(n)
        for i, current in enumerate(NEG_CURRENTS):
            summary_rows.append({
                'condition':       cond_name,
                'current_nA':      round(float(current), 4),
                'mean_delta_v_mV': mean[i],
                'sem_delta_v_mV':  sem[i],
                'n_cells':         n,
            })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, f'{prefix}_{suffix}_MEAN_SEM.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f'Mean±SEM saved  → {summary_path}')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_population(population_data, cell_type, save_path=None):
    """
    Plot mean ± SEM I-V curves for the 4 conditions.
    Shaded band = ±1 SEM around the mean.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond_name, matrix in population_data.items():
        n     = matrix.shape[0]
        mean  = np.mean(matrix, axis=0)
        sem   = np.std(matrix, axis=0, ddof=1) / np.sqrt(n)
        color = CONDITION_COLORS.get(cond_name, '#333333')

        ax.plot(NEG_CURRENTS, mean, 'o-',
                color=color, linewidth=2, markersize=5,
                label=f"{cond_name} (n={n})")
        ax.fill_between(NEG_CURRENTS,
                        mean - sem, mean + sem,
                        color=color, alpha=0.18)

    ax.set_xlabel('Current (nA)', fontsize=18)
    ax.set_ylabel('ΔV (mV)',      fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=11,
              framealpha=0.95, edgecolor='gray', fancybox=False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved → {save_path}")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_inputs')
    os.makedirs(out_dir, exist_ok=True)
    t_overall = time.time()

    for cell_type in ['dmsn', 'imsn']:
        pop_data = run_population(cell_type)

        if interrupted:
            print("Run interrupted — skipping plot.")
            break

        save_csvs(pop_data, cell_type, out_dir)

        plot_population(
            pop_data,
            cell_type,
            save_path=os.path.join(
                out_dir,
                f'Figure3_IV_Kir_Scaling_Population_{cell_type.upper()}.png'
            )
        )

    print(f"\nTotal elapsed time: {time.time() - t_overall:.1f}s")
