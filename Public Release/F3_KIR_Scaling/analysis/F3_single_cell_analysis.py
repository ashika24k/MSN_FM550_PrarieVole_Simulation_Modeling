
"""
MSN Rectification Analysis with KIR Channel Scaling
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from msn.cell import MSN
from msn.instrumentation import Stim
from neuron import h
import time
import sys
import signal

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
plt.rcParams['grid.alpha']       = 0.3

# Color palette for conditions
CONDITION_COLORS = ['#285C81', '#9D2222', '#2E7D32', '#F57C00']  # Dark blue, dark red, dark green, dark orange


# Global variable to track interruption
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\n\n⚠️  Interruption requested! Finishing current simulation step...")

signal.signal(signal.SIGINT, signal_handler)

def show_progress_spinner(duration=1.0, message="Setting up cell..."):
    chars = "|/-\\"
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        print(f"\r{message} {chars[i % len(chars)]}", end="")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    print(f"\r{message} ✓")


def modify_kir_conductance(cell, scale_factor=1.0, compartment='all'):
    for section in cell.all:
        if _should_modify_section(section, compartment):
            if hasattr(section, 'gbar_kir'):
                for seg in section:
                    seg.gbar_kir *= scale_factor


def set_kir_conductance(cell, conductance, compartment='all'):
    for section in cell.all:
        if _should_modify_section(section, compartment):
            if hasattr(section, 'gbar_kir'):
                for seg in section:
                    seg.gbar_kir = conductance


def set_kir_modulation(cell, max_mod=1.0, level=0.0, enable=True, compartment='all'):
    for section in cell.all:
        if _should_modify_section(section, compartment):
            if hasattr(section, 'damod_kir'):
                for seg in section:
                    seg.damod_kir = 1 if enable else 0
                    seg.maxMod_kir = max_mod
                    seg.level_kir = level


def apply_kir_gradient(cell, soma_conductance, distal_conductance):
    # Calculate distances from soma
    h.distance(sec=cell.soma)
    max_distance = 0
    for section in cell.all:
        if 'dend' in section.name():
            for seg in section:
                distance = h.distance(seg.x, sec=section)
                max_distance = max(max_distance, distance)
    
    # Apply gradient to dendrites
    for section in cell.all:
        if 'dend' in section.name():
            for seg in section:
                distance = h.distance(seg.x, sec=section)
                norm_dist = distance / max_distance if max_distance > 0 else 0
                conductance = (soma_conductance * (1 - norm_dist) + 
                             distal_conductance * norm_dist)
                seg.gbar_kir = conductance


def get_kir_conductances(cell):
    conductances = {'soma': [], 'dend': [], 'axon': []}
    
    for section in cell.all:
        if hasattr(section, 'gbar_kir'):
            section_type = None
            if 'soma' in section.name():
                section_type = 'soma'
            elif 'dend' in section.name():
                section_type = 'dend'
            elif 'axon' in section.name():
                section_type = 'axon'
            
            if section_type:
                for seg in section:
                    conductances[section_type].append(seg.gbar_kir)
    
    # Calculate averages
    avg_conductances = {}
    for comp, values in conductances.items():
        if values:
            avg_conductances[comp] = np.mean(values)
        else:
            avg_conductances[comp] = 0.0
    
    return avg_conductances


def create_modern_summary_table(results):
    """Create a modern, publication-ready summary table."""
    print(f"\n{'='*85}")
    print(f"{'QUANTITATIVE ANALYSIS SUMMARY':^85}")
    print(f"{'='*85}")
    
    # Header
    header = f"{'Condition':<20} {'Avg IR (MΩ)':<12} {'KIR Soma':<12} {'KIR Dend':<12} {'IR Range':<15}"
    print(header)
    print(f"{'='*85}")
    
    # Data rows with enhanced formatting
    for condition, result in results.items():
        kir_cond = result['kir_conductances']
        ir_values = result['all_inward_rectifications']
        ir_range = f"{min(ir_values):.1f}-{max(ir_values):.1f}"
        
        row = (f"{str(condition):<20} "
               f"{result['average_inward_rectification']:<12.2f} "
               f"{kir_cond['soma']:<12.6f} "
               f"{kir_cond['dend']:<12.6f} "
               f"{ir_range:<15}")
        print(row)
    
    print(f"{'='*85}")
    
    # Calculate and display percentage differences
    if len(results) > 1:
        print(f"\n{'RELATIVE CHANGES FROM BASELINE':^85}")
        print(f"{'='*85}")
        
        baseline_condition = list(results.keys())[0]
        baseline_ir = results[baseline_condition]['average_inward_rectification']
        
        for condition, result in results.items():
            if condition != baseline_condition:
                current_ir = result['average_inward_rectification']
                percent_change = ((current_ir - baseline_ir) / baseline_ir) * 100
                direction = "↑" if percent_change > 0 else "↓"
                print(f"{condition:<20} {direction} {abs(percent_change):>6.1f}% change from {baseline_condition}")
        
        print(f"{'='*85}")


def find_equilibrium_voltage(cell_type, cell_index, kir_modification=None):
    """Find the natural equilibrium voltage for a single MSN cell with optional KIR modification."""
    cell = MSN(cell_type, cell_index)
    
    # Apply KIR modification if specified
    if kir_modification is not None:
        if isinstance(kir_modification, dict):
            if 'scale_factor' in kir_modification:
                compartment = kir_modification.get('compartment', 'all')
                modify_kir_conductance(cell, kir_modification['scale_factor'], compartment)
    
    stim = Stim(cell)
    stim.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
    stim.run()
    
    # Get equilibrated voltage (average over last 50 ms)
    v_array = stim.v.as_numpy()
    t_array = stim.t.as_numpy()
    eq_mask = t_array >= 150
    equilibrated_voltage = np.mean(v_array[eq_mask])
    
    return equilibrated_voltage


def _should_modify_section(section, compartment):
    if compartment == 'all':
        return True
    elif compartment == 'soma':
        return 'soma' in section.name()
    elif compartment == 'dend':
        return 'dend' in section.name()
    elif compartment == 'axon':
        return 'axon' in section.name()
    else:
        return False


def apply_kir_preset(cell, preset_name):
    presets = {
        'normal': {'scale': 1.0, 'compartment': 'all'},
        'enhanced': {'scale': 2.0, 'compartment': 'all'},
        'reduced': {'scale': 0.5, 'compartment': 'all'},
        'blocked': {'scale': 0.0, 'compartment': 'all'},
        'soma_enhanced': {'scale': 3.0, 'compartment': 'soma'},
        'dend_reduced': {'scale': 0.3, 'compartment': 'dend'},
        'modulated_up': {'modulation': {'max_mod': 1.5, 'level': 1.0, 'enable': True}},
        'modulated_down': {'modulation': {'max_mod': 0.7, 'level': 1.0, 'enable': True}}
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    preset = presets[preset_name]
    
    if 'scale' in preset:
        modify_kir_conductance(cell, preset['scale'], preset['compartment'])
    elif 'modulation' in preset:
        mod = preset['modulation']
        set_kir_modulation(cell, mod['max_mod'], mod['level'], mod['enable'])


def calculate_rectification(cell_type='imsn', cell_index=21, v_init=-82.8, 
                           kir_modification=None, use_female_params=False):
    # Setup cell and stim
    print(f"Setting up {cell_type.upper()} cell #{cell_index}...", end=" ")
    setup_start = time.time()
    
    cell = MSN(cell_type, cell_index, v_init=v_init)
    
    setup_time = time.time() - setup_start
    print(f"✓ ({setup_time:.1f}s)")

    # Apply KIR modifications if specified
    kir_description = "Normal KIR"
    if kir_modification is not None:
        print(f"⚠️  Applying KIR modification: ", end="")
        if isinstance(kir_modification, dict) and 'scale_factor' in kir_modification:
            compartment = kir_modification.get('compartment', 'all')
            modify_kir_conductance(cell, kir_modification['scale_factor'], compartment)
            kir_description = f"{kir_modification['scale_factor']}x KIR"
            print(f"Scale {kir_modification['scale_factor']}x in {compartment}")
    else:
        print("✓ Using normal KIR conductances")
    
    # Get current KIR conductances
    kir_conductances = get_kir_conductances(cell)

    # Parameters based on cell type (Female vs normal)
    if use_female_params:
        rheobase_pA = 153.0
        scaling_factor = 163.8 / 33.216
        kir_description = f"Female FM550"
        print(f"✓ Using Female cell parameters (scaling_factor={scaling_factor:.3f})")
    else:
        rheobase_pA = 112.5
        scaling_factor = 211.7 / 32.205
        print(f"✓ Using standard cell parameters (scaling_factor={scaling_factor:.3f})")
    
    neg_currents = np.arange(-0.09, 0, 0.01)
    rheobase_nA = rheobase_pA / 1000.0

    # Stimulus parameters
    pre_equilibration_time = 200
    delay = 10
    duration = 150
    tmax = pre_equilibration_time + delay + duration + 150

    # Step 1: Calculate holding current using SAME cell
    print("Calculating holding current...", end=" ")
    stim = Stim(cell)
    stim.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
    stim.run()
    
    # Measure natural equilibrium (without holding current)
    v_test_array = stim.v.as_numpy()
    natural_equilibrium = np.mean(v_test_array[-100:])
    
    # Initial estimate of holding current
    holding_current = (v_init - natural_equilibrium) / 100.0
    
    # Iterative refinement - converge to exact v_init, REUSING same cell
    for iteration in range(10):
        stim.set_stim(delay=0, duration=200, amplitude=holding_current, tmax=200, add_rheob=False)
        stim.run()
        
        # Update natural_equilibrium after each test
        natural_equilibrium = np.mean(stim.v.as_numpy()[-100:])
        
        # Check convergence
        if abs(natural_equilibrium - v_init) < 0.01:
            break
        
        # Adjust holding current
        holding_current += (v_init - natural_equilibrium) / 100.0
    
    print(f"✓ Holding current = {holding_current:.4f} nA")
    
    # Step 2: Use SAME cell for main simulation
    # Set rheobase
    if hasattr(cell, 'rheobase'):
        cell.rheobase = rheobase_nA
    
    # Create IClamp once and reuse
    iclamp = h.IClamp(cell.soma(0.5))
    iclamp.delay = 0
    iclamp.dur = tmax
    
    # Create recording vectors once
    v_rec = h.Vector()
    t_rec = h.Vector()
    v_rec.record(cell.soma(0.5)._ref_v)
    t_rec.record(h._ref_t)
    
    # Pre-calculate steady-state time window
    steady_start_time = pre_equilibration_time + delay + duration - 50
    steady_end_time = pre_equilibration_time + delay + duration

    # Storage
    inward_rectifications = []
    delta_v_values = []

    # Progress tracking
    total_steps = len(neg_currents)
    print(f"Running {total_steps} current injection steps...")
    start_time = time.time()

    # Run simulation
    for step_idx, i_test in enumerate(neg_currents):
        # Check for interruption
        global interrupted
        if interrupted:
            print(f"\n❌ Simulation interrupted at step {step_idx + 1}/{total_steps}")
            print("Returning partial results...")
            break
            
        # Progress indicator
        if step_idx % 2 == 0 or step_idx == total_steps - 1:
            percent_complete = (step_idx + 1) / total_steps * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (step_idx + 1) * total_steps if step_idx > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            print(f"  Step {step_idx + 1}/{total_steps} ({percent_complete:.1f}%) - "
                  f"Current: {i_test:.3f} nA - "
                  f"Elapsed: {elapsed_time:.1f}s - "
                  f"ETA: {remaining_time:.1f}s", end="\r")
            sys.stdout.flush()
        
        # Update current amplitude (holding + test current)
        iclamp.amp = holding_current + i_test
        
        # Run simulation
        h.finitialize(v_init)
        h.continuerun(tmax)
        
        # Convert NEURON vectors to numpy arrays
        v_array = v_rec.as_numpy()
        t_array = t_rec.as_numpy()

        # Use searchsorted for fast indexing
        steady_start_idx = np.searchsorted(t_array, steady_start_time)
        steady_end_idx = np.searchsorted(t_array, steady_end_time)
        v_steady = np.mean(v_array[steady_start_idx:steady_end_idx])
        
        # Calculate delta_v with scaling factor
        delta_v = (v_steady - v_init) * scaling_factor
        inward_rectification = abs(delta_v / i_test)

        inward_rectifications.append(inward_rectification)
        delta_v_values.append(delta_v)

    # Clear progress line
    if not interrupted:
        print(f"\n✓ Simulation completed in {time.time() - start_time:.1f} seconds")
    else:
        print(f"\n⚠️  Partial simulation completed in {time.time() - start_time:.1f} seconds")

    # Rectification calculations
    average_inward_rectification = np.mean(inward_rectifications)
    IR1 = inward_rectifications[0]
    IR2 = inward_rectifications[-1]

    percent_IR_steps = []
    for i in range(len(inward_rectifications) - 1):
        r1, r2 = inward_rectifications[i], inward_rectifications[i + 1]
        percent_IR = (r1 / r2) * 100 if r2 != 0 else np.nan
        percent_IR_steps.append(percent_IR)
    average_percent_IR = np.nanmean(percent_IR_steps) if percent_IR_steps else np.nan

    # Print summary
    print(f"\n=== {cell_type.upper()} #{cell_index} | v_init={v_init} mV Analysis Results ===")
    print(f"KIR Configuration: {kir_description}")
    print(f"KIR Conductances (S/cm2) - Soma: {kir_conductances['soma']:.6f}, "
          f"Dend: {kir_conductances['dend']:.6f}, Axon: {kir_conductances['axon']:.6f}")
    print(f"Rheobase = {rheobase_pA:.1f} pA")
    print(f"Avg IR = {average_inward_rectification:.2f} MΩ")
    print(f"IR1 = {IR1:.2f} MΩ, IR2 = {IR2:.2f} MΩ")
    print(f"Average Percent IR = {average_percent_IR:.2f}%")
    print("=" * 80)

    # No individual plotting - results will be shown in comparison plot only

    # Cleanup
    print(f"✓ Analysis complete. KIR modifications were isolated to this cell instance.")

    return {
        'average_inward_rectification': average_inward_rectification,
        'IR1': IR1,
        'IR2': IR2,
        'percent_inward_rectification_stepwise': percent_IR_steps,
        'average_percent_inward_rectification': average_percent_IR,
        'all_inward_rectifications': inward_rectifications,
        'kir_description': kir_description,
        'kir_conductances': kir_conductances,
        'analysis_data': {
            'currents': neg_currents,
            'inward_rectifications': inward_rectifications,
            'delta_v_values': delta_v_values
        }
    }


def compare_kir_conditions(cell_type='imsn', cell_index=21, v_init=-82.8, 
                          conditions=None):
    if conditions is None:
        conditions = ['normal', 'enhanced', 'reduced', 'blocked']
    
    results = {}
    
    print(f"\n=== COMPARING KIR CONDITIONS FOR {cell_type.upper()} #{cell_index} ===")
    print(f"Total conditions to test: {len(conditions)}")
    
    for condition_idx, condition in enumerate(conditions):
        print(f"\n[{condition_idx + 1}/{len(conditions)}] Running condition: {condition}")
        condition_start_time = time.time()
        
        result = calculate_rectification(cell_type, cell_index, v_init, condition)
        results[str(condition)] = result
        
        condition_time = time.time() - condition_start_time
        print(f"✓ Condition '{condition}' completed in {condition_time:.1f} seconds")
    
    # Create I-V relationship plot only
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, (condition, result) in enumerate(results.items()):
        color = MODERN_COLORS[i % len(MODERN_COLORS)]
        data = result['analysis_data']
        
        # I-V relationship plots using actual delta_v values
        ax.plot(data['currents'], data['delta_v_values'], 
                'o-', color=color, label=f"{condition}", alpha=0.8,
                linewidth=2.5, markersize=6, markerfacecolor='white', 
                markeredgecolor=color, markeredgewidth=2)
    
    # Format plot with modern styling
    ax.set_xlabel('Current (nA)', fontsize=12)
    ax.set_ylabel('ΔV (mV)', fontsize=12)
    ax.set_title(f'Current-Voltage Relationship - {cell_type.upper()} #{cell_index}', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='gray', fancybox=False)
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_kir_modifications():
    print("\n" + "="*60)
    print("DEMONSTRATING KIR CHANNEL MODIFICATIONS")
    print("="*60)
    
    # Preset comparisons
    print("\n1. COMPARING KIR PRESETS:")
    compare_kir_conditions(conditions=['normal', 'enhanced', 'reduced', 'blocked'])
    
    # Custom scaling
    print("\n2. CUSTOM KIR SCALING:")
    custom_conditions = [
        {'scale_factor': 1.0},   # Normal
        {'scale_factor': 1.5},   # 1.5x
        {'scale_factor': 2.0},   # 2x
        {'scale_factor': 0.25}   # 0.25x
    ]
    compare_kir_conditions(conditions=custom_conditions)
    
    # Compartment-specific modifications
    print("\n3. COMPARTMENT-SPECIFIC MODIFICATIONS:")
    compartment_conditions = [
        {'scale_factor': 1.0, 'compartment': 'all'},   # Normal
        {'scale_factor': 3.0, 'compartment': 'soma'},  # Soma enhanced
        {'scale_factor': 0.3, 'compartment': 'dend'},  # Dend reduced
        {'scale_factor': 2.0, 'compartment': 'dend'}   # Dend enhanced
    ]
    compare_kir_conditions(conditions=compartment_conditions)
    
    # Modulation effects
    print("\n4. KIR MODULATION EFFECTS:")
    modulation_conditions = [
        None,  # Normal
        {'modulation': {'max_mod': 1.3, 'level': 1.0, 'enable': True}},  # 30% increase
        {'modulation': {'max_mod': 0.7, 'level': 1.0, 'enable': True}},  # 30% decrease
        {'modulation': {'max_mod': 1.5, 'level': 0.5, 'enable': True}}   # 25% increase (50% of 50%)
    ]
    compare_kir_conditions(conditions=modulation_conditions)


if __name__ == '__main__':
    print("=" * 60)
    print("MSN RECTIFICATION ANALYSIS WITH KIR MODIFICATIONS")
    print("=" * 60)

    analysis_start = time.time()

    condition_colors = {
        'Female Vehicle (Control)': '#285C81',
        '1.25x KIR':                '#F57C00',
        '1.5x KIR':                 '#2E7D32',
        'Female FM550':             '#9D2222',
    }
    markevery_map = {
        'Female Vehicle (Control)': None,
        '1.25x KIR':                None,
        '1.5x KIR':                 (0, 2),
        'Female FM550':             (1, 2),
    }
    condition_specs = [
        ('Female Vehicle (Control)', dict(v_init=-82.8, kir_modification=None,                      use_female_params=False)),
        ('1.25x KIR',               dict(v_init=-82.8, kir_modification={'scale_factor': 1.25},    use_female_params=False)),
        ('1.5x KIR',                dict(v_init=-82.8, kir_modification={'scale_factor': 1.5},     use_female_params=False)),
        ('Female FM550',            dict(v_init=-82.3, kir_modification=None,                      use_female_params=True)),
    ]

    # Run for both cell types: dMSN → panel A, IMSN → panel B
    for cell_type, panel_label, fname_suffix in [('dmsn', 'A', 'DMSN'), ('imsn', 'B', 'IMSN')]:
        print(f"\n=== COMPARING KIR CONDITIONS FOR {cell_type.upper()} #21 ===")
        results = {}

        for idx, (cond_name, kwargs) in enumerate(condition_specs, 1):
            print(f"\n[{idx}/4] Running condition: {cond_name}")
            t0 = time.time()
            results[cond_name] = calculate_rectification(cell_type=cell_type, cell_index=21, **kwargs)
            print(f"✓ Condition '{cond_name}' completed in {time.time() - t0:.1f} seconds")

        fig, ax = plt.subplots(figsize=(10, 8))

        legend_entries = []
        for cond_name, result in results.items():
            color = condition_colors.get(cond_name, '#333333')
            data  = result['analysis_data']
            me    = markevery_map.get(cond_name, None)
            ax.plot(data['currents'], data['delta_v_values'],
                    'o-', color=color, alpha=0.8, linewidth=2, markersize=6,
                    **(dict(markevery=me) if me is not None else {}))
            legend_entries.append(plt.Line2D([0], [0], color=color, linewidth=2,
                                             marker='o', markersize=6, label=cond_name))

        # Panel label (A for dMSN, B for IMSN)
        ax.text(-0.2, 1.15, panel_label, transform=ax.transAxes,
                fontsize=28, fontweight='bold', fontfamily='Calibri', va='top', ha='left')

        ax.set_xlabel('Current (nA)', fontsize=22, fontweight='bold', fontfamily='Calibri')
        ax.set_ylabel('\u0394V (mV)',      fontsize=22, fontweight='bold', fontfamily='Calibri')
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=5)
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.legend(handles=legend_entries, loc='lower right', fontsize=16,
                  prop={'family': 'Calibri', 'weight': 'bold', 'size': 16},
                  framealpha=0.95, edgecolor='gray', fancybox=False)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'Figure3_{fname_suffix}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Figure saved as {output_path}')

    total_time = time.time() - analysis_start
    print(f"\n{'='*85}")
    print(f"{'ANALYSIS COMPLETED SUCCESSFULLY!':^85}")
    print(f"{'Total execution time: ' + f'{total_time:.1f} seconds':^85}")
    print(f"{'='*85}")
