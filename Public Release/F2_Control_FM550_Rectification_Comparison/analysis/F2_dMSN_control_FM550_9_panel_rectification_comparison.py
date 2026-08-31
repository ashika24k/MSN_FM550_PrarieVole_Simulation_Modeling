import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from msn.cell import MSN
from msn.instrumentation import Stim

# Set Times New Roman font for all text
plt.rcParams['font.family'] = 'Times New Roman'


def calculate_rectification_9_19_logic(cell_type='dmsn', cell_index=21, v_init=-82.8, scaling_factor=211.7/46.26):
    """
    Calculate rectification using exact 9.19 script logic with holding current to maintain v_init.
    OPTIMIZED FOR SPEED.
    """
    from neuron import h
    
    # Rheobase setup
    rheobase_pA = 112.5 if v_init == -82.8 else 153.0
    rheobase_nA = rheobase_pA / 1000.0
    
    # Step 1: Find holding current (FAST - fewer iterations, looser tolerance)
    print(f"  Setting up {cell_type.upper()} cell #{cell_index}, v_init={v_init:.1f} mV...")
    cell_test = MSN(cell_type, cell_index, v_init=v_init)
    if hasattr(cell_test, 'rheobase'):
        cell_test.rheobase = rheobase_nA
    
    # Quick initial equilibration test
    print(f"  Testing natural equilibration...")
    stim_test = Stim(cell_test)
    stim_test.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
    stim_test.run()
    
    v_test_array = np.array(stim_test.v)
    natural_equilibrium = np.mean(v_test_array[-100:])  # Last 100 points instead of masking
    
    # Fast convergence - fewer iterations, practical tolerance
    print(f"  Finding holding current (target: {v_init:.1f} mV, natural: {natural_equilibrium:.2f} mV)...")
    holding_current = (v_init - natural_equilibrium) / 100.0  # Initial estimate
    
    for iteration in range(10):  # Up to 10 iterations for precision
        cell_test2 = MSN(cell_type, cell_index, v_init=v_init)
        if hasattr(cell_test2, 'rheobase'):
            cell_test2.rheobase = rheobase_nA
        
        stim_test2 = Stim(cell_test2)
        stim_test2.set_stim(delay=0, duration=200, amplitude=holding_current, tmax=200, add_rheob=False)
        stim_test2.run()
        
        # Update natural_equilibrium with this holding current
        natural_equilibrium = np.mean(np.array(stim_test2.v)[-100:])
        
        if abs(natural_equilibrium - v_init) < 0.01:  # 0.01 mV precision
            print(f"  ✓ Converged in {iteration + 1} iterations (error: {abs(natural_equilibrium - v_init):.4f} mV)")
            break
        
        # Adjust holding current based on error
        holding_current += (v_init - natural_equilibrium) / 100.0
    
    print(f"  Holding current: {holding_current:.4f} nA (final voltage: {natural_equilibrium:.3f} mV)")
    
    # Stimulus protocol parameters    # Stimulus protocol parameters
    pre_equilibration_time = 200
    delay = 10
    duration = 150
    tmax = pre_equilibration_time + delay + duration + 150
    neg_currents = np.arange(-0.09, 0, 0.01)
    
    # Pre-allocate arrays for speed
    num_steps = len(neg_currents)
    voltages = np.zeros(num_steps)
    inward_rectifications = np.zeros(num_steps)
    voltage_changes = np.zeros(num_steps)
    all_traces = []
    
    # Create ONE cell for all simulations
    cell_reuse = MSN(cell_type, cell_index, v_init=v_init)
    if hasattr(cell_reuse, 'rheobase'):
        cell_reuse.rheobase = rheobase_nA
    
    # Setup stimulation once
    iclamp_hold = h.IClamp(cell_reuse.soma(0.5))
    iclamp_hold.delay = 0
    iclamp_hold.dur = tmax
    iclamp_hold.amp = holding_current
    
    iclamp_test = h.IClamp(cell_reuse.soma(0.5))
    iclamp_test.delay = pre_equilibration_time + delay
    iclamp_test.dur = duration
    
    v_rec = h.Vector()
    t_rec = h.Vector()
    v_rec.record(cell_reuse.soma(0.5)._ref_v)
    t_rec.record(h._ref_t)
    
    # Pre-calculate indices for steady-state measurement
    test_end = pre_equilibration_time + delay + duration
    steady_start = test_end - 50
    
    # Run all simulations
    print(f"  Running {num_steps} current steps...")
    for idx, i_test in enumerate(neg_currents):
        iclamp_test.amp = i_test
        
        # Progress update
        if idx % 2 == 0 or idx == num_steps - 1:
            print(f"    Step {idx + 1}/{num_steps}: {i_test:.3f} nA", end="\r")
        
        h.tstop = tmax
        h.v_init = v_init
        h.finitialize(v_init)
        h.run()
        
        # Convert NEURON vectors to numpy arrays
        v_array = np.array(v_rec.as_numpy())
        t_array = np.array(t_rec.as_numpy())
        
        # Only store traces for plotting (keep this minimal)
        plot_start_idx = np.searchsorted(t_array, pre_equilibration_time)
        all_traces.append((t_array[plot_start_idx:] - pre_equilibration_time, 
                          v_array[plot_start_idx:], i_test))
        
        # Fast steady-state calculation using indices
        steady_start_idx = np.searchsorted(t_array, steady_start)
        steady_end_idx = np.searchsorted(t_array, test_end)
        v_steady = np.mean(v_array[steady_start_idx:steady_end_idx])
        
        delta_v = (v_steady - v_init) * scaling_factor
        
        voltages[idx] = v_steady
        voltage_changes[idx] = delta_v
        inward_rectifications[idx] = abs(delta_v / i_test)
    
    # Fast calculations
    average_inward_rectification = np.mean(inward_rectifications)
    IR1 = inward_rectifications[0]
    IR2 = inward_rectifications[-1]
    
    # Vectorized percent IR calculation
    percent_IR_steps = (inward_rectifications[:-1] / inward_rectifications[1:]) * 100
    average_percent_IR = np.mean(percent_IR_steps)
    
    print(f"\n  ✓ Analysis complete: Avg IR = {average_inward_rectification:.1f} MΩ, %IR = {average_percent_IR:.1f}%")
    
    return {
        'currents': neg_currents,
        'voltages': voltages,
        'voltage_changes': voltage_changes,
        'inward_rectifications': inward_rectifications,
        'all_traces': all_traces,
        'average_ir': average_inward_rectification,
        'ir1': IR1,
        'ir2': IR2,
        'percent_ir_steps': percent_IR_steps,
        'average_percent_ir': average_percent_IR,
        'v_init': v_init,
        'rheobase_pA': rheobase_pA,
        'scaling_factor': scaling_factor
    }


def plot_condition_row(axes_row, data, color, condition_name):
    """
    Plot a single row of analysis for one condition using 9.19 logic.
    """
    ax1, ax2, ax3 = axes_row
    
    # Voltage traces
    for t_array, v_array, i_test in data['all_traces']:
        ax1.plot(t_array, v_array, color=color, alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Time (ms)', fontsize=14)
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    
    # I-V relationship - using RAW voltage changes (9.19 logic)
    ax2.plot(data['currents'], data['voltage_changes'], 'o-', color=color, linewidth=1.5, markersize=5)
    ax2.set_xlabel('Current (nA)', fontsize=14)
    ax2.set_ylabel('Voltage Change (ΔV, mV)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    
    # IR profile
    ax3.plot(data['currents'], data['inward_rectifications'], 'o-', color=color, linewidth=1.5, markersize=5)
    ax3.set_xlabel('Current (nA)', fontsize=14)
    ax3.set_ylabel('Inward Rectification (MΩ)', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=13)


def plot_comparison_row(axes_row, data1, data2, color1, color2):
    """
    Plot comparison row with overlaid data from both conditions.
    """
    ax7, ax8, ax9 = axes_row
    
    # Selected voltage traces comparison
    sample_currents = [-0.09, -0.07, -0.05, -0.03, -0.02]
    for t_array, v_array, i_test in data1['all_traces']:
        if abs(i_test - np.array(sample_currents)).min() < 0.005:
            ax7.plot(t_array, v_array, color=color1, alpha=0.7, linewidth=1.5)
    for t_array, v_array, i_test in data2['all_traces']:
        if abs(i_test - np.array(sample_currents)).min() < 0.005:
            ax7.plot(t_array, v_array, color=color2, alpha=0.7, linewidth=1.5)
    ax7.set_xlabel('Time (ms)', fontsize=14)
    ax7.set_ylabel('Membrane Potential (mV)', fontsize=14)
    ax7.tick_params(axis='both', which='major', labelsize=13)
    
    # I-V comparison - RAW voltage changes
    ax8.plot(data1['currents'], data1['voltage_changes'], 'o-', color=color1, linewidth=2, markersize=5)
    ax8.plot(data2['currents'], data2['voltage_changes'], 'o-', color=color2, linewidth=2, markersize=5)
    ax8.set_xlabel('Current (nA)', fontsize=14)
    ax8.set_ylabel('Voltage Change (ΔV, mV)', fontsize=14)
    ax8.tick_params(axis='both', which='major', labelsize=13)
    
    # IR comparison
    ax9.plot(data1['currents'], data1['inward_rectifications'], 'o-', color=color1, linewidth=2, markersize=5)
    ax9.plot(data2['currents'], data2['inward_rectifications'], 'o-', color=color2, linewidth=2, markersize=5)
    ax9.set_xlabel('Current (nA)', fontsize=14)
    ax9.set_ylabel('Inward Rectification (MΩ)', fontsize=14)
    ax9.tick_params(axis='both', which='major', labelsize=13)


def generate_3x3_analysis(cell_type='dmsn', cell_index=21):
    """
    Generate 3x3 grid analysis using exact 9.19 logic for both conditions.
    """
    print(f"\n=== 3x3 Grid Analysis using 9.19 Logic ===")
    print(f"Cell: {cell_type.upper()} #{cell_index}")
    print("="*60)
    
    # Define conditions with exact 9.19 parameters
    conditions = {
        'condition1': {
            'v_init': -82.8,
            'scaling_factor': 211.7 / 32.205,  # Adjusted to hit IR_large=211.7
            'color': "#285C81",  # Dark blue color
            'name': 'Control Profile'
        },
        'condition2': {
            'v_init': -82.3,
            'scaling_factor': 163.8 / 33.216,  # Adjusted to hit IR_large=163.8
            'color':  "#9D2222",  # Dark red color
            'name': 'FM550 Profile'
        }
    }
    
    # Run analysis for both conditions
    results = {}
    for name, params in conditions.items():
        print(f"\nRunning {params['name']}...")
        results[name] = calculate_rectification_9_19_logic(
            cell_type=cell_type,
            cell_index=cell_index,
            v_init=params['v_init'],
            scaling_factor=params['scaling_factor']
        )
    
    # Print summary
    data1, data2 = results['condition1'], results['condition2']
    print(f"\n=== Analysis Summary ===")
    print(f"Vehicle (-82.8 mV): Avg IR={data1['average_ir']:.2f} MΩ, %IR={data1['average_percent_ir']:.2f}%, "
          f"IR1={data1['ir1']:.2f} MΩ, IR2={data1['ir2']:.2f} MΩ")
    print(f"FM550 (-82.3 mV):   Avg IR={data2['average_ir']:.2f} MΩ, %IR={data2['average_percent_ir']:.2f}%, "
          f"IR1={data2['ir1']:.2f} MΩ, IR2={data2['ir2']:.2f} MΩ")
    print(f"\nDifference in Average IR: {abs(data1['average_ir'] - data2['average_ir']):.2f} MΩ")
    print(f"Scaling factors used: Vehicle={data1['scaling_factor']:.3f}, FM550={data2['scaling_factor']:.3f}")
    print("="*80)
    
    # Create 3x3 plot
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # Row 1: Vehicle condition (blue)
    plot_condition_row(axes[0], data1, conditions['condition1']['color'], 
                      conditions['condition1']['name'])
    
    # Row 2: FM550 condition (red)
    plot_condition_row(axes[1], data2, conditions['condition2']['color'],
                      conditions['condition2']['name'])
    
    # Row 3: Comparison overlays
    plot_comparison_row(axes[2], data1, data2,
                       conditions['condition1']['color'],
                       conditions['condition2']['color'])

    # Column titles (shared for each column)
    axes[0, 0].set_title('Hyperpolarizing Current Injection vs Membrane Potential',
                         fontsize=14, fontweight='bold', pad=16)
    axes[0, 1].set_title('Hyperpolarizing Current Injection vs Change in Voltage',
                         fontsize=14, fontweight='bold', pad=16)
    axes[0, 2].set_title('Hyperpolarizing Current vs Inward Rectification',
                         fontsize=14, fontweight='bold', pad=16)

    # Panel labels A-I in upper-left of each subplot
    panel_labels = list("ABCDEFGHI")
    for idx, ax in enumerate(axes.flat):
        ax.text(-0.20, 1.10, panel_labels[idx], transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='left')

    # Legends inside first graph of row 1 and row 2
    control_handle = Line2D([0], [0], color=conditions['condition1']['color'], lw=2.5, label='Control')
    fm550_handle = Line2D([0], [0], color=conditions['condition2']['color'], lw=2.5, label='FM550')
    axes[0, 0].legend(handles=[control_handle], loc='lower right', frameon=True,
                      fontsize=13, borderpad=0.6)
    axes[1, 0].legend(handles=[fm550_handle], loc='lower right', frameon=True,
                      fontsize=13, borderpad=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.07, left=0.11, right=0.96, hspace=0.45, wspace=0.38)
    plt.show()
    
    return {
        'vehicle_condition': data1,
        'fm550_condition': data2,
        'summary': {
            'cell_type': cell_type,
            'cell_index': cell_index,
            'ir_difference': abs(data1['average_ir'] - data2['average_ir']),
            'vehicle_scaling': data1['scaling_factor'],
            'fm550_scaling': data2['scaling_factor']
        }
    }


if __name__ == '__main__':
    # Run the 3x3 analysis using exact 9.19 logic
    results = generate_3x3_analysis(cell_type='dmsn', cell_index=21)
    
    # Optional: Print detailed step-by-step results
    print(f"\n=== Detailed Step-by-Step Results ===")
    for condition_name, data in [('Vehicle', results['vehicle_condition']), 
                                ('FM550', results['fm550_condition'])]:
        print(f"\n{condition_name} Condition:")
        print(f"Scaling factor: {data['scaling_factor']:.3f}")
        print("Step-by-step IR values:")
        for i, (curr, ir) in enumerate(zip(data['currents'], data['inward_rectifications'])):
            print(f"  {curr:.2f} nA → {ir:.2f} MΩ")
        print(f"Percent IR steps: {[f'{v:.1f}' for v in data['percent_ir_steps']]}")