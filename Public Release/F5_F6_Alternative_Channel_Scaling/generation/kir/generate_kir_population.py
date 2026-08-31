"""
Population Analysis: Percent Difference for all DMSN and IMSN cells
Comparing FM550 vs Control, 2x kir vs Control, and 3x kir vs Control

This script calculates the percent difference in average inward rectification
for all 71 DMSN cells and 34 IMSN cells under different conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from msn.cell import MSN
from msn.instrumentation import Stim
from neuron import h
import time
import sys
import signal
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Modern color palette with high contrast and differentiation
MODERN_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Set modern plotting style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'axes.linewidth': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'legend.framealpha': 0.9,
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC'
})

# Global variable to track interruption
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\n\n⚠️  Interruption requested! Finishing current simulation step...")

signal.signal(signal.SIGINT, signal_handler)

def modify_kir_conductance(cell, scale_factor=1.0, compartment='all'):
    """Modify kir conductance by a scale factor."""
    for section in cell.all:
        if _should_modify_section(section, compartment):
            if hasattr(section, 'gbar_kir'):
                for seg in section:
                    seg.gbar_kir *= scale_factor

def _should_modify_section(section, compartment):
    """Helper function to determine if section should be modified."""
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

def calculate_rectification_fast(cell_type='dmsn', cell_index=0, v_init=-82.8, 
                                kir_modification=None, use_female_params=False):
    """
    Calculate rectification for a single cell with optimized parameters and exact holding current.
    """
    global interrupted
    if interrupted:
        return None
        
    try:
        # Parameters based on cell type
        if use_female_params:
            rheobase_pA = 153.0
            scaling_factor = 163.8 / 33.216
        else:
            rheobase_pA = 112.5
            scaling_factor = 211.7 / 32.205
        
        neg_currents = np.arange(-0.09, -0.01, 0.01)  # Full 8 steps for accuracy
        rheobase_nA = rheobase_pA / 1000.0
        
        # Stimulus parameters
        pre_equilibration_time = 200
        delay = 10
        duration = 150
        tmax = pre_equilibration_time + delay + duration + 150
        
        # Pre-allocate arrays
        num_steps = len(neg_currents)
        inward_rectifications = np.zeros(num_steps)
        
        # CREATE CELL ONCE - reuse for holding current calculation AND main simulation
        cell = MSN(cell_type, cell_index, v_init=v_init)
        
        # Apply kir modifications once
        if kir_modification is not None and 'scale_factor' in kir_modification:
            modify_kir_conductance(cell, kir_modification['scale_factor'])
        
        # Step 1: Calculate holding current using SAME cell
        stim = Stim(cell)
        stim.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
        stim.run()
        
        # Measure natural equilibrium (without holding current)
        v_test_array = np.array(stim.v)
        natural_equilibrium = np.mean(v_test_array[-100:])
        
        # Initial estimate of holding current
        holding_current = (v_init - natural_equilibrium) / 100.0
        
        # Iterative refinement - converge to exact v_init, REUSING same cell
        for iteration in range(10):
            stim.set_stim(delay=0, duration=200, amplitude=holding_current, tmax=200, add_rheob=False)
            stim.run()
            
            # Update natural_equilibrium after each test
            natural_equilibrium = np.mean(np.array(stim.v)[-100:])
            
            # Check convergence
            if abs(natural_equilibrium - v_init) < 0.01:
                break
            
            # Adjust holding current
            holding_current += (v_init - natural_equilibrium) / 100.0
        
        # Step 2: Use SAME cell for main simulation (no need to recreate!)
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
        
        for step_idx, i_test in enumerate(neg_currents):
            if interrupted:
                break
            
            # Update current amplitude (holding + test current)
            iclamp.amp = holding_current + i_test
            
            # Run simulation
            h.finitialize(v_init)
            h.continuerun(tmax)
            
            # Convert NEURON vectors to numpy arrays
            v_array = np.array(v_rec.as_numpy())
            t_array = np.array(t_rec.as_numpy())
            
            # Use searchsorted for fast indexing (faster than masking)
            steady_start_idx = np.searchsorted(t_array, steady_start_time)
            steady_end_idx = np.searchsorted(t_array, steady_end_time)
            v_steady = np.mean(v_array[steady_start_idx:steady_end_idx])
            
            # Calculate delta_v with scaling factor
            delta_v = (v_steady - v_init) * scaling_factor
            
            # Store results
            inward_rectifications[step_idx] = abs(delta_v / i_test)
        
        if len(inward_rectifications) == 0:
            return None, None
            
        average_inward_rectification = np.mean(inward_rectifications)
        
        return average_inward_rectification, holding_current
        
    except Exception as e:
        print(f"Error in cell {cell_index}: {e}")
        return None

def run_population_analysis(cell_types=['dmsn', 'imsn'], cell_indices=None, sample_size=None):
    """
    Run population analysis for MSN cells comparing different conditions.
    
    Parameters:
    -----------
    cell_types : list
        List of cell types to analyze ('dmsn', 'imsn')
    cell_indices : dict, optional
        Dictionary with cell_type as key and list of indices as values
        e.g., {'dmsn': [0, 1, 2], 'imsn': [0, 1, 2]}
    sample_size : dict or int, optional
        Dictionary with cell_type as key and sample size as value
        e.g., {'dmsn': 20, 'imsn': 10} or single int for same size for all
    """
    
    # Define available cells for each type
    available_cells = {'dmsn': 71, 'imsn': 34}
    
    # Determine which cells to analyze for each type
    analysis_plan = {}
    
    for cell_type in cell_types:
        if cell_indices is not None and cell_type in cell_indices:
            analysis_plan[cell_type] = cell_indices[cell_type]
        elif sample_size is not None:
            if isinstance(sample_size, dict):
                size = sample_size.get(cell_type, 10)
            else:
                size = sample_size
            # Random sample
            np.random.seed(42)  # For reproducibility
            max_cells = available_cells[cell_type]
            size = min(size, max_cells)
            selected_cells = np.random.choice(max_cells, size, replace=False)
            analysis_plan[cell_type] = sorted(selected_cells)
        else:
            # All available cells
            analysis_plan[cell_type] = list(range(available_cells[cell_type]))
    
    print(f"\n{'='*80}")
    print(f"POPULATION ANALYSIS: MSN CELLS")
    for cell_type in cell_types:
        cells = analysis_plan[cell_type]
        print(f"{cell_type.upper()}: {len(cells)} cells - {cells[:10]}{'...' if len(cells) > 10 else ''}")
    print(f"Total cells to analyze: {sum(len(analysis_plan[ct]) for ct in cell_types)}")
    print(f"{'='*80}")
    
    # Storage for results
    results_data = {
        'cell_type': [],
        'cell_index': [],
        'control_ir': [],
        'fm550_ir': [],
        'kir125x_ir': [],
        'kir15x_ir': [],
        'control_holding_current': [],
        'fm550_holding_current': [],
        'kir125x_holding_current': [],
        'kir15x_holding_current': [],
        'fm550_vs_control_percent': [],
        'kir125x_vs_control_percent': [],
        'kir15x_vs_control_percent': []
    }
    
    total_cells = sum(len(analysis_plan[ct]) for ct in cell_types)
    cell_counter = 0
    start_time = time.time()
    
    for cell_type in cell_types:
        cells_to_analyze = analysis_plan[cell_type]
        print(f"\n{'='*60}")
        print(f"ANALYZING {cell_type.upper()} CELLS")
        print(f"{'='*60}")
        
        for i, cell_idx in enumerate(cells_to_analyze):
            cell_counter += 1
            
            if interrupted:
                print(f"\n❌ Analysis interrupted at cell {cell_counter}/{total_cells}")
                break
                
            print(f"\n[{cell_counter}/{total_cells}] Analyzing {cell_type.upper()} Cell #{cell_idx}")
            cell_start_time = time.time()
            
            try:
                # Condition 1: Control (Female Vehicle)
                print(f"  Running Control condition...", end=" ")
                control_ir, control_holding_current = calculate_rectification_fast(
                    cell_type=cell_type, cell_index=cell_idx, v_init=-82.8,
                    kir_modification=None, use_female_params=False
                )
                print("✓")
                
                if control_ir is None:
                    print(f"  ❌ Failed to calculate control for {cell_type} cell {cell_idx}")
                    continue
                
                # Condition 2: Female FM550
                print(f"  Running FM550 condition...", end=" ")
                fm550_ir, fm550_holding_current = calculate_rectification_fast(
                    cell_type=cell_type, cell_index=cell_idx, v_init=-82.3,
                    kir_modification=None, use_female_params=True
                )
                print("✓")
                
                if fm550_ir is None:
                    print(f"  ❌ Failed to calculate FM550 for {cell_type} cell {cell_idx}")
                    continue
                
                # Condition 3: 1.25x kir
                print(f"  Running 1.25x kir condition...", end=" ")
                kir125x_ir, kir125x_holding_current = calculate_rectification_fast(
                    cell_type=cell_type, cell_index=cell_idx, v_init=-82.8,
                    kir_modification={'scale_factor': 1.25}, use_female_params=False
                )
                print("✓")
                
                if kir125x_ir is None:
                    print(f"  ❌ Failed to calculate 1.25x kir for {cell_type} cell {cell_idx}")
                    continue
                
                # Condition 4: 1.5x kir
                print(f"  Running 1.5x kir condition...", end=" ")
                kir15x_ir, kir15x_holding_current = calculate_rectification_fast(
                    cell_type=cell_type, cell_index=cell_idx, v_init=-82.8,
                    kir_modification={'scale_factor': 1.5}, use_female_params=False
                )
                print("✓")
                
                if kir15x_ir is None:
                    print(f"  ❌ Failed to calculate 1.5x kir for {cell_type} cell {cell_idx}")
                    continue
                
                # Calculate percent differences
                fm550_percent = ((fm550_ir - control_ir) / control_ir) * 100
                kir125x_percent = ((kir125x_ir - control_ir) / control_ir) * 100
                kir15x_percent = ((kir15x_ir - control_ir) / control_ir) * 100
                
                # Store results
                results_data['cell_type'].append(cell_type)
                results_data['cell_index'].append(cell_idx)
                results_data['control_ir'].append(control_ir)
                results_data['fm550_ir'].append(fm550_ir)
                results_data['kir125x_ir'].append(kir125x_ir)
                results_data['kir15x_ir'].append(kir15x_ir)
                results_data['control_holding_current'].append(control_holding_current)
                results_data['fm550_holding_current'].append(fm550_holding_current)
                results_data['kir125x_holding_current'].append(kir125x_holding_current)
                results_data['kir15x_holding_current'].append(kir15x_holding_current)
                results_data['fm550_vs_control_percent'].append(fm550_percent)
                results_data['kir125x_vs_control_percent'].append(kir125x_percent)
                results_data['kir15x_vs_control_percent'].append(kir15x_percent)
                
                cell_time = time.time() - cell_start_time
                print(f"  ✓ {cell_type.upper()} {cell_idx} completed in {cell_time:.1f}s")
                print(f"    Control: {control_ir:.2f} MΩ | FM550: {fm550_ir:.2f} MΩ ({fm550_percent:+.1f}%)")
                print(f"    1.25x kir: {kir125x_ir:.2f} MΩ ({kir125x_percent:+.1f}%) | 1.5x kir: {kir15x_ir:.2f} MΩ ({kir15x_percent:+.1f}%)")
                
                # Progress update
                elapsed_time = time.time() - start_time
                cells_remaining = total_cells - cell_counter
                if cell_counter > 1:
                    avg_time_per_cell = elapsed_time / cell_counter
                    estimated_remaining_time = avg_time_per_cell * cells_remaining
                    print(f"    Progress: {cell_counter}/{total_cells} ({(cell_counter/total_cells)*100:.1f}%) | "
                          f"Elapsed: {elapsed_time/60:.1f}min | ETA: {estimated_remaining_time/60:.1f}min")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {cell_type} cell {cell_idx}: {e}")
                continue
        
        if interrupted:
            break
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results_data)
    
    if len(df) == 0:
        print("❌ No successful analyses completed!")
        return None
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETED!")
    dmsn_count = len(df[df['cell_type'] == 'dmsn'])
    imsn_count = len(df[df['cell_type'] == 'imsn'])
    print(f"Successfully analyzed: {len(df)} total cells")
    print(f"  - DMSN: {dmsn_count} cells")
    print(f"  - IMSN: {imsn_count} cells")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"{'='*80}")
    
    return df

def create_population_plots(df):
    """Create comprehensive population plots for both DMSN and IMSN with 1.25x and 1.5x kir."""
    
    # Separate data by cell type
    dmsn_df = df[df['cell_type'] == 'dmsn'].copy()
    imsn_df = df[df['cell_type'] == 'imsn'].copy()
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"POPULATION STATISTICS SUMMARY")
    print(f"{'='*80}")
    
    for cell_type, sub_df in [('DMSN', dmsn_df), ('IMSN', imsn_df)]:
        if len(sub_df) > 0:
            print(f"\n{cell_type} (n={len(sub_df)}):")
            print(f"  FM550 vs Control:")
            print(f"    Mean ± SEM: {sub_df['fm550_vs_control_percent'].mean():.1f} ± {sub_df['fm550_vs_control_percent'].sem():.1f}%")
            print(f"    Range: {sub_df['fm550_vs_control_percent'].min():.1f}% to {sub_df['fm550_vs_control_percent'].max():.1f}%")
            print(f"  1.25x kir vs Control:")
            print(f"    Mean ± SEM: {sub_df['kir125x_vs_control_percent'].mean():.1f} ± {sub_df['kir125x_vs_control_percent'].sem():.1f}%")
            print(f"    Range: {sub_df['kir125x_vs_control_percent'].min():.1f}% to {sub_df['kir125x_vs_control_percent'].max():.1f}%")
            print(f"  1.5x kir vs Control:")
            print(f"    Mean ± SEM: {sub_df['kir15x_vs_control_percent'].mean():.1f} ± {sub_df['kir15x_vs_control_percent'].sem():.1f}%")
            print(f"    Range: {sub_df['kir15x_vs_control_percent'].min():.1f}% to {sub_df['kir15x_vs_control_percent'].max():.1f}%")
    
    # Statistical tests
    print(f"\nSTATISTICAL TESTS:")
    
    for cell_type, sub_df in [('DMSN', dmsn_df), ('IMSN', imsn_df)]:
        if len(sub_df) > 1:
            print(f"\n{cell_type}:")
            # One-sample t-test against zero (no change)
            fm550_tstat, fm550_pval = stats.ttest_1samp(sub_df['fm550_vs_control_percent'], 0)
            kir125x_tstat, kir125x_pval = stats.ttest_1samp(sub_df['kir125x_vs_control_percent'], 0)
            kir15x_tstat, kir15x_pval = stats.ttest_1samp(sub_df['kir15x_vs_control_percent'], 0)
            
            print(f"  FM550 vs Control: t({len(sub_df)-1}) = {fm550_tstat:.3f}, p = {fm550_pval:.6f}")
            print(f"  1.25x kir vs Control: t({len(sub_df)-1}) = {kir125x_tstat:.3f}, p = {kir125x_pval:.6f}")
            print(f"  1.5x kir vs Control: t({len(sub_df)-1}) = {kir15x_tstat:.3f}, p = {kir15x_pval:.6f}")
            
            # Paired t-tests between conditions
            fm550_kir125x_tstat, fm550_kir125x_pval = stats.ttest_rel(sub_df['fm550_vs_control_percent'], sub_df['kir125x_vs_control_percent'])
            fm550_kir15x_tstat, fm550_kir15x_pval = stats.ttest_rel(sub_df['fm550_vs_control_percent'], sub_df['kir15x_vs_control_percent'])
            kir125x_kir15x_tstat, kir125x_kir15x_pval = stats.ttest_rel(sub_df['kir125x_vs_control_percent'], sub_df['kir15x_vs_control_percent'])
            
            print(f"  FM550 vs 1.25x kir: t({len(sub_df)-1}) = {fm550_kir125x_tstat:.3f}, p = {fm550_kir125x_pval:.6f}")
            print(f"  FM550 vs 1.5x kir: t({len(sub_df)-1}) = {fm550_kir15x_tstat:.3f}, p = {fm550_kir15x_pval:.6f}")
            print(f"  1.25x kir vs 1.5x kir: t({len(sub_df)-1}) = {kir125x_kir15x_tstat:.3f}, p = {kir125x_kir15x_pval:.6f}")
    
    # Between cell type comparisons if both types available
    if len(dmsn_df) > 0 and len(imsn_df) > 0:
        print(f"\nBETWEEN CELL TYPE COMPARISONS:")
        # Independent t-tests between DMSN and IMSN
        fm550_tstat, fm550_pval = stats.ttest_ind(dmsn_df['fm550_vs_control_percent'], imsn_df['fm550_vs_control_percent'])
        kir125x_tstat, kir125x_pval = stats.ttest_ind(dmsn_df['kir125x_vs_control_percent'], imsn_df['kir125x_vs_control_percent'])
        kir15x_tstat, kir15x_pval = stats.ttest_ind(dmsn_df['kir15x_vs_control_percent'], imsn_df['kir15x_vs_control_percent'])
        
        print(f"  FM550 (DMSN vs IMSN): t = {fm550_tstat:.3f}, p = {fm550_pval:.6f}")
        print(f"  1.25x kir (DMSN vs IMSN): t = {kir125x_tstat:.3f}, p = {kir125x_pval:.6f}")
        print(f"  1.5x kir (DMSN vs IMSN): t = {kir15x_tstat:.3f}, p = {kir15x_pval:.6f}")
    
    print(f"{'='*80}")
    
    # Create the main overlay figure
    fig_overlay = plt.figure(figsize=(18, 12))
    
    # Create subplots: one for each cell type and one combined
    gs = fig_overlay.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Define colors for each cell type and condition (publishing standard)
    # DMSN = red hues, IMSN = blue hues
    dmsn_fm550_color = '#8B0000'      # Dark red
    dmsn_kir125x_color = '#DC143C'    # Crimson
    dmsn_kir15x_color = '#FF6B6B'     # Light red
    imsn_fm550_color = '#00008B'      # Dark blue
    imsn_kir125x_color = '#1E90FF'    # Dodger blue
    imsn_kir15x_color = '#87CEEB'     # Sky blue
    
    # Combined plot (top spanning both columns)
    ax_combined = fig_overlay.add_subplot(gs[0, :])
    
    # Plot DMSN data
    if len(dmsn_df) > 0:
        x_dmsn = np.arange(len(dmsn_df))
        ax_combined.scatter(x_dmsn, dmsn_df['fm550_vs_control_percent'], 
                          color=dmsn_fm550_color, alpha=0.8, s=60, marker='o', 
                          edgecolors='black', linewidth=1, 
                          label=f'DMSN FM550 vs Control (n={len(dmsn_df)})')
        ax_combined.scatter(x_dmsn, dmsn_df['kir125x_vs_control_percent'], 
                          color=dmsn_kir125x_color, alpha=0.8, s=60, marker='o', 
                          edgecolors='black', linewidth=1,
                          label=f'DMSN 1.25x kir vs Control (n={len(dmsn_df)})')
        ax_combined.scatter(x_dmsn, dmsn_df['kir15x_vs_control_percent'], 
                          color=dmsn_kir15x_color, alpha=0.8, s=60, marker='o', 
                          edgecolors='black', linewidth=1,
                          label=f'DMSN 1.5x kir vs Control (n={len(dmsn_df)})')
        
        # Connect paired points
        for i in range(len(dmsn_df)):
            ax_combined.plot([x_dmsn[i], x_dmsn[i]], 
                           [dmsn_df['fm550_vs_control_percent'].iloc[i], 
                            dmsn_df['kir125x_vs_control_percent'].iloc[i]], 
                           color='red', alpha=0.2, linewidth=1)
            ax_combined.plot([x_dmsn[i], x_dmsn[i]], 
                           [dmsn_df['kir125x_vs_control_percent'].iloc[i], 
                            dmsn_df['kir15x_vs_control_percent'].iloc[i]], 
                           color='red', alpha=0.2, linewidth=1)
    
    # Plot IMSN data (offset for visibility)
    if len(imsn_df) > 0:
        x_offset = len(dmsn_df) if len(dmsn_df) > 0 else 0
        x_imsn = np.arange(len(imsn_df)) + x_offset + 1  # Small gap between cell types
        
        ax_combined.scatter(x_imsn, imsn_df['fm550_vs_control_percent'], 
                          color=imsn_fm550_color, alpha=0.8, s=60, marker='o', 
                          edgecolors='black', linewidth=1, 
                          label=f'IMSN FM550 vs Control (n={len(imsn_df)})')
        ax_combined.scatter(x_imsn, imsn_df['kir125x_vs_control_percent'], 
                          color=imsn_kir125x_color, alpha=0.8, s=60, marker='o', 
                          edgecolors='black', linewidth=1,
                          label=f'IMSN 1.25x kir vs Control (n={len(imsn_df)})')
        ax_combined.scatter(x_imsn, imsn_df['kir15x_vs_control_percent'], 
                          color=imsn_kir15x_color, alpha=0.8, s=60, marker='o', 
                          edgecolors='black', linewidth=1,
                          label=f'IMSN 1.5x kir vs Control (n={len(imsn_df)})')
        
        # Connect paired points
        for i in range(len(imsn_df)):
            ax_combined.plot([x_imsn[i], x_imsn[i]], 
                           [imsn_df['fm550_vs_control_percent'].iloc[i], 
                            imsn_df['kir125x_vs_control_percent'].iloc[i]], 
                           color='green', alpha=0.2, linewidth=1)
            ax_combined.plot([x_imsn[i], x_imsn[i]], 
                           [imsn_df['kir125x_vs_control_percent'].iloc[i], 
                            imsn_df['kir15x_vs_control_percent'].iloc[i]], 
                           color='green', alpha=0.2, linewidth=1)
    
    # Add horizontal reference lines
    ax_combined.axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='No Change (0%)')
    
    # Add mean lines for each condition and cell type
    if len(dmsn_df) > 0:
        dmsn_fm550_mean = dmsn_df['fm550_vs_control_percent'].mean()
        dmsn_kir125x_mean = dmsn_df['kir125x_vs_control_percent'].mean()
        dmsn_kir15x_mean = dmsn_df['kir15x_vs_control_percent'].mean()
        ax_combined.axhline(dmsn_fm550_mean, color=dmsn_fm550_color, linestyle='-', alpha=0.7, linewidth=3)
        ax_combined.axhline(dmsn_kir125x_mean, color=dmsn_kir125x_color, linestyle='-', alpha=0.7, linewidth=3)
        ax_combined.axhline(dmsn_kir15x_mean, color=dmsn_kir15x_color, linestyle='-', alpha=0.7, linewidth=3)
    
    if len(imsn_df) > 0:
        imsn_fm550_mean = imsn_df['fm550_vs_control_percent'].mean()
        imsn_kir125x_mean = imsn_df['kir125x_vs_control_percent'].mean()
        imsn_kir15x_mean = imsn_df['kir15x_vs_control_percent'].mean()
        ax_combined.axhline(imsn_fm550_mean, color=imsn_fm550_color, linestyle='-', alpha=0.7, linewidth=3)
        ax_combined.axhline(imsn_kir125x_mean, color=imsn_kir125x_color, linestyle='-', alpha=0.7, linewidth=3)
        ax_combined.axhline(imsn_kir15x_mean, color=imsn_kir15x_color, linestyle='-', alpha=0.7, linewidth=3)
    
    # Formatting for combined plot
    ax_combined.set_xlabel('Cell Number (DMSN then IMSN)', fontweight='bold', fontsize=12)
    ax_combined.set_ylabel('Percent Change from Control (%)', fontweight='bold', fontsize=12)
    ax_combined.set_title(f'Population Analysis: FM550 vs 1.25x kir vs 1.5x kir Effects\n'
                         f'DMSN (n={len(dmsn_df)}) and IMSN (n={len(imsn_df)}) Combined', 
                         fontweight='bold', fontsize=14, pad=15)
    ax_combined.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, fontsize=8)
    ax_combined.grid(True, alpha=0.3)
    ax_combined.set_facecolor('#FAFAFA')
    
    # Individual cell type plots
    cell_type_data = [('DMSN', dmsn_df, dmsn_fm550_color, dmsn_kir125x_color, dmsn_kir15x_color), 
                      ('IMSN', imsn_df, imsn_fm550_color, imsn_kir125x_color, imsn_kir15x_color)]
    
    for idx, (cell_type, sub_df, fm550_color, kir125x_color, kir15x_color) in enumerate(cell_type_data):
        if len(sub_df) == 0:
            continue
            
        ax = fig_overlay.add_subplot(gs[1, idx])
        
        x_positions = np.arange(len(sub_df))
        
        # Plot data points
        ax.scatter(x_positions, sub_df['fm550_vs_control_percent'], 
                  color=fm550_color, alpha=0.8, s=70, marker='o', 
                  edgecolors='black', linewidth=1, 
                  label=f'FM550 vs Control')
        ax.scatter(x_positions, sub_df['kir125x_vs_control_percent'], 
                  color=kir125x_color, alpha=0.8, s=70, marker='o', 
                  edgecolors='black', linewidth=1,
                  label=f'1.25x kir vs Control')
        ax.scatter(x_positions, sub_df['kir15x_vs_control_percent'], 
                  color=kir15x_color, alpha=0.8, s=70, marker='o', 
                  edgecolors='black', linewidth=1,
                  label=f'1.5x kir vs Control')
        
        # Connect paired points
        for i in range(len(sub_df)):
            ax.plot([x_positions[i], x_positions[i]], 
                   [sub_df['fm550_vs_control_percent'].iloc[i], 
                    sub_df['kir125x_vs_control_percent'].iloc[i]], 
                   color='gray', alpha=0.3, linewidth=1)
            ax.plot([x_positions[i], x_positions[i]], 
                   [sub_df['kir125x_vs_control_percent'].iloc[i], 
                    sub_df['kir15x_vs_control_percent'].iloc[i]], 
                   color='gray', alpha=0.3, linewidth=1)
        
        # Reference lines
        ax.axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        
        # Mean lines
        fm550_mean = sub_df['fm550_vs_control_percent'].mean()
        kir125x_mean = sub_df['kir125x_vs_control_percent'].mean()
        kir15x_mean = sub_df['kir15x_vs_control_percent'].mean()
        ax.axhline(fm550_mean, color=fm550_color, linestyle='-', alpha=0.8, linewidth=3)
        ax.axhline(kir125x_mean, color=kir125x_color, linestyle='-', alpha=0.8, linewidth=3)
        ax.axhline(kir15x_mean, color=kir15x_color, linestyle='-', alpha=0.8, linewidth=3)
        
        # Formatting
        ax.set_xlabel(f'{cell_type} Cell Number', fontweight='bold', fontsize=11)
        ax.set_ylabel('Percent Change (%)', fontweight='bold', fontsize=11)
        ax.set_title(f'{cell_type} Population (n={len(sub_df)})', fontweight='bold', fontsize=12)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        
        # Add statistics text box
        fm550_sem = sub_df['fm550_vs_control_percent'].sem()
        kir125x_sem = sub_df['kir125x_vs_control_percent'].sem()
        kir15x_sem = sub_df['kir15x_vs_control_percent'].sem()
        
        textstr = f'FM550: {fm550_mean:.1f} ± {fm550_sem:.1f}%\n' \
                  f'1.25x kir: {kir125x_mean:.1f} ± {kir125x_sem:.1f}%\n' \
                  f'1.5x kir: {kir15x_mean:.1f} ± {kir15x_sem:.1f}%'
        
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig_overlay

def save_results_to_file(df, filename=None):
    """Save results to separate CSV files for each cell type."""
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "MSN_Population_Analysis_Results_125x15x.csv")
    # Save separate files for each cell type
    dmsn_df = df[df['cell_type'] == 'dmsn']
    imsn_df = df[df['cell_type'] == 'imsn']
    
    if len(dmsn_df) > 0:
        dmsn_filename = filename.replace('.csv', '_DMSN.csv')
        dmsn_df.to_csv(dmsn_filename, index=False)
        print(f"✅ DMSN results saved to: {dmsn_filename}")
    
    if len(imsn_df) > 0:
        imsn_filename = filename.replace('.csv', '_IMSN.csv')
        imsn_df.to_csv(imsn_filename, index=False)
        print(f"✅ IMSN results saved to: {imsn_filename}")

if __name__ == '__main__':
    print("=" * 80)
    print("MSN POPULATION ANALYSIS: FM550 vs 1.25x kir vs 1.5x kir")
    print("DMSN and IMSN Cell Types")
    print("=" * 80)
    
    # You can modify these parameters:
    # - To run all cells: df = run_population_analysis()
    # - To run sample sizes: df = run_population_analysis(sample_size={'dmsn': 10, 'imsn': 5})
    # - To run specific cells: df = run_population_analysis(cell_indices={'dmsn': [0, 1, 2], 'imsn': [0, 1, 2]})
    # - To run only one cell type: df = run_population_analysis(cell_types=['dmsn'])
    
    # Running COMPLETE POPULATION ANALYSIS - ALL CELLS
    print("Running COMPLETE POPULATION ANALYSIS:")
    print("  - ALL 71 DMSN cells")
    print("  - ALL 34 IMSN cells")
    print("  - Total: 105 cells")
    print("  - Conditions: Control, FM550, 1.25x kir, 1.5x kir")
    print("This will take approximately 40-50 minutes to complete.")
    print("Press Ctrl+C to interrupt if needed.")
    
    df = run_population_analysis()
    
    if df is not None:
        # Create and display plots
        fig_overlay = create_population_plots(df)
        
        # Save results
        save_results_to_file(df)
        
        print(f"\n{'='*80}")
        print("🎉 MSN POPULATION ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
        print(f"{'='*80}")
    else:
        print("❌ Analysis failed - no results to display")
