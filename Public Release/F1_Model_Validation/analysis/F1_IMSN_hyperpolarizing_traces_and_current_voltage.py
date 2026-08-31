import numpy as np
import matplotlib.pyplot as plt
from msn.cell import MSN  
from msn.instrumentation import Stim  

# Set Times New Roman font for all text
plt.rcParams['font.family'] = 'Times New Roman'

# === Create a Medium Spiny Neuron (MSN) ===
cell_type = 'imsn'  
cell_index =  32
v_init = -84.0  # Target initial voltage
cell = MSN(cell_type, cell_index, v_init=v_init)  

# === Calculate holding current to maintain v_init ===
print("Calculating holding current to maintain initial voltage...")

# Quick initial equilibration test
stim_test = Stim(cell)
stim_test.set_stim(delay=0, duration=200, amplitude=0.0, tmax=200, add_rheob=False)
stim_test.run()

# Measure natural equilibrium (without holding current)
v_test_array = np.array(stim_test.v)
natural_equilibrium = np.mean(v_test_array[-100:])

# Initial estimate of holding current
holding_current = (v_init - natural_equilibrium) / 100.0

# Iterative refinement - converge to exact v_init
for iteration in range(10):
    # Create new cell for each test
    cell_test2 = MSN(cell_type, cell_index, v_init=v_init)
    stim_test2 = Stim(cell_test2)
    stim_test2.set_stim(delay=0, duration=200, amplitude=holding_current, tmax=200, add_rheob=False)
    stim_test2.run()
    
    # Update natural_equilibrium with this holding current
    natural_equilibrium = np.mean(np.array(stim_test2.v)[-100:])
    
    # Check convergence
    if abs(natural_equilibrium - v_init) < 0.01:
        print(f"Converged in {iteration + 1} iterations (error: {abs(natural_equilibrium - v_init):.4f} mV)")
        break
    
    # Adjust holding current
    holding_current += (v_init - natural_equilibrium) / 100.0

print(f"Holding current: {holding_current:.4f} nA")
print(f"Natural equilibrium: {natural_equilibrium:.2f} mV")
print(f"Target v_init: {v_init:.2f} mV\n")

# === Set up the plot ===
plt.figure(figsize=(8, 6))  

# Define dark blue color for all traces
dark_blue = "#285C81"
dark_red = "#9D2222"  # Dark red color

# Scaling factor to amplify voltage changes (makes rectification more visible)
scaling_factor = 1

# Store the voltage changes and current amplitudes for non-linearity analysis
current_injections = []
voltage_changes = []

# === Setup simulation parameters ===
from neuron import h

pre_equilibration_time = 200
delay = 10
duration = 150
tmax = pre_equilibration_time + delay + duration + 150

# Create ONE cell for all simulations
cell_reuse = MSN(cell_type, cell_index, v_init=v_init)

# Setup TWO IClamps: one for holding current (continuous), one for test current (pulsed)
iclamp_hold = h.IClamp(cell_reuse.soma(0.5))
iclamp_hold.delay = 0
iclamp_hold.dur = tmax
iclamp_hold.amp = holding_current

iclamp_test = h.IClamp(cell_reuse.soma(0.5))
iclamp_test.delay = pre_equilibration_time + delay
iclamp_test.dur = duration

# Recording vectors
v_rec = h.Vector()
t_rec = h.Vector()
v_rec.record(cell_reuse.soma(0.5)._ref_v)
t_rec.record(h._ref_t)

# === Loop over negative current injections ===
for i, amplitude in enumerate(np.arange(-0.01, -1.0, -0.10)):  

    iclamp_test.amp = amplitude
    
    h.tstop = tmax
    h.v_init = v_init
    h.finitialize(v_init)
    h.run()
    
    # Convert to arrays
    t_array = np.array(t_rec.as_numpy())
    v_array = np.array(v_rec.as_numpy())
    
    # Only plot from after pre-equilibration
    plot_start_idx = np.searchsorted(t_array, pre_equilibration_time)
    t_plot = t_array[plot_start_idx:] - pre_equilibration_time
    v_plot = v_array[plot_start_idx:]
    
    # Calculate steady-state voltage during stimulation
    test_end = delay + duration
    steady_start_time = test_end - 50
    steady_mask = (t_array >= (pre_equilibration_time + steady_start_time)) & (t_array <= (pre_equilibration_time + test_end))
    v_steady = np.mean(v_array[steady_mask])
    
    # Calculate scaled voltage change (amplifies the deflection)
    delta_v = (v_steady - v_init) * scaling_factor
    
    # Apply scaling only during stimulus period, keep baseline and recovery at v_init
    v_plot_scaled = v_plot.copy()
    
    # Apply scaling factor to the deflection during stimulus only
    stim_start_idx = np.searchsorted(t_plot, delay)
    stim_end_idx = np.searchsorted(t_plot, delay + duration)
    
    # Scale the stimulus period
    for idx in range(stim_start_idx, stim_end_idx):
        deflection = v_plot[idx] - v_init
        v_plot_scaled[idx] = v_init + deflection * scaling_factor

    # Store the data for non-linearity plot
    current_injections.append(amplitude)
    voltage_changes.append(delta_v)

    print(f"Test current: {amplitude:.2f} nA, v_steady: {v_steady:.2f} mV, Scaled ΔV: {delta_v:.2f} mV")
    
    # === Plot scaled voltage response ===
    plt.plot(t_plot, v_plot_scaled, color=dark_blue, linewidth=1.5)

plt.xlabel("Time (ms)", fontsize=18)  
plt.ylabel("Membrane potential (mV)", fontsize=18)  
plt.tight_layout()
plt.show()

# === Plot the non-linearity of rectification ===
plt.figure(figsize=(8, 6))
plt.plot(current_injections, voltage_changes, marker='o', color=dark_blue, linestyle='-', markersize=6, linewidth=1.5)

plt.xlabel("Injected Current (nA)", fontsize=18)
plt.ylabel("Change in Membrane Potential (mV)", fontsize=18)
plt.tight_layout()
plt.show()