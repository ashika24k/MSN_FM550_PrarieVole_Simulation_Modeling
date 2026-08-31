import numpy as np
import matplotlib.pyplot as plt
from msn.cell import MSN
from msn.instrumentation import Stim

# Create a MSN neuron
cell_type = 'dmsn'
cell_index = 21
cell = MSN(cell_type, cell_index)

# Define current injection
current_injection = 0.05  # Single current injection at 0.05 nA

print(f"Current injection: {current_injection} nA")
print(f"Cell rheobase: {cell.rheobase:.1f} pA")

# Set Times New Roman font for all text
plt.rcParams['font.family'] = 'Times New Roman'

# Initialize the plot
plt.figure(figsize=(12, 8))

# Get initial membrane voltage by running a zero current injection
stim_init = Stim(cell)
stim_init.set_stim(delay=10, duration=150, amplitude=0.0, tmax=200, add_rheob=False)
stim_init.run()
initial_voltage = float(np.mean(np.array(stim_init.v)[:100]))  # Average voltage before stimulus
print(f"Initial membrane voltage: {initial_voltage:.2f} mV")

# Plot current injection in dark red
# Plot current injection in dark red
print(f"Running current injection at {current_injection} nA...")
dark_red = "#9D2222"  # Dark red color
stim = Stim(cell)
stim.set_stim(delay=10, duration=150, amplitude=current_injection, tmax=200, add_rheob=True)
stim.run()

# Plot with dark red color
plt.plot(stim.t, stim.v, color=dark_red, linewidth=1.5)

# Formatting the plot
plt.xlabel("Time (ms)", fontsize=18)
plt.ylabel("Membrane Potential (mV)", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.show()

print("\n=== SUMMARY ===")
print(f"Current injection: {current_injection} nA")
print(f"Stimulus period: 10-160 ms")
print(f"Initial membrane voltage: {initial_voltage:.2f} mV")