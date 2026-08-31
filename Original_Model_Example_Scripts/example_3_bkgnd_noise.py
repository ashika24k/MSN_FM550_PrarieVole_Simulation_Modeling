"""
Add background noise to a MSN model.

This script illustrates how to add background noise (in the form of
random GABA and glutamate inputs) to a MSN model.

Note that, depending on the computer, the script may take a good number
of seconds before the results are displayed.

author: Antonio Gonzalez
"""
import numpy as np
from neuron import h
import matplotlib.pyplot as plt

from msn.cell import MSN

# Make a MSN model.
cell_type = 'dmsn'
cell_index = 18
cell = MSN(cell_type, cell_index)

# Create recording vectors: time and voltage (measured at the soma).
t  = h.Vector()
v  = h.Vector()
t.record(h._ref_t)
v.record(cell.soma(0.5)._ref_v)

# Add background noise to the cell with these GABA and glut input
# frequencies.
fglut = 12
fgaba = 24
cell.add_bg_noise(glut_freq=fglut, gaba_freq=fgaba)

# Run the simulation for these many ms and plot.
tstop = 1000
h.finitialize(cell.v_init)
while h.t < tstop:
    h.fadvance()
plt.plot(t, v, label=f'Glut={fglut}, GABA={fgaba}')

# Remove the background noise to see how membrane potential looks like
# without any noise (also useful for testing that the remove function
# works as expected). Run the simulation again and plot.
# cell.remove_bg_noise()
# h.finitialize(cell.v_init)
# while h.t < tstop:
#     h.fadvance()
# plt.plot(t, v, label='No noise')

# Modify background noise: this time there is more glutamate activity.
# Run again the simulation.
fglut = 36
fgaba = 24
cell.add_bg_noise(glut_freq=fglut, gaba_freq=fgaba)
h.finitialize(cell.v_init)
while h.t < tstop:
    h.fadvance()
plt.plot(t, v, label=f'Glut={fglut}, GABA={fgaba}')

# Label the plot and display the results.
plt.title(f'{cell_type} #{cell_index}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.legend()
plt.show()
