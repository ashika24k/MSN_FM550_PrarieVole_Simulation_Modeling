"""
Simulate dopamine (DA) or acetylcholine (ACh) modulation.

In this script a MSN is stimulated (current step) before and after
acetylcholine or dopamine modulation. The resulting voltage traces are
plotted together to compare the effects of modulation. The action
potentials are counted and displayed in the figure legend.
"""
import numpy as np
import matplotlib.pyplot as plt

from msn.cell import MSN
from msn.instrumentation import Stim, ActionPotentials
from msn.modulation import Dopamine, Acetylcholine

# Select whether to add dopamine or acetylcholine modulation; `modulate`
# should be 'DA' or 'ACh'.
# modulate = 'DA'
modulate = 'ACh'

# Make a MSN with background noise (without this background noise the
# cell will have no GABA or glut inputs to modulate).
cell_type = 'dmsn'
cell_index = 28
cell = MSN(cell_type, cell_index)
cell.add_bg_noise(gaba_freq=24, glut_freq=12)

# Setup stimulation protocol (current clamp step).
stim = Stim(cell)
stim.set_stim(delay=40, duration=250, amplitude=0.015, tmax=290,
              add_rheob=True)
ax = plt.figure().add_subplot(111)

# Run simulation.
stim.run()

# Get the action potentials in the voltage trace
ap = ActionPotentials(stim.t, stim.v)

# Plot, including the number of action potentials (n) in the label.
stim.plot(ax=ax, label=f'Control (n={ap.n})')

# Add modulation to the cell and run the simulation again.
if modulate == 'DA':
    mod = Dopamine(cell)
elif modulate == 'ACh':
    mod = Acetylcholine(cell)
stim.run()
ap = ActionPotentials(stim.t, stim.v)
stim.plot(ax=ax, label=f'+{modulate} (n={ap.n})')

# Add a title and display
ax.set_title('{} #{}'.format(cell_type, cell_index))
plt.show()
