"""
Effects of channel conductance on cell excitability.

This script illustrates how to build a MSN and use that model to study
the effects of ion conductances on cell excitability.

A MSN is created and a current step is applied to its soma to stimulate
the cell. Then, a potassium conductance (kaf) is increased before
repeating the stimulation protocol. Both simulations are plotted to
compare the effect of `kaf` on cell activity.

A Gonzalez
"""
import numpy as np
import matplotlib.pyplot as plt

from msn.cell import MSN
from msn.instrumentation import Stim

# Make a MSN.
cell_type = 'dmsn'
cell_index = 40
cell = MSN(cell_type, cell_index)

# Setup stimulation protocol (current clamp step).
stim = Stim(cell)
stim.set_stim(delay=10, duration=200, amplitude=0.1, tmax=220)

# Create a figure for plotting the results.
ax = plt.figure().add_subplot(111)

# Model cell activity, first under control conditions (i.e. kaf maximum
# conductance gmax=100%), then after increasing this conductance by 20%
# in all cell segments.
for gmax in (1, 1.2):
    for section in cell.all:
        for segment in section:
            if hasattr(segment, 'kir'):
                segment.kaf.gbar *= gmax
    stim.run()

    # Plot the results
    stim.plot(ax=ax, label=f'kir {gmax:.0%}')

# Add a title and display
ax.set_title('{} #{}'.format(cell_type, cell_index))
plt.show()
