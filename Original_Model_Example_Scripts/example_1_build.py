"""
Build a MSN.

This script shows how to build a model of a MSN. To observe cell
activity, a current step is applied to the soma of the cell and the
cell's membrane potential is plotted.

A Gonzalez
"""
import numpy as np 
import matplotlib.pyplot as plt 

from msn.cell import MSN
from msn.instrumentation import Stim

# Make a MSN. Cell type can be 'imsn' or 'dmsn'. There are 71 different
# sets of parameters for modelling dMSNs and 34 different sets for
# iMSNs. Which one of these to use is selected with the `cell_index`
# variable.
cell_type = 'dmsn'
cell_index = 21
cell = MSN(cell_type, cell_index)

# Setup and run the stimulation protocol (current clamp step). The delay and
# duration of the step are in ms. The amplitude is the current (in nA) applied
# to the cell; by default this current is added on top of the cell's
# rheobase (change this behaviour with the `add_rheob` setting). `tmax`
# sets the total duration (in ms) of the simulation.
stim = Stim(cell)
stim.set_stim(delay=10, duration=150, amplitude=0.1, tmax=200,
              add_rheob=True)
stim.run()

# Plot the results, add a title and display. Note how the function
# `stim.plot()` adds by default suitable labels to the x and y axes.
stim.plot()
plt.title('{} #{}'.format(cell_type, cell_index))
plt.show()
