"""
Find frequency-current (gain) in a MSN.

Gain is a measure of an output in relation to a signal (input). In
neurones, this is usually a measure of cell activity (e.g. number of
action potentials) elicited by a particular stimulus (e.g. current); see
[@Chance2002].

In this script a MSN is stimulated with varying levels of driving
current to trigger action potentials. To investigate gain, the cell's
firing rate is plotted against the current.

Note
----
In addition to the packages used in previous examples, this script
requires the Python package
[scikit-learn](http://scikit-learn.sourceforge.net) for linear
regression.

References
----------
[@Chance2002]: Chance FS, Abbott LF & Reyes AD (2002). Gain modulation
from background synaptic input. Neuron 35, 773-782.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from msn.cell import MSN
from msn.instrumentation import Stim, ActionPotentials

# Choose simulation parameters. Stimulus amplitude is in nA and time is
# in ms.
stim_duration = 250
stim_delay = 50
stim_amplitudes = np.arange(0.4, 0.8, 0.1)
cell_type = 'dmsn'
cell_index = 3

# Make a MSN (with background noise) and setup the stimulation.
cell = MSN(cell_type, cell_index)
cell.add_bg_noise(gaba_freq=24, glut_freq=12)

# Create axes for plotting: one axis for the gain data and one to
# display some traces.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Create an array for storing the results.
freq = np.empty_like(stim_amplitudes, dtype=float)

# Run the simulation.
stim = Stim(cell)
for index, amplitude in enumerate(stim_amplitudes):
    stim.set_stim(delay=stim_delay, duration=stim_duration,
                  amplitude=amplitude, tmax=stim_duration+stim_delay,
                  add_rheob=False)
    stim.run()

    # Calculate firing frequency: number of action potentials in one
    # second. The stimulus duration is in ms so this value has to be
    # converted to seconds.
    ap = ActionPotentials(stim.t, stim.v)
    freq[index] = ap.n/(stim_duration/1000)

    # Plot the first and last trace.
    if (index == 0) or (index == len(stim_amplitudes)-1):
        stim.plot(ax1)

# Fit a straight line to the frequency-current data. The slope of that
# line is the cell's gain.
model = linear_model.LinearRegression()
x = np.array(stim_amplitudes).reshape(-1, 1)
model.fit(x, freq)
freq_fit = model.predict(x)
slope = model.coef_[0]

# Plot gain data.
ax2.plot(stim_amplitudes, freq, 'o')
ax2.plot(stim_amplitudes, freq_fit)
ax2.set_title('{} #{}'.format(cell_type, cell_index))
ax2.set_xlabel('Driving current (nA)')
ax2.set_ylabel('Firing rate (Hz)')
ax2.text(x=stim_amplitudes[-3], y=freq[0],
         s=f'Gain = {slope:.1f} Hz/nA',
         fontsize='small')

plt.show()
