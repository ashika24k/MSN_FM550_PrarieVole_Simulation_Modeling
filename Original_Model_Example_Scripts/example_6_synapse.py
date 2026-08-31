"""
Single synaptic input.

This script illustrates how to create a synapse and connect it to one
single dendritic section. This synapse then delivers a train of
excitatory (glutamatergic) inputs and the voltage (i.e. the excitatory
post-synaptic potential) at the soma is recorded.

Three simulations take place as the synapse is connected to a dendrite
close to the soma, then to another dendrite further from the soma, and
finally to a third dendrite even further away. The resulting
post-synaptic potentials are plotted together for comparison: the
further from the soma the wider the potential, similar to Figure 2(a)
and 2(b) in Lindroos and Hellgren Kotaleski (2021).

author: Antonio Gonzalez
"""

import numpy as np
from neuron import h
import matplotlib.pyplot as plt
from msn.cell import MSN
from msn.cell import synaptic_input

# Make a MSN
cell_type = 'dmsn'
cell_index = 13
cell = MSN(cell_type, cell_index)
# cell.add_bg_noise(gaba_freq=24, glut_freq=12)

# These are the 3 dendrites that will receive the excitatory synaptic
# input in this example. They are respectively close, far, further, from
# the soma, in the model of a dMSN. To find out which dendrites are
# close or not to the soma, or whether they form part of the same tree,
# the easiest way is to build the cell (i.e. the lines of code above
# this comment) in an interactive session (e.g. ipython or a Jupyter
# notebook) and then type `h.topology()` to see how the cell sections
# are connected together.
dendrites = [6, 8, 10]

# Create recording vectors: time and voltage (measured at the soma).
t = h.Vector()
v = h.Vector()
t.record(h._ref_t)
v.record(cell.soma(0.5)._ref_v)


# Create a function to ease running the simulation more than once.
def run(cell, dend_num, tstop=350):
    global t, v

    # Loop along all cell sections and stop as soon as the dendrite we
    # want is found.
    dend_name = f'dend[{dend_num}]'
    for dend in cell.dend:
        if dend_name in dend.name():
            break

    # Calculate somatic distance.
    somatic_distance = h.distance(cell.soma(0.5), dend(0.5))

    # Membrane potential takes a few milliseconds to settle. Take this
    # into account so that the synaptic stimulus starts some time after
    # this, and also to remove the settling time from the plot later.
    settle_time = 50  # in ms
    start_time = settle_time + 20  # in ms

    connections = []

    # In Lindroos and Hellgren Kotaleski (2020), Section 2.6, the
    # synaptic conductances used to model dendritic plateaus are:
    #   NMDA, 1500 pS = 1.5e-3 uS
    #   AMPA, 500 pS = 5e-4 uS
    # (This implies an AMPA:NMDA ratio of 1/3.)
    # They also use an interval of 1 ms and 16 spikes in their model.
    # That is where these parameters come from.
    ampa_synapse, __, netcon1 = synaptic_input(
        section=dend, stype='ampa', x=0.5, interval=1, number=16,
        start=start_time, noise=0, delay=0, weight=5e-4)
    connections.append(netcon1)

    nmda_synapse, __, netcon2 = synaptic_input(
        section=dend, stype='nmda', x=0.5, interval=1, number=16,
        start=start_time, noise=0, delay=0, weight=1.5e-3)
    connections.append(netcon2)

    # Run the simulation.
    h.finitialize(cell.v_init)
    while h.t < tstop:
        h.fadvance()

    # Before plotting remove the first few ms of data during which the
    # membrane potential was settling.
    x = t.as_numpy()
    y = v.as_numpy()
    y = y[x > settle_time]
    x = x[x > settle_time]
    x -= start_time

    # Plot the data, adding somatic distance to the label.
    lbl = f'{dend_name}, distance={somatic_distance:.0f}'
    plt.plot(x, y, label=lbl)

    # Set connection weight to 0 to inactivate these synapses. This has
    # to be done before the next one is added or else the synapses
    # build up.
    for netcon in connections:
        netcon.weight[0] = 0


# Run the simulation as many times as there are dendrites to receive a
# the synaptic stimuli.
for dend_num in dendrites:
    run(cell, dend_num)

# Label the plot and display.
plt.title(f'{cell_type} #{cell_index}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')

plt.legend()
plt.show()
