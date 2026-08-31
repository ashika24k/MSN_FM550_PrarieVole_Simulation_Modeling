import neuron as nrn
from neuron import h
import numpy as np
import matplotlib.pyplot as plt


class ActionPotentials:
    """
    Action potentials in a voltage trace.

    Attributes
    ----------
    n : int
        The number of action potentials in the trace.
    threshold : numeric, default=0
        The voltage threshold used to detect aciton potetnials.
    timestamps : array
        Timestamps of detected action potentials.
    """

    def __init__(self, x, y, threshold=0):
        """
        Parameters
        ----------
        x : HocObject
            A NEURON vector of time values.
        y : HocObject
            A NEURON vector of voltage values.
        threshold : numeric, default=0
            Voltage threshold for detecting action potentials.
        """
        self._x = x
        self._y = y
        self._threshold = threshold

        self._n_spikes = 0
        self._timestamps = np.nan
        self._get_action_potentials()

    def _get_action_potentials(self):
        is_above_threshold = np.where(
            self._y.as_numpy() > self.threshold, 1, 0)
        is_upstroke = np.diff(is_above_threshold) == 1
        self._n_spikes = is_upstroke.sum()
        self._timestamps = self._x.as_numpy()[:-1][is_upstroke]

    @property
    def n(self):
        """
        Number of action potentials.
        """
        return self._n_spikes

    @property
    def timestamps(self):
        """
        Time of action potentials.
        """
        return self._timestamps

    @property
    def threshold(self):
        """
        Voltage threshold for detecting action potentials.

        The action potentials in the voltage trace will be detected
        whenever this value is set.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """
        Set voltage threshold and detect acion potentials based on this
        threshold.
        """
        self._threshold = value
        self._get_action_potentials()


class Stim:
    """
    A current clamp stimulus.

    Attributes
    ----------
    cell : object
        The NEURON model cell to stimulate
    stim : object
        The NEURON IClamp object
    t : array_like
        Time vector
    v : array_like
        Voltage vector

    Methods
    -------
    set_stim(delay=10, duration=100, amplitude=0.25, tmax=150,
             add_rheob=True)
        Set stimulation parameters
    run(v_init=None)
        Run the stimulation
    plot(ax=None, label='', **kwargs)
        Display the results

    Examples
    --------
    Create a model cell
    >>> cell = MSN(cell_type='dmsn', cell_index=12)

    Model a current clamp step
    >>> stim = Stim(cell)
    >>> stim.set_stim(delay=10, duration=100, amplitude=200, tmax=150)
    >>> stim.run()

    Plot the results
    >>> stim.plot()
    """

    def __init__(self, cell, section='soma'):
        """
        Parameters
        ----------
        cell : object
            A NEURON model cell
        section : str, default='soma'
            Cell section where stimulus will be applied. It should be in
            NEURON's standard naming format for sections, e.g.
            'dend[45]' or 'axon[0]'.
        """
        self.cell = cell
        stim = None
        if section == 'soma':
            stim = h.IClamp(0.5, sec=cell.soma)
        else:
            for sec in cell.all:
                if section in sec.name():
                    stim = h.IClamp(0.5, sec=sec)
                    break
        if stim is None:
            raise ValueError(f'{section} is not a section in cell.')
        self.stim = stim

        # Recording vectors
        self.t = h.Vector()
        self.t.record(h._ref_t)
        self.v = h.Vector()
        self.v.record(cell.soma(0.5)._ref_v)

    def set_stim(self, delay=10, duration=100, amplitude=0.25,
                 tmax=150, add_rheob=True):
        """
        Set the stimulus parameters.

        Parameters
        ----------
        delay : numeric, default=10
            Stimulus delay
        duration : numeric, default=100
            Stimulus duration
        amplitude : numeric, default=0.25
            Stimulus amplitude in nA (which are the standard stimulation
            units in NEURON).
        tmax : numeric, default=150
            Length of simulation
        add_rheob : bool, default=True
            If True (default), the stimulus amplitude will be added on
            top of the cell's rheobase.
        """
        self.stim.delay = delay
        self.stim.dur = duration
        self.tmax = tmax
        self.amp = amplitude
        if add_rheob is True:
            # NEURON expects amplitude in nA; the rheobase in the
            # Lindroos et al. model seems to be in pA.
            self.stim.amp = (self.cell.rheobase * 1e-3) + amplitude
        else:
            self.stim.amp = amplitude

    def run(self, v_init=None):
        """
        Run the simulation.

        Parameters
        ----------
        v_init : None or numeric, default=None
            Membrane voltage for initialising the simulation. If none is
            given then the simulated cell's `v_init` attribute will be
            used.
        """
        if v_init is None:
            h.finitialize(self.cell.v_init)
        else:
            h.finitialize(v_init)
        while h.t < self.tmax:
            h.fadvance()
        # h.run()

    def plot(self, ax=None, label='', **kwargs):
        """
        Plot the simulation results.

        Parameters
        ----------
        ax : None or matplotlib axes object
            Matplotlib axes to use for plotting. If None (default), one
            will be created.
        label : str
            Legend label.
        **kwargs :
            Additional keyword arguments passed on to the plot function.

        Returns
        -------
        ax : matplotlib axes object
            The matplotlib axes used for plotting.
        """
        if ax is None:
            ax = plt.figure().add_subplot(111)
        ax.plot(self.t, self.v, label=label, **kwargs)
        if label:
            ax.legend()
        # Label the axes if these are not set:
        if ax.get_xlabel() == '':
            ax.set_xlabel('Time (ms)')
        if ax.get_ylabel() == '':
            ax.set_ylabel('Membrane potential (mV)')
        return ax