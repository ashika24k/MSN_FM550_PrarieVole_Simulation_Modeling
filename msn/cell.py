"""
Build a model of a medium spiny neuron (MSN).

author: Antonio Gonzalez
"""
import numpy as np
import pandas as pd
import neuron as nrn
from neuron import h
import matplotlib.pyplot as plt

from msn import paths
from msn.params import ModelParameters


h.load_file('stdrun.hoc')
h.load_file('import3d.hoc')
nrn.load_mechanisms(paths['mechanisms'])


def synaptic_input(section, stype, x=0.5, interval=10, number=10,
                   start=50, noise=0, threshold=10, delay=1, weight=0):
    """
    Connect a synapse to a cell section and deliver synaptic stimuli.

    Parameters
    ----------
    section : object
        The cell section that will receive synaptic input.
    stype : {'gaba', 'ampa', 'nmda'}
        The type of synapse.
    x : float, range [0, 1], default=0.5
        Location on `section` where the synapse will be created.
    interval : numeric, default=10
        Mean time between spikes in ms.
    number : int, default=10
       Average number of spikes.
    start : numeric, default=50
        Start time of first spike in ms.
    noise : float, range [0, 1], default=0
        Fractional randomness.
    threshold : numeric, default=10
        Connection threshold in mV. If presynaptic variable crosses this
        threshold, the procedure NET_RECEIVE in the target is called,
        i.e. the synapse is activated.
    delay : numeric, default=1
        Connection delay. Delay (after `start`) of the stimulus in ms.
        Represents conduction latency + synaptic latency. It should be
        >= 0.
    weight : numeric, default=0
        Connection weight in uS. Passed on to the NET_RECEIVE procedure
        in the target point.

    Returns
    -------
    synapse : object
        The synapse created
    stim : object
        The NetStim object (NEURON's spike generator object)
    conn : object
        The NetCon object, connecting `synapse` and `stim` (NEURON's
        network connection object)

    Notes
    -----
    The default values are NEURON's default values for NetStim and
    NetCon objects.

    Based on the [online
    tutorial](https://neuron.yale.edu/neuron/docs/ball-and-stick-model-part-2)
    on NEURON's website and on Chapter 10 in The NEURON Book (esp.
    Section 10.1.3.2). To understand the function and its parameters it
    is useful to read the
    [NetStim](https://neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/mech.html#NetStim)
    and
    [NetCon](https://neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/network/netcon.html#NetCon)
    documentation.
    """
    # Create the synapse.
    if stype == 'ampa':
        synapse = h.ampa(x, sec=section)
    elif stype == 'nmda':
        synapse = h.nmda(x, sec=section)
    elif stype == 'gaba':
        synapse = h.gaba(x, sec=section)
    else:
        raise ValueError("Synapse type `stype` must be 'ampa', 'nmda' "
                         "or 'gaba'")

    # Create the stimulus (NetStim - spike generator)
    stim = h.NetStim()
    stim.interval = interval
    stim.number = number
    stim.start = start
    stim.noise = noise

    # Connect the stimulus to the synapse (NetCon - connection object)
    conn = h.NetCon(stim, synapse)
    conn.threshold = threshold
    conn.delay = delay
    conn.weight[0] = weight

    return synapse, stim, conn


def uniform_fun(distance, args, gbar):
    p0 = args[0]
    return 10**p0 * gbar * 10


def step_fun(distance, args, gbar):
    p0, p1, p2, p3 = args
    if (distance > p2) and (distance < p3):
        value = p0
    else:
        value = p1
    return value * gbar


def get_channel_density(distance, params):
    """
    Get ion channel density (gbar, pbar) as a function of somatic
    distance.

    Parameters
    ----------
    distance : numeric
        Distance from the soma.
    params : list
        A list with four elements: [compartment, mechanism,
        density_args, gbar]; see notes below.

    Returns
    -------
    density : numeric
        The channel density value.

    Notes
    -----
    Ion channel densities are calculated for each compartment and
    mechanism as a function of somatic distance according to sigmodial,
    step, linear, or exponential functions; see Table 2 in Lindroos et
    al. (2018) and Table 1 in Lindroos and Hellgren Kotaleski (2020).
    The parameters used for these calculations are named p0, p1, p2, and
    p3 in that same Table, and that is the convention followed in this
    function. Peak conductance density is also required; the function
    get_density_params() from params.ModelParameters returns the data
    needed for these calculations in a suitable format.

    See also
    --------
    .params.ModelParameters
    """
    compartment, mech, args, gbar = params
    #if in the dendrite and is one of the mechanisms, then use specialized function to get density (2021)
    if compartment == 'dend':
        if mech == 'naf':
            p0, p1, p2, p3 = args
            # Exponential sigmoidal function
            density = gbar * 10**p0 * (1 - p1 + p1 / (
                1 + np.exp((distance-p2)/p3)))
        elif mech == 'kaf':
            p0, p1, p2, p3 = args
            # Exponential sigmoidal function
            density = gbar * 10**p0 * (1 + p1 / (
                1 + np.exp((distance-p2)/p3)))
        elif mech == 'kas':
            p0, p2, p3 = args
            # Exponential sigmoidal function (scaled between 0.1 and 1.0)
            density = gbar * 10**p0 * (0.1 + 0.9 / (
                1 + np.exp((distance-p2)/p3)))
        elif mech == 'kir': 
            p0 = args[0]
            # Uniform function (constant)
            density = gbar * 10**(p0)
        elif mech == 'sk':
            p0 = args[0]
            # Uniform function (constant)
            density = gbar * 10**(p0)
        elif mech == 'can':
            p0, p1, p2, p3 = args
            # Exponential sigmoidal function
            density = 10**p0 * (1 - p1 + p1 / (
                1 + np.exp((distance-p2)/p3)))
        elif mech == 'cav32' or mech == 'cav33':
            p0, p2, p3 = args
            # Exponential sigmoidal decay function
            density = 10**p0 / (1 + np.exp((distance-p2)/p3))
        else:
            # Custom uniform function fallback
            density = uniform_fun(distance, args, gbar)
    #if in axon
    elif compartment == 'axon':
        if mech == 'kas' or mech == 'km':
            # Uniform function
            density = uniform_fun(distance, args, gbar)
        elif mech == 'naf':
            # Step function
            density = step_fun(distance, args, gbar)
    #if in soma
    elif compartment == 'soma':
    # Uniform function
        density = uniform_fun(distance, args, gbar)


    # In the function `calculate_distribution` (in Lindroos's
    # MSN_builder.py) the resulting density may be negative, in which
    # case they set the density to 0 instead. I do not know why or when
    # the density may be negative. This assert is to look into this if
    # that ever happened.
    assert (density >= 0), "Ion channel density is negative"

    return density


class MSN:
    """
    Build a model of a MSN.

    Attributes
    ----------
    type : str
        Cell type, 'imsn' or 'dmsn'
    index : int
        Cell index
    distrib_params : list
        Channel distribution parameters (compartment, mechanism, args,
        gbar)
    rheobase : numeric
    v_init : numeric
        Initialisation membrane voltage

    Methods
    -------
    add_bg_noise(gaba_freq=4, glut_freq=12, syn_fact=[],
                 gaba_mod=0, dend_only=False, delays=[]
        Add background noise
    remove_bg_noise()
        Remove background noise

    Notes
    -----
    This class is based on 'MSN_builder.py', distributed with the NEURON
    model by Lindroos and Hellgren Kotaleski (2020), available from
    ModelDB (accession number 266775).
    """
    def __init__(self, cell_type, cell_index, v_init= -85):
        """
        Parameters
        ----------
        cell_type : str
            Cell type to model, one of 'dmsn' or 'imsn'.
        cell_index : int
            Cell to model, from the set of iMSN and dMSN provided by
            Lindroos et al. In that set there are n=71 dMSNs and n=34
            iMSNs.
        v_init : numeric, default=-80
            Initialisation membrane voltage.
        """
        # self._gid = gid
        self.type = cell_type
        self.index = cell_index

        # Load parameters
        params = ModelParameters()
        self.density_params = params.get_density_params(cell_type, cell_index)
        self.rheobase = params.get_rheobase(cell_type, cell_index)
        self._morphology_file = params.get_morphology_path(cell_type)
        gbar_pas = params.get_gbar(cell_type) 
        self._gbar_pas = gbar_pas.value[gbar_pas.mechanism == 'pas'].values[0] 

        # Create cell
        self._setup_morphology()
        self._setup_mechanisms()
        self._setup_biophysics()
        self._setup_density()
        h.celsius = 35
        self.v_init = v_init

        # Additional containers
        self._bg_noise = []

    def _setup_morphology(self):
        # Import morphology from SWC file.
        cell = h.Import3d_SWC_read()
        cell.input(self._morphology_file)
        i3d = h.Import3d_GUI(cell, 0)
        i3d.instantiate(self)
        # h.define_shape() # Is this needed? This line is in the
        # original Lindroos MSN class but I am not sure it is required.
        # NEURON's documentation suggests this is used when cell
        # geometry is specified by methods "where 3D shape is
        # irrelevant", not when using 3D reconstruciton data as it is
        # done here; see
        # https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/topology/geometry.html#geometry-geometry

        # There should only be one soma
        assert(len(self.soma) == 1)
        self.soma = self.soma[0]

        # Spatial discretisation.
        for sec in self.all:
            if 'axon' in sec.name():
                sec.nseg = 2
            else:
                sec.nseg = 2 * int(sec.L/40) + 1

    def _setup_mechanisms(self):
        dendritic_channels = (['naf', 'kaf', 'kas', 'kir', 'sk', 'can',
                               'cav32', 'cav33', 'kdr', 'cal12', 'cal13',
                               'car', 'bk'] +
                              ['cadyn', 'caldyn'])
        somatic_channels = (['naf', 'kaf', 'kas', 'kdr', 'bk', 'cal12',
                             'cal13', 'car', 'can', 'sk', 'kir'] +
                            ['cadyn', 'caldyn'])
        axonal_channels = ['naf', 'kas', 'km']

        for sec in self.all:
            if 'soma' in sec.name():
                mechanisms = somatic_channels
            elif 'axon' in sec.name():
                mechanisms = axonal_channels
            elif 'dend' in sec.name():
                mechanisms = dendritic_channels
            else:
                raise ValueError('Unknown section: ' + sec.name())
            for mechanism in mechanisms:
                sec.insert(mechanism)

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 150
            sec.cm = 1
            sec.insert('pas')
            for seg in sec:
                seg.pas.g = self._gbar_pas
                seg.pas.e = -70
            sec.ena = 50
            sec.ek = -85

    def _setup_density(self):
        """
        Setup ion channel density.
        """
        for params in self.density_params:
            compartment, mechanism, args, gbar = params
            if mechanism.startswith('ca'):
                prefix = 'pbar_'
            else:
                prefix = 'gbar_'
            h.distance(sec=self.soma)
            for section in self.all:
                if compartment in section.name():
                    for segment in section:
                        distance = h.distance(segment.x, sec=section)
                        density = get_channel_density(distance, params)
                        cmd = f'segment.{prefix}{mechanism} = {density}'
                        exec(cmd)

    def add_bg_noise(self, gaba_freq=4, glut_freq=12, dend_only=False,
                     ampa_scale_factor=None, nmda_scale_factor=None,
                     gaba_scale_factor=None, delays=[]):
        """
        Add background noise.

        Creates random GABA and glutamate synaptic inputs and connects these
        to the cell sections to create background noise.

        Parameters
        ----------
        gaba_freq : numeric, default=4
            Frequency of GABAergic inputs (number of inputs per s)
        glut_freq : numeric, default=12
            Frequency of glutamatergic inputs (number of input per s)
        dend_only : bool, default=False
            Whether glut and GABA inputs are applied only to dendrites or to
            all cell sections.
        ampa_scale_factor,
        nmda_scale_factor : None or numeric, default=None
            If not None, scale glutamatergic input by these factors.
            (These two were parameter `syn_fact` in the original
            Lindroos function.)
        gaba_scale_factor : None or numeric, default=None
            If not None, scale GABA input by these factors.
            (This was parameter `gabaMod` in the original Lindroos
            function.)
        delays : list, default=[]
            TODO: What are `delays`?
            Looks like it should be a list with the same number of
            elements as there are sections in the cell. Then, each
            element will set the value of `NetStim.start` for each
            section. (It is confusing that in the random_synapse()
            function there is a NetCon.delay parameter but this delays
            here seem to be unrelated.)

        Notes
        -----
        Based on the function set_bg_noise() in common_functions.py,
        from Lindroos et al. and available from
        [GitHub](https://github.com/robban80/striatal_SPN_lib).

        The synaptic input parameters not specified in that function
        are the default parameters in the function random_synapse()
        found in that same common_functions.py file.
        """
        if dend_only is True:
            sections = [sec for sec in self.all if 'dend' in sec.name()]
        else:
            sections = self.all

        # Make sure to remove previous noise first.
        self.remove_bg_noise()

        # The default synaptic conductance in the synapses that generate
        # this background noise ("synaptic bombardment") quoted in
        # Section 2.6 in Lindroos and Hellgren Kotaleski (2020) are:
        #
        # AMPA, 300 pS = 3e-4 uS
        # NMDA, 300 pS = 3e-4 uS
        # GABA, 900 pS = 9e-4 uS
        #
        # (This implies an AMPA:NMDA ratio of 1.)
        #
        # It is not clear there whether these values apply to both dMSN
        # and iMSN or not but they have separate default values in
        # their function `set_bg_noise()`:
        if self.type == 'dmsn':
            gbase = 3e-4  # in uS
        elif self.type == 'imsn':
            gbase = 2e-4  # in uS

        for indx, sec in enumerate(sections):
            if len(delays) == 0:
                delay = 0
            else:
                delay = delays[indx]

            # AMPA synapse
            synapse, netstim, netcon = synaptic_input(
                sec, stype='ampa', x=0.5, interval=1000/glut_freq,
                number=1000, start=delay, noise=1, threshold=0.1,
                delay=0, weight=gbase)
            if ampa_scale_factor:
                synapse.scale_factor = ampa_scale_factor
            self._bg_noise.append([synapse, netstim, netcon])

            # NMDA synapse
            synapse, netstim, netcon = synaptic_input(
                sec, stype='nmda', x=0.5, interval=1000/glut_freq,
                number=1000, start=delay, noise=1, threshold=0.1,
                delay=0, weight=gbase)
            if nmda_scale_factor:
                synapse.scale_factor = nmda_scale_factor
            self._bg_noise.append([synapse, netstim, netcon])

            # GABA synapse
            # It is not clear why GABA synaptic conductance is
            # calculated as it is done here. GABA conductance should be
            # 9e-4 uS according to the Lindroos and Hellgren Kotaleski
            # paper.
            if gaba_scale_factor:
                conductance = gbase * 3 * gaba_scale_factor  # Why times 3?
            else:
                conductance = gbase * 5  # Why times 5?
            synapse, netstim, netcon = synaptic_input(
                sec, stype='gaba', x=0.1, interval=1000/gaba_freq,
                number=1000,  start=delay, noise=1, threshold=0.1,
                delay=0, weight=conductance)
            self._bg_noise.append([synapse, netstim, netcon])

    def remove_bg_noise(self):
        """
        Removes background noise from the cell.

        Notes
        -----
        The way to remove synaptic inputs from a cell model seems to be
        to just set NetCon weight to 0, as mentioned in this post:

        https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3576
        """
        if len(self._bg_noise) > 0:
            for line in self._bg_noise:
                netcon = line[2]
                netcon.weight[0] = 0
            self._bg_noise = []