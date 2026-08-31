"""
Dopaminergic and cholinergic modulation of medium spiny neurones (MSNs).

Dopamine (DA) and acetylcholine (ACh) affect the activity of MSNs by
modulating several ion currents and synaptic mechanisms: thus, DA
decreases Naf, Kas and CaN, and increases Kir and CaL (Lindroos2018 [1]
Table 3), whereas ACh affects Kaf, Kir, Km, Naf CaN, CaL (Lindroos2021
[2]). Both also modulate glutamate and GABA receptors.

The functions and classes here implement these modulatory effects, which
are effectively changes in conductance, as described in those two
papers, and are based on the files made available by Lindroos2021 [2]
on GitHub [3].

References
----------
[1]: Lindroos R, Dorst MC, Du K, Filipović M, Keller D, Ketzef M,
Kozlov AK, Kumar A, Lindahl M, Nair AG, Pérez-Fernández J, Grillner S,
Silberberg G & Hellgren Kotaleski J (2018). Basal ganglia
neuromodulation over multiple temporal and structural scales-simulations
of direct pathway MSNs investigate the fast onset of dopaminergic
effects and predict the role of Kv4.2. Front Neural Circuits 12, 3.

[2]: Lindroos R & Hellgren Kotaleski J (2021). Predicting complex spikes
in striatal projection neurons of the direct pathway following
neuromodulation by acetylcholine and dopamine. Eur J Neurosci 53,
2117–2134 (https://doi.org/10.1111/ejn.14891).

[3]: https://github.com/robban80/striatal_SPN_lib


author: Antonio Gonzalez
"""
import numpy as np
from neuron import h

# Setup a uniform random number generator
rand_uniform = np.random.default_rng().uniform


def get_modulation_params(cell_type, neurotransmitter):
    """
    Modulation parameters for dopamine (DA) and acetylcholine (ACh) for
    MSNs.

    Return modulation parameters for MSNs. These values are drawn at
    random from a uniform distribution within the boundaries set
    in Table 3 in Lindroos & Hellgren Kotaleski (2020).

    Parameters
    ----------
    cell_type : str
        One of 'dmsn' or 'imsn'.
    neurotransmitter : str
        'ACh' or 'DA'.

    Returns
    -------
    modulation : dict
        Random modulation parameters, instrinsic and synaptic, for the
        cell type and neurotransmitter specified.

    Notes
    -----
    There are no values for acetylcholine modulation of NMDA, GABA or
    AMPA in dMSNs in Table 3 [Lindroos2020]. The implication is that
    ACh does not modulate these synaptic channels in dMSNs.
    TODO: check that this is intended -- does ACh have no effects on
    synaptic inputs in dMSNs?
    """
    if cell_type == 'imsn':
        if neurotransmitter == 'DA':
            modulation = {
                'intrinsic': {
                    'naf': rand_uniform(0.95, 1.1),
                    'kaf': rand_uniform(1, 1.1),
                    'kas': rand_uniform(1, 1.1),
                    'kir': rand_uniform(0.8, 1.0),
                    'cal12': rand_uniform(0.7, 0.8),
                    'cal13': rand_uniform(0.7, 0.8),
                    'can': rand_uniform(0.9, 1.0),
                    'car': rand_uniform(0.6, 0.8)},
                'synaptic': {
                    'NMDA': rand_uniform(0.85, 1.05),
                    'AMPA': rand_uniform(0.7, 0.9),
                    'GABA': rand_uniform(0.90, 1.1)}
            }
        elif neurotransmitter == 'ACh':
            modulation = {
                'intrinsic': {
                    'naf': rand_uniform(1, 1.2),
                    'kir': rand_uniform(0.5, 0.7),
                    'cal12': rand_uniform(0.3, 0.7),
                    'cal13': rand_uniform(0.3, 0.7),
                    'can': rand_uniform(0.65, 0.85),
                    'km': rand_uniform(0, 0.4)},
                'synaptic': {
                    'NMDA': rand_uniform(1, 1.05),
                    'AMPA': rand_uniform(0.99, 1.01),
                    'GABA': rand_uniform(0.99, 1.01)}
            }

    elif cell_type == 'dmsn':
        if neurotransmitter == 'DA':
            modulation = {
                'intrinsic': {
                    'naf': rand_uniform(0.6, 0.8),
                    'kaf': rand_uniform(0.75, 0.85),
                    'kas': rand_uniform(0.65, 0.85),
                    'kir': rand_uniform(0.85, 1.25),
                    'cal12': rand_uniform(1, 2),
                    'cal13': rand_uniform(1, 2),
                    'can': rand_uniform(0.2, 1)},
                'synaptic': {
                    'NMDA': 1.3,
                    'AMPA': 1.2,
                    'GABA': 0.8}
            }
        elif neurotransmitter == 'ACh':
            modulation = {
                'intrinsic': {
                    'naf': rand_uniform(1, 1.2),
                    # 'kaf': shift in voltage dependence
                    'kir': rand_uniform(0.8, 1),
                    'cal12': rand_uniform(0.3, 0.7),
                    'cal13': rand_uniform(0.3, 0.7),
                    'can': rand_uniform(0.65, 0.85),
                    'km': rand_uniform(0, 0.4)},
                'synaptic': {
                    # There are no values for cholinergic modulation of
                    # synaptic channels in dMSNs in Table 3
                    # Lindroos2020. Setting these values to 1 here thus
                    # implies no cholinergic modulation (see `maxMod`
                    # variables in gaba.mod and glutamate.mod)
                    'NMDA': 1,
                    'AMPA': 1,
                    'GABA': 1}
            }
    return modulation


class Dopamine:
    """
    Dopamine (DA) modulation of a model neuron.

    DA modulates Naf, Kas, CaN, Kir and CaL currents in dMSNs, and all
    these plus CaR in iMSN. This class alters these currents by a
    random amount in a model of a MSN to simulate DA action.

    Additional modulation of GABA- and glutamate-activated currents is
    also possible. By default, both 'intrinsic' (Naf, Kir, etc) and
    'syanptic' (GABA, glut) are set. Also by default the axon is
    excluded from the modulation.

    Attributes
    ----------
    cell : object
        Neuron model to modulate
    params : dict
        Set of modulation parameters
    play : list
        Not sure
    dt : numeric
        Time interval for playing `play`

    Methods
    -------
    reset()
        Reset modulation

    See also
    --------
    get_modulation_params : Modulation parameters sampled at random

    Notes
    -----
    The methods defined here for modulating channels work by setting the
    value of attributes with very obscure names. To understand what
    these mean it is necessary to refer to the original NEURON (.mod)
    files, see e.g. naf.mod. A summary of what these attributes mean
    follows:

    damod : [0, 1]
        Switch modulation on (1) or off (0).
    maxMod :
        Maximum modulation for the channel, according to the value
        provided by get_modulation_params().
    level : numeric, range 0 to 1
        An "additional parameter for scaling modulation". It can be set
        to anything between 0 and 1 to e.g. "simulate non-static
        moduladion".

    Notes
    -----
    Based on the class DA in modulation_lib.py by Lindroos and Hellgren
    Kotaleski (2020), available as part of their
    [striatal_SPN_lib](https://github.com/robban80/striatal_SPN_lib).
    """

    def __init__(self, cell, modulate='no_axon',
                 intrinsic_modulation=True, gaba_modulation=True,
                 glut_modulation=True, play=[], dt=h.dt):
        """
        Parameters
        ----------
        cell : object
            Model cell to modulate.
        modulate : str ['no_axon', 'all'], default='no_axon'
            Whether to modulate all cell sections or exclude the axon.
        intrinsic_modulation : bool, default=True
            If True, modulate intrinsic currents (Naf, Kir, etc).
        gaba_modulation : bool, default=True
            If True, modulate GABA-activated currents.
        glut_modulation : bool, default=True
            If True, modulate gluatame-activated currents.
        play : list, default=[]
            Looking at naf.mod, `play` seems to be a value between 0 and
            1 that, when changed gradually, can be used to "simulate non
            static modulation". It is used to change the parameter
            `level` (for DA) or `lev2` (for ACh) in the corresponding
            mod file.
        dt : numeric, default=h.dt
            Time interval for playing `play`. Defaults to NEURON's dt
            value, accessed via `neuron.h`.
            (TODO: I don't really know what use is this)
        """
        self.cell = cell
        self.params = get_modulation_params(
            cell.type, neurotransmitter='DA')
        self.play = play
        self.dt = dt

        if modulate == 'all':
            self._sections = cell.all
        elif modulate == 'no_axon':
            self._sections = [cell.soma] + cell.dend

        for section in self._sections:
            for segment in section:
                if intrinsic_modulation is True:
                    self._modulate_instrinsic(segment)
                if gaba_modulation is True:
                    self._modulate_gaba(segment)
                if glut_modulation is True:
                    self._modulate_glut(segment)

    def _modulate_instrinsic(self, segment, reset=False):
        for mech in segment:
            if mech.name() in self.params['intrinsic']:
                mech.damod = 1
                mech.maxMod = self.params['intrinsic'][mech.name()]
                if reset is True:
                    mech.level = 0
                elif len(self.play) and mech.name() in self.play:
                    self.play[mech.name()].play(mech._ref_level, self.dt)
                else:
                    mech.level = 1

    def _modulate_gaba(self, segment, reset=False):
        for syn in segment.point_processes():
            if 'gaba' in syn.hname():
                syn.damod = 1
                syn.maxMod = self.params['synaptic']['GABA']
                if reset is True:
                    syn.level = 0
                elif len(self.play) and 'gaba' in self.play:
                    self.play['gaba'].play(syn._ref_level, self.dt)
                else:
                    syn.level = 1

    def _modulate_glut(self, segment, reset=False):
        for syn in segment.point_processes():
            if 'ampa' in syn.hname():
                syn.damod = 1
                syn.maxMod = self.params['synaptic']['AMPA']
                if reset is True:
                    syn.level = 0
                elif len(self.play) and 'glut' in self.play:
                    self.play['glut'].play(syn._ref_level, self.dt)
                else:
                    syn.level = 1
            elif 'nmda' in syn.hname():
                syn.damod = 1
                syn.maxMod = self.params['synaptic']['NMDA']
                if reset is True:
                    syn.level = 0
                elif len(self.play) and 'glut' in self.play:
                    self.play['glut'].play(syn._ref_level, self.dt)
                else:
                    syn.level = 1

    def reset(self):
        if len(self.play):
            for trans in self.play.values():
                trans.play_remove()
        for section in self._sections:
            for segment in section:
                self._modulate_instrinsic(segment, reset=True)
                self._modulate_glut(segment, reset=True)
                self._modulate_gaba(segment, reset=True)


class Acetylcholine:
    """
    Acetylcholine (ACh) modulation of a model neuron.

    ACh in MSN modulates Naf, Kir, CaL, CaN, Kaf and Km currents.  Here,
    these currents are modified to simulate ACh action. (Kaf  modulation
    is simulated not by changing conductance parameters but by shifting
    membrane potential.)

    In addition, modulation of GABA- and glutamate-activated currents is
    also possible. By default, both 'intrinsic' (Naf, Kir, etc) and
    'syanptic' (GABA, glut) are set in all cell sections (axon, soma,
    dend).
    
    NOTE: ACh does not modulate NMDA, AMPA or GABA in dMSNs according to
    Lindroos and Hellgren Kotaleski (2020) (see Table 3). To change this
    default behaviour the modulation values returned by
    get_modulation_params() must be changed.

    Attributes
    ----------
    cell : object
        Neuron model to modulate
    params : dict
        Set of modulation parameters
    play : list
        Not sure
    dt : numeric
        Time interval for playing `play`

    Methods
    -------
    reset()
        Reset modulation

    See also
    --------
    get_modulation_params : Modulation parameters sampled at random

    Notes
    -----
    Based on the class ACh in modulation_lib.py by Lindroos and Hellgren
    Kotaleski (2020), available as part of their
    [striatal_SPN_lib](https://github.com/robban80/striatal_SPN_lib).
    """

    def __init__(self, cell, modulate='all', intrinsic_modulation=True,
                 gaba_modulation=True, glut_modulation=True,
                 shift_kaf=-10, play=[], dt=h.dt):
        """
        Parameters
        ----------
        cell : object
            Model cell to modulate.
        modulate : ['no_axon', 'all'], default='all'
            Whether to modulate all cell sections or exclude the axon.
        intrinsic_modulation : bool, default=True
            If True, modulate intrinsic currents (Naf, Kir, etc).
        gaba_modulation : bool, default=True
            If True, modulate GABA-activated currents.
        glut_modulation : bool, default=True
            If True, modulate gluatame-activated currents.
        shift_kaf : numeric, default=-10
            Shift in mV in kaf voltage dependence. This only applies to
            dMSNs; see note below.
        play : list, default=[]
            Looking at naf.mod, `play` seems to be a value between 0 and
            1 that, when changed gradually, can be used to "simulate non
            static modulation". It is used to change the parameter
            `level` (for DA) or `lev2` (for ACh) in the corresponding
            mod file. 
        dt : numeric, default=h.dt
            Time interval for playing `play`. Defaults to NEURON's dt
            value, accessed via `neuron.h`.
            (TODO: I don't really know what use is this)

        Note
        ----
        Acetylcholine modulation of Kaf in dMSNs is implemented as a
        shift in that channel's voltage dependence; see the original
        mechanism implementation (kaf.mod) and also Table 3 and Section
        2.9 in [Lindroos2020], where the default shift value is -10 mV.
        Note, however, than in the original class `ACh` in
        `modulation_lib.py` by the same authors the default value is
        +20 mV.

        References
        ----------
        [Lindroos2020]: Lindroos and Hellgren Kotaleski (2020).
        """
        self.cell = cell
        self.params = get_modulation_params(
            cell.type, neurotransmitter='ACh')
        self.play = play
        self.dt = dt

        if modulate == 'all':
            self._sections = cell.all
        elif modulate == 'no_axon':
            self._sections = [cell.soma] + cell.dend

        for section in self._sections:
            for segment in section:
                if intrinsic_modulation is True:
                    self._modulate_intrinsic(segment)
                if gaba_modulation is True:
                    self._modulate_gaba(segment)
                if glut_modulation is True:
                    self._modulate_glut(segment)
                if shift_kaf:
                    self._shift_kaf(segment, by_mV=shift_kaf)

    def _modulate_intrinsic(self, seg, reset=False):
        for mech in seg:
            if mech.name() in self.params['intrinsic']:
                mech.damod = 1
                mech.max2 = self.params['intrinsic'][mech.name()]
                if reset is True:
                    mech.damod = 0
                    mech.lev2 = 0
                if len(self.play) and mech.name() in self.play:
                    self.play[mech.name()].play(mech._ref_lev2, self.dt)
                else:
                    mech.lev2 = 1

    def _shift_kaf(self, seg, by_mV=-10, reset=False):
        """
        Shift kaf voltage dependence in dMSNs.

        Parameters
        ----------
        by_mV : numeric, default=-10
            Amount of shift in mV

        Note
        ----
        Acetylcholine-dependent kaf modulation in dMSNs is simulated
        with the parameter modShift (see kaf.mod in mechanisms) which
        shifts the volatge dependence of the channel. The default value
        (-10 mV) is from Table 3 in Lindroos and Hellgren Kotaleski
        (2020).
        """
        if self.cell.type == 'dmsn':
            for mech in seg:
                if mech.name() == 'kaf':
                    if reset is True:
                        mech.damod = 0
                        mech.modShift = 0
                    elif len(self.play) and 'kaf' in self.play:
                        self.play['kaf'].play(mech._ref_modShift, self.dt)
                    else:
                        mech.modShift = by_mV

    def _modulate_gaba(self, seg, reset=False):
        for syn in seg.point_processes():
            if 'gaba' in syn.hname():
                syn.damod = 1
                syn.max2 = self.params['synaptic']['GABA']
                if reset is True:
                    syn.damod = 0
                    syn.lev2 = 0
                elif len(self.play) and 'gaba' in self.play:
                    self.play['gaba'].play(syn._ref_lev2, self.dt)
                else:
                    syn.lev2 = 1

    def _modulate_glut(self, seg, reset=False):
        for syn in seg.point_processes():
            if 'ampa' in syn.hname():
                syn.damod = 1
                syn.max2 = self.params['synaptic']['AMPA']
                if reset is True:
                    syn.damod = 0
                    syn.lev2 = 0
                elif len(self.play) and 'glut' in self.play:
                    self.play['glut'].play(syn._ref_lev2, self.dt)
                else:
                    syn.lev2 = 1
            elif 'nmda' in syn.hname():
                syn.damod = 1
                syn.max2 = self.params['synaptic']['NMDA']
                if reset is True:
                    syn.damod = 0
                    syn.lev2 = 0
                elif len(self.play) and 'glut' in self.play:
                    self.play['glut'].play(syn._ref_lev2, self.dt)
                else:
                    syn.lev2 = 1

    def reset(self, what='all'):
        """
        Switch off synaptic modulation

        Parameters
        ----------
        what : str, {'all', 'intrinsic', 'gaba', 'glut'}, default='all'
            What modulation to reset.
        """
        if what not in ['all', 'gaba', 'glut', 'intrinsic']:
            raise ValueError("'what' must be ['all', 'gaba', "
                             "'glut', 'intrinsic']")
        if len(self.play):
            for trans in self.play.values():
                trans.play_remove()
        for sec in self._sections:
            for seg in sec:
                if what == 'all' or what == 'intrinsic':
                    self._modulate_intrinsic(seg, reset=True)
                    self._shift_kaf(seg, reset=True)
                if what == 'all' or what == 'glut':
                    self._modulate_glut(seg, reset=True)
                if what == 'all' or what == 'gaba':
                    self._modulate_gaba(seg, reset=True)