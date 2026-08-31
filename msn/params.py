"""
Manage parameters needed for modelling the Lindroos et al medium spiny
neurones (MSNs).

author: Antonio Gonzalez
"""
import numpy as np
import pandas as pd

from . import paths


class ModelParameters:
    """
    Manage cell model parameters in the Lindroos et al data set.

    In addition to the core NEURON (.mod) files that define the
    biophysical mechanisms, Lindroos et al provide several extra files
    needed to model MSNs. This class provides a common point to manage
    those files and their data.

    Methods
    -------
    get_density_params(cell_type, cell_index)
        Get ion channel density parameters for the cell specified.
    get_gbar(cell_type)
        Get values of maximal conductance (gbar).
    get_morphology_path(cell_type)
        Get the path to the cell's morphology (swc) file.
    get_rheobase(cell_type, cell_index)
        Get the cell's rheobase.

    Example
    -------
    >>> cell_type = 'dmsn'
    >>> cell_index = 30
    >>> params = ModelParameters()
    >>> rheobase = params.get_rheobase(cell_type, cell_index)

    Notes
    -----
    The files used to create the [Lindroos model] of a MSN fall in 3
    groups. The way in which this class is organised reflects these 3
    sets of data sources:

    - Mechanisms. These are the NEURON (.mod) files that implement the
      biophysical mechanisms of the MSN.

    - Morphology. Microscopy images of real neurones can be digitally
      reconstructed, and these reconstructions are often saved as SWC
      files (see e.g. the [NeuroMorpho] database). The [Lindroos model]
      includes two such SWC files, one of a dMSN and one of a iMSN.

    - Parameters.

        - Density parameters. Ion channel densities in the MSN model are
          a function of somatic distance and are calculated according to
          sigmodial, linear, or exponential functions; see
          [Lindroos2021] for details. Lindroos et al. provide density
          distribution parameters (named p0, p1, p2, and p3 in Table 1,
          [Lindroos2021]) to model 71 different dMSNs and 34 different
          iMSNs. These parameters are contained in the `variables`
          subset in the pickled files `D1_71bestFit_updRheob.pkl` and
          `D2_34bestFit_updRheob.pkl`.

        - Conductance. The peak conductance density (gbar, also maximal
          conductance) in the Lindroos et al model is defined for
          different cell compartments and for the two types of MSNs in
          the files `params_dMSN.json` and `params_iMSN.json`. There are
          additional (default) gbar values harcoded in the class MSN
          (file MSN_builder.py in the [Lindroos model]). To make this
          simpler I put together all these peak conductance values in
          one single file, `conductances.tsv`, which thus replaces the
          two params_*.json files.

    References
    ----------
    [NeuroMorpho]: http://www.neuromorpho.org

    [Lindroos model]: http://modeldb.yale.edu/266775

    [Lindroos2018]: Lindroos R, Dorst MC, Du K, Filipović M, Keller D,
    Ketzef M, Kozlov AK, Kumar A, Lindahl M, Nair AG, Pérez-Fernández J,
    Grillner S, Silberberg G & Hellgren Kotaleski J (2018). Basal
    ganglia neuromodulation over multiple temporal and structural
    scales-simulations of direct pathway MSNs investigate the fast onset
    of dopaminergic effects and predict the role of Kv4.2. Front Neural
    Circuits 12, 3.

    [Lindroos2021]: Lindroos R & Hellgren Kotaleski J (2021). Predicting
    complex spikes in striatal projection neurons of the direct pathway
    following neuromodulation by acetylcholine and dopamine. Eur J
    Neurosci 53, 2117–2134 (https://doi.org/10.1111/ejn.14891).

    """

    def __init__(self):
        # Load channel density parameters.
        self._params = {}
        for cell_type, path in paths['parameters'].items():
            params = pd.read_pickle(path)
            self._params[cell_type] = {}
            for key, val in params.items():
                self._params[cell_type][key] = {}
                self._params[cell_type][key]['rheobase'] = val['rheobase']
                these_params = val['variables']
                # In their MSN model, Lindroos et al name the CaT3.2 and
                # CaT3.3 currents 'cav32' and 'cav33'. However, in their
                # parameters (*.pkl) files these same currents are named
                # 'c32' and 'c33'. Here these currents are renamed to
                # their 'cav' nomenclature to match the rest of the
                # model.
                these_params['cav33'] = these_params.pop('c33')
                these_params['cav32'] = these_params.pop('c32')
                self._params[cell_type][key]['density_params'] = these_params

        # Load conductances.
        self._conductances = pd.read_csv(paths['conductances'],
                                         delimiter='\t', comment='#')

    def get_rheobase(self, cell_type, cell_index):
        """
        Get the cell's rheobase.

        Input
        -----
        cell_type : str
            One of 'dmsn' or 'imsn'
        cell_index : int
            The index of the cell parameters. There are 71 different
            sets of parameters for dMSNs and 34 for iMSNs in the
            Lindroos dataset.

        Returns
        --------
        rheobase : numeric
            The value of rheobase of the selected cell.
        """
        rheobase = self._params[cell_type][cell_index]['rheobase']
        return rheobase

    def get_morphology_path(self, cell_type):
        """
        Returns the path to the morphology (SWC) file for the given
        cell type.

        Input
        -----
        cell_type : str
            One of 'dmsn' or 'imsn'

        Returns
        -------
        morph : str
            The path to the SWC file.
        """
        morph = str(paths['morphologies'][cell_type])
        return morph

    def get_density_params(self, cell_type, cell_index):
        """
        Return the parameters required for calculating ion channel
        density.

        Input
        -----
        cell_type : str
            One of 'dmsn' or 'imsn'
        cell_index : int
            The index of the cell parameters. There are 71 different
            sets of parameters for dMSNs and 34 for iMSNs in the
            Lindroos dataset.

        Returns
        -------
        params : list
            A list of lists, where each row is
            [compartment, mechanism, args, gbar]:

            compartment : str
                One of 'axon', 'soma', 'dend' or 'all'
            mechanism : str
                E.g. 'naf', 'kir', etc.
            args : list
                A list with one to four elements, referred as p0, p1,
                p2, p3 in Table 1 in Lindroos2021, used to calculate
                ion channel density.
            gbar : numeric
                The value of gbar (or pbar)
        """
        density_params = self._params[cell_type][cell_index]['density_params']
        params = []

        # Dendrites
        for mech in ['naf', 'kaf', 'kas', 'kir', 'sk', 'can', 'cav32',
                     'cav33', 'kdr', 'cal12', 'cal13', 'car', 'bk']:
            if mech in density_params:
                args = density_params[mech]
            else:
                # If the mechanism is not specified, the default value
                # is 1 in Lindroos et al MSN_build.py file. They use
                # that 1 to substitue for the value of 10**p0 in their
                # density calculations. Here, I set the default to 0
                # because the formula to calculate ion channel density
                # uses this value, p0, as the exponent in 10**p0, and
                # 10**0 = 1.
                args = [0]
            params.append(['dend', mech, args])

        # Soma. For the soma, all channels take the default argument 0
        # except for sk and kir.
        for mech in ['naf', 'kaf', 'kas', 'kdr', 'bk', 'cal12',
                     'cal13', 'car', 'can']:
            args = [0]
            params.append(['soma', mech, args])
        for mech in ['sk', 'kir']:
            args = density_params[mech]
            params.append(['soma', mech, args])

        # Axon. These take default arguments hardcoded in the original
        # MSN_build.py file.
        params += [
                ['axon', 'kas', [0]],
                ['axon', 'naf', [1, 1.1, 30, 500]],
                ['axon', 'km', [0]]]

        # Now append the value of gbar (or pbar) to each line.
        conductances = self.get_gbar(cell_type)
        for line in params:
            compartment, mechanism, __ = line
            gbar = conductances[
                    (conductances.mechanism == mechanism) &
                    (conductances.compartment == compartment)]
            if len(gbar) == 0:
                gbar = np.nan
            else:
                gbar = gbar.value.values[0]
            line.append(gbar)
        return params

    def get_gbar(self, cell_type):
        """
        Get peak conductance density (maximal conductance).

        Returns the maximal conductance (gbar) or maximal permeability
        (pbar, for calcium mechanisms) pre-defined for iMSNs or dDMSNs.

        Parameters
        ----------
        cell_type : str, ['dmsn', 'imsn']
            The type of cell, dMSN or iMSN.

        Returns
        -------
        gbar : pandas dataframe
            Maximum conductance values in a pandas dataframe with
            columns: [mechanism, compartment, value].
        """
        gbar = self._conductances[
            (self._conductances.cell == cell_type) |
            (self._conductances.cell == 'all')]
        gbar = gbar.drop('cell', axis=1)
        return gbar