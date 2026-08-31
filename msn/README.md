# msn

A library for modelling striatial medium spiny neurons (MSNs).

The original NEURON model (mechanisms/\*.mod), cell morphologies
(morphologies/\*.swc), and best-fit parameters (parameters/\*.pkl) are
from Lindroos and Hellgren Kotaleski [Lindroos2021], available from
ModelDB [(model 266775)](http://modeldb.yale.edu/266775). Several Python
functions are based on code made available online [on
GitHub](https://github.com/robban80/striatal_SPN_lib) by the same
authors and also on their previous work [Lindroos2018].

(C) 2021-2022 [Antonio González](mailto:antgon@cantab.net)

## Requirements

* [Python 3 (>= 3.7)](https://www.python.org/)
* [Numpy](http://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [NEURON](https://neuron.yale.edu/neuron/)

## References

[Lindroos2018]: Lindroos R, Dorst MC, Du K, Filipović M, Keller D,
Ketzef M, Kozlov AK, Kumar A, Lindahl M, Nair AG, Pérez-Fernández J,
Grillner S, Silberberg G & Hellgren Kotaleski J (2018). Basal ganglia
neuromodulation over multiple temporal and structural scales-simulations
of direct pathway MSNs investigate the fast onset of dopaminergic
effects and predict the role of Kv4.2. Front Neural Circuits 12, 3
(https://doi.org/10.3389/fncir.2018.00003).

[Lindroos2021]: Lindroos R & Hellgren Kotaleski J (2021). Predicting
complex spikes in striatal projection neurons of the direct pathway
following neuromodulation by acetylcholine and dopamine. Eur J Neurosci
53, 2117–2134 (https://doi.org/10.1111/ejn.14891).