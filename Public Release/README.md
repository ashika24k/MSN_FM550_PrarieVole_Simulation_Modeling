# Public Release: Firemaster® 550 (FM550) Model Analyses

This directory contains the code used to generate the computational analyses for the paper **Employing computational simulations of prairie vole nucleus accumbens medium spiny neurons to assess inwardly rectifying potassium channels as a target of perinatal flame retardant exposure**. In this investigation, the simulation workflows are used to study prairie vole (*Microtus ochrogaster*) nucleus accumbens medium spiny neurons (MSNs) under FM550-associated and ion-channel conductance manipulations.

These scripts are a focused analysis resource, not a comprehensive model of prairie vole MSN biology. They are intended to show how the model was applied in this investigation and to provide a starting point for researchers who want to adapt comparable simulations and analyses for prairie vole MSNs. Results should be interpreted within the assumptions, cell selections, stimulus protocols, and conductance manipulations defined in the individual figure packages.

## Original Model and Code Credits

The `msn/` package in the repository root is required to run these analyses. It was obtained from Antonio González's [msn-model repository](https://github.com/antgon/msn-model) (version retrieved [Jan 2025]) and is included for reproducibility; it may differ from the current upstream version. The `msn/` package remains under the copyright of its original authors and is not covered by this repository's license.

That implementation incorporates NEURON mechanisms (`.mod` files), morphologies (`.swc` files), and parameter sets associated with the Lindroos and Hellgren Kotaleski model available as [ModelDB model 266775](http://modeldb.yale.edu/266775). Its Python support code also builds on the authors' [striatal_SPN_lib repository](https://github.com/robban80/striatal_SPN_lib). The original sources and their scientific references below should remain credited.

## References

Lindroos R, Dorst MC, Du K, Filipovic M, Keller D, Ketzef M, Kozlov AK, Kumar A, Lindahl M, Nair AG, Perez-Fernandez J, Grillner S, Silberberg G, and Hellgren Kotaleski J. Basal ganglia neuromodulation over multiple temporal and structural scales-simulations of direct pathway MSNs investigate the fast onset of dopaminergic effects and predict the role of Kv4.2. *Frontiers in Neural Circuits*. 2018;12:3. https://doi.org/10.3389/fncir.2018.00003

Lindroos R and Hellgren Kotaleski J. Predicting complex spikes in striatal projection neurons of the direct pathway following neuromodulation by acetylcholine and dopamine. *European Journal of Neuroscience*. 2021;53:2117-2134. https://doi.org/10.1111/ejn.14891

Krentzel AA, Kimble LC, Dorris DM, Horman BM, Meitzen J, Patisaul HB. FireMaster® 550 (FM 550) exposure during the perinatal period impacts partner preference behavior and nucleus accumbens core medium spiny neuron electrophysiology in adult male and female prairie voles, Microtus ochrogaster. Horm Behav. 2021 Aug;134:105019. doi: 10.1016/j.yhbeh.2021.105019. Epub 2021 Jun 25. PMID: 34182292; PMCID: PMC8403633.

Kamjula A, Gaeta EM, Meitzen J. Employing computational simulations of prairie vole nucleus accumbens medium spiny neurons to assess inwardly rectifying potassium channels as a target of perinatal flame retardant exposure. *Manuscript under review.*

## Requirements

1. Use Python ≥ 3.12 (tested with 3.12.8) with the `msn/` model package present at the repository root.
2. Compile the NEURON mechanisms before running simulations. From the repository root:

	```bash
	cd msn/mechanisms
	nrnivmodl
	cd ../..
	```

3. Install the Python dependencies:

	```bash
	python -m pip install -r "Public Release/requirements.txt"
	```

## Figure Packages

Each package has its own README with the exact scripts, required inputs, run order, and interpretation guidance.

| Package | Analysis |
| --- | --- |
| `F1_Model_Validation/` | dMSN action-potential and iMSN inward-rectification model validation |
| `F2_Control_FM550_Rectification_Comparison/` | Control and FM550 rectification comparison |
| `F3_KIR_Scaling/` | Single-cell and population KIR-scaling current-voltage analyses |
| `F4_KIR_Scaling_Population/` | KIR-scaled population statistics and plots |
| `F5_F6_Alternative_Channel_Scaling/` | Alternative ion-channel population scaling analyses |

## Reproducibility and Data Statement

The underlying model files and baseline parameterization used in this investigation come from the original [msn-model repository](https://github.com/antgon/msn-model). For this project, selected simulation conditions and analyses were adapted to investigate prairie vole MSNs in the context of FM550 exposure and to reflect findings reported by [Krentzel et al. (2021)](https://doi.org/10.1016/j.yhbeh.2021.105019). These project-specific adaptations should be understood as analysis choices for this investigation, not as a replacement for the original model.

Figures 1-3 simulate results directly from the `msn` model. Figures 4-6 first produce per-cell population simulation results, then use the included input-building scripts to create the tables used by the plotting and statistical analyses. Follow each figure package README in order because later stages require outputs from the earlier simulation stages.

Generated model-result tables, assembled plotting inputs, statistics exports, and figure-image outputs are not included in this repository. They can be regenerated by running the supplied workflow with the required model package and dependencies. Users may also adjust simulation conditions, cell selections, conductance scaling, or analysis settings to generate comparable data and extend the analyses for their own research questions. Any modified analysis should document its parameter changes and retain credit to the original model and relevant source data.


## License

The analysis code in `Public Release/` is released under the MIT License (see `LICENSE`). This license does not apply to the `msn/` package or `Original_Model_Example_Scripts/`, which remain under the copyright of their original authors.
