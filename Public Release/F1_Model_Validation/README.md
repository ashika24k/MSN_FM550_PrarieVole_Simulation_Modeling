# Figure 1: Model Validation

This folder contains the simulations used to validate the model's membrane-voltage responses. It does not use experimental input tables: each script simulates the required medium spiny neuron response from the root `msn/` model package.

## Run Instructions

1. Start in the repository root, which must contain the `msn/` package.
2. Install the release dependencies once:

	```bash
	python -m pip install -r "Public Release/requirements.txt"
	```

3. Run the complete three-panel Figure 1 and save `Figure1_Combined_ABC.png` in the `analysis/` directory:

	```bash
	python "Public Release/F1_Model_Validation/analysis/F1_model_validation_combined_figure.py"
	```

4. Run only the dMSN action-potential validation trace when that panel is needed. This displays the trace without saving an image:

	```bash
	python "Public Release/F1_Model_Validation/analysis/F1_dMSN_rheobase_evoked_action_potential.py"
	```

5. Run only the IMSN hyperpolarizing-current analysis when those panels are needed. This displays the voltage-trace plot and the current-voltage plot separately:

	```bash
	python "Public Release/F1_Model_Validation/analysis/F1_IMSN_hyperpolarizing_traces_and_current_voltage.py"
	```

## Analyses and Interpretation

`F1_dMSN_rheobase_evoked_action_potential.py` simulates dMSN cell 21 with a $0.05$ nA depolarizing injection and rheobase enabled. The membrane-potential trace should show the model's response during the 10-160 ms stimulus period; an evoked action potential supports that the model produces an excitable dMSN response under this protocol.

`F1_IMSN_hyperpolarizing_traces_and_current_voltage.py` holds IMSN cell 32 at $-84$ mV, then applies negative current steps from $-0.01$ to $-0.91$ nA. The trace plot shows the time-resolved hyperpolarizing responses. The current-voltage plot reports steady-state $\Delta V$ for each injection. A curved or changing-slope relationship indicates that the voltage response is not purely linear across the tested current range, which is the modeled inward-rectification behavior assessed by this validation.

`F1_model_validation_combined_figure.py` runs both protocols and places them in one figure: Panel A is the dMSN action-potential trace, Panel B is the IMSN hyperpolarizing response family, and Panel C is the IMSN steady-state current-voltage relationship.

Generated images are outputs of the analysis and are intentionally not stored in the public release.