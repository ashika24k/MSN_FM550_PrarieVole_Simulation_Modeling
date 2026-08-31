# Figure 2: Control vs FM550 Rectification Comparison

This folder contains dMSN simulations comparing the control and FM550 model conditions across hyperpolarizing current steps. Both analyses calculate voltage traces, steady-state voltage change, and inward rectification directly from the root `msn/` model package. No tabular input data are required.

## Run Instructions

1. Start in the repository root, which must contain the `msn/` package.
2. Install the release dependencies once:

	```bash
	python -m pip install -r "Public Release/requirements.txt"
	```

3. Run the complete nine-panel condition comparison when you need each condition and their direct overlays:

	```bash
	python "Public Release/F2_Control_FM550_Rectification_Comparison/analysis/F2_dMSN_control_FM550_9_panel_rectification_comparison.py"
	```

4. Run the compact three-panel overlay when you need only the final control-versus-FM550 comparison figure:

	```bash
	python "Public Release/F2_Control_FM550_Rectification_Comparison/analysis/F2_dMSN_control_FM550_3_panel_overlay.py"
	```

5. Review the printed summary from the nine-panel script for the mean inward rectification, stepwise percent inward rectification, and difference in mean inward rectification between conditions.

## Simulation Protocol

Both scripts simulate dMSN cell 21. The control condition uses $V_{init}=-82.8$ mV and a rheobase of $112.5$ pA; the FM550 condition uses $V_{init}=-82.3$ mV and a rheobase of $153.0$ pA. Each condition first finds a holding current that maintains its target initial voltage, then applies nine negative current steps from $-0.09$ to $-0.01$ nA after a 200 ms pre-equilibration period. Steady-state voltage is measured during the final 50 ms of the 150 ms test pulse.

## Analyses and Interpretation

`F2_dMSN_control_FM550_9_panel_rectification_comparison.py` displays three rows. The first row shows control traces, the control current-voltage relationship, and control inward rectification. The second row presents the same analyses for FM550. The third row overlays selected traces and both conditions' current-voltage and inward-rectification curves. Compare the curves at the same injected current: separation between conditions indicates the modeled effect of the FM550 parameter set.

`F2_dMSN_control_FM550_3_panel_overlay.py` produces the compact final comparison. Panel A overlays representative voltage traces, Panel B compares steady-state $\Delta V$ over injected current, and Panel C compares inward rectification calculated as $|\Delta V / I|$ in M$\Omega$. A changing slope across negative current steps indicates a non-linear voltage response; a systematic separation of the control and FM550 curves indicates a condition-dependent change in the modeled rectification.

The nine-panel script displays the result without saving an image. The three-panel overlay saves `Figure2_OverlayRow.png` in its `analysis/` directory. Generated images are intentionally not stored in the public release.