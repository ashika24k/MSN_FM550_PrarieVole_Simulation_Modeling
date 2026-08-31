# Figure 3: KIR Scaling

This folder tests how scaling KIR conductance changes current-voltage responses in dMSN and IMSN model populations. Each analysis compares four conditions: control, 1.25x KIR, 1.5x KIR, and FM550.

## Run Instructions

1. Start in the repository root, which must contain the `msn/` package.
2. Install the release dependencies once:

	```bash
	python -m pip install -r "Public Release/requirements.txt"
	```

3. Before running either population plotting script or the combined figure, generate the required dMSN and IMSN per-cell CSVs. This is the first required analysis step and can take substantial time because it simulates all 71 dMSNs and 34 IMSNs:

	```bash
	python "Public Release/F3_KIR_Scaling/analysis/F3_generate_population_data.py"
	```

	The script writes `Figure3C_DMSN_RAW.csv` and `Figure3D_IMSN_RAW.csv` to `analysis/generated_inputs/`.

4. To plot only the generated population data, run:

	```bash
	python "Public Release/F3_KIR_Scaling/analysis/F3_plot_population_data.py"
	```

5. After step 3, run the combined four-panel figure:

	```bash
	python "Public Release/F3_KIR_Scaling/analysis/F3_combined_figure.py"
	```

6. `analysis/F3_single_cell_analysis.py` provides the reusable single-cell simulation function used by the combined figure's Panels A and B. Its direct entry point runs additional exploratory KIR presets, so use the combined figure command for the Figure 3 single-cell result.

## Analyses and Interpretation

`F3_generate_population_data.py` performs the population-level simulation and calculates steady-state $\Delta V$ at negative current steps from $-0.09$ to $-0.01$ nA. It retains one response row per simulated cell and condition, which is the input required for the population summaries.

`F3_plot_population_data.py` reads those generated inputs and shows mean $\pm$ SEM current-voltage curves separately for dMSNs and IMSNs. Compare the position and curvature of the 1.25x KIR and 1.5x KIR curves against control to assess the modeled effect of increasing KIR conductance. The FM550 curve provides the model's FM550 reference condition. Error bars show the standard error across cells.

`F3_combined_figure.py` creates the publication-style summary. Panels A and B are single-cell dMSN and IMSN current-voltage responses, respectively. Panels C and D are the dMSN and IMSN population mean $\pm$ SEM responses, respectively. A systematic separation of the KIR-scaled curves from control supports an effect of the conductance manipulation; overlap relative to the SEM indicates less separation across the modeled population.

Generated population CSVs and image outputs are intentionally not stored in the public release.