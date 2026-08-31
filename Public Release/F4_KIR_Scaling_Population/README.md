# Figure 4: KIR Scaling Population

This folder evaluates population-level inward rectification after scaling KIR conductance. It compares control, FM550, 1.25x KIR, and 1.5x KIR conditions in dMSN and IMSN model populations.

## Run Instructions

1. Start in the repository root, which must contain the `msn/` package.
2. Install the release dependencies once:

  ```bash
  python -m pip install -r "Public Release/requirements.txt"
  ```

3. Generate the required per-cell KIR population tables. This simulates all 71 dMSNs and 34 IMSNs, so it can take approximately 40-50 minutes:

  ```bash
  python "Public Release/F4_KIR_Scaling_Population/generation/kir/F4_generate_population_data.py"
  ```

  This creates `MSN_Population_Analysis_Results_125x15x_DMSN.csv` and `MSN_Population_Analysis_Results_125x15x_IMSN.csv` in `generation/kir/`.

4. Build the two plotting-input CSVs from the population tables generated in step 3:

  ```bash
  python "Public Release/F4_KIR_Scaling_Population/F4_build_plotting_inputs.py" \
    --kir-dmsn "Public Release/F4_KIR_Scaling_Population/generation/kir/MSN_Population_Analysis_Results_125x15x_DMSN.csv" \
    --kir-imsn "Public Release/F4_KIR_Scaling_Population/generation/kir/MSN_Population_Analysis_Results_125x15x_IMSN.csv"
  ```

  The builder writes `Figure4_Input_IR_DMSN.csv` and `Figure4_Input_IR_IMSN.csv` in this Figure 4 folder.

5. Run the dMSN strip plot and paired comparisons after step 4:

  ```bash
  python "Public Release/F4_KIR_Scaling_Population/F4_plot_dmsn_paired_tests.py"
  ```

6. Run the IMSN strip plot, one-way ANOVA, and Tukey HSD comparisons after step 4:

  ```bash
  python "Public Release/F4_KIR_Scaling_Population/F4_plot_imsn_anova_tukey.py"
  ```

## Analyses and Interpretation

`F4_generate_population_data.py` calculates average inward rectification for each modeled dMSN and IMSN under all four conditions. Its two generated tables contain the per-cell values that preserve the paired structure used by the Figure 4 analyses.

`F4_build_plotting_inputs.py` performs only a transparent column selection: control, FM550, 1.25x KIR, and 1.5x KIR inward-rectification values. It does not calculate or modify any values.

`F4_plot_dmsn_paired_tests.py` displays each dMSN population value with reproducible horizontal jitter and compares each non-control condition against control with a paired t-test. The horizontal black line is the condition mean. Significance brackets show the result of each comparison; `ns` means the test was not significant at $\alpha = 0.05$, while one to three asterisks indicate increasingly small p-values.

`F4_plot_imsn_anova_tukey.py` displays the IMSN population values, tests whether any of the four group means differ with a one-way ANOVA, and, when that test is significant, uses Tukey HSD to identify the individual group pairs. Interpret the ANOVA p-value as evidence for or against an overall condition effect; interpret Tukey-adjusted p-values only as the follow-up pairwise comparisons.

The empty input schema is in `input_templates/`. Generated population tables, plotting inputs, statistics exports, and images are intentionally excluded from this release.