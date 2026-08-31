# Figures 5-6: Alternative Channel Scaling

This folder tests how scaling individual ion channels changes population inward rectification in dMSN and IMSN model cells. Each analysis includes control and FM550 reference conditions, then compares 1.25x and 1.5x scaling of KIR, KAF, KAS, KDR, NAF, SK, or BK conductance.

## Run Instructions

1. Start in the repository root, which must contain the `msn/` package.
2. Install the release dependencies once:

  ```bash
  python -m pip install -r "Public Release/requirements.txt"
  ```

3. Generate population results for every channel. These are the required inputs for the next step. Each command simulates 71 dMSNs and 34 IMSNs and can take substantial time:

  ```bash
  python "Public Release/F5_F6_Alternative_Channel_Scaling/generation/kir/generate_kir_population.py"
  python "Public Release/F5_F6_Alternative_Channel_Scaling/generation/kaf/generate_kaf_population.py"
  python "Public Release/F5_F6_Alternative_Channel_Scaling/generation/kas/generate_kas_population.py"
  python "Public Release/F5_F6_Alternative_Channel_Scaling/generation/kdr/generate_kdr_population.py"
  python "Public Release/F5_F6_Alternative_Channel_Scaling/generation/naf/generate_naf_population.py"
  python "Public Release/F5_F6_Alternative_Channel_Scaling/generation/sk/generate_sk_population.py"
  python "Public Release/F5_F6_Alternative_Channel_Scaling/generation/bk/generate_bk_population.py"
  ```

  Each generator writes its dMSN and IMSN CSV files to its own `generation/<channel>/` directory.

4. Build the four figure-ready input tables only after all seven generators finish:

  ```bash
  python "Public Release/F5_F6_Alternative_Channel_Scaling/F5_F6_build_inputs.py"
  ```

  This creates one input table for each cell type and scaling factor: dMSN 1.25x, dMSN 1.5x, IMSN 1.25x, and IMSN 1.5x.

5. Create the complete 2x2 population comparison figure and run one-way ANOVA followed by Tukey HSD:

  ```bash
  python "Public Release/F5_F6_Alternative_Channel_Scaling/F5_F6_plot_all_conditions.py"
  ```

6. To inspect a single cell-type/scaling condition, first select one of the four `Figure5&6_Input_IR_*.csv` values in `CSV_FILE` near the top of `F5_F6_plot_one_condition.py`, then run:

  ```bash
  python "Public Release/F5_F6_Alternative_Channel_Scaling/F5_F6_plot_one_condition.py"
  ```

## Analyses and Interpretation

Each `generate_<channel>_population.py` script calculates the average inward rectification for every model cell under control, FM550, and 1.25x and 1.5x scaling of its named channel. The output retains per-cell values, allowing the figure inputs to represent the simulated population rather than only a summary statistic.

`F5_F6_build_inputs.py` creates the plotting tables by selecting the FM550 inward-rectification value plus the matching scaled value from each channel analysis. The 1.25x and 1.5x outputs therefore compare the same magnitude of conductance scaling across all seven channels.

`F5_F6_plot_all_conditions.py` creates Panels A-D: dMSN 1.25x, dMSN 1.5x, IMSN 1.25x, and IMSN 1.5x. Each strip plot shows individual simulated cell values and a horizontal mean line. The one-way ANOVA tests whether any channel condition differs within a panel. When significant, Tukey HSD identifies the pairwise differences; brackets and stars mark significant comparisons involving FM550.

`F5_F6_plot_one_condition.py` provides the same ANOVA and Tukey HSD analysis for the selected single input table. Use it when a full four-panel comparison is not needed.

Generated population tables, assembled input tables, statistics exports, and image outputs are intentionally not stored in the public release.