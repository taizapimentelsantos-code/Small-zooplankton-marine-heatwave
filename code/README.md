# Reproducible analysis code

## Files

- `integrated_analysis.py`: rebuilds the biological descriptors, monthly integration, community-level AR(1) models, cyclic-shift probabilities, Benjamini–Hochberg corrections, partial RDA, taxon-level AR(1) models, Bray–Curtis PCoA, summaries, and local-temperature validation.
- `rebuild_mhw_conventional.py`: reconstructs the MHW event inventory from the daily threshold exceedances and joins qualifying events separated by gaps of up to two days.
- `validate_results.py`: compares the regenerated outputs with the archived corrected result workbook.
- `requirements.txt`: records the tested Python dependencies.

## Run the analysis

From the repository root:

```bash
python -m venv .venv
python -m pip install -r code/requirements.txt
python code/integrated_analysis.py --package-root . --output-dir reproduced_results
```

The deposited thermal workbook already contains the conventional event definition. To rebuild it from a prior no-gap version, run:

```bash
python code/rebuild_mhw_conventional.py input_thermal.xlsx data/03_thermal_exposure_Copernicus_NOAA_1982_2024_EN.xlsx
```

## Validate the results

```bash
python code/validate_results.py \
  --reference results/04_corrected_reanalysis_results_2021_2024_EN.xlsx \
  --reproduced-dir reproduced_results \
  --report validation_report.json
```

The supplied validation report passed all required checks.

## Statistical implementation

- Continuous responses and predictors are standardized with population standard deviations.
- MHW events require at least five above-threshold days; qualifying events separated by at most two days are joined (`joinAcrossGaps = TRUE`, `maxGap = 2`). Gap days are included in joined-event duration and monthly MHW-day totals.
- AR(1) generalized least-squares models use exact calendar-month gaps and constrain the autocorrelation parameter to −0.94–0.94.
- Cyclic probabilities use all 47 nonzero circular shifts of each 48-month thermal series. Every shifted AR(1) model is refitted, using the absolute standardized thermal coefficient as the test statistic.
- Partial RDA is applied to Hellinger-transformed taxon densities after residualizing both the community matrix and thermal predictor against the nuisance covariates.
- Taxon-level models are restricted to taxa present in at least 25% of biological campaigns and use Hellinger-transformed values.
- Benjamini–Hochberg adjustment is performed within each predefined model family.
- PCoA uses Bray–Curtis distances calculated from the Hellinger-transformed density matrix.

