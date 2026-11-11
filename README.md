# Tamandaré Reef Small-Zooplankton and Thermal-Exposure Dataset

## Overview

This repository provides the data, corrected analytical results, and Python scripts supporting the study of small-zooplankton dynamics and marine heatwave exposure at Tamandaré Reef, northeastern Brazil, from January 2021 to December 2024.

The repository includes biological, environmental, and satellite-derived thermal datasets, together with corrected analytical outputs and reproducible code.

The public spreadsheets are provided in English, use machine-readable field names, preserve ISO-compatible Excel dates, and contain fixed values to avoid differences among spreadsheet software. Original calculated environmental values are retained and documented in a dedicated audit sheet.

## Study period and data coverage

- Biological sampling period: January 2021 to December 2024.
- Biological dataset: 47 monthly sampling campaigns.
- Taxonomic and developmental categories: 55.
- June 2024: no biological sampling was conducted; this month was not interpolated.
- Local environmental dataset: 48 monthly records, from January 2021 to December 2024.
- Thermal exposure dataset: Copernicus Marine Service and NOAA Coral Reef Watch products, covering the period indicated in the thermal workbook.

## Repository contents

### Data (`data/`)

| File | Description |
|---|---|
| `01_local_environmental_data_2021_2024_EN.xlsx` | Monthly local environmental observations, quality-control decisions, and audit information for formula-derived values. |
| `02_zooplankton_data_2021_2024_EN.xlsx` | Raw counts, corrected densities, sampling metadata, density validation, and taxonomic dictionary for the 47 analyzed campaigns. |
| `03_thermal_exposure_Copernicus_NOAA_1982_2024_EN.xlsx` | Copernicus sea-surface temperature data, NOAA Coral Reef Watch indicators, marine heatwave events, summaries, methods, and sources. |

### Results (`results/`)

| File | Description |
|---|---|
| `04_corrected_reanalysis_results_2021_2024_EN.xlsx` | Corrected integrated data, AR(1) results, partial RDA, taxon-level models, diagnostics, sensitivity analyses, ordination scores, and key analytical results. |

### Analysis code (`code/`)

| File | Description |
|---|---|
| `integrated_analysis.py` | Reproducible implementation of the corrected analytical workflow. |
| `rebuild_mhw_conventional.py` | Reconstructs marine heatwave events and daily, monthly, and annual summaries using the conventional two-day gap-joining rule. |
| `validate_results.py` | Compares regenerated outputs with the archived corrected results. |
| `requirements.txt` | Tested Python dependencies and versions. |
| `README.md` | Instructions for running and validating the analyses. |

## Data quality control

The following decisions were applied to the deposited datasets and corrected analyses:

- Biological records contain 47 analyzed monthly campaigns and 55 taxonomic or developmental categories.
- June 2024 was not sampled biologically and was not interpolated.
- The local environmental series contains 48 monthly records.
- Local temperature is retained for descriptive purposes and thermal-series validation but was not used as an analytical covariate. January 2024 was treated as a local-temperature outlier; Copernicus sea-surface temperature was used as the primary thermal-exposure series.
- pH was excluded from the analyses because extreme values and calibration or quality-control information could not be verified.
- Conductivity and total dissolved solids were retained for transparency but excluded from the models because they were redundant with salinity.
- Oxygen saturation was retained for transparency but excluded because it duplicated dissolved oxygen information and had a primary salinity-related covariate.
- Calculated spreadsheet values were converted to fixed values for repository stability. Their original expressions are documented as non-executable text in the `Formula_Derived_Values` audit sheet.
- Corrected zooplankton density was calculated from raw counts, dilution volume, aliquot volume, and filtered volume in cubic metres. Campaign-level density totals were validated against the cached source calculations.

## Marine heatwave definition

Marine heatwave events follow the definition proposed by Hobday et al. (2016). Qualifying events separated by gaps of up to two days were joined using the conventional gap-joining rule (`joinAcrossGaps = TRUE`, `maxGap = 2`).

The thermal dataset includes the relevant sea-surface temperature information and thermal-stress indicators from Copernicus Marine Service and NOAA Coral Reef Watch. Detailed sources and definitions are provided in the thermal workbook.

## Analytical workflow

The corrected analytical results include:

- Community-level AR(1) generalized least-squares models.
- Seasonally adjusted secondary models.
- Partial redundancy analysis (partial RDA) with cyclic-shift inference.
- Taxon-level AR(1) models.
- Benjamini–Hochberg multiple-testing correction.
- Model diagnostics and sensitivity analyses, including exclusion of February 2021.
- Bray–Curtis principal coordinates analysis (PCoA).
- Community diversity descriptors and phase summaries.

The `integrated_analysis.py` script reproduces the corrected community AR(1), partial-RDA, taxon-level, diversity, phase-summary, and PCoA results using the deposited data files.

## Reproducibility

The analyses were implemented in Python. The required dependencies and tested versions are listed in `code/requirements.txt`.

From the repository root, create a Python environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install -r code/requirements.txt
```

Run the integrated analysis:

```bash
python code/integrated_analysis.py --package-root . --output-dir reproduced_results
```

Validate the regenerated results against the archived corrected workbook:

```bash
python code/validate_results.py \
  --reference results/04_corrected_reanalysis_results_2021_2024_EN.xlsx \
  --reproduced-dir reproduced_results \
  --report validation_report.json
```

The supplied validation report records the outcome of the validation checks. See `validation_report.json` for details.

Additional execution and methodological information is available in `code/README.md` and `HOBDAY_GAP_JOINING_UPDATE.md`.

## Citation and acknowledgements

The manuscript citation, author information, institutional affiliations, ORCID identifiers, funding information, and permanent repository DOI will be added upon finalization of the publication and data deposit.

## Licenses and access

The repository is currently maintained as a private repository during preparation and review. Data and code licensing and public access conditions should be established before public release.
