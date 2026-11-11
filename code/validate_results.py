#!/usr/bin/env python3
"""Compare reproduced results with the archived corrected result workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHECKS = {
    "Community_AR1": {"keys": ["scope", "response", "thermal_metric", "lag_months"], "columns": ["rho_ar1", "r2_marginal", "beta_standardized", "se", "t", "p", "p_cyclic", "p_bh_within_scope"], "tolerance": 1e-5},
    "Partial_RDA": {"keys": ["scope", "thermal_metric", "lag_months"], "columns": ["partial_r2", "pseudo_f", "p_cyclic", "p_bh_within_scope"], "tolerance": 1e-8},
    "Taxon_AR1": {"keys": ["taxon", "lag_months"], "columns": ["rho_ar1", "beta_standardized", "se", "p", "ci95_low", "ci95_high", "p_bh"], "tolerance": 1e-5},
    "PCoA_Scores": {"keys": ["date"], "columns": ["PCoA1", "PCoA2", "PCoA1_explained_pct", "PCoA2_explained_pct"], "tolerance": 1e-8},
    "Annual_Summary": {"keys": ["year"], "columns": ["density_median_org_m3", "density_mean_org_m3", "richness_median", "shannon_median", "mhw_days_total", "anomaly_mean_c", "dhw_max_c_weeks"], "tolerance": 1e-6},
    "Phase_Summary": {"keys": ["phase"], "columns": ["density_median_org_m3", "richness_median", "shannon_median", "mhw_days_total", "anomaly_mean_c", "dhw_max_c_weeks"], "tolerance": 1e-6},
    "Temp_Validation": {"keys": ["set"], "columns": ["pearson_r", "p"], "tolerance": 5e-5},
    "Sensitivity": {"keys": ["analysis", "sensitivity_set", "lag_months"], "columns": ["n", "beta_standardized", "p", "rho_ar1", "partial_r2", "pseudo_f", "design_rank", "design_columns", "max_vif", "condition_number", "cyclic_permutations"], "tolerance": 1e-5},
}


def compare(reference_file: Path, reproduced_dir: Path):
    report = {"reference": reference_file.name, "status": "passed", "checks": {}}
    for sheet, specification in CHECKS.items():
        reference = pd.read_excel(reference_file, sheet_name=sheet)
        reproduced = pd.read_csv(reproduced_dir / f"{sheet}.csv")
        for key in specification["keys"]:
            if key == "date":
                reference[key] = pd.to_datetime(reference[key]).dt.strftime("%Y-%m-%d")
                reproduced[key] = pd.to_datetime(reproduced[key]).dt.strftime("%Y-%m-%d")
        merged = reference.merge(reproduced, on=specification["keys"], suffixes=("_reference", "_reproduced"))
        maxima = {}
        for column in specification["columns"]:
            difference = np.abs(
                pd.to_numeric(merged[f"{column}_reference"], errors="coerce")
                - pd.to_numeric(merged[f"{column}_reproduced"], errors="coerce")
            )
            maxima[column] = float(np.nanmax(difference))
        passed = len(reference) == len(reproduced) == len(merged) and max(maxima.values(), default=0.0) <= specification["tolerance"]
        report["checks"][sheet] = {
            "reference_rows": len(reference), "reproduced_rows": len(reproduced), "matched_rows": len(merged),
            "tolerance": specification["tolerance"], "maximum_absolute_differences": maxima,
            "status": "passed" if passed else "failed",
        }
        if not passed:
            report["status"] = "failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reproduced-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("validation_report.json"))
    args = parser.parse_args()
    report = compare(args.reference, args.reproduced_dir)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

