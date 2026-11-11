#!/usr/bin/env python3
"""Reproducible analysis for the Tamandaré small-zooplankton dataset.

The script rebuilds biological descriptors from raw counts, merges the local
environmental and daily thermal records, fits the AR(1) model families, runs
partial RDA with deterministic cyclic shifts, calculates taxon-level models,
and produces Bray–Curtis PCoA scores.

Example
-------
python code/integrated_analysis.py --package-root . --output-dir reproduced_results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, pearsonr, shapiro, t as student_t


THERMAL_METRICS = ("mhw_days", "cop_anomaly_mean_c", "noaa_dhw_max_c_weeks")
RESPONSES = ("log1p_density", "taxon_richness", "shannon", "pielou", "simpson_1d")
ENVIRONMENTAL_COVARIATES = ("salinity", "log1p_turbidity", "dissolved_oxygen_mg_l", "orp_mv")
PHASES = {
    "Pre-main-event (2023-11 to 2024-01)": ("2023-11-01", "2024-01-01"),
    "Main-event (2024-02 to 2024-05)": ("2024-02-01", "2024-05-01"),
    "Post-main-event (2024-07 to 2024-09)": ("2024-07-01", "2024-09-01"),
}


def zscore(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Cannot standardize a constant or non-finite vector")
    return (values - values.mean()) / sd


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def signed_ar1_correlation(rho: float, time_index: np.ndarray) -> np.ndarray:
    gaps = np.abs(time_index[:, None] - time_index[None, :]).astype(int)
    matrix = np.sign(rho) ** gaps * np.abs(rho) ** gaps
    np.fill_diagonal(matrix, 1.0)
    return matrix


@dataclass
class AR1Fit:
    beta: np.ndarray
    se: np.ndarray
    t_value: np.ndarray
    p_value: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    rho: float
    fitted: np.ndarray
    residuals: np.ndarray
    r2: float
    aic: float
    df_residual: int


def fit_ar1(y: np.ndarray, predictors: list[np.ndarray], time_index: np.ndarray) -> AR1Fit:
    y = zscore(y)
    time_index = np.asarray(time_index, dtype=float)
    x = np.column_stack([np.ones(len(y))] + [zscore(v) for v in predictors])
    n, p = x.shape

    def evaluate(rho: float):
        correlation = signed_ar1_correlation(rho, time_index)
        factor = cho_factor(correlation, lower=True, check_finite=False)
        inverse_x = cho_solve(factor, x, check_finite=False)
        inverse_y = cho_solve(factor, y, check_finite=False)
        beta = np.linalg.solve(x.T @ inverse_x, x.T @ inverse_y)
        residuals = y - x @ beta
        quadratic = float(residuals @ cho_solve(factor, residuals, check_finite=False))
        sigma2_ml = quadratic / n
        logdet = 2.0 * np.log(np.diag(factor[0])).sum()
        minus_two_loglik = n * np.log(2.0 * np.pi) + logdet + n * np.log(sigma2_ml) + n
        return minus_two_loglik, beta, residuals, quadratic, inverse_x

    optimum = minimize_scalar(
        lambda rho: evaluate(rho)[0], bounds=(-0.94, 0.94), method="bounded", options={"xatol": 1e-12}
    )
    minus_two_loglik, beta, residuals, quadratic, inverse_x = evaluate(float(optimum.x))
    df_residual = n - p
    covariance = (quadratic / df_residual) * np.linalg.inv(x.T @ inverse_x)
    se = np.sqrt(np.diag(covariance))
    t_value = beta / se
    p_value = 2.0 * student_t.sf(np.abs(t_value), df_residual)
    critical = student_t.ppf(0.975, df_residual)
    fitted = x @ beta
    r2 = 1.0 - float(residuals @ residuals) / float(y @ y)
    # The working analysis counted the AR parameter in addition to regression
    # coefficients and used the residual-variance estimate in the AIC display.
    sigma2_df = quadratic / df_residual
    correlation = signed_ar1_correlation(float(optimum.x), time_index)
    logdet = np.linalg.slogdet(correlation)[1]
    aic = (
        n * np.log(2.0 * np.pi)
        + logdet
        + n * np.log(sigma2_df)
        + quadratic / sigma2_df
        + 2.0 * (p + 1)
    )
    return AR1Fit(
        beta=beta,
        se=se,
        t_value=t_value,
        p_value=p_value,
        ci_low=beta - critical * se,
        ci_high=beta + critical * se,
        rho=float(optimum.x),
        fitted=fitted,
        residuals=residuals,
        r2=r2,
        aic=float(aic),
        df_residual=df_residual,
    )


def fit_t_with_fixed_rho(y, predictors, time_index, rho) -> float:
    y = zscore(y)
    time_index = np.asarray(time_index, dtype=float)
    x = np.column_stack([np.ones(len(y))] + [zscore(v) for v in predictors])
    n, p = x.shape
    correlation = signed_ar1_correlation(rho, time_index)
    factor = cho_factor(correlation, lower=True, check_finite=False)
    inverse_x = cho_solve(factor, x, check_finite=False)
    inverse_y = cho_solve(factor, y, check_finite=False)
    beta = np.linalg.solve(x.T @ inverse_x, x.T @ inverse_y)
    residuals = y - x @ beta
    quadratic = float(residuals @ cho_solve(factor, residuals, check_finite=False))
    covariance = (quadratic / (n - p)) * np.linalg.inv(x.T @ inverse_x)
    return float(beta[1] / np.sqrt(covariance[1, 1]))


def diagnostic_statistics(y, predictors, time_index, fit: AR1Fit):
    """Diagnostics based on the whitened GLS residuals used in the audit."""
    y = zscore(y)
    time_index = np.asarray(time_index, dtype=float)
    x = np.column_stack([np.ones(len(y))] + [zscore(v) for v in predictors])
    correlation = signed_ar1_correlation(fit.rho, time_index)
    lower = cholesky(correlation, lower=True, check_finite=False)
    whitened_residuals = solve_triangular(lower, fit.residuals, lower=True, check_finite=False)
    whitened_x = solve_triangular(lower, x, lower=True, check_finite=False)
    n, p = x.shape

    centered = whitened_residuals - whitened_residuals.mean()
    denominator = float(centered @ centered)
    ljung = {}
    for lag in (6, 12):
        correlations = [
            float(centered[k:] @ centered[:-k]) / denominator for k in range(1, lag + 1)
        ]
        q = n * (n + 2) * sum(value * value / (n - k) for k, value in enumerate(correlations, 1))
        df = lag - 1
        ljung[lag] = (q, df, float(chi2.sf(q, df)))

    shapiro_p = float(shapiro(whitened_residuals).pvalue)
    squared = whitened_residuals**2
    auxiliary_beta = np.linalg.lstsq(x, squared, rcond=None)[0]
    auxiliary_fitted = x @ auxiliary_beta
    r2_auxiliary = 1.0 - float(np.sum((squared - auxiliary_fitted) ** 2)) / float(
        np.sum((squared - squared.mean()) ** 2)
    )
    bp_lm = n * r2_auxiliary
    bp_df = p - 1
    bp_p = float(chi2.sf(bp_lm, bp_df))

    hat = np.diag(whitened_x @ np.linalg.inv(whitened_x.T @ whitened_x) @ whitened_x.T)
    mse = float(whitened_residuals @ whitened_residuals) / (n - p)
    cooks = (whitened_residuals**2 / (p * mse)) * hat / (1.0 - hat) ** 2
    gaps = np.diff(time_index)
    return {
        "ljung_box_q6": ljung[6][0], "ljung_box_df6": ljung[6][1], "ljung_box_p6": ljung[6][2],
        "ljung_box_q12": ljung[12][0], "ljung_box_df12": ljung[12][1], "ljung_box_p12": ljung[12][2],
        "shapiro_wilk_p": shapiro_p,
        "breusch_pagan_lm": bp_lm, "breusch_pagan_df": bp_df, "breusch_pagan_p": bp_p,
        "max_cooks_distance": float(cooks.max()),
        "cooks_threshold_4_over_n": 4.0 / n,
        "n_cooks_gt_4_over_n": int(np.sum(cooks > 4.0 / n)),
        "max_vif": float(np.diag(np.linalg.inv(np.corrcoef(x[:, 1:], rowvar=False))).max()),
        "condition_number": float(np.linalg.cond(x[:, 1:])),
        "optimizer_success": True,
        "rho_near_boundary": abs(fit.rho) >= 0.94,
        "max_gap_months": int(gaps.max()) if len(gaps) else 0,
        "n_gaps_gt1": int(np.sum(gaps > 1)),
    }


def diagnostics(integrated: pd.DataFrame, density: pd.DataFrame, bio_dates: pd.Series):
    community_rows = []
    for scope in ("seasonal-adjusted", "environment-adjusted"):
        for response in RESPONSES:
            for metric in THERMAL_METRICS:
                for lag in (0, 1):
                    frame = model_frame(integrated, response, metric, lag)
                    predictors, _ = predictors_for(frame, scope)
                    fit = fit_ar1(frame[response], predictors, frame["time_index"])
                    row = {
                        "model_id": f"{scope}|{response}|{metric}|lag{lag}",
                        "model_family": "community-level AR(1)", "scope": scope,
                        "response": response, "thermal_metric": metric, "lag_months": lag,
                        "n": len(frame), "rho_ar1": fit.rho,
                    }
                    row.update(diagnostic_statistics(frame[response], predictors, frame["time_index"], fit))
                    community_rows.append(row)

    density_by_date = density.copy()
    density_by_date.index = pd.to_datetime(bio_dates)
    hellinger = np.sqrt(density_by_date.div(density_by_date.sum(axis=1), axis=0))
    prevalence = (density_by_date > 0).sum(axis=0)
    eligible = prevalence[prevalence / len(density_by_date) >= 0.25].index
    taxon_rows = []
    for taxon in eligible:
        integrated_taxon = integrated.copy()
        integrated_taxon["taxon_response"] = integrated_taxon["date"].map(hellinger[taxon])
        for lag in (0, 1):
            frame = model_frame(integrated_taxon, "taxon_response", "mhw_days", lag)
            predictors, _ = predictors_for(frame, "environment-adjusted")
            fit = fit_ar1(frame["taxon_response"], predictors, frame["time_index"])
            row = {
                "model_id": f"taxon|{taxon}|mhw_days|lag{lag}",
                "model_family": "taxon-level AR(1)", "scope": "environment-adjusted",
                "response": taxon, "thermal_metric": "mhw_days", "lag_months": lag,
                "n": len(frame), "rho_ar1": fit.rho,
            }
            row.update(diagnostic_statistics(frame["taxon_response"], predictors, frame["time_index"], fit))
            taxon_rows.append(row)
    return pd.DataFrame(community_rows), pd.DataFrame(taxon_rows)


def read_inputs(package_root: Path):
    data_dir = package_root / "data" if (package_root / "data").exists() else package_root
    environment = pd.read_excel(data_dir / "01_local_environmental_data_2021_2024_EN.xlsx", sheet_name="Monthly_Data")
    counts = pd.read_excel(data_dir / "02_zooplankton_data_2021_2024_EN.xlsx", sheet_name="Raw_Counts")
    metadata = pd.read_excel(data_dir / "02_zooplankton_data_2021_2024_EN.xlsx", sheet_name="Sampling_Metadata")
    thermal_file = data_dir / "03_thermal_exposure_Copernicus_NOAA_1982_2024_EN.xlsx"
    daily_mhw = pd.read_excel(thermal_file, sheet_name="Daily_MHW_2021_2024")
    daily_noaa = pd.read_excel(thermal_file, sheet_name="Daily_NOAA_2021_2024")
    for frame in (environment, counts, metadata, daily_mhw, daily_noaa):
        frame["date"] = pd.to_datetime(frame["date"])
    return environment, counts, metadata, daily_mhw, daily_noaa


def monthly_thermal(daily_mhw: pd.DataFrame, daily_noaa: pd.DataFrame) -> pd.DataFrame:
    mhw = daily_mhw.copy()
    noaa = daily_noaa.copy()
    mhw["date"] = pd.to_datetime(mhw["date"])
    noaa["date"] = pd.to_datetime(noaa["date"])
    mhw["month_date"] = mhw["date"].dt.to_period("M").dt.to_timestamp()
    noaa["month_date"] = noaa["date"].dt.to_period("M").dt.to_timestamp()

    m1 = mhw.groupby("month_date").agg(
        cop_sst_mean_c=("sst_c", "mean"),
        cop_sst_max_c=("sst_c", "max"),
        cop_anomaly_mean_c=("intensity_above_climatology_c", "mean"),
        cop_anomaly_max_c=("intensity_above_climatology_c", "max"),
        mhw_days=("event_id", lambda x: int(x.notna().sum())),
        mhw_event_count=("event_id", lambda x: x.dropna().nunique()),
    )
    event_intensity = mhw.assign(
        event_intensity=np.where(mhw["event_id"].notna(), mhw["intensity_above_climatology_c"], 0.0)
    ).groupby("month_date")["event_intensity"].sum()
    m1["mhw_cumulative_intensity_c_days"] = event_intensity

    m2 = noaa.groupby("month_date").agg(
        noaa_sst90_mean_c=("sst_at_p90_hotspot_pixel_c", "mean"),
        noaa_hotspot_max_c=("hotspot_c", "max"),
        noaa_dhw_max_c_weeks=("dhw_c_weeks", "max"),
        noaa_days_hotspot_ge_1=("hotspot_c", lambda x: int((x >= 1.0).sum())),
        noaa_days_warning_or_higher=("legacy_7d_baa_code", lambda x: int((x >= 2).sum())),
        noaa_days_alert1_or_higher=("legacy_7d_baa_code", lambda x: int((x >= 3).sum())),
        noaa_baa_max_code=("legacy_7d_baa_code", "max"),
        noaa_current_days_alert1_or_higher=("current_7d_baa_code", lambda x: int((x >= 3).sum())),
        noaa_current_baa_max_code=("current_7d_baa_code", "max"),
    )
    thermal = m1.join(m2, how="inner").reset_index().rename(columns={"month_date": "date"})
    thermal["year"] = thermal["date"].dt.year
    thermal["month"] = thermal["date"].dt.month
    return thermal.sort_values("date").reset_index(drop=True)


def biological_data(counts: pd.DataFrame, metadata: pd.DataFrame):
    taxa = [c for c in counts.columns if c != "date"]
    merged = counts.merge(metadata[["date", "sample", "density_scale_factor"]], on="date", validate="one_to_one")
    density = merged[taxa].astype(float).mul(merged["density_scale_factor"], axis=0)
    total = density.sum(axis=1)
    relative = density.div(total, axis=0)
    positive = density.gt(0)
    richness = positive.sum(axis=1)
    shannon = -(relative.where(relative > 0, 1.0) * np.log(relative.where(relative > 0, 1.0))).sum(axis=1)
    pielou = shannon / np.log(richness)
    simpson = 1.0 - (relative**2).sum(axis=1)
    bio = pd.DataFrame(
        {
            "date": merged["date"],
            "sample": merged["sample"],
            "density_org_m3": total,
            "log1p_density": np.log1p(total),
            "taxon_richness": richness,
            "shannon": shannon,
            "pielou": pielou,
            "simpson_1d": simpson,
        }
    )
    return bio, density.set_axis(taxa, axis=1), taxa


def integrated_table(environment, bio, thermal):
    env = environment.rename(
        columns={"sample_id": "environment_sample", "turbidity_raw": "turbidity_raw"}
    ).copy()
    env["log1p_turbidity"] = np.log1p(env["turbidity_raw"])
    calendar = pd.DataFrame({"date": pd.date_range("2021-01-01", "2024-12-01", freq="MS")})
    table = calendar.merge(thermal, on="date", how="left").merge(env, on="date", how="left").merge(bio, on="date", how="left")
    table["time_index"] = (table["date"].dt.year - 2021) * 12 + table["date"].dt.month - 1
    table["sin_month"] = np.sin(2 * np.pi * (table["date"].dt.month - 1) / 12)
    table["cos_month"] = np.cos(2 * np.pi * (table["date"].dt.month - 1) / 12)
    table["phase_2024"] = "Other"
    for label, (start, end) in PHASES.items():
        table.loc[table["date"].between(start, end), "phase_2024"] = label
    return table


def model_frame(integrated: pd.DataFrame, response: str, metric: str, lag: int):
    frame = integrated.copy()
    frame["thermal"] = frame[metric].shift(lag)
    required = [response, "thermal", "time_index", "sin_month", "cos_month", *ENVIRONMENTAL_COVARIATES]
    return frame.dropna(subset=required).copy()


def predictors_for(frame: pd.DataFrame, scope: str, thermal_values=None):
    thermal = frame["thermal"].to_numpy(float) if thermal_values is None else np.asarray(thermal_values, float)
    values = [thermal, frame["time_index"], frame["sin_month"], frame["cos_month"]]
    names = ["thermal", "time", "sin_month", "cos_month"]
    if scope == "environment-adjusted":
        values.extend(frame[c] for c in ENVIRONMENTAL_COVARIATES)
        names.extend(ENVIRONMENTAL_COVARIATES)
    return values, names


def shifted_thermal_for_dates(integrated: pd.DataFrame, metric: str, lag: int, shift: int, dates: pd.Series):
    calendar = integrated.set_index("date")[metric].sort_index()
    shifted = pd.Series(np.roll(calendar.to_numpy(float), shift), index=calendar.index).shift(lag)
    return dates.map(shifted).to_numpy(float)


def community_ar1(integrated: pd.DataFrame):
    rows, coefficient_rows = [], []
    for scope in ("environment-adjusted", "seasonal-adjusted"):
        scope_start = len(rows)
        for response in RESPONSES:
            for metric in THERMAL_METRICS:
                for lag in (0, 1):
                    frame = model_frame(integrated, response, metric, lag)
                    predictors, names = predictors_for(frame, scope)
                    fit = fit_ar1(frame[response].to_numpy(float), predictors, frame["time_index"].to_numpy(float))
                    permuted_beta = []
                    for shift in range(1, 48):
                        shifted = shifted_thermal_for_dates(integrated, metric, lag, shift, frame["date"])
                        perm_predictors, _ = predictors_for(frame, scope, shifted)
                        permuted_fit = fit_ar1(
                            frame[response].to_numpy(float),
                            perm_predictors,
                            frame["time_index"].to_numpy(float),
                        )
                        permuted_beta.append(abs(permuted_fit.beta[1]))
                    p_cyclic = (1 + sum(v >= abs(fit.beta[1]) - 1e-12 for v in permuted_beta)) / 48
                    model_id = f"{scope}|{response}|{metric}|lag{lag}"
                    rows.append(
                        {
                            "model_id": model_id,
                            "scope": scope,
                            "response": response,
                            "thermal_metric": metric,
                            "lag_months": lag,
                            "n": len(frame),
                            "rho_ar1": fit.rho,
                            "r2_marginal": fit.r2,
                            "aic": fit.aic,
                            "beta_standardized": fit.beta[1],
                            "se": fit.se[1],
                            "t": fit.t_value[1],
                            "p": fit.p_value[1],
                            "p_cyclic": p_cyclic,
                            "cyclic_permutations": 47,
                            "ci95_low": fit.ci_low[1],
                            "ci95_high": fit.ci_high[1],
                            "direction": "positive" if fit.beta[1] > 0 else "negative",
                        }
                    )
                    for index, term in enumerate(["intercept", *names]):
                        coefficient_rows.append(
                            {
                                "model_id": model_id,
                                "scope": scope,
                                "response": response,
                                "thermal_metric": metric,
                                "lag_months": lag,
                                "term": term,
                                "beta_standardized": fit.beta[index],
                                "se": fit.se[index],
                                "t": fit.t_value[index],
                                "p": fit.p_value[index],
                                "ci95_low": fit.ci_low[index],
                                "ci95_high": fit.ci_high[index],
                            }
                        )
        scope_end = len(rows)
        adjusted = bh_adjust([r["p_cyclic"] for r in rows[scope_start:scope_end]])
        for row, value in zip(rows[scope_start:scope_end], adjusted):
            row["p_bh_within_scope"] = value
    return pd.DataFrame(rows), pd.DataFrame(coefficient_rows)


def residualize(matrix: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    return matrix - nuisance @ np.linalg.lstsq(nuisance, matrix, rcond=None)[0]


def rda_statistic(y: np.ndarray, thermal: np.ndarray, nuisance: np.ndarray):
    y_residual = residualize(y, nuisance)
    x_residual = residualize(thermal[:, None], nuisance)
    fitted = x_residual @ np.linalg.lstsq(x_residual, y_residual, rcond=None)[0]
    residuals = y_residual - fitted
    effect_ss = float(np.sum(fitted**2))
    residual_ss = float(np.sum(residuals**2))
    rank = int(np.linalg.matrix_rank(np.column_stack([nuisance, thermal])))
    partial_r2 = effect_ss / (effect_ss + residual_ss)
    pseudo_f = effect_ss / (residual_ss / (len(y) - rank))
    return partial_r2, pseudo_f, rank


def partial_rda(integrated: pd.DataFrame, density: pd.DataFrame, bio_dates: pd.Series):
    relative = density.div(density.sum(axis=1), axis=0)
    hellinger_by_date = pd.DataFrame(np.sqrt(relative.to_numpy(float)), index=pd.to_datetime(bio_dates), columns=density.columns)
    rows = []
    for scope in ("environment-adjusted", "seasonal-adjusted"):
        start = len(rows)
        for metric in THERMAL_METRICS:
            for lag in (0, 1):
                frame = model_frame(integrated, "density_org_m3", metric, lag)
                y = hellinger_by_date.loc[frame["date"]].to_numpy(float)
                nuisance_columns = ["time_index", "sin_month", "cos_month"]
                if scope == "environment-adjusted":
                    nuisance_columns.extend(ENVIRONMENTAL_COVARIATES)
                nuisance = np.column_stack([np.ones(len(frame))] + [zscore(frame[c]) for c in nuisance_columns])
                thermal = zscore(frame["thermal"])
                partial_r2, pseudo_f, rank = rda_statistic(y, thermal, nuisance)
                permuted = []
                for shift in range(1, 48):
                    shifted = shifted_thermal_for_dates(integrated, metric, lag, shift, frame["date"])
                    permuted.append(rda_statistic(y, zscore(shifted), nuisance)[1])
                p_cyclic = (1 + sum(value >= pseudo_f - 1e-12 for value in permuted)) / 48
                standardized_design = np.column_stack([zscore(frame[c]) for c in ["thermal", *nuisance_columns]])
                correlation = np.corrcoef(standardized_design, rowvar=False)
                vif = np.diag(np.linalg.inv(correlation))
                condition_number = np.linalg.cond(standardized_design)
                rows.append(
                    {
                        "scope": scope,
                        "thermal_metric": metric,
                        "lag_months": lag,
                        "n": len(frame),
                        "taxa": y.shape[1],
                        "partial_r2": partial_r2,
                        "pseudo_f": pseudo_f,
                        "df_effect": 1,
                        "df_residual": len(frame) - rank,
                        "p_cyclic": p_cyclic,
                        "cyclic_permutations": 47,
                        "design_rank": rank,
                        "design_columns": nuisance.shape[1] + 1,
                        "max_vif": float(vif.max()),
                        "condition_number": float(condition_number),
                    }
                )
        adjusted = bh_adjust([r["p_cyclic"] for r in rows[start:]])
        for row, value in zip(rows[start:], adjusted):
            row["p_bh_within_scope"] = value
    return pd.DataFrame(rows)


def taxon_ar1(integrated: pd.DataFrame, density: pd.DataFrame, bio_dates: pd.Series):
    density_by_date = density.copy()
    density_by_date.index = pd.to_datetime(bio_dates)
    hellinger_by_date = np.sqrt(density_by_date.div(density_by_date.sum(axis=1), axis=0))
    prevalence = (density_by_date > 0).sum(axis=0)
    eligible = prevalence[prevalence / len(density_by_date) >= 0.25].index
    rows = []
    for taxon in eligible:
        integrated_taxon = integrated.copy()
        integrated_taxon["taxon_response"] = integrated_taxon["date"].map(hellinger_by_date[taxon])
        for lag in (0, 1):
            frame = model_frame(integrated_taxon, "taxon_response", "mhw_days", lag)
            predictors, _ = predictors_for(frame, "environment-adjusted")
            fit = fit_ar1(frame["taxon_response"], predictors, frame["time_index"])
            rows.append(
                {
                    "taxon": taxon,
                    "prevalence_months": int(prevalence[taxon]),
                    "prevalence_pct": 100 * prevalence[taxon] / len(density_by_date),
                    "lag_months": lag,
                    "n": len(frame),
                    "rho_ar1": fit.rho,
                    "beta_standardized": fit.beta[1],
                    "se": fit.se[1],
                    "p": fit.p_value[1],
                    "ci95_low": fit.ci_low[1],
                    "ci95_high": fit.ci_high[1],
                    "direction": "positive" if fit.beta[1] > 0 else "negative",
                }
            )
    out = pd.DataFrame(rows)
    out["p_bh"] = bh_adjust(out["p"])
    return out.sort_values("p").reset_index(drop=True)


def pcoa_scores(density: pd.DataFrame, bio_dates: pd.Series, integrated: pd.DataFrame):
    relative = density.div(density.sum(axis=1), axis=0)
    hellinger = np.sqrt(relative.to_numpy(float))
    distances = squareform(pdist(hellinger, metric="braycurtis"))
    n = len(distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    matrix = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 0
    explained = 100 * eigenvalues[positive] / eigenvalues[positive].sum()
    scores = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    # Eigenvector signs are arbitrary; orient axes to the archived result convention.
    if scores[0, 0] > 0:
        scores[:, 0] *= -1
    if scores[0, 1] > 0:
        scores[:, 1] *= -1
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(bio_dates),
            "PCoA1": scores[:, 0],
            "PCoA2": scores[:, 1],
            "PCoA1_explained_pct": explained[0],
            "PCoA2_explained_pct": explained[1],
        }
    )
    return result.merge(integrated[["date", "phase_2024", "mhw_days", "cop_anomaly_mean_c", "year", "month"]], on="date")


def sensitivity_analysis(integrated: pd.DataFrame, density: pd.DataFrame, bio_dates: pd.Series):
    relative = density.div(density.sum(axis=1), axis=0)
    hellinger = pd.DataFrame(np.sqrt(relative.to_numpy(float)), index=pd.to_datetime(bio_dates), columns=density.columns)
    rows = []
    for label, excluded_date in (("full", None), ("excluding 2021-02 density outlier", pd.Timestamp("2021-02-01"))):
        for lag in (0, 1):
            frame = model_frame(integrated, "log1p_density", "mhw_days", lag)
            if excluded_date is not None:
                frame = frame[frame["date"] != excluded_date].copy()
            predictors, _ = predictors_for(frame, "environment-adjusted")
            fit = fit_ar1(frame["log1p_density"], predictors, frame["time_index"])
            diagnostic = diagnostic_statistics(frame["log1p_density"], predictors, frame["time_index"], fit)
            rows.append({
                "analysis": "univariate density", "sensitivity_set": label, "lag_months": lag, "n": len(frame),
                "beta_standardized": fit.beta[1], "p": fit.p_value[1], "rho_ar1": fit.rho,
                "ljung_box_p12": diagnostic["ljung_box_p12"],
                "shapiro_wilk_p": diagnostic["shapiro_wilk_p"],
                "breusch_pagan_p": diagnostic["breusch_pagan_p"],
                "max_cooks_distance": diagnostic["max_cooks_distance"],
                "partial_r2": np.nan, "pseudo_f": np.nan, "design_rank": np.nan,
                "design_columns": np.nan, "max_vif": np.nan, "condition_number": np.nan,
                "cyclic_permutations": np.nan,
            })

            y = hellinger.loc[frame["date"]].to_numpy(float)
            nuisance_columns = ["time_index", "sin_month", "cos_month", *ENVIRONMENTAL_COVARIATES]
            nuisance = np.column_stack([np.ones(len(frame))] + [zscore(frame[c]) for c in nuisance_columns])
            thermal = zscore(frame["thermal"])
            partial_r2, pseudo_f, rank = rda_statistic(y, thermal, nuisance)
            permuted = []
            for shift in range(1, 48):
                shifted = shifted_thermal_for_dates(integrated, "mhw_days", lag, shift, frame["date"])
                permuted.append(rda_statistic(y, zscore(shifted), nuisance)[1])
            p_cyclic = (1 + sum(v >= pseudo_f - 1e-12 for v in permuted)) / 48
            design = np.column_stack([zscore(frame[c]) for c in ["thermal", *nuisance_columns]])
            vif = np.diag(np.linalg.inv(np.corrcoef(design, rowvar=False)))
            rows.append({
                "analysis": "community partial RDA", "sensitivity_set": label, "lag_months": lag, "n": len(frame),
                "beta_standardized": np.nan, "p": p_cyclic, "rho_ar1": np.nan,
                "ljung_box_p12": np.nan, "shapiro_wilk_p": np.nan,
                "breusch_pagan_p": np.nan, "max_cooks_distance": np.nan,
                "partial_r2": partial_r2, "pseudo_f": pseudo_f, "design_rank": rank,
                "design_columns": nuisance.shape[1] + 1, "max_vif": float(vif.max()),
                "condition_number": float(np.linalg.cond(design)), "cyclic_permutations": 47,
            })
    order = {"univariate density": 0, "community partial RDA": 1}
    set_order = {"full": 0, "excluding 2021-02 density outlier": 1}
    result = pd.DataFrame(rows)
    result["_set"] = result["sensitivity_set"].map(set_order)
    result["_analysis"] = result["analysis"].map(order)
    return result.sort_values(["_set", "_analysis", "lag_months"]).drop(columns=["_set", "_analysis"]).reset_index(drop=True)


def summaries(integrated: pd.DataFrame, density: pd.DataFrame, bio_dates: pd.Series):
    observed = integrated.dropna(subset=["density_org_m3"]).copy()
    annual = observed.groupby(observed["date"].dt.year).agg(
        n_samples=("density_org_m3", "size"),
        density_median_org_m3=("density_org_m3", "median"),
        density_mean_org_m3=("density_org_m3", "mean"),
        richness_median=("taxon_richness", "median"),
        richness_mean=("taxon_richness", "mean"),
        shannon_median=("shannon", "median"),
        shannon_mean=("shannon", "mean"),
        pielou_median=("pielou", "median"),
        simpson_median=("simpson_1d", "median"),
        simpson_mean=("simpson_1d", "mean"),
        mhw_days_total=("mhw_days", "sum"),
        anomaly_mean_c=("cop_anomaly_mean_c", "mean"),
        dhw_max_c_weeks=("noaa_dhw_max_c_weeks", "max"),
    ).reset_index(names="year")
    phase_rows = []
    for phase in PHASES:
        group = observed[observed["phase_2024"] == phase]
        phase_rows.append(
            {
                "phase": phase,
                "n": len(group),
                "density_median_org_m3": group["density_org_m3"].median(),
                "richness_median": group["taxon_richness"].median(),
                "shannon_median": group["shannon"].median(),
                "pielou_median": group["pielou"].median(),
                "simpson_median": group["simpson_1d"].median(),
                "mhw_days_total": group["mhw_days"].sum(),
                "anomaly_mean_c": group["cop_anomaly_mean_c"].mean(),
                "dhw_max_c_weeks": group["noaa_dhw_max_c_weeks"].max(),
            }
        )
    return annual, pd.DataFrame(phase_rows)


def validate_temperature(integrated: pd.DataFrame):
    matched = integrated.dropna(subset=["local_temperature_c", "cop_sst_mean_c"])
    rows = []
    for label, frame in (
        ("all matched months", matched),
        ("excluding January 2024", matched[matched["date"] != pd.Timestamp("2024-01-01")]),
    ):
        r, p = pearsonr(frame["local_temperature_c"], frame["cop_sst_mean_c"])
        rows.append({"comparison": "Local temperature vs Copernicus SST", "set": label, "n": len(frame), "pearson_r": r, "p": p})
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, output_dir: Path, name: str):
    frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")


def run(package_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    environment, counts, metadata, daily_mhw, daily_noaa = read_inputs(package_root)
    thermal = monthly_thermal(daily_mhw, daily_noaa)
    bio, density, taxa = biological_data(counts, metadata)
    integrated = integrated_table(environment, bio, thermal)
    integrated["year"] = integrated["date"].dt.year
    integrated["month"] = integrated["date"].dt.month

    community, coefficients = community_ar1(integrated)
    rda = partial_rda(integrated, density, bio["date"])
    taxon = taxon_ar1(integrated, density, bio["date"])
    pcoa = pcoa_scores(density, bio["date"], integrated)
    annual, phase = summaries(integrated, density, bio["date"])
    temperature = validate_temperature(integrated)
    sensitivity = sensitivity_analysis(integrated, density, bio["date"])
    community_diagnostics, taxon_diagnostics = diagnostics(integrated, density, bio["date"])

    outputs = {
        "Integrated_Monthly": integrated,
        "Species_Density": pd.concat([bio[["date"]].reset_index(drop=True), density.reset_index(drop=True)], axis=1),
        "Annual_Summary": annual,
        "Phase_Summary": phase,
        "Community_AR1": community,
        "AR1_Coefficients": coefficients,
        "Partial_RDA": rda,
        "Taxon_AR1": taxon,
        "PCoA_Scores": pcoa,
        "Temp_Validation": temperature,
        "Sensitivity": sensitivity,
        "Community_Diagnostics": community_diagnostics,
        "Taxon_Diagnostics": taxon_diagnostics,
    }
    for name, frame in outputs.items():
        write_csv(frame, output_dir, name)

    metadata_out = {
        "biological_campaigns": int(len(bio)),
        "taxa": int(len(taxa)),
        "calendar_months": int(len(integrated)),
        "cyclic_shifts": 47,
        "software": {"python": "3.12+", "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    print(json.dumps(metadata_out, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced_results"))
    args = parser.parse_args()
    run(args.package_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
