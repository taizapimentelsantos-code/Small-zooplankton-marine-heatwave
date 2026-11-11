#!/usr/bin/env python3
"""Rebuild MHW diagnostics using the conventional Hobday gap-joining rule.

Events are first defined as runs of at least five days above the seasonally
varying 90th-percentile threshold. Consecutive qualifying events separated by
at most two days are then joined. The intervening gap days are included in the
joined event duration, as in the conventional marineHeatWaves implementation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


MIN_DURATION = 5
MAX_GAP = 2


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def detect_events(daily: pd.DataFrame) -> list[tuple[int, int]]:
    initial = [
        (start, end)
        for start, end in contiguous_runs(daily["above_p90"].to_numpy(bool))
        if end - start + 1 >= MIN_DURATION
    ]
    if not initial:
        return []
    joined = [initial[0]]
    for start, end in initial[1:]:
        previous_start, previous_end = joined[-1]
        gap = start - previous_end - 1
        if gap <= MAX_GAP:
            joined[-1] = (previous_start, end)
        else:
            joined.append((start, end))
    return joined


def category_label(code: int) -> str:
    return {
        0: "No MHW",
        1: "I - Moderate",
        2: "II - Strong",
        3: "III - Severe",
        4: "IV - Extreme",
    }[min(max(int(code), 0), 4)]


def rebuild_daily_and_events(daily: pd.DataFrame, noaa: pd.DataFrame):
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    noaa = noaa.copy()
    noaa["date"] = pd.to_datetime(noaa["date"])
    daily["event_id"] = pd.NA
    events = []

    for number, (start, end) in enumerate(detect_events(daily), start=1):
        event_id = f"MHW-{number:02d}"
        daily.loc[start:end, "event_id"] = event_id
        block = daily.loc[start:end]
        peak_index = block["intensity_above_climatology_c"].idxmax()
        noaa_block = noaa[noaa["date"].between(block["date"].iloc[0], block["date"].iloc[-1])]
        max_ratio = float(block["category_ratio"].max())
        max_code = min(max(int(np.floor(max_ratio)) + 1, 1), 4)
        events.append(
            {
                "event_id": event_id,
                "start_date": block["date"].iloc[0],
                "end_date": block["date"].iloc[-1],
                "duration_days": len(block),
                "peak_date": daily.loc[peak_index, "date"],
                "mean_intensity_c": block["intensity_above_climatology_c"].mean(),
                "maximum_intensity_c": block["intensity_above_climatology_c"].max(),
                "cumulative_intensity_c_days": block["intensity_above_climatology_c"].sum(),
                "maximum_p90_exceedance_c": block["p90_exceedance_c"].max(),
                "maximum_category_code": max_code,
                "maximum_category": category_label(max_code),
                "noaa_hotspot_max_c": noaa_block["hotspot_c"].max(),
                "noaa_dhw_max_c_weeks": noaa_block["dhw_c_weeks"].max(),
                "legacy_baa_code": int(noaa_block["legacy_7d_baa_code"].max()),
                "legacy_baa_label": noaa_block.loc[noaa_block["legacy_7d_baa_code"].idxmax(), "legacy_baa_label"],
                "current_baa_code": int(noaa_block["current_7d_baa_code"].max()),
                "current_baa_label": noaa_block.loc[noaa_block["current_7d_baa_code"].idxmax(), "current_7d_baa_label"],
            }
        )
    return daily, pd.DataFrame(events)


def summaries(daily: pd.DataFrame, events: pd.DataFrame, noaa: pd.DataFrame):
    daily = daily.copy()
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["is_mhw_day"] = daily["event_id"].notna()
    daily["event_intensity"] = np.where(
        daily["is_mhw_day"], daily["intensity_above_climatology_c"], 0.0
    )
    noaa = noaa.copy()
    noaa["year"] = noaa["date"].dt.year
    noaa["month"] = noaa["date"].dt.month

    monthly = daily.groupby(["year", "month"]).agg(
        copernicus_sst_mean_c=("sst_c", "mean"),
        copernicus_sst_max_c=("sst_c", "max"),
        sst_anomaly_mean_c=("intensity_above_climatology_c", "mean"),
        sst_anomaly_max_c=("intensity_above_climatology_c", "max"),
        mhw_days=("is_mhw_day", "sum"),
        mhw_events_present=("event_id", lambda x: x.dropna().nunique()),
        cumulative_intensity_c_days=("event_intensity", "sum"),
    ).reset_index()
    monthly["mhw_days"] = monthly["mhw_days"].astype(int)
    nmonthly = noaa.groupby(["year", "month"]).agg(
        noaa_sst_p90_mean_c=("sst_at_p90_hotspot_pixel_c", "mean"),
        noaa_hotspot_max_c=("hotspot_c", "max"),
        noaa_dhw_max_c_weeks=("dhw_c_weeks", "max"),
        days_hotspot_ge_1_c=("hotspot_c", lambda x: int((x >= 1).sum())),
        legacy_days_warning_or_higher=("legacy_7d_baa_code", lambda x: int((x >= 2).sum())),
        legacy_baa_max_code=("legacy_7d_baa_code", "max"),
        current_days_alert1_or_higher=("current_7d_baa_code", lambda x: int((x >= 3).sum())),
        current_baa_max_code=("current_7d_baa_code", "max"),
    ).reset_index()
    labels = noaa.sort_values("current_7d_baa_code").groupby(["year", "month"]).tail(1)
    labels = labels[["year", "month", "current_7d_baa_label"]].rename(
        columns={"current_7d_baa_label": "current_maximum_alert_level"}
    )
    monthly = monthly.merge(nmonthly, on=["year", "month"]).merge(labels, on=["year", "month"])

    annual = daily.groupby("year").agg(
        copernicus_sst_mean_c=("sst_c", "mean"),
        copernicus_sst_max_c=("sst_c", "max"),
        sst_anomaly_mean_c=("intensity_above_climatology_c", "mean"),
        sst_anomaly_max_c=("intensity_above_climatology_c", "max"),
        mhw_days=("is_mhw_day", "sum"),
        cumulative_intensity_c_days=("event_intensity", "sum"),
    ).reset_index()
    annual["mhw_days"] = annual["mhw_days"].astype(int)
    annual["mhw_events"] = annual["year"].map(
        lambda year: int(((events["start_date"].dt.year <= year) & (events["end_date"].dt.year >= year)).sum())
    )
    nannual = noaa.groupby("year").agg(
        noaa_sst_p90_mean_c=("sst_at_p90_hotspot_pixel_c", "mean"),
        noaa_hotspot_max_c=("hotspot_c", "max"),
        noaa_dhw_max_c_weeks=("dhw_c_weeks", "max"),
        days_hotspot_ge_1_c=("hotspot_c", lambda x: int((x >= 1).sum())),
        legacy_days_warning_or_higher=("legacy_7d_baa_code", lambda x: int((x >= 2).sum())),
        legacy_days_alert1_or_higher=("legacy_7d_baa_code", lambda x: int((x >= 3).sum())),
        legacy_baa_max_code=("legacy_7d_baa_code", "max"),
        current_days_alert1_or_higher=("current_7d_baa_code", lambda x: int((x >= 3).sum())),
        current_baa_max_code=("current_7d_baa_code", "max"),
    ).reset_index()
    alabels = noaa.sort_values("current_7d_baa_code").groupby("year").tail(1)
    alabels = alabels[["year", "current_7d_baa_label"]].rename(
        columns={"current_7d_baa_label": "current_maximum_alert_level"}
    )
    annual = annual.merge(nannual, on="year").merge(alabels, on="year")
    order = [
        "year", "copernicus_sst_mean_c", "copernicus_sst_max_c", "sst_anomaly_mean_c",
        "sst_anomaly_max_c", "mhw_days", "mhw_events", "cumulative_intensity_c_days",
        "noaa_sst_p90_mean_c", "noaa_hotspot_max_c", "noaa_dhw_max_c_weeks",
        "days_hotspot_ge_1_c", "legacy_days_warning_or_higher", "legacy_days_alert1_or_higher",
        "legacy_baa_max_code", "current_days_alert1_or_higher", "current_baa_max_code",
        "current_maximum_alert_level",
    ]
    return annual[order], monthly


def replace_sheet_data(workbook_path: Path, sheet: str, frame: pd.DataFrame):
    wb = load_workbook(workbook_path)
    ws = wb[sheet]
    max_rows = max(ws.max_row, len(frame) + 1)
    max_cols = max(ws.max_column, len(frame.columns))
    for row in ws.iter_rows(min_row=1, max_row=max_rows, min_col=1, max_col=max_cols):
        for cell in row:
            cell.value = None
    for column, value in enumerate(frame.columns, start=1):
        ws.cell(1, column, value)
    for row_index, record in enumerate(frame.itertuples(index=False, name=None), start=2):
        for column_index, value in enumerate(record, start=1):
            if pd.isna(value):
                value = None
            elif isinstance(value, pd.Timestamp):
                value = value.to_pydatetime()
            elif isinstance(value, np.generic):
                value = value.item()
            ws.cell(row_index, column_index, value)
    wb.save(workbook_path)


def update_notes(workbook_path: Path):
    wb = load_workbook(workbook_path)
    replacements = {
        "Excursions below the threshold interrupted the event; short gaps were not bridged.":
            "Qualifying events separated by gaps of up to two days were joined (joinAcrossGaps = TRUE; maxGap = 2).",
        "Short gaps below the threshold were not bridged.":
            "Qualifying events separated by gaps of up to two days were joined (joinAcrossGaps = TRUE; maxGap = 2).",
    }
    for ws in (wb["README"], wb["Methods_Sources"]):
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    for old, new in replacements.items():
                        if old in cell.value:
                            cell.value = cell.value.replace(old, new)
    wb.save(workbook_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, args.output)

    daily = pd.read_excel(args.input, sheet_name="Daily_MHW_2021_2024")
    noaa = pd.read_excel(args.input, sheet_name="Daily_NOAA_2021_2024")
    daily["date"] = pd.to_datetime(daily["date"])
    noaa["date"] = pd.to_datetime(noaa["date"])
    revised_daily, events = rebuild_daily_and_events(daily, noaa)
    annual, monthly = summaries(revised_daily, events, noaa)
    replace_sheet_data(args.output, "Daily_MHW_2021_2024", revised_daily.drop(columns=["is_mhw_day", "event_intensity"], errors="ignore"))
    replace_sheet_data(args.output, "MHW_Events", events)
    replace_sheet_data(args.output, "Annual_Summary", annual)
    replace_sheet_data(args.output, "Monthly_Summary", monthly)
    update_notes(args.output)
    print(annual[["year", "mhw_days", "mhw_events", "cumulative_intensity_c_days"]].to_string(index=False))
    print(events[["event_id", "start_date", "end_date", "duration_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
