from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends.file_manager import FILE_CACHE


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MERRA_DIR = ROOT_DIR / "Dataset" / "merra-2-till-DEC2025"
DEFAULT_SENTINEL_REFERENCE = ROOT_DIR / "Dataset" / "processed" / "sentinel_5d.nc"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "Dataset" / "processed" / "merra2_5d_aligned.nc"
DATE_PATTERN = re.compile(r"(\d{8})")


def _extract_file_date(file_path: Path) -> str | None:
    match = DATE_PATTERN.search(file_path.name)
    return match.group(1) if match else None


def _deduplicate_merra_files(files: list[Path]) -> list[Path]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        date_token = _extract_file_date(file_path)
        if date_token is not None:
            grouped[date_token].append(file_path)

    deduplicated: list[Path] = []
    for date_token in sorted(grouped):
        candidates = sorted(
            grouped[date_token],
            key=lambda path: (
                "(" in path.stem,
                len(path.name),
                path.name,
            ),
        )
        deduplicated.append(candidates[0])
    return deduplicated


def _discover_spatial_variables(dataset: xr.Dataset) -> list[str]:
    variables: list[str] = []
    for name, data_array in dataset.data_vars.items():
        dims = set(data_array.dims)
        if {"lat", "lon"}.issubset(dims) or {"time", "lat", "lon"}.issubset(dims):
            variables.append(name)
    if not variables:
        raise ValueError("No MERRA-2 variables with spatial dimensions were found.")
    return variables


def _preprocess_merra_dataset(dataset: xr.Dataset) -> xr.Dataset:
    variables = _discover_spatial_variables(dataset)
    dataset = dataset[variables]

    if "time" in dataset.dims and dataset.sizes.get("time", 0) > 0:
        timestamp = pd.to_datetime(dataset["time"].values[0]).normalize() + pd.Timedelta(hours=12)
        dataset = dataset.mean(dim="time", keep_attrs=True).expand_dims(time=[timestamp])
    elif "time" not in dataset.coords:
        raise ValueError("MERRA-2 input file is missing a time coordinate.")

    return dataset


def _align_merra_to_reference_times(
    merra_daily: xr.Dataset,
    reference_times: pd.DatetimeIndex,
    window_days: int = 5,
) -> xr.Dataset:
    half_window = pd.Timedelta(days=window_days // 2)
    aligned_slices: list[xr.Dataset] = []

    for timestamp in reference_times:
        window = merra_daily.sel(time=slice(timestamp - half_window, timestamp + half_window))
        if window.sizes.get("time", 0) == 0:
            continue
        averaged = window.mean(dim="time", keep_attrs=True).expand_dims(time=[timestamp])
        aligned_slices.append(averaged)

    if not aligned_slices:
        raise ValueError("No overlapping MERRA-2 windows were found for the Sentinel reference timeline.")

    aligned = xr.concat(aligned_slices, dim="time")
    return aligned.sortby("time")


def process_merra2(
    merra_dir: Path | str = DEFAULT_MERRA_DIR,
    sentinel_reference_path: Path | str = DEFAULT_SENTINEL_REFERENCE,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    window_days: int = 5,
) -> xr.Dataset:
    merra_dir = Path(merra_dir)
    sentinel_reference_path = Path(sentinel_reference_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merra_files = _deduplicate_merra_files(sorted(merra_dir.glob("*.nc")))
    if not merra_files:
        raise FileNotFoundError(f"No MERRA-2 NetCDF files were found in {merra_dir}.")

    sentinel_reference = xr.open_dataset(sentinel_reference_path)
    reference_times = pd.DatetimeIndex(pd.to_datetime(sentinel_reference["time"].values))
    target_lats = sentinel_reference["lat"].values
    target_lons = sentinel_reference["lon"].values

    merra_daily = xr.open_mfdataset(
        [str(file_path) for file_path in merra_files],
        combine="nested",
        concat_dim="time",
        preprocess=_preprocess_merra_dataset,
        compat="override",
        coords="minimal",
        data_vars="minimal",
        join="override",
        chunks={"time": 1},
    )
    merra_daily = merra_daily.sortby("time")

    merra_interpolated = merra_daily.interp(
        lat=target_lats,
        lon=target_lons,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    ).fillna(0)

    merra_5d = _align_merra_to_reference_times(merra_interpolated, reference_times, window_days=window_days)
    merra_5d = merra_5d.fillna(0)
    merra_5d.attrs.update(
        {
            "source_directory": str(merra_dir),
            "aligned_to": str(sentinel_reference_path),
            "window_days": window_days,
            "feature_variables": json.dumps(list(merra_5d.data_vars)),
        }
    )
    merra_5d = merra_5d.load()
    sentinel_reference.close()
    merra_daily.close()
    FILE_CACHE.clear()
    gc.collect()
    if output_path.exists():
        output_path.unlink()
    merra_5d.to_netcdf(output_path)
    return merra_5d


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process MERRA-2 files and align them to Sentinel 5-day timestamps.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_MERRA_DIR)
    parser.add_argument("--sentinel-reference", type=Path, default=DEFAULT_SENTINEL_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--window-days", type=int, default=5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    processed = process_merra2(args.source_dir, args.sentinel_reference, args.output, args.window_days)
    print(
        f"Processed MERRA-2 data: time={processed.sizes.get('time', 0)}, "
        f"lat={processed.sizes.get('lat', 0)}, lon={processed.sizes.get('lon', 0)}, "
        f"features={len(processed.data_vars)}"
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
