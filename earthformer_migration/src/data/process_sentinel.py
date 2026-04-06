from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends.file_manager import FILE_CACHE


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SENTINEL_PATH = ROOT_DIR / "Dataset" / "S5PL2_5D.nc"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "Dataset" / "processed" / "sentinel_5d.nc"
DEFAULT_TARGET_VARIABLE = "AER_AI_340_380"


def _discover_spatial_variables(dataset: xr.Dataset) -> list[str]:
    variables: list[str] = []
    for name, data_array in dataset.data_vars.items():
        dims = set(data_array.dims)
        if {"time", "lat", "lon"}.issubset(dims):
            variables.append(name)
    if not variables:
        raise ValueError("No Sentinel variables with time/lat/lon dimensions were found.")
    return variables


def _normalize_time_index(dataset: xr.Dataset) -> xr.Dataset:
    normalized_time = pd.to_datetime(dataset["time"].values).normalize() + pd.Timedelta(hours=12)
    return dataset.assign_coords(time=normalized_time).sortby("time")


def _is_effectively_5_day_series(time_values: np.ndarray) -> bool:
    if len(time_values) < 2:
        return True
    diffs = np.diff(pd.to_datetime(time_values))
    diff_days = np.array([delta / np.timedelta64(1, "D") for delta in diffs], dtype=float)
    return bool(np.all((diff_days >= 4.5) & (diff_days <= 5.5)))


def process_sentinel(
    source_path: Path | str = DEFAULT_SENTINEL_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
) -> xr.Dataset:
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = xr.open_dataset(source_path)
    spatial_variables = _discover_spatial_variables(dataset)
    dataset = _normalize_time_index(dataset[spatial_variables])
    dataset = dataset.fillna(0)

    if _is_effectively_5_day_series(dataset["time"].values):
        sentinel_5d = dataset
    else:
        sentinel_5d = dataset.resample(time="5D").mean(keep_attrs=True)

    feature_variables = [variable for variable in spatial_variables if variable != target_variable]
    if target_variable not in sentinel_5d.data_vars:
        raise ValueError(f"Target variable '{target_variable}' was not found after Sentinel processing.")

    sentinel_5d = sentinel_5d.fillna(0)
    sentinel_5d.attrs.update(
        {
            "source_path": str(source_path),
            "target_variable": target_variable,
            "feature_variables": json.dumps(feature_variables),
            "all_variables": json.dumps(list(sentinel_5d.data_vars)),
            "time_start": str(pd.to_datetime(sentinel_5d.time.values[0])) if sentinel_5d.sizes.get("time", 0) else "",
            "time_end": str(pd.to_datetime(sentinel_5d.time.values[-1])) if sentinel_5d.sizes.get("time", 0) else "",
        }
    )
    sentinel_5d = sentinel_5d.load()
    dataset.close()
    FILE_CACHE.clear()
    gc.collect()
    if output_path.exists():
        output_path.unlink()
    sentinel_5d.to_netcdf(output_path)
    return sentinel_5d


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Sentinel-5P data into a 5-day aligned NetCDF.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SENTINEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-variable", default=DEFAULT_TARGET_VARIABLE)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    processed = process_sentinel(args.source, args.output, args.target_variable)
    print(
        f"Processed Sentinel data: time={processed.sizes.get('time', 0)}, "
        f"lat={processed.sizes.get('lat', 0)}, lon={processed.sizes.get('lon', 0)}"
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
