from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends.file_manager import FILE_CACHE

try:
    from .process_merra2 import DEFAULT_OUTPUT_PATH as DEFAULT_MERRA_PATH, process_merra2
    from .process_sentinel import (
        DEFAULT_OUTPUT_PATH as DEFAULT_SENTINEL_PATH,
        DEFAULT_TARGET_VARIABLE,
        process_sentinel,
    )
except ImportError:
    from process_merra2 import DEFAULT_OUTPUT_PATH as DEFAULT_MERRA_PATH, process_merra2
    from process_sentinel import (
        DEFAULT_OUTPUT_PATH as DEFAULT_SENTINEL_PATH,
        DEFAULT_TARGET_VARIABLE,
        process_sentinel,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Dataset" / "processed"
DEFAULT_FUSED_PATH = DEFAULT_OUTPUT_DIR / "fused_5D.nc"
DEFAULT_TRAIN_PATH = DEFAULT_OUTPUT_DIR / "train_5D.nc"
DEFAULT_TEST_PATH = DEFAULT_OUTPUT_DIR / "test_5D.nc"
DEFAULT_AEROSOL_SPECIES = ["BCCMASS", "DUCMASS", "OCCMASS", "SO4CMASS", "SSCMASS"]


def _load_json_attr(dataset: xr.Dataset, name: str) -> list[str]:
    value = dataset.attrs.get(name, "[]")
    return list(json.loads(value))


def _compute_dominant_aerosol(dataset: xr.Dataset, aerosol_species: list[str]) -> xr.DataArray:
    available_species = [species for species in aerosol_species if species in dataset.data_vars]
    if not available_species:
        raise ValueError("No aerosol species variables were found for classification target creation.")

    stacked = xr.concat([dataset[species] for species in available_species], dim="species")
    stacked = stacked.assign_coords(species=available_species)
    dominant = stacked.argmax(dim="species").astype(np.int32)
    dominant.name = "DOMINANT_AEROSOL_TYPE"
    dominant.attrs["class_names"] = json.dumps(available_species)
    return dominant


def _split_train_test(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    train_mask = np.asarray(times <= pd.Timestamp("2022-12-31 23:59:59"))
    test_mask = np.asarray(times.year == 2023)
    split_summary: dict[str, int | str] = {
        "requested_train_end": "2022-12-31",
        "requested_test_year": 2023,
    }

    if test_mask.sum() == 0:
        fallback_test_year = int(times.year.max())
        test_mask = np.asarray(times.year == fallback_test_year)
        train_mask = np.asarray(times.year < fallback_test_year)
        split_summary.update(
            {
                "fallback_test_year": fallback_test_year,
                "split_note": "No overlapping 2023 data was available; using the latest available year for testing.",
            }
        )

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Unable to create non-empty train/test splits from the fused timeline.")

    split_summary.update(
        {
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
        }
    )
    return train_mask, test_mask, split_summary


def _safe_minmax_scale(dataset: xr.Dataset, variables: list[str], train_mask: np.ndarray) -> tuple[xr.Dataset, dict[str, dict[str, float]]]:
    scaling_stats: dict[str, dict[str, float]] = {}
    scaled = dataset.copy()
    train_subset = dataset.isel(time=np.where(train_mask)[0])

    for variable in variables:
        train_data = train_subset[variable]
        min_value = float(train_data.min().compute().item())
        max_value = float(train_data.max().compute().item())
        scale = max(max_value - min_value, 1e-8)
        scaled[variable] = ((dataset[variable] - min_value) / scale).clip(0.0, 1.0)
        scaling_stats[variable] = {"min": min_value, "max": max_value}

    return scaled, scaling_stats


def _assert_dataset_validity(dataset: xr.Dataset, feature_variables: list[str], target_variable: str) -> None:
    expected_shape = (
        dataset.sizes.get("time", 0),
        1,
        dataset.sizes.get("lat", 0),
        dataset.sizes.get("lon", 0),
        len(feature_variables),
    )
    assert expected_shape[2] == 291 and expected_shape[3] == 512, (
        "Expected a 291x512 grid after fusion, "
        f"but received {expected_shape[2]}x{expected_shape[3]}."
    )

    for variable in feature_variables + [target_variable, "DOMINANT_AEROSOL_TYPE"]:
        if variable not in dataset.data_vars:
            raise AssertionError(f"Required variable '{variable}' is missing from the fused dataset.")
        contains_nan = bool(dataset[variable].isnull().any().compute().item())
        assert not contains_nan, f"Variable '{variable}' contains NaN values after fusion."


def _write_dataset_safely(dataset: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    # Clear xarray's shared file cache so previously opened NetCDF files do not
    # keep Windows file handles alive across notebook cells.
    FILE_CACHE.clear()
    gc.collect()

    if temp_path.exists():
        temp_path.unlink()

    dataset.to_netcdf(temp_path)

    if output_path.exists():
        output_path.unlink()

    temp_path.replace(output_path)


def fuse_datasets(
    sentinel_path: Path | str = DEFAULT_SENTINEL_PATH,
    merra_path: Path | str = DEFAULT_MERRA_PATH,
    fused_output_path: Path | str = DEFAULT_FUSED_PATH,
    train_output_path: Path | str = DEFAULT_TRAIN_PATH,
    test_output_path: Path | str = DEFAULT_TEST_PATH,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    aerosol_species: list[str] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    aerosol_species = aerosol_species or DEFAULT_AEROSOL_SPECIES
    sentinel_path = Path(sentinel_path)
    merra_path = Path(merra_path)
    fused_output_path = Path(fused_output_path)
    train_output_path = Path(train_output_path)
    test_output_path = Path(test_output_path)

    sentinel = xr.open_dataset(sentinel_path).sortby("time")
    merra = xr.open_dataset(merra_path).sortby("time")

    sentinel_feature_variables = _load_json_attr(sentinel, "feature_variables")
    if not sentinel_feature_variables:
        sentinel_feature_variables = [name for name in sentinel.data_vars if name != target_variable]

    merra_feature_variables = list(merra.data_vars)
    merged = xr.merge(
        [
            sentinel[sentinel_feature_variables + [target_variable]],
            merra[merra_feature_variables],
        ],
        join="inner",
        compat="override",
    ).sortby("time")

    if merged.sizes.get("time", 0) == 0:
        raise ValueError("The fused dataset is empty after temporal alignment.")

    dominant_aerosol = _compute_dominant_aerosol(merged, aerosol_species)
    merged = xr.merge([merged, dominant_aerosol], compat="override")
    merged = merged.fillna(0)

    times = pd.DatetimeIndex(pd.to_datetime(merged["time"].values))
    train_mask, test_mask, split_summary = _split_train_test(times)

    feature_variables = sentinel_feature_variables + [
        variable for variable in merra_feature_variables if variable not in sentinel_feature_variables
    ]
    scaled, scaling_stats = _safe_minmax_scale(merged, feature_variables + [target_variable], train_mask)

    scaled.attrs.update(
        {
            "feature_variables": json.dumps(feature_variables),
            "target_variable": target_variable,
            "class_names": dominant_aerosol.attrs.get("class_names", json.dumps(aerosol_species)),
            "num_classes": len(json.loads(dominant_aerosol.attrs.get("class_names", json.dumps(aerosol_species)))),
            "scaling_stats": json.dumps(scaling_stats),
            "split_summary": json.dumps(split_summary),
        }
    )

    train_dataset = scaled.isel(time=np.where(train_mask)[0])
    test_dataset = scaled.isel(time=np.where(test_mask)[0])

    _assert_dataset_validity(train_dataset, feature_variables, target_variable)
    _assert_dataset_validity(test_dataset, feature_variables, target_variable)

    _write_dataset_safely(scaled, fused_output_path)
    _write_dataset_safely(train_dataset, train_output_path)
    _write_dataset_safely(test_dataset, test_output_path)
    return scaled, train_dataset, test_dataset


def run_full_fusion_pipeline(
    sentinel_source_path: Path | str | None = None,
    merra_source_dir: Path | str | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    sentinel_source_path = Path(sentinel_source_path) if sentinel_source_path else None
    merra_source_dir = Path(merra_source_dir) if merra_source_dir else None

    sentinel_output_path = Path(DEFAULT_SENTINEL_PATH)
    merra_output_path = Path(DEFAULT_MERRA_PATH)

    process_sentinel(
        source_path=sentinel_source_path or ROOT_DIR / "Dataset" / "S5PL2_5D.nc",
        output_path=sentinel_output_path,
    )
    process_merra2(
        merra_dir=merra_source_dir or ROOT_DIR / "Dataset" / "merra-2-till-DEC2025",
        sentinel_reference_path=sentinel_output_path,
        output_path=merra_output_path,
    )

    return fuse_datasets(
        sentinel_path=sentinel_output_path,
        merra_path=merra_output_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse processed Sentinel and MERRA-2 datasets into train/test NetCDF files.")
    parser.add_argument("--sentinel", type=Path, default=DEFAULT_SENTINEL_PATH)
    parser.add_argument("--merra", type=Path, default=DEFAULT_MERRA_PATH)
    parser.add_argument("--fused-output", type=Path, default=DEFAULT_FUSED_PATH)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--target-variable", default=DEFAULT_TARGET_VARIABLE)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    fused, train_dataset, test_dataset = fuse_datasets(
        sentinel_path=args.sentinel,
        merra_path=args.merra,
        fused_output_path=args.fused_output,
        train_output_path=args.train_output,
        test_output_path=args.test_output,
        target_variable=args.target_variable,
    )
    print(
        f"Fused samples={fused.sizes.get('time', 0)}, "
        f"train={train_dataset.sizes.get('time', 0)}, test={test_dataset.sizes.get('time', 0)}"
    )
    print(f"Saved fused dataset to {args.fused_output}")


if __name__ == "__main__":
    main()
