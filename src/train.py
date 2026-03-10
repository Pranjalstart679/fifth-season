from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import xarray as xr

from model.convlstm import AverageSSIM, combined_regression_loss, compile_mtl_unet_convlstm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "Dataset" / "processed"
DEFAULT_TRAIN_DATASET = DEFAULT_DATA_DIR / "train_5D.nc"
DEFAULT_VAL_DATASET = DEFAULT_DATA_DIR / "test_5D.nc"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts"


class WeatherDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataset_path: Path | str,
        feature_variables: list[str],
        target_variable: str,
        sequence_length: int = 5,
        batch_size: int = 1,
        shuffle: bool = False,
        max_samples: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.dataset = xr.open_dataset(self.dataset_path)
        self.feature_variables = feature_variables
        self.target_variable = target_variable
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.available_indices = np.arange(max(self.dataset.sizes["time"] - self.sequence_length, 0))
        if max_samples is not None:
            self.available_indices = self.available_indices[:max_samples]
        self.target_times = [self.dataset["time"].values[index + self.sequence_length] for index in self.available_indices]
        self.on_epoch_end()

    def __len__(self) -> int:
        if len(self.available_indices) == 0:
            return 0
        return int(np.ceil(len(self.available_indices) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        batch_indices = self.epoch_indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_x: list[np.ndarray] = []
        batch_y_regression: list[np.ndarray] = []
        batch_y_classification: list[np.ndarray] = []

        for start_idx in batch_indices:
            x_window = self.dataset.isel(time=slice(start_idx, start_idx + self.sequence_length))
            y_window = self.dataset.isel(time=start_idx + self.sequence_length)

            stacked_x = np.stack(
                [x_window[feature_name].values.astype(np.float32) for feature_name in self.feature_variables],
                axis=-1,
            )
            regression_target = y_window[self.target_variable].values.astype(np.float32)[np.newaxis, ..., np.newaxis]
            classification_target = y_window["DOMINANT_AEROSOL_TYPE"].values.astype(np.int32)[np.newaxis, ...]

            batch_x.append(stacked_x)
            batch_y_regression.append(regression_target)
            batch_y_classification.append(classification_target)

        return np.asarray(batch_x, dtype=np.float32), {
            "severity_output": np.asarray(batch_y_regression, dtype=np.float32),
            "identity_output": np.asarray(batch_y_classification, dtype=np.int32),
        }

    def on_epoch_end(self) -> None:
        self.epoch_indices = np.copy(self.available_indices)
        if self.shuffle and len(self.epoch_indices) > 0:
            np.random.shuffle(self.epoch_indices)


def _load_dataset_config(dataset_path: Path | str) -> tuple[list[str], str, int]:
    dataset = xr.open_dataset(dataset_path)
    feature_variables = json.loads(dataset.attrs["feature_variables"])
    target_variable = dataset.attrs["target_variable"]
    num_classes = int(dataset.attrs.get("num_classes", 5))
    dataset.close()
    return feature_variables, target_variable, num_classes


def train_model(
    train_dataset_path: Path | str = DEFAULT_TRAIN_DATASET,
    val_dataset_path: Path | str = DEFAULT_VAL_DATASET,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    sequence_length: int = 5,
    batch_size: int = 1,
    epochs: int = 20,
    learning_rate: float = 1e-5,
    max_samples: int | None = None,
    dry_run: bool = False,
    early_stopping_patience: int = 4,
    early_stopping_monitor: str = "val_loss",
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-run timestamped directory ─────────────────────────────────────────
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Record configuration before training starts
    run_config: dict = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "hyperparameters": {
            "epochs": 2 if dry_run else epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "early_stopping_patience": 1 if dry_run else early_stopping_patience,
            "early_stopping_monitor": early_stopping_monitor,
            "max_samples": max_samples,
        },
        "dataset": {
            "train": str(train_dataset_path),
            "val": str(val_dataset_path),
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    feature_variables, target_variable, num_classes = _load_dataset_config(train_dataset_path)

    train_generator = WeatherDataGenerator(
        train_dataset_path,
        feature_variables=feature_variables,
        target_variable=target_variable,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True,
        max_samples=10 if dry_run and max_samples is None else max_samples,
    )
    val_generator = WeatherDataGenerator(
        val_dataset_path,
        feature_variables=feature_variables,
        target_variable=target_variable,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        max_samples=10 if dry_run and max_samples is None else max_samples,
    )

    if len(train_generator) == 0 or len(val_generator) == 0:
        raise ValueError("Training and validation datasets must each contain more than one forecasting window.")

    sample_batch, _ = train_generator[0]
    input_shape = sample_batch.shape[1:]
    model = compile_mtl_unet_convlstm(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_monitor,
            patience=early_stopping_patience if not dry_run else 1,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor=early_stopping_monitor,
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=2 if dry_run else epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(run_dir / "final_model.keras")
    (run_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    # ── Summary with results ──────────────────────────────────────────────────
    monitor_values = history.history.get(early_stopping_monitor, [])
    best_epoch = int(np.argmin(monitor_values)) + 1 if monitor_values else len(history.history.get("loss", []))
    run_config["results"] = {
        "epochs_trained": len(history.history.get("loss", [])),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history.history.get("val_loss", [float("nan")]), default=float("nan"))),
        "best_val_severity_mse": float(min(history.history.get("val_severity_output_mse", [float("nan")]), default=float("nan"))),
        "best_val_severity_ssim": float(max(history.history.get("val_severity_output_average_ssim", [float("nan")]), default=float("nan"))),
        "best_val_identity_accuracy": float(max(history.history.get("val_identity_output_accuracy", [float("nan")]), default=float("nan"))),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    # ── Copy latest artifacts to output_dir root for backward compatibility ──
    if (run_dir / "best_model.keras").exists():
        shutil.copy2(run_dir / "best_model.keras", output_dir / "best_model.keras")
    shutil.copy2(run_dir / "final_model.keras", output_dir / "final_model.keras")
    shutil.copy2(run_dir / "history.json", output_dir / "history.json")
    shutil.copy2(run_dir / "training_log.csv", output_dir / "training_log.csv")

    print(f"Run artifacts saved to: {run_dir}")
    return model, history


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the multi-task U-Net ConvLSTM smog forecaster.")
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_VAL_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-monitor", default="val_loss")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    _, history = train_model(
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_monitor=args.early_stopping_monitor,
    )
    print(f"Training finished with metrics: {list(history.history.keys())}")


if __name__ == "__main__":
    main()
