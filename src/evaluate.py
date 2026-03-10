from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xarray as xr

from model.convlstm import (
    AverageSSIM,
    _expand_time_axis_output_shape,
    _flatten_spatial_output_shape,
    _restore_spatial_output_shape,
    combined_regression_loss,
    expand_time_axis,
    flatten_spatial,
    restore_spatial,
)
from train import WeatherDataGenerator, _load_dataset_config


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "Dataset" / "processed"
DEFAULT_TEST_DATASET = DEFAULT_DATA_DIR / "test_5D.nc"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "best_model.keras"
DEFAULT_HISTORY_PATH = DEFAULT_ARTIFACT_DIR / "history.json"


def _compute_sample_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_tensor = tf.convert_to_tensor(y_true[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    y_pred_tensor = tf.convert_to_tensor(y_pred[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    return float(tf.reduce_mean(tf.image.ssim(y_true_tensor, y_pred_tensor, max_val=1.0)).numpy())


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    test_dataset_path: Path | str = DEFAULT_TEST_DATASET,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
    output_dir: Path | str = DEFAULT_ARTIFACT_DIR / "evaluation",
    sequence_length: int = 5,
    max_samples: int | None = None,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_variables, target_variable, _ = _load_dataset_config(test_dataset_path)
    generator = WeatherDataGenerator(
        test_dataset_path,
        feature_variables=feature_variables,
        target_variable=target_variable,
        sequence_length=sequence_length,
        batch_size=1,
        shuffle=False,
        max_samples=max_samples,
    )
    if len(generator) == 0:
        raise ValueError("The evaluation dataset does not contain any valid forecasting windows.")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "combined_regression_loss": combined_regression_loss,
            "AverageSSIM": AverageSSIM,
            "flatten_spatial": flatten_spatial,
            "restore_spatial": restore_spatial,
            "expand_time_axis": expand_time_axis,
            "_flatten_spatial_output_shape": _flatten_spatial_output_shape,
            "_restore_spatial_output_shape": _restore_spatial_output_shape,
            "_expand_time_axis_output_shape": _expand_time_axis_output_shape,
        },
    )

    mse_scores: list[float] = []
    ssim_scores: list[float] = []
    true_series: list[float] = []
    pred_series: list[float] = []
    center_pixel = None
    first_batch: tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray] | None = None

    for batch_index in range(len(generator)):
        batch_x, batch_y = generator[batch_index]
        pred_severity, pred_identity = model.predict(batch_x, verbose=0)

        true_severity = batch_y["severity_output"][0, 0, ..., 0]
        pred_severity_map = pred_severity[0, 0, ..., 0]

        mse_scores.append(float(np.mean((true_severity - pred_severity_map) ** 2)))
        ssim_scores.append(_compute_sample_ssim(true_severity, pred_severity_map))

        if center_pixel is None:
            center_pixel = (true_severity.shape[0] // 2, true_severity.shape[1] // 2)
        true_series.append(float(true_severity[center_pixel]))
        pred_series.append(float(pred_severity_map[center_pixel]))

        if first_batch is None:
            first_batch = (batch_x, batch_y, pred_severity, pred_identity)

    metrics = {
        "mse": float(np.mean(mse_scores)),
        "average_ssim": float(np.mean(ssim_scores)),
        "num_windows": float(len(mse_scores)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if Path(history_path).exists():
        history = json.loads(Path(history_path).read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.get("loss", []), label="Train loss")
        axes[0].plot(history.get("val_loss", []), label="Validation loss")
        axes[0].set_title("Loss over epochs")
        axes[0].legend()

        axes[1].plot(history.get("severity_output_mse", []), label="Train MSE")
        axes[1].plot(history.get("val_severity_output_mse", []), label="Validation MSE")
        axes[1].set_title("Severity MSE over epochs")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_mse_over_epochs.png", dpi=150)
        plt.close(fig)

    if first_batch is not None:
        _, batch_y, pred_severity, pred_identity = first_batch
        true_severity = batch_y["severity_output"][0, 0, ..., 0]
        pred_severity_map = pred_severity[0, 0, ..., 0]
        true_identity = batch_y["identity_output"][0, 0, ...]
        pred_identity_map = np.argmax(pred_identity[0, 0], axis=-1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        im0 = axes[0, 0].imshow(true_severity, cmap="magma")
        axes[0, 0].set_title("Ground truth aerosol index")
        fig.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(pred_severity_map, cmap="magma")
        axes[0, 1].set_title("Predicted aerosol index")
        fig.colorbar(im1, ax=axes[0, 1])

        im2 = axes[1, 0].imshow(true_identity, cmap="tab10")
        axes[1, 0].set_title("Ground truth dominant aerosol type")
        fig.colorbar(im2, ax=axes[1, 0])

        im3 = axes[1, 1].imshow(pred_identity_map, cmap="tab10")
        axes[1, 1].set_title("Predicted dominant aerosol type")
        fig.colorbar(im3, ax=axes[1, 1])
        fig.tight_layout()
        fig.savefig(output_dir / "ground_truth_vs_prediction.png", dpi=150)
        plt.close(fig)

    evaluation_dataset = xr.open_dataset(test_dataset_path)
    target_times = [str(np.datetime_as_string(value, unit="D")) for value in generator.target_times]
    evaluation_dataset.close()

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(target_times, true_series, label="Validation AI")
    axis.plot(target_times, pred_series, label="Predicted AI")
    axis.set_title("Validation AI vs predicted AI at a sample location")
    axis.set_xlabel("Forecast timestep")
    axis.set_ylabel("Normalized aerosol index")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_vs_predicted_ai_timeseries.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(target_times, ssim_scores, label="SSIM")
    axis.set_title("SSIM over evaluation timesteps")
    axis.set_xlabel("Forecast timestep")
    axis.set_ylabel("SSIM")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ssim_over_timesteps.png", dpi=150)
    plt.close(fig)

    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the trained U-Net ConvLSTM smog forecaster.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_TEST_DATASET)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "evaluation")
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = evaluate_model(
        model_path=args.model,
        test_dataset_path=args.test_dataset,
        history_path=args.history,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        max_samples=args.max_samples,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
