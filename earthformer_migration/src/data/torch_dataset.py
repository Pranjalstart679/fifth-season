import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class EarthformerWeatherDataset(Dataset):
    def __init__(self, dataset_path, feature_variables, target_variable, sequence_length=5):
        self.dataset = xr.open_dataset(dataset_path)
        self.feature_variables = feature_variables
        self.target_variable = target_variable
        self.sequence_length = sequence_length
        self.available_indices = np.arange(max(self.dataset.sizes["time"] - self.sequence_length, 0))

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, idx):
        start_idx = self.available_indices[idx]
        x_window = self.dataset.isel(time=slice(start_idx, start_idx + self.sequence_length))
        y_window = self.dataset.isel(time=start_idx + self.sequence_length)

        # PyTorch often likes channels first for images, so we transpose: 
        # (Timesteps, Channels, Height, Width)
        stacked_x = np.stack([x_window[f].values.astype(np.float32) for f in self.feature_variables], axis=0) # [C, T, H, W]
        stacked_x = np.transpose(stacked_x, (1, 0, 2, 3)) # -> [T, C, H, W]
        
        # Regression Target: Severity Map
        severity_y = y_window[self.target_variable].values.astype(np.float32)[np.newaxis, ...] # [1, H, W]
        
        # Classification Target: Dominant Aerosol Type
        identity_y = y_window["DOMINANT_AEROSOL_TYPE"].values.astype(np.int64) # [H, W]

        return (
            torch.from_numpy(stacked_x),
            {
                "severity": torch.from_numpy(severity_y),
                "identity": torch.from_numpy(identity_y)
            }
        )
