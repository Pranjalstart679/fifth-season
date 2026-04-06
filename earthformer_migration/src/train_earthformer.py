import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.torch_dataset import EarthformerWeatherDataset
from model.earthformer_wrapper import MultiTaskEarthformer

def load_dataset_config(dataset_path):
    import xarray as xr
    with xr.open_dataset(dataset_path) as ds:
        feature_variables = json.loads(ds.attrs["feature_variables"])
        target_variable = ds.attrs["target_variable"]
        num_classes = int(ds.attrs.get("num_classes", 5))
    return feature_variables, target_variable, num_classes

def combined_loss(severity_pred, severity_target, identity_pred, identity_target, alpha=0.65):
    # Mean Squared Error for severity
    mse_loss = nn.functional.mse_loss(severity_pred, severity_target)
    
    # You can implement differentiable SSIM in PyTorch (e.g., using torchmetrics.image.StructuralSimilarityIndexMeasure)
    # Here is a placeholder for SSIM blending
    ssim_loss = 0.0 # Replace with actual 1 - SSIM code

    reg_loss = (1.0 - alpha) * mse_loss + alpha * ssim_loss
    
    # Cross Entropy for identity
    ce_loss = nn.functional.cross_entropy(identity_pred, identity_target)
    
    return reg_loss + ce_loss, reg_loss, ce_loss

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    loop = tqdm(dataloader, desc="Training")
    for x, y in loop:
        x = x.to(device)
        severity_target = y["severity"].to(device)
        identity_target = y["identity"].to(device)
        
        optimizer.zero_grad()
        severity_pred, identity_pred = model(x)
        
        loss, r_loss, c_loss = combined_loss(
            severity_pred, severity_target, 
            identity_pred, identity_target
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_path = "../../Dataset/processed/train_5D.nc"
    if not Path(train_path).exists():
        print(f"Data not found at {train_path}.")
        return

    features, target_var, num_classes = load_dataset_config(train_path)
    
    # Setup Dataset and DataLoader
    train_dataset = EarthformerWeatherDataset(train_path, features, target_var)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=4)

    # Setup Model
    model = MultiTaskEarthformer(in_channels=len(features), seq_length=5, num_classes=num_classes)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 10
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()