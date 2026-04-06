# Earthformer Migration Plan

This directory contains everything you need to recreate the fifth-season geospatial ML forecasting project using the new **Amazon Earthformer** space-time transformer architecture instead of the existing TensorFlow U-Net ConvLSTM setup. This uses PyTorch.

## What is in this directory?

- `src/data/`: A complete copy of your working data processing pipeline. This means you do not need to rewrite the Sentinel-5P / MERRA-2 fusion logic. It will generate the exact same`.nc` NetCDF files you already know.
- `src/data/torch_dataset.py`: A new PyTorch `Dataset` wrapper that reads from the XArray `.nc` files constructed by the data pipeline, and yields the appropriate tensor shapes `(Timesteps, Channels, Height, Width)` required by modern video transformers.
- `src/model/earthformer_wrapper.py`: The architecture skeleton for substituting the `CuboidTransformerModel` (Earthformer) as a multi-task backbone with two task heads (severity and dominant aerosol classification).
- `src/train_earthformer.py`: The new PyTorch training loop that orchestrates the dataloader and runs the multi-task loss.
- `environment.earthformer.yml`: A PyTorch conda environment configuration that pulls the requirements for running PyTorch Lightning and the Earthformer GitHub package.

## Steps for the New Device

1. **Move this Folder**: Move this complete `earthformer_migration` folder to the target device. It is standalone except for the raw NetCDF data you will also need to bring over.
2. **Environment Setup**: 
   Ensure you adjust the `pytorch-cuda` version inside `environment.earthformer.yml` to match the native CUDA version on the new machine.
   ```bash
   conda env create -f environment.earthformer.yml
   conda activate earthformer-env
   ```
3. **Bring the Dataset**: Create the folder `Dataset/processed` near the root and place your merged `train_5D.nc` and `test_5D.nc` there, OR re-run the `src/data/fuse_datasets.py` pipeline to regenerate them.
4. **Implement the Backbone**: Open `src/model/earthformer_wrapper.py`. Currently, `self.backbone` is an `nn.Identity()`. You must import `CuboidTransformerModel` from the `earthformer` library and configure its block depths, channels, and spatial resolution (291x512).
5. **Run Training**: 
   ```bash
   python src/train_earthformer.py
   ```
   *Note on SSIM:* `train_earthformer.py` currently holds a placeholder for SSIM regression loss. Make sure to plug in `torchmetrics.image.StructuralSimilarityIndexMeasure` to replicate the old TF SSIM+MSE combined loss exactly.
