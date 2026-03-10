# CUDA Setup For fifth-season

This project uses TensorFlow.

Important constraint:

1. Native CUDA acceleration is not supported on Windows for TensorFlow 2.11 and newer.
2. Your current project stack uses TensorFlow 2.21, so the current Windows environment will run on CPU only.
3. If you must stay on native Windows, the only CUDA path is a legacy TensorFlow 2.10 setup.

## Native Windows options

You have two realistic choices on Windows:

1. Stay with the current project stack and use CPU.
2. Create a separate legacy Windows CUDA environment using TensorFlow 2.10.

The second option is the only native Windows CUDA path, but it is a compatibility path, not the preferred path. On a very new GPU like an RTX 5060, legacy TensorFlow 2.10 with CUDA 11.2 may not be reliable.

## Recommended path

If you can use WSL2, use WSL2. If you cannot, use the legacy Windows CUDA path below and expect potential compatibility problems.

## Native Windows legacy CUDA path

Use the native Windows file:

`environment.windows-cuda.yml`

This environment is for TensorFlow 2.10 on Python 3.10.

### Required versions

Install these system-level NVIDIA components on Windows:

1. NVIDIA driver: keep your current modern driver.
2. CUDA Toolkit 11.2
3. cuDNN 8.1 for CUDA 11.2

These must be installed on Windows outside conda.

### Create the legacy Windows CUDA environment

From the repo root in PowerShell:

```powershell
& "C:\Users\Pranjal\miniconda3\Scripts\conda.exe" env create -f environment.windows-cuda.yml
conda activate fifth-season-windows-cuda
```

### Register the Jupyter kernel

```powershell
python -m ipykernel install --user --name fifth-season-windows-cuda --display-name "Python (fifth-season-windows-cuda)"
```

### Verify TensorFlow GPU visibility

```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

If this prints an empty list, native CUDA is not working in that environment.

### Important warning for your GPU

Your machine is using an RTX 5060 with a modern driver. TensorFlow 2.10 is tied to an older CUDA stack. That means this path may fail even if installation succeeds.

If it fails, the failure is not in your project code. It is a platform compatibility limitation between:

1. Native Windows TensorFlow
2. TensorFlow 2.10's supported CUDA stack
3. A newer GPU generation

## WSL2 path

If you later decide to use WSL2, use `environment.cuda.yml` and follow the Linux CUDA path below.

## WSL2 installation

In an elevated PowerShell window:

```powershell
wsl --install
```

Reboot if Windows asks for it, then open Ubuntu.

## Verify GPU inside WSL

In Ubuntu:

```bash
nvidia-smi
```

If this works, WSL can see the GPU.

## Install Miniconda inside WSL

In Ubuntu:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

## Create the CUDA environment

From the repo root inside WSL:

```bash
conda env create -f environment.cuda.yml
conda activate fifth-season-cuda
```

## Register the Jupyter kernel

```bash
python -m ipykernel install --user --name fifth-season-cuda --display-name "Python (fifth-season-cuda)"
```

## Quick TensorFlow GPU check

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see at least one GPU device.

## Run the project

From the repo root inside WSL:

```bash
conda activate fifth-season-cuda
jupyter notebook
```

Or run scripts directly:

```bash
python src/train.py
python src/evaluate.py
```

## Important note

The current Windows conda environment can still be used for CPU runs, but it will not use CUDA with TensorFlow 2.21.