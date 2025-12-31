# DAPoinTr Installation Guide

This document provides detailed instructions for setting up the environment and installing dependencies for the DAPoinTr project.

## Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU with compute capability 6.0 or higher
- GCC compiler for building CUDA extensions

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/DAPoinTr.git
cd DAPoinTr
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
conda create -n dapointr python=3.7
conda activate dapointr
```

### 3. Install PyTorch with CUDA support

```bash
# For CUDA 11.1
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

# Or for other CUDA versions, visit https://pytorch.org/
```

### 4. Install other Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Install PointNet++ CUDA extensions

This project uses PointNet++ CUDA operations. You need to install the `pointnet2_ops` package:

```bash
# Install from pip
pip install pointnet2_ops

# Or build from source if needed
pip install --upgrade git+https://github.com/erikwijmans/Pointnet2_PyTorch.git
```

### 6. Build CUDA extensions for the project

The project includes several custom CUDA extensions that need to be compiled:

```bash
# Build Chamfer Distance
cd extensions/chamfer_dist
python setup.py install
cd ../..

# Build EMD (Earth Mover's Distance)
cd extensions/emd
python setup.py install
cd ../..

# Build Cubic Feature Sampling
cd extensions/cubic_feature_sampling
python setup.py install
cd ../..

# Build Gridding
cd extensions/gridding
python setup.py install
cd ../..

# Build Gridding Loss
cd extensions/gridding_loss
python setup.py install
cd ../..
```

### 7. Verify Installation

You can verify that the installation was successful by running a simple test:

```bash
python -c "import torch; import numpy as np; print('Basic imports successful')"
```

## Training and Testing

### Training

To train the model, use the provided scripts:

```bash
# Single GPU training
bash scripts/train.sh 0 --config cfgs/KITTI_models/DAPoinTr.yaml

# Multi-GPU training
bash scripts/dist_train.sh 2 12345 --config cfgs/KITTI_models/DAPoinTr.yaml
```

### Testing

To test a trained model:

```bash
bash scripts/test.sh 0 --ckpts path/to/checkpoint --config cfgs/KITTI_models/DAPoinTr.yaml
```

## Configuration Files

The project uses YAML configuration files located in:
- `cfgs/KITTI_models/` - Model configurations for KITTI dataset
- `cfgs/CRN_models/` - Model configurations for CRN dataset
- `cfgs/ModelNet_models/` - Model configurations for ModelNet dataset
- `cfgs/dataset_configs/` - Dataset configurations

## Troubleshooting

### CUDA Compilation Issues
- Ensure you have a compatible CUDA toolkit installed
- Make sure nvcc --version matches your PyTorch CUDA version
- Check that your GCC version is compatible (GCC 7-9 are typically supported)

### Memory Issues
- Reduce batch size in the configuration files if you encounter out-of-memory errors
- Use fewer GPUs for distributed training if memory is limited

### Missing Dependencies
If you encounter import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```