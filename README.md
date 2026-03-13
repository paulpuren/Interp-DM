# Generative Refinement Learning for Continuous Temporal Interpolation

This repository implements generative refinement learning approaches for high-quality temporal interpolation of spatio-temporal data, with a focus on fluid dynamics simulations and weather data. The project provides multiple model architectures and evaluation frameworks for interpolating between temporal frames in scientific datasets.

## Features

### Models
- **FLEX**: Flexible interpolation model with MLP-based refinement
- **UNet**: Convolutional neural network for spatial feature extraction
- **SuperSloMo**: Optical flow-based frame interpolation
- **Diffusion Models**: Generative diffusion-based interpolation using denoising processes

### Datasets
- **NSKT**: Navier-Stokes Kolmogorov Turbulence data at various Reynolds numbers (600-36000)
- **Shanghai**: Urban-scale real-world radar dataset
- **SST**: Sea Surface Temperature data

### Key Capabilities
- Multi-scale temporal interpolation (T=8, T=16, T=20 frames)
- Distributed training with PyTorch DDP
- Comprehensive evaluation metrics (RFNE, R² scores)
- WandB integration for experiment tracking
- Patch-based training for large-scale data

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended for training)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Interp-DM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For evaluation and analysis:
```bash
pip install scipy seaborn tqdm lpips
```

## Usage

### Data Preparation
Place your datasets in the `datasets/` directory. The code expects HDF5 files for NSKT data with the following structure:
- Vorticity field (`w`)
- Velocity components (`u`, `v`)

### Training

#### Basic Training
```bash
python train.py --config config/nskt/flex_small.yaml
```

#### Distributed Training
```bash
torchrun --nproc_per_node=4 train.py --config config/nskt/flex_small.yaml
```

#### Available Training Scripts
- `train.py`: Main training script with DDP support
- `train_flex_unet.py`: FLEX + UNet combined training
- `train_super_slomo.py`: SuperSloMo model training
- `train_unet.py`: UNet-only training

### Evaluation

#### Single Model Evaluation
```bash
python eval.py --model flex_small --data nskt --re 1000
```

#### Batch Evaluation
```bash
python eval_all.py
```

#### Dataset-Specific Evaluation
- `eval_nskt.py`: NSKT dataset evaluation
- `eval_shanghai.py`: Shanghai dataset evaluation
- `eval_sst.py`: SST dataset evaluation

### Configuration
Configuration files are located in `config/` directory:
- `nskt/`: NSKT dataset configurations
- `shanghai/`: Shanghai dataset configurations
- `sst/`: SST dataset configurations

Key parameters:
- `model`: Model architecture (flex_small, flex_medium, unet, super_slomo)
- `T`: Number of interpolation frames
- `stride`: Patch stride for training
- `lr`: Learning rate
- `epochs`: Training epochs

## Results

Evaluation results are stored in `assets/` directory with detailed metrics:

### Performance Metrics
- **RFNE (Relative Frobenius Norm Error)**: Measures interpolation accuracy
- **R² Score**: Coefficient of determination for prediction quality
- **Runtime**: Inference time per sample

### Example Results (NSKT Re=1000)
| Model | Avg RFNE | Avg R² | Runtime (s) |
|-------|----------|--------|-------------|
| FLEX Small | 0.004 | 0.99999 | 1.47 |
| FLEX Medium | 0.003 | 0.99999 | 2.1 |
| UNet | 0.005 | 0.99998 | 0.9 |

## Project Structure

```
Interp-DM/
├── src/                    # Source code
│   ├── flex.py            # FLEX model implementation
│   ├── unet.py            # UNet architecture
│   ├── diffusion_model.py # Diffusion model
│   ├── super_slomo.py     # SuperSloMo implementation
│   └── metrics.py         # Evaluation metrics
├── datasets/              # Dataset loaders
│   ├── data_nskt.py      # NSKT data loader
│   └── data_era5_z500.py  # ERA5 data loader
├── config/                # Configuration files
├── checkpoints/           # Model checkpoints
├── assets/                # Evaluation results
├── analysis/              # Jupyter notebooks for analysis
└── utils/                 # Utility functions
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{generative-refinement-interpolation,
  title={Generative Refinement Learning for Continuous Temporal Interpolation},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-repo/Interp-DM}
}
```

## Acknowledgments

- Built on PyTorch and Diffusers library
- Inspired by video frame interpolation methods like SuperSloMo and EDEN
- Uses NSKT turbulence data for benchmarking



