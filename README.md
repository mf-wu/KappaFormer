# KappaFormer

> This repository is the implementation of KappaFormer for the paper "*KappaFormer: Physics-aware Transformer for lattice thermal conductivity via cross-domain transfer learning*".  
- arXiv: [10.48550/arXiv.2604.03547](https://arxiv.org/abs/2604.03547)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

Machine learning has been widely used for predicting material properties. However, efficient prediction of lattice thermal conductivity (κ<sub>L</sub>) remains a long-standing challenge, primarily due to the scarcity of high-quality training data. 

**KappaFormer** is a physics-aware Transformer architecture that embeds the harmonic–anharmonic decomposition of κ<sub>L</sub> within the network. It comprises:

- A **harmonic branch** pre-trained on large-scale elastic property (bulk modulus B and shear modulus G) data
- An **anharmonic branch** fine-tuned on limited experimental κ<sub>L</sub> data

This two-stage training strategy enables effective knowledge transfer and enhanced generalization. High-throughput screening with KappaFormer identifies multiple candidates with ultralow κ<sub>L</sub>, which are further confirmed by first-principles calculations. Physics interpretability further elucidates the vibrational mechanisms governing thermal transport suppression, linking structural motifs to strong anharmonicity.

## Key Features

- ⚛️ **Physics-aware Architecture**: Embeds harmonic-anharmonic decomposition of κ<sub>L</sub> directly into the network
- 🔄 **Cross-domain Transfer Learning**: Leverages large-scale elastic property data to improve κ<sub>L</sub> prediction
- 🚀 **High-throughput Screening**: Capable of screening thousands of materials efficiently
- 📊 **Interpretability**: Provides harmonic and anharmonic interpretability for physics insights

## Architecture

KappaFormer is built on an equivariant Transformer backbone with the following key components:

1. **Shared Backbone**: SO(2) equivariant graph attention blocks for material structure encoding
2. **Harmonic Branch**: Pre-trained on bulk modulus (B) and shear modulus (G) prediction
3. **Anharmonic Branch**: Fine-tuned for κ<sub>L</sub> prediction using transfer learning
4. **Mixture of Experts (MoE)**: Specialized expert networks for harmonic/anharmonic feature processing
5. **Physics-guided Fusion**: GRU-based fusion module combining harmonic and anharmonic contributions

## Installation

### Environment Setup

We provide the full environment specification in `environment.txt`. To install:

```bash
pip install -r environment.txt
```

### Key Dependencies

- Python 3.9+
- PyTorch 2.4.0+ (CUDA 12.4 recommended)
- torch_geometric 2.5.3
- e3nn 0.5.0
- ASE 3.24.0
- pymatgen 2022.5.26
- lmdb 1.6.2
- pandas, numpy, scipy, tqdm

## Dataset Preparation

### Download Datasets

Training datasets are available for download from Zenodo:

| Dataset | Description | Download |
|---------|-------------|----------|
| Stage I | B/G elastic properties dataset | [Zenodo](https://zenodo.org/records/21502039) |
| Stage II | Experimental κ<sub>L</sub> dataset | [Zenodo](https://zenodo.org/records/21502039) |

### Data Format

Datasets are stored in LMDB format. Raw XYZ files can be converted using the provided utilities:

```bash
# For Stage I (B/G) data
python utils/xyz_to_lmdb.py

# For Stage II (kappa) data
python utils/xyz_to_lmdb_kappa.py
```

The expected XYZ format follows the standard extended XYZ format with material properties in the `info` dictionary.

## Pretrained Models

Pretrained model weights are available for download:

| Model | Description | Download |
|-------|-------------|----------|
| Stage I | B/G prediction checkpoint | [Zenodo](https://zenodo.org/records/21502118) |
| Stage II | κ<sub>L</sub> prediction checkpoint | [Zenodo](https://zenodo.org/records/21502118) |

Download and place the checkpoints in the `pretrained_models/` directory.

## Training

KappaFormer training follows a two-stage process.

### Stage I: Pre-training on Elastic Properties

Train the backbone and harmonic branch on large-scale B/G data:

```bash
python train.py \
    --save_dir ./ckpt_bg \
    --save_epoch_interval 1 \
    --begin_save_epoch 50 \
    --enable_kappa False \
    --train_batch_size 4 \
    --accu_steps 1 \
    --lr 1e-4
```

**Key parameters:**
- `--enable_kappa False`: Train only on B/G prediction task
- `--train_batch_size 4`: Batch size per GPU
- `--lr 1e-4`: Learning rate for Stage I

### Stage II: Fine-tuning on κ<sub>L</sub> Prediction

Fine-tune using the Stage I checkpoint with the anharmonic branch:

```bash
python train.py \
    --save_dir ./ckpt_kappa \
    --save_epoch_interval 1 \
    --begin_save_epoch 50 \
    --enable_kappa True \
    --train_batch_size 1 \
    --accu_steps 1 \
    --lr 1e-3 \
    --use_checkpoints ./pretrained_models/stage_I_bg.pt
```

**Key parameters:**
- `--enable_kappa True`: Enable kappa prediction task
- `--use_checkpoints`: Path to Stage I pre-trained checkpoint
- `--lr 1e-3`: Higher learning rate for fine-tuning

### Additional Training Options

```bash
# Distributed training
python train.py --strategy ddp ...

# Mixed precision training
python train.py --fp16 True ...

# Resume from checkpoint
python train.py --ifresume True --ckpt_path ./path/to/ckpt.pt ...
```

## Inference

### Predict Thermal Conductivity

Use the trained model to predict κ<sub>L</sub> for new materials:

```bash
python return_infer_embedding.py \
    --input prediction_dataset.xyz \
    --ckpt_path ./pretrained_models/stage_II_kappa.pt \
    --output_path ./preds_embeddings.csv
```

### Output Format

The script generates a CSV file containing:

| Column | Description |
|--------|-------------|
| `mp_id` | Material Project ID |
| `material` | Chemical formula |
| `nsites` | Number of atomic sites |
| `Ehull` | Energy above convex hull (eV/atom) |
| `dim` | Dimensionality |
| `b_pred` | Predicted bulk modulus (GPa) |
| `g_pred` | Predicted shear modulus (GPa) |
| `kappa_pred` | Predicted lattice thermal conductivity (W/mK) |
| `Harm_emb_*` | Harmonic branch embeddings |
| `Anharm_emb_*` | Anharmonic branch embeddings |

### Using as ASE Calculator

```python
from ase.io import read
from models.kappaformer.kappaformer import Kappaformer
from return_infer_embedding import InferCalculator

# Load model
model = Kappaformer(enable_kappa=True, return_embedding=True)
calc = InferCalculator(model, load_path="pretrained_models/stage_II_kappa.pt", device="cuda")

# Predict for a single structure
atoms = read("your_structure.xyz")
mp_id, mat, nsites, ehull, dim, b, g, kappa_log, harm_emb, anharm_emb = calc.calculate(atoms)
kappa = 10 ** kappa_log
```

## Project Structure

```
KappaFormer/
├── train.py                      # Training script
├── return_infer_embedding.py     # Inference script
├── environment.txt               # Python environment specification
├── LICENSE                       # MIT License
├── README.md                     # This file
├── models/
│   └── kappaformer/
│       ├── kappaformer.py        # Main model architecture
│       ├── transformer_block.py  # Transformer block implementations
│       ├── so3.py               # SO(3) equivariant operations
│       ├── gaussian_rbf.py      # Radial basis functions
│       └── ...
├── engine/
│   ├── trainer.py               # Training loop
│   ├── scheduler.py             # Learning rate schedulers
│   ├── accelerator.py           # Training acceleration
│   └── logging/                 # Logging utilities
├── utils/
│   ├── lmdb_dataset.py          # LMDB dataset loader
│   ├── xyz_to_lmdb.py           # XYZ to LMDB converter
│   └── xyz_to_lmdb_kappa.py     # Kappa-specific converter
├── data/                         # Dataset storage
└── pretrained_models/            # Pretrained model checkpoints
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use KappaFormer in your research, please cite:

```bibtex
@article{wu2026kappaformer,
  title={KappaFormer: Physics-aware Transformer for Lattice Thermal Conductivity via Cross-Domain Transfer Learning},
  author={Wu, Mengfan and Tan, Junfu and Ren, Jie and others},
  journal={arXiv preprint arXiv:2604.03547},
  year={2026}
}
```

## Contact

For questions or issues, please contact us:
- **Mengfan Wu** - mfwu@tongji.edu.cn
- **Junfu Tan** - jtan370@gatech.edu
- **Jie Ren** - Xonics@tongji.edu.cn
