# RLPL: Representation Learning with Pseudo-Labeling

This repository implements RLPL (Representation Learning with Pseudo-Labeling), a semi-supervised learning approach that combines representation learning techniques with pseudo-labeling for improved performance on image classification tasks.

## Overview

RLPL framework incorporates an auxiliary loss function with consistency loss and pseudo-labeling techniques. The approach consists of three main stages:

1. **Unsupervised Representation Learning**: Training a model on unlabeled data to learn meaningful representations.
2. **Downstream Task Training**: Fine-tuning the pre-trained model on a small labeled dataset.
3. **Pseudo-Labeling**: Using the trained model to generate pseudo-labels for unlabeled data, improving model performance through semi-supervised learning.

## Repository Structure

- `rlpl_main.py`: Core implementation of the RLPL algorithm
- `rlpl_unsup.py`: Script for unsupervised representation learning stage
- `rlpl_downstream.py`: Script for training on downstream tasks with labeled data
- `pl.py`: Implementation of pseudo-labeling for final model training

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- tqdm

## Installation

```bash
git clone https://github.com/yourusername/rlpl.git
cd rlpl
pip install -r requirements.txt
```

## Usage

### 1. Unsupervised Representation Learning

```bash
python rlpl_unsup.py --path /path/to/dataset/ --batchsz 128 --epochs 50 --method RLPL
```

Key arguments:
- `--path`: Path to the dataset directory
- `--resize`: Image size (default: 256)
- `--batchsz`: Batch size (default: 128)
- `--epochs`: Number of training epochs (default: 50)
- `--method`: Method to use, either 'RLPL' or 'BYOL' (default: 'RLPL')
- `--save_path`: Path to save the pre-trained model (default: './model/rlpl_unsup.pt')

### 2. Downstream Task Training

```bash
python rlpl_downstream.py --data_path /path/to/dataset/ --num_epochs 25 --model_checkpoint ./model/rlpl_unsup.pt
```

Key arguments:
- `--data_path`: Path to the dataset directory
- `--batch_size`: Batch size (default: 10)
- `--num_epochs`: Number of training epochs (default: 25)
- `--model_checkpoint`: Path to the pre-trained model checkpoint
- `--save_path`: Path to save the downstream model (default: './model/downstream.pt')

### 3. Pseudo-Labeling

```bash
python pl.py --data_path /path/to/dataset/ --model_path ./model/downstream.pt --threshold 0.95
```

Key arguments:
- `--data_path`: Path to the dataset directory
- `--batch_size`: Batch size for labeled data (default: 32)
- `--unlbl_batch_size`: Batch size for unlabeled data (default: 128)
- `--num_epochs`: Number of training epochs (default: 151)
- `--threshold`: Confidence threshold for pseudo-labeling (default: 0.95)
- `--model_path`: Path to the downstream model
- `--save_path`: Path to save the final model (default: './model/final.pt')

## Dataset Structure

The dataset should be organized as follows:

```
dataset_folder/
├── train/  # Labeled training data
│   ├── class1/
│   ├── class2/
│   └── ...
├── test/   # Test data
│   ├── class1/
│   ├── class2/
│   └── ...
└── unsup/  # Unlabeled data
    └── dummy_class/  # Single folder containing all unlabeled images
```

## Model Performance

RLPL has been shown to outperform traditional supervised learning approaches, especially in scenarios with limited labeled data and computationally constrained environments. The method leverages unlabeled data to improve model generalization and achieves competitive results compared to state-of-the-art semi-supervised learning techniques.

