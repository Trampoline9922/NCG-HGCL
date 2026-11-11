# NCG-HGCL: Node-Conditioned Gated Heterogeneous Graph Contrastive Learning

## Overview

NCG-HGCL is a lightweight heterogeneous graph contrastive learning framework designed for efficient node representation learning and node classification.

The framework combines offline multi-path feature pre-aggregation, node-conditioned semantic gating, and feature-level contrastive learning to improve the representation of heterogeneous graph nodes while reducing repeated graph computation during training.

The model is evaluated on three heterogeneous graph benchmark datasets: ACM, DBLP, and IMDB.

## Key Components

**1. Offline Feature Pre-aggregation**

NCG-HGCL performs parameter-free feature aggregation along heterogeneous graph relations and meta-paths before neural network training. The resulting features capture different structural and semantic relationships and are reused during online training.

**2. Node-Conditioned Gating**

A node-conditioned gating mechanism learns adaptive fusion weights for different semantic channels based on target-node features. This enables node-specific integration of heterogeneous semantic information.

**3. Feature-Level Contrastive Learning**

Feature-level stochastic augmentation is used to construct different views for contrastive learning without repeatedly modifying the original graph topology.

**4. Joint Optimization**

The framework jointly optimizes the supervised node classification objective and a symmetric InfoNCE-based contrastive learning objective.

## Repository Structure

The following structure is recommended for the complete repository:

```text
NCG-HGCL/
├── data/
│   ├── data_loader.py
│   ├── ACM/
│   ├── DBLP/
│   └── IMDB/
├── hgb/
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── sparse_tools/
├── README.md
└── .gitattributes
```

The exact structure should reflect the final uploaded implementation.

## Requirements

The project uses Python and PyTorch for model training and evaluation.

Main dependencies include:

- Python 3.9
- PyTorch
- DGL
- PyTorch Sparse (`torch_sparse`)
- NumPy
- SciPy
- scikit-learn
- tqdm

The experiments were developed in an environment using CUDA 11.8.

Please install compatible versions of PyTorch, DGL, and PyTorch Sparse according to the available CUDA configuration. Some sparse operations may require compiling the extensions provided in `sparse_tools/`.

## Datasets

The experiments use three heterogeneous graph benchmarks.

| Dataset | Target Node | Classification Setting | Number of Classes |
|---|---|---|---|
| ACM | Paper | Single-label | 3 |
| DBLP | Author | Single-label | 4 |
| IMDB | Movie | Multi-label | 5 |

The datasets are expected to be organized as follows:

```text
data/
├── ACM/
│   ├── node.dat
│   ├── link.dat
│   ├── label.dat
│   └── label.dat.test
├── DBLP/
│   ├── node.dat
│   ├── link.dat
│   ├── label.dat
│   └── label.dat.test
└── IMDB/
    ├── node.dat
    ├── link.dat
    ├── label.dat
    └── label.dat.test
```

The data loader also supports an optional `label.dat.test_full` file.

Users should ensure that the node types, relation types, features, and label splits correspond to the datasets used by the training scripts.

## Running the Model

The following examples assume that the training scripts are located in the `hgb/` directory and that the datasets are stored in `../data/`.

Navigate to the training directory:

```bash
cd hgb
```

### ACM

```bash
python main.py --dataset ACM --seeds 1
```

### DBLP

```bash
python main.py --dataset DBLP --seeds 1
```

### IMDB

```bash
python main.py --dataset IMDB --seeds 1
```

### Multiple Random Seeds

The training script supports repeated experiments using different random seeds.

For example:

```bash
python main.py --dataset ACM --seeds 1 2 3 4 5 6 7 8 9 10
```

## Reproducibility

For reproducible experiments, users should record the following information:

- Dataset versions and label splits
- Random seeds
- Model hyperparameters
- Python and library versions
- Hardware configuration
- Training and evaluation settings

The released implementation and configuration files should be checked against the final manuscript before reproducing reported experimental results.

