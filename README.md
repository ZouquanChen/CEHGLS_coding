# CEHGLS: Communication-Efficient Head Gradient Linear Search for Personalized Federated Learning

## Introduction

CEHGLS is a communication-efficient personalized federated learning system developed based on [PFLlib](https://github.com/TsingZ0/PFLlib.git) [1][2]. This strategy treats the drift introduced during the global aggregation process as an explicit optimization target. By actively utilizing peer gradients for local error rectification, it bridges the gap between global consensus and local adaptation.

This project implements two core enhancement components:
1. **HGLS (Head Gradient Linear Search)**
2. **HGC (Head Gradient Compression)**

---

## Core Components and Methodology

### 1. HGLS (Head Gradient Linear Search)
The HGLS component serves as the core execution layer of CEHGLS, transforming the model optimization task into a linear search problem within the parameter space.
* **Mechanism**: Upon receiving head gradients from peer clients, the client executes a linear search algorithm to determine the optimal linear combination of these gradients.
* **Optimization Objective**: Formulated as `L(w_{t+1}) = min{ L(w_t), L(w_t - α_j · g_j) }`, aiming to identify the update direction that minimizes the local loss.
* **Directional Consistency Constraint**: It introduces the constraint `<g_j, G_H> > 0` and utilizes cosine similarity (`alpha_j`) to measure directional consistency, preventing the model from degrading due to conflicting external information. If no candidate gradient satisfies the condition, the system defaults to a local update.

### 2. HGC (Head Gradient Compression)
To ensure HGLS acts as a "burden-free" plug-and-play component, HGC is designed to serialize high-dimensional gradient matrices into a highly compact format, significantly reducing communication bottlenecks.
* **Quantization and Binning Strategy**: It employs a row-wise norm quantization strategy. The component extracts the original sign matrix `S = sign(G)` for binarization, computes the L2 norm for each row, and maps these norms into `K` uniformly distributed bins.
* **Data Structure**: Ultimately, it only transmits a lightweight bin count array `C`, a highly sparse binary code `M` indicating non-empty bins, the sign matrix `S`, and the maximum norm value `r_max`.
* **Geometric Calibration Reconstruction**: At the receiver side, gradient reconstruction is performed using the formula `v_k = b_k / sqrt(c)` (where `c` is the column dimension), strictly preserving the row-wise structural intensity and significant optimization directions. It supports bit-widths of 8, 16, 32, and 64, achieving approximately 4-8x compression ratios.

---

## Project Directory Structure

```text
CEHGLS/
├── system/
│   ├── main.py                 # Training entry point
│   ├── flcore/
│   │   ├── servers/
│   │   │   ├── serverbase.py   # Base server class, controls global model distribution/aggregation
│   │   │   └── serveravg.py    # FedAvg server implementation, includes memory/communication monitoring
│   │   ├── clients/
│   │   │   ├── clientbase.py   # Base client class, implements HGLS search and HGC compress/decompress algorithms
│   │   │   └── clientavg.py    # FedAvg client implementation, handles local training
│   │   ├── trainmodel/
│   │   │   ├── models.py       # Model definitions (e.g., CNN)
│   │   │   ├── resnet.py       # ResNet models
│   │   │   └── vit.py          # Vision Transformer (ViT)
│   │   └── optimizers/
│   │       └── fedoptimizer.py # Federated optimizers
│   └── utils/
│       ├── ALA.py              # Adaptive Local Aggregation (ALA)
│       ├── data_utils.py       # Dataset processing and loading
│       ├── dlg.py              # DLG attack implementation
│       ├── encode.py           # LFSR encoder for gradient compression
│       ├── mem_utils.py        # Memory and communication resource monitoring
│       └── result_utils.py     # Results processing and saving
├── dataset/                    # Dataset directory
│   └── <DatasetName>/          # Dataset (e.g., Cifar10)
│       ├── config.json         # Dataset configuration
│       ├── data_log            # Data partitioning log
│       ├── rawdata/            # Raw dataset files
│       ├── train/              # Training data splits
│       └── test/               # Test data splits
└── results/                    # Output directory for experimental results (.h5 files)
```

---

## Supported Environment

* **Supported Algorithms**: The current backbone code is streamlined and retains `FedAvg` as the foundational aggregation algorithm.
* **Supported Datasets**: MNIST / FashionMNIST, CIFAR10 / CIFAR100, TinyImageNet, ImageNet.
* **Supported Models**: CNN (FedAvgCNN), ResNet10, Vision Transformer (ViT).

---

## Quick Start

All experimental entry points are located in the `system/` directory. Please navigate to this directory first:

```bash
cd system

```

### 1. Run Basic FedAvg

Run without any enhancement components to establish a performance baseline:

```bash
python main.py -go test -data CIFAR10 -model CNN -nb 10 -lbs 10 \
    -gr 200 -algo FedAvg -nc 20 -did 0

```

### 2. Enable HGLS (Head Gradient Linear Search)

Add the `-uhg` flag to enable HGLS. The server will broadcast the head gradients of all clients, and each client will perform linear search optimization locally:

```bash
python main.py -go test -data CIFAR10 -model CNN -nb 10 -lbs 10 \
    -gr 200 -algo FedAvg -nc 20 -did 0 -uhg

```

### 3. Enable Full CEHGLS (HGLS + HGC Compression)

Add both `-uhg` and `-ugc` flags to perform head gradient interactions with extremely low communication overhead. The quantization bit-width can be specified via `-bw` (default is 16):

```bash
python main.py -go test -data CIFAR10 -model CNN -nb 10 -lbs 10 \
    -gr 200 -algo FedAvg -nc 20 -did 0 -uhg -ugc -bw 16

```

---

## Core Arguments

| Argument | Abbr. | Description | Default |
| --- | --- | --- | --- |
| `--algorithm` | `-algo` | Federated learning algorithm | `FedAvg` |
| `--dataset` | `-data` | Dataset name | `MNIST` |
| `--model` | `-m` | Model architecture (CNN/ResNet10/ViT) | `CNN` |
| `--global_rounds` | `-gr` | Number of global communication rounds | `100` |
| `--num_clients` | `-nc` | Total number of participating clients | `20` |
| **`--use_head_grad`** | **`-uhg`** | **Enable Head Gradient Linear Search (HGLS) enhancement** | `False` |
| **`--use_grad_compress`** | **`-ugc`** | **Enable Head Gradient Compression (HGC)** | `False` |
| **`--bit_width`** | **`-bw`** | **Quantization bit-width (8, 16, 32, 64)** | `16` |

---

## References

[1] Zhang, J., Liu, Y., Hua, Y., Wang, H., Song, T., Xue, Z., Ma, R., & Cao, J. (2025). PFLlib: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark. *Journal of Machine Learning Research*, 26(50), 1-10.

[2] Zhang, J., Wu, X., Zhou, Y., Sun, X., Cai, Q., Liu, Y., Hua, Y., Zheng, Z., Cao, J., & Yang, Q. (2025). HtFLlib: A Comprehensive Heterogeneous Federated Learning Library and Benchmark. *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.
