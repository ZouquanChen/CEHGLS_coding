# CEHGLS - Personalized Federated Learning

## Project Description

CEHGLS is a personalized federated learning system based on [PFLlib](https://github.com/AIx4Noobs/PFLlib). It implements two main enhancements:

1. **HGLS (Head Gradient Linear Search)**: Clients perform linear search to select optimal head gradient from all clients
   - Formula: `L(w_{t+1}) = min{ L(w_t), L(w_t - α_j · g_j) }` with constraint `<g_j, G_H> > 0`
   
2. **HGC (Head Gradient Compression)**: Quantization-based compression for head gradients
   - Supports bit widths: 8, 16, 32, 64
   - Achieves ~4-8x compression ratio

## Project Structure

```
CEHGLS/
├── system/
│   ├── main.py                 # Entry point
│   ├── flcore/
│   │   ├── servers/
│   │   │   ├── serverbase.py   # Base server class
│   │   │   └── serveravg.py    # FedAvg implementation
│   │   ├── clients/
│   │   │   ├── clientbase.py   # Base client with HGLS/HGC
│   │   │   └── clientavg.py    # FedAvg client
│   │   ├── trainmodel/
│   │   │   ├── models.py       # Model definitions
│   │   │   ├── resnet.py       # ResNet models
│   │   │   └── vit.py          # Vision Transformer
│   │   └── optimizers/
│   │       └── fedoptimizer.py # Optimizers
│   └── utils/
│       ├── ALA.py              # Adaptive local aggregation
│       ├── data_utils.py       # Data utilities
│       ├── dlg.py              # DLG attack
│       ├── encode.py           # LFSR encoder for compression
│       ├── mem_utils.py        # Memory utilities
│       └── result_utils.py     # Result processing
└── results/                    # Output directory
```

## Usage

### Basic FedAvg

```bash
cd system
python main.py -go test -data CIFAR10 -model CNN -nb 100 -lbs 10 \
    -gr 200 -algo FedAvg -nc 20 -did 0
```

### With HGLS (Head Gradient Enhancement)

```bash
python main.py -go test -data CIFAR10 -model CNN -nb 100 -lbs 10 \
    -gr 200 -algo FedAvg -nc 20 -did 0 -uhg
```

### With HGLS + HGC (Gradient Compression)

```bash
python main.py -go test -data CIFAR10 -model CNN -nb 100 -lbs 10 \
    -gr 200 -algo FedAvg -nc 20 -did 0 -uhg -ugc -bw 16
```

## Command Line Arguments

### General
| Argument | Description | Default |
|----------|-------------|---------|
| `-go`, `--goal` | Goal (test/train/...) | test |
| `-dev`, `--device` | Device (cpu/cuda) | cuda |
| `-did`, `--device_id` | GPU ID | 0 |
| `-data`, `--dataset` | Dataset name | MNIST |
| `-nb`, `--num_classes` | Number of classes | 10 |
| `-m`, `--model` | Model type (CNN/ResNet10/ViT) | CNN |
| `-lbs`, `--batch_size` | Local batch size | 10 |
| `-lr`, `--local_learning_rate` | Local learning rate | 0.01 |
| `-ld`, `--learning_rate_decay` | Enable LR decay | False |
| `-gr`, `--global_rounds` | Number of global rounds | 100 |
| `-ls`, `--local_epochs` | Local epochs | 2 |
| `-algo`, `--algorithm` | Algorithm (FedAvg) | FedAvg |
| `-jr`, `--join_ratio` | Join ratio | 1.0 |
| `-nc`, `--num_clients` | Total number of clients | 20 |
| `-eg`, `--eval_gap` | Evaluation gap (rounds) | 1 |

### HGLS / HGC Components
| Argument | Description | Default |
|----------|-------------|---------|
| `-ugc`, `--use_grad_compress` | Enable gradient compression | False |
| `-uhg`, `--use_head_grad` | Enable head gradient enhancement | False |
| `-bw`, `--bit_width` | Bit width for quantization | 16 |

## Supported Datasets

- MNIST / FashionMNIST
- CIFAR10 / CIFAR100
- TinyImageNet
- ImageNet

## Supported Models

- CNN (FedAvgCNN)
- ResNet10
- Vision Transformer (ViT)

## Memory Tracking

The system tracks memory usage:
- `train_base_m`: Baseline FedAvg communication (MB)
- `train_base_glsm`: With head gradients (uncompressed) (MB)
- `train_base_chgc`: With head gradients (compressed) (MB)

## Development Notes

- Only FedAvg algorithm is kept (other FL algorithms removed)
- All code comments are in English
- Results are saved in `results/` directory
- Use `-uhg` flag to enable HGLS
- Use `-ugc` flag to enable HGC compression
- Compression ratio is printed each round when enabled
