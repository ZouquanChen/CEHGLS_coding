# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#!/usr/bin/env python
import copy
import torch
import torch.nn as nn
import argparse
import os
import time
import warnings
import numpy as np
import logging

from torchvision.transforms import transforms

from flcore.servers.serveravg import FedAvg

from flcore.trainmodel.models import *

from flcore.trainmodel.resnet import *
from flcore.trainmodel.vit import VisionTransformerModel
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "CNN":  # non-convex
            if "MNIST" in args.dataset or "FashionMNIST" in args.dataset:
                args.model = FedAvgCNN(
                    in_features=1, num_classes=args.num_classes, dim=1024
                ).to(args.device)
            elif "cifar" in args.dataset.lower():
                args.model = FedAvgCNN(
                    in_features=3, num_classes=args.num_classes, dim=1600
                ).to(args.device)

        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "ViT":
            # Enhanced dataset config check
            dataset_config = {
                "fashionmnist": (1, 28, 4),
                "cifar10": (3, 32, 8),
                "cifar100": (3, 32, 8),
                "tinyimagenet": (3, 64, 16),
                "imagenet": (3, 224, 16),
            }

            # Get config (lowercase handling)
            dataset_key = args.dataset.lower()
            in_channels, image_size, patch_size = dataset_config.get(
                dataset_key, (3, 224, 16)
            )

            if image_size % patch_size != 0:
                raise ValueError(
                    f"Invalid config: {args.dataset} image_size({image_size}) cannot be divided by patch_size({patch_size})"
                )

            num_patches = (image_size // patch_size) ** 2
            print(
                f"[Debug] ViT config: image_size={image_size}, patch_size={patch_size}, patches={num_patches}"
            )

            args.model = VisionTransformerModel(
                image_size=image_size,
                patch_size=patch_size,
                in_channels=in_channels,
                d_model=args.feature_dim,
                nhead=8,
                nlayers=6,
                num_classes=args.num_classes,
                d_hid=args.feature_dim * 4,
            ).to(args.device)

            args.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406] if in_channels == 3 else [0.5],
                        std=[0.229, 0.224, 0.225] if in_channels == 3 else [0.5],
                    ),
                ]
            )

        else:
            raise NotImplementedError

        print(args.model)

        # wrap model for FedAvg
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)

        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(
        dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times
    )

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "-go", "--goal", type=str, default="test", help="The goal for this experiment"
    )
    parser.add_argument(
        "-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("-data", "--dataset", type=str, default="MNIST")
    parser.add_argument("-nb", "--num_classes", type=int, default=10)
    parser.add_argument("-m", "--model", type=str, default="CNN")
    parser.add_argument("-lbs", "--batch_size", type=int, default=10)
    parser.add_argument(
        "-lr",
        "--local_learning_rate",
        type=float,
        default=0.01,
        help="Local learning rate",
    )
    parser.add_argument("-ld", "--learning_rate_decay", type=bool, default=False)
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument("-gr", "--global_rounds", type=int, default=100)
    parser.add_argument(
        "-ls",
        "--local_epochs",
        type=int,
        default=2,
        help="Multiple update steps in one local epoch.",
    )
    parser.add_argument("-algo", "--algorithm", type=str, default="FedAvg")
    parser.add_argument(
        "-jr",
        "--join_ratio",
        type=float,
        default=1.0,
        help="Ratio of clients per round",
    )
    parser.add_argument(
        "-rjr",
        "--random_join_ratio",
        type=bool,
        default=False,
        help="Random ratio of clients per round",
    )
    parser.add_argument(
        "-nc", "--num_clients", type=int, default=20, help="Total number of clients"
    )
    parser.add_argument(
        "-pv", "--prev", type=int, default=0, help="Previous Running times"
    )
    parser.add_argument("-t", "--times", type=int, default=1, help="Running times")
    parser.add_argument(
        "-eg", "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
    )
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="items")
    parser.add_argument("-ab", "--auto_break", type=bool, default=False)
    parser.add_argument("-dlg", "--dlg_eval", type=bool, default=False)
    parser.add_argument("-dlgg", "--dlg_gap", type=int, default=100)
    parser.add_argument("-bnpc", "--batch_num_per_client", type=int, default=2)
    parser.add_argument("-nnc", "--num_new_clients", type=int, default=0)
    parser.add_argument("-ften", "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument("-fd", "--feature_dim", type=int, default=512)
    # practical
    parser.add_argument(
        "-cdr",
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Rate for clients that train but drop out",
    )
    parser.add_argument(
        "-tsr",
        "--train_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when training locally",
    )
    parser.add_argument(
        "-ssr",
        "--send_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when sending global model",
    )
    parser.add_argument(
        "-ts",
        "--time_select",
        type=bool,
        default=False,
        help="Whether to group and select clients at each round according to time cost",
    )
    parser.add_argument(
        "-tth",
        "--time_threthold",
        type=float,
        default=10000,
        help="The threthold for droping slow clients",
    )
    # HGLS / HGC components
    parser.add_argument(
        "-ugc",
        "--use_grad_compress",
        action="store_true",
        default=False,
        help="Enable gradient compression for head gradients",
    )
    parser.add_argument(
        "-uhg",
        "--use_head_grad",
        action="store_true",
        default=False,
        help="Enable head gradient aggregation enhancement",
    )
    parser.add_argument(
        "-bw",
        "--bit_width",
        type=int,
        default=16,
        choices=[8, 16, 32, 64],
        help="Bit width for gradient quantization",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))
    print("=" * 50)

    run(args)
    print("=" * 50)

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
