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

import copy
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import os

from bitarray import bitarray
from pympler import asizeof
from torch import cosine_similarity
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data

def quantize_matrix(matrix, bit_width=16):
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    row_length = np.linalg.norm(uns_matrix, axis=1)
    r_max = np.max(row_length)
    r_max = np.ceil(r_max)
    if r_max == 0:
        r_max = 1
    index = np.linspace(0, r_max, num=bit_width, endpoint=True)
    uns_matrix = np.digitize(row_length, index)
    result = np.zeros_like(index, dtype=int)
    for i in range(len(uns_matrix)):
        result[uns_matrix[i]] += 1
    code = [1 if i != 0 else 0 for i in result]
    for i in range(len(og_sign)):
        for j in range(len(og_sign[i])):
            if og_sign[i][j] == -1:
                og_sign[i][j] = 0

    return result, og_sign, r_max, code

def unquantize_matrix(matrix, bit_width=16, r_max=0.5):
    matrix = matrix.astype(int)
    og_sign = np.sign(matrix)
    uns_matrix = matrix * og_sign
    uns_result = uns_matrix * r_max / (pow(2, bit_width - 1) - 1.0)
    result = og_sign * uns_result
    return result.astype(np.float32)

def numpy_array_to_bitstream(arr):
    flat_arr = arr.flatten().astype(np.uint8)

    bit_stream = bitarray()

    bit_stream.extend(flat_arr)

    return bit_stream.tobytes()

class Client(object):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs["train_slow"]
        self.send_slow = kwargs["send_slow"]
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        self.grad_list = []
        self.grad_shape = None
        self.code_shape = None
        self.bit_width = args.bit_width

    def set_grad(self, grads):
        self.grad_list = grads

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def set_base_parameters(self, model):
        for new_param, old_param in zip(
            model.base.parameters(), self.model.base.parameters()
        ):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(
            item,
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"),
        )

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt")
        )

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_like_list(self, grads):
        local_grads = [param.grad.data for param in self.model.head.parameters()]
        like_list = []
        for grad in grads:
            sim = cosine_similarity(
                grad[0].view(1, -1), local_grads[0].view(1, -1)
            ).item()
            like_list.append(sim)
        return like_list

    def local_test(self, test_loader, test_model=None):
        model = self.model if test_model is None else test_model
        model.eval()
        device = self.device
        correct = 0
        total = len(test_loader.dataset)
        loss_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss_test.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        return acc, sum(loss_test) / len(loss_test)

    def head_grad_liner_search(self):
        grads = self.grad_list

        local_grads = [param.grad.data for param in self.model.head.parameters()]
        G_H = torch.cat([g.flatten() for g in local_grads])
        G_H_norm = torch.norm(G_H)

        if G_H_norm == 0:
            return

        train_loader = self.load_train_data()

        original_state = copy.deepcopy(self.model.state_dict())
        best_model_state = copy.deepcopy(original_state)

        _, best_loss = self.local_test(train_loader, self.model)

        w_t_state = copy.deepcopy(original_state)

        tmp_model = copy.deepcopy(self.model)

        for j in range(len(grads)):
            g_j = grads[j]

            g_j_flat = torch.cat([g.flatten() for g in g_j])

            inner_product = torch.dot(g_j_flat, G_H)
            if inner_product <= 0:
                continue

            g_j_norm = torch.norm(g_j_flat)
            if g_j_norm == 0:
                continue
            alpha_j = (inner_product / (g_j_norm * G_H_norm)).item()

            tmp_model.load_state_dict(w_t_state)
            with torch.no_grad():
                for param, grad in zip(tmp_model.head.parameters(), g_j):
                    param.data -= alpha_j * grad

            _, new_loss = self.local_test(train_loader, tmp_model)

            if new_loss < best_loss:
                best_loss = new_loss
                w_t_state = copy.deepcopy(tmp_model.state_dict())
                best_model_state = copy.deepcopy(w_t_state)

        self.model.load_state_dict(best_model_state)

    def compress_grad(self):
        grads = [
            param.grad.data.cpu().numpy() for param in self.model.head.parameters()
        ]
        grads[1] = grads[1][:, np.newaxis]
        merged_array = np.hstack((grads[0], grads[1]))

        self.grad_shape = merged_array.shape
        result, og_sign, r_max, code = self.quantize_grad_batch(
            merged_array, bit_width=self.bit_width
        )
        r_max = int(r_max)
        sum = 0
        for grad in grads:
            sum += self.total_size_of_numpy_array(grad)
        compress_sum = (
            self.total_size_of_numpy_array(result)
            + asizeof.asizeof(r_max)
            + len(code)
            + len(og_sign)
        )
        print(f"compress_sum: {compress_sum:.2f} B")
        print(f"sum: {sum:.2f} B")
        return result, og_sign, r_max, code, sum, compress_sum, self.bit_width

    def quantize_grad_batch(self, A, batch_size=10, bit_width=None):
        og_shape = A.shape

        if len(A.shape) == 1:
            A = np.expand_dims(A, axis=0)

        bw = bit_width if bit_width is not None else self.bit_width

        result, og_sign, r_max, code = quantize_matrix(A, bw)

        self.code_shape = np.array(code).shape

        code = numpy_array_to_bitstream(np.array(code))
        og_sign = numpy_array_to_bitstream(og_sign)

        return result, og_sign, r_max, code

    @staticmethod
    def total_size_of_numpy_array(arr):
        try:
            if isinstance(arr, np.ndarray):
                return arr.nbytes
            elif isinstance(arr, torch.Tensor):
                return arr.element_size() * arr.numel()
            else:
                return sys.getsizeof(arr)
        except Exception as e:
            print(f"Warning: Cannot calculate array size, error: {e}")
            return 0

    def decompress_grad(self, quantized_result, og_sign, r_max, code, bit_width):
        decoded_code = self.bitstream_to_numpy_array(code)

        decoded_matrix = self.decode_quantized_matrix(
            quantized_result, decoded_code, bit_width, r_max
        )

        og_sign = self.bitstream_to_numpy_array(og_sign)
        og_sign = og_sign[: self.grad_shape[0] * self.grad_shape[1]]
        og_sign[og_sign == 0] = -1
        og_sign = og_sign.reshape(self.grad_shape)

        decoded_matrix = decoded_matrix * og_sign
        grads = [
            torch.Tensor(decoded_matrix[:, :-1]).to(self.device),
            torch.Tensor(decoded_matrix[:, -1]).to(self.device),
        ]

        return grads

    def decode_quantized_matrix(self, result, code, bit_width, r_max):
        index = np.linspace(0, r_max, num=bit_width, endpoint=True)

        decoded_values = np.zeros(bit_width)
        for i, val in enumerate(code):
            if i < bit_width and val:
                decoded_values[i] = index[i]

        num_cols = self.grad_shape[-1]
        sqrt_cols = np.sqrt(num_cols)
        grads = []
        for i in range(bit_width):
            val = decoded_values[i] / sqrt_cols
            for j in range(result[i]):
                grads.append([val for _ in range(num_cols)])
        return np.array(grads) if len(grads) > 0 else np.zeros(self.grad_shape)

    @staticmethod
    def bitstream_to_numpy_array(bit_stream):
        bit_arr = bitarray()
        bit_arr.frombytes(bit_stream)

        result_list = [int(bit) for bit in bit_arr]
        return np.array(result_list)
