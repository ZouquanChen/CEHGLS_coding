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

import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, last_n=10):
    test_acc, test_acc_std = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    best_round_std = []
    for i in range(times):
        best_idx = int(np.argmax(test_acc[i]))
        max_accurancy.append(test_acc[i][best_idx])
        if test_acc_std[i] is not None and len(test_acc_std[i]) > best_idx:
            best_round_std.append(test_acc_std[i][best_idx])

    if times > 1:
        print("std for best accurancy (across runs):", np.std(max_accurancy))
    else:
        tail = test_acc[0][-last_n:]
        print(f"std for best accurancy (last {last_n} rounds):", np.std(tail))

    if best_round_std:
        print("mean client std at best round:", np.mean(best_round_std))

    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    test_acc_std = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        acc, acc_std = read_data_then_delete(file_name, delete=False)
        test_acc.append(np.array(acc))
        test_acc_std.append(acc_std)

    return test_acc, test_acc_std


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))
        rs_test_acc_std = np.array(hf.get('rs_test_acc_std')) if 'rs_test_acc_std' in hf else None

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc, rs_test_acc_std