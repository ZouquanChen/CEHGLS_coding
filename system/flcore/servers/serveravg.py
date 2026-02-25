import time
import torch
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server

from pympler import asizeof


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.train_base_m = []
        self.train_base_glsm = []
        self.train_base_chgc = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            grads = []
            results_set = []
            m_sum = 0
            m_grad_sum = 0
            m_grad_c_sum = 0

            # Client local training phase (upload)
            for client in self.selected_clients:
                client.train()
                m_sum += self.mm(client.model.base)

                if self.use_grad_compress:
                    results = client.compress_grad()
                    results_set.append(results)
                    m_grad_sum += self.mm(client.model.base) + results[4]
                    m_grad_c_sum += self.mm(client.model.base) + results[5]
                else:
                    grads.append(
                        [
                            param.grad.data.clone()
                            for param in client.model.head.parameters()
                        ]
                    )
                    head_grad_mem = sum(
                        p.grad.data.numel() * p.grad.data.element_size()
                        for p in client.model.head.parameters()
                        if p.grad is not None
                    )
                    m_grad_sum += self.mm(client.model.base) + head_grad_mem

            # Server receive parameters phase
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # Client receive adjustment phase (download)
            if self.use_grad_compress:
                for client, result_tmp in zip(self.selected_clients, results_set):
                    grads.append(
                        client.decompress_grad(
                            result_tmp[0],
                            result_tmp[1],
                            result_tmp[2],
                            result_tmp[3],
                            result_tmp[6],
                        )
                    )

            # Head gradient aggregation
            if len(grads) > 0:
                if self.use_head_grad:
                    # HGLS: Server broadcasts all head gradients, client performs linear search
                    if self.use_grad_compress:
                        # Compressed mode: each client downloads base + all N compressed head gradients
                        total_orig_size = sum(r[4] for r in results_set)
                        total_comp_size = sum(r[5] for r in results_set)
                        for client in self.selected_clients:
                            m_sum += self.mm(client.model.base)
                            m_grad_sum += self.mm(client.model.base) + total_orig_size
                            m_grad_c_sum += self.mm(client.model.base) + total_comp_size
                            client.set_grad(grads)
                            client.head_grad_liner_search()
                    else:
                        # Uncompressed mode: each client downloads base + all N original head gradients
                        grad_mem = self.get_grad_memory(grads)
                        for client in self.selected_clients:
                            m_sum += self.mm(client.model.base)
                            m_grad_sum += self.mm(client.model.base) + grad_mem
                            client.set_grad(grads)
                            client.head_grad_liner_search()
                else:
                    # Simple aggregation: average all head gradients then download and apply
                    num_grads = len(grads)
                    avg_grad = [
                        sum(grads[k][j] for k in range(num_grads)) / num_grads
                        for j in range(len(grads[0]))
                    ]
                    avg_grad_mem = sum(g.numel() * g.element_size() for g in avg_grad)

                    if self.use_grad_compress:
                        # Compressed mode: each client downloads base + 1 averaged head gradient (compressed)
                        avg_comp_size = sum(r[5] for r in results_set) / len(
                            results_set
                        )
                        for client in self.selected_clients:
                            m_sum += self.mm(client.model.base)
                            m_grad_sum += self.mm(client.model.base) + avg_grad_mem
                            m_grad_c_sum += self.mm(client.model.base) + avg_comp_size
                    else:
                        # Uncompressed mode: each client downloads base + 1 averaged head gradient
                        for client in self.selected_clients:
                            m_sum += self.mm(client.model.base)
                            m_grad_sum += self.mm(client.model.base) + avg_grad_mem

            # Calculate compression ratio
            ratio = 0.0
            if self.use_grad_compress and len(results_set) > 0:
                total_orig = sum(r[4] for r in results_set)
                total_comp = sum(r[5] for r in results_set)
                ratio = total_orig / total_comp if total_comp > 0 else 0.0
                self.train_compress_ratio.append(ratio)

            self.Budget.append(time.time() - s_t)
            self.train_base_m.append(m_sum)
            self.train_base_glsm.append(m_grad_sum)
            self.train_base_chgc.append(m_grad_c_sum)

            print(f"Round {i} summary:")
            print(f"  FedAvg baseline communication: {m_sum / (1024**2):.2f} MB")
            print(f"  + Head gradient (uncompressed): {m_grad_sum / (1024**2):.2f} MB")
            if self.use_grad_compress:
                print(
                    f"  + Head gradient (compressed): {m_grad_c_sum / (1024**2):.2f} MB"
                )
                print(
                    f"  Head gradient compression ratio: {ratio:.4f}x (original/compressed)"
                )

            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        best_idx = int(np.argmax(self.rs_test_acc))
        print(
            f"{max(self.rs_test_acc)} (std: {self.rs_test_acc_std[best_idx]:.4f}, round: {best_idx})"
        )
        print("\nAverage time cost per round.")
        print(
            f"FedAvg baseline communication: {sum(self.train_base_m) / (1024**2):.2f} MB"
        )
        print(
            f"+ Head gradient (uncompressed): {sum(self.train_base_glsm) / (1024**2):.2f} MB"
        )
        if self.use_grad_compress:
            print(
                f"+ Head gradient (compressed): {sum(self.train_base_chgc) / (1024**2):.2f} MB"
            )
            if self.train_compress_ratio:
                avg_ratio = sum(self.train_compress_ratio) / len(
                    self.train_compress_ratio
                )
                print(
                    f"Head gradient average compression ratio: {avg_ratio:.4f}x (original/compressed)"
                )
        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
