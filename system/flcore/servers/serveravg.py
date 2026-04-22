import sys
import time
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

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

            for client in self.selected_clients:
                client.train()

                if self.use_grad_compress:
                    results = client.compress_grad()
                    results_set.append(results)
                else:
                    grads.append(
                        [
                            param.grad.data.clone()
                            for param in client.model.head.parameters()
                        ]
                    )

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

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

            if len(grads) > 0:
                if self.use_head_grad:
                    for client in self.selected_clients:
                        client.set_grad(grads)
                        client.head_grad_liner_search()

            self.Budget.append(time.time() - s_t)

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

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
