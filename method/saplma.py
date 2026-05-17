import os
import argparse
import json
import time
from typing import Optional, List, Dict
import torch.autograd as autograd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skdim.id import TwoNN
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from LID import build_parser,LID
from src.metrics import roc
from src.utils import last_token_stack,get_least_used_gpu,MODEL2LAYER
import copy
import matplotlib.pyplot as plt
# import wandb
# Known transformer layer counts for supported models


def build_saplma_parser() -> argparse.ArgumentParser:
    parser=build_parser()
    # parser.add_argument("--result_name", type=str, default="saplma_results.json", help="Name of the results file.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for optimizer."
    )
    parser.add_argument("-b","--batch_size",type=int,default=2048)
    return parser


class _LogReg(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.num_layers = len(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.net[0].weight.dtype)
        return self.net(x)

    @torch.no_grad()
    def last_hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns features before the final classification layer (Linear(64->2)): shape [B, 64]
        """
        x = x.to(self.net[0].weight.dtype)
        h = x
        # run through all modules except the last one (i.e., ReLU after 128->64)
        for layer in self.net[:-1]:
            h = layer(h)
        return h


class SAPLMA(LID):
    """
    Implementation of the method from "The Internal State of an LLM Knows When It's Lying"
    (Azaria & Mitchell, 2023): train a lightweight classifier on hidden layer
    activations to predict truthfulness/hallucination labels.

    This class consumes activations saved by xp_prj1/get_activation.py, using
    predicted_activations as features and labels as binary targets.
    """

    def __init__(
        self,
        args: Optional[argparse.Namespace] = None,
    ) -> None:
        super().__init__(args)

        self.layer_num=MODEL2LAYER[args.model]
        self.epochs = args.epochs
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay

    def run_saplma(self):
        """
        Trains a small classifier on activations for each layer, evaluates test AUROC,
        and saves per-layer metrics with timing info to CSV and a plot.
        """
        device = torch.device({
            'cuda': f'cuda:{get_least_used_gpu()}',
            'cpu': 'cpu'
        }['cuda' if torch.cuda.is_available() else 'cpu'])

        metrics_rows = []  # one row per layer: layer, auroc_saplma, best_val_loss, rgn, timing
        hidden_dim1 = None
        total_training_process_time_sec = 0.0
        total_best_val_loss_compute_time_sec = 0.0
        total_rgn_compute_time_sec = 0.0
        total_snr_compute_time_sec = 0.0

        for layer in tqdm(range(self.layer_num), desc="saplma for each layer"):
            train_acts, train_labels, test_acts, test_labels = self._load_layer_data(layer)

            # ========= prepare train / val / test data =========
            train_X_all = train_acts.to(device=device, dtype=torch.float32)   # [N_train, D]
            train_y_all = train_labels.to(device=device, dtype=torch.long)    # [N_train]

            test_X = test_acts.to(device=device, dtype=torch.float32)         # [N_test, D]
            test_y = test_labels.cpu().numpy()                                # [N_test]

            # ---- split train / val ----
            val_ratio = 0.1
            n_train_total = train_X_all.size(0)
            n_val = max(1, int(n_train_total * val_ratio))

            perm = torch.randperm(n_train_total, device=device)
            val_idx_local   = perm[:n_val]
            train_idx_local = perm[n_val:]

            train_X = train_X_all[train_idx_local]
            train_y = train_y_all[train_idx_local]
            val_X   = train_X_all[val_idx_local]
            val_y   = train_y_all[val_idx_local]

            # ========= define model, loss, optimizer =========
            model = _LogReg(train_X.shape[1]).to(device)
            if hidden_dim1 is None:
                hidden_dim1 = model.net[0].out_features

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            batch_size = self.args.batch_size

            train_loader = DataLoader(
                TensorDataset(train_X, train_y),
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(val_X, val_y),
                batch_size=batch_size,
                shuffle=False,
            )

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.1
            )

            best_val_loss = float("inf")
            best_state_dict = None

            # ========= training =========
            self._sync_device(device)
            training_process_start = time.perf_counter()
            for epoch in range(self.epochs):
                # ---- train epoch ----
                model.train()
                # train_loss_sum = 0.0
                # train_count = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    out = model(batch_X)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()

                    # train_loss_sum += loss.item() * batch_X.size(0)
                    # train_count += batch_X.size(0)

                # mean_train_loss = train_loss_sum / train_count

                # ---- validation epoch ----
                model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)

                        out = model(batch_X)
                        loss = criterion(out, batch_y)
                        val_loss_sum += loss.item() * batch_X.size(0)
                        val_count += batch_X.size(0)

                mean_val_loss = val_loss_sum / max(1, val_count)

                # save best val loss checkpoint
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

                scheduler.step()
            self._sync_device(device)
            training_process_time_sec = time.perf_counter() - training_process_start

            # ========= evaluate AUROC on test set using best val loss checkpoint =========
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            self._sync_device(device)
            best_val_loss_start = time.perf_counter()
            best_val_loss = self._compute_validation_loss(
                model=model,
                criterion=criterion,
                val_X=val_X,
                val_y=val_y,
                device=device,
                batch_size=batch_size,
            )
            self._sync_device(device)
            best_val_loss_compute_time_sec = time.perf_counter() - best_val_loss_start

            self._sync_device(device)
            rgn_start = time.perf_counter()
            rgn_layer = self._compute_rgn_for_layer(
                model=model,
                criterion=criterion,
                val_X=val_X,
                val_y=val_y,
                device=device,
                batch_size=1,
            )
            self._sync_device(device)
            rgn_compute_time_sec = time.perf_counter() - rgn_start

            self._sync_device(device)
            snr_start = time.perf_counter()
            snr_layer = self._compute_snr_for_layer(
                model=model,
                criterion=criterion,
                val_X=val_X,
                val_y=val_y,
                device=device,
                batch_size=1,
            )
            self._sync_device(device)
            snr_compute_time_sec = time.perf_counter() - snr_start

            total_time_sec = (
                training_process_time_sec
                + best_val_loss_compute_time_sec
                + rgn_compute_time_sec
                + snr_compute_time_sec
            )
            # id_lastfeat = self._twonn_id_from_last_features(
            #     model=model,
            #     X=train_X_all,
            #     device=device,
            #     batch_size=4096,
            #     max_points=20000,
            #     random_state=0,
            # )
            model.eval()
            test_dataset = TensorDataset(test_X)
            test_loader  = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

            pos_scores_list = []
            with torch.no_grad():
                for (batch_X,) in test_loader:
                    batch_X = batch_X.to(device)
                    logits = model(batch_X)                  # [b, 2]
                    probs  = torch.softmax(logits, dim=-1)   # [b, 2]
                    pos    = probs[:, 1]                     # positive class probability [b]
                    pos_scores_list.append(pos.cpu().numpy())

            pos_scores = np.concatenate(pos_scores_list, axis=0)  # [N_test]
            auroc_saplma = roc(test_y, pos_scores)

            # record one row of metrics for this layer
            metrics_rows.append(
                {
                    "layer": int(layer),
                    "auroc_saplma": float(auroc_saplma),
                    "best_val_loss": float(best_val_loss),
                    "rgn": float(rgn_layer),
                    "snr": float(snr_layer),
                    # "id_lastfeat_twonn": float(id_lastfeat),
                    "training_process_time_sec": float(training_process_time_sec),
                    "best_val_loss_compute_time_sec": float(best_val_loss_compute_time_sec),
                    "rgn_compute_time_sec": float(rgn_compute_time_sec),
                    "snr_compute_time_sec": float(snr_compute_time_sec),
                    "total_time_sec": float(total_time_sec),
                }
            )
            total_training_process_time_sec += training_process_time_sec
            total_best_val_loss_compute_time_sec += best_val_loss_compute_time_sec
            total_rgn_compute_time_sec += rgn_compute_time_sec
            total_snr_compute_time_sec += snr_compute_time_sec

        # ========= save results =========
        os.makedirs(self.results_dir, exist_ok=True)

        results_df = pd.DataFrame(
            metrics_rows,
            columns=[
                "layer",
                "auroc_saplma",
                "best_val_loss",
                "rgn",
                "snr",
                "id_lastfeat_twonn",
                "training_process_time_sec",
                "best_val_loss_compute_time_sec",
                "rgn_compute_time_sec",
                "snr_compute_time_sec",
                "total_time_sec",
            ],
        )
        csv_path = os.path.join(
            self.results_dir,
            f"auroc_valLoss_rgn_snr_lr{self.lr}_epochs{self.epochs}"
            f"_h1={hidden_dim1}_layer{self.layer_num}.csv",
        )
        results_df.to_csv(csv_path, index=False)
        print(f"Saved SAPLMA metrics CSV to {csv_path}")

        summary_path = os.path.join(
            self.results_dir,
            f"auroc_valLoss_rgn_snr_lr{self.lr}_epochs{self.epochs}"
            f"_h1={hidden_dim1}_layer{self.layer_num}_summary.json",
        )
        summary = {
            "total_training_process_time_sec": float(total_training_process_time_sec),
            "total_best_val_loss_compute_time_sec": float(total_best_val_loss_compute_time_sec),
            "total_rgn_compute_time_sec": float(total_rgn_compute_time_sec),
            "total_snr_compute_time_sec": float(total_snr_compute_time_sec),
            "total_time_sec": float(
                total_training_process_time_sec
                + total_best_val_loss_compute_time_sec
                + total_rgn_compute_time_sec
                + total_snr_compute_time_sec
            ),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved SAPLMA timing summary JSON to {summary_path}")

        # plot
        self._plot_layer_metrics(results_df, hidden_dim1)
        print(f"SAPLMA computation complete for model={self.model}, dataset={self.dataset}")

    @staticmethod
    def _sync_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)



    def _compute_validation_loss(
        self,
        model: nn.Module,
        criterion: nn.Module,
        val_X: torch.Tensor,
        val_y: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> float:
        model.eval()

        val_dataset = TensorDataset(val_X, val_y)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                out = model(batch_X)
                loss = criterion(out, batch_y)
                val_loss_sum += loss.item() * batch_X.size(0)
                val_count += batch_X.size(0)

        return float(val_loss_sum / max(1, val_count))

    def _compute_rgn_for_layer(
        self,
        model: nn.Module,
        criterion: nn.Module,
        val_X: torch.Tensor,
        val_y: torch.Tensor,
        device: torch.device,
        batch_size: int,
        eps: float = 1e-12,
    ) -> float:
        """
        Runs backward on the validation set loss of the trained probe
        to estimate RGN for this layer: ||g||_2 / ||theta||_2.
        """
        model.eval()

        val_dataset = TensorDataset(val_X, val_y)
        val_loader  = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        rgn_vals = []

        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # clear gradients
            model.zero_grad(set_to_none=True)

            # forward + backward
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()

            # ===== collect all gradients & parameters, then flatten and concatenate =====
            all_g = []
            all_theta = []
            for p in model.parameters():
                # if p.grad is None:
                #     continue
                # cast to float() to avoid numerical issues with half/bfloat16
                all_g.append(p.grad.detach().float().view(-1))
                all_theta.append(p.detach().float().view(-1))

            g_flat = torch.cat(all_g)        # [total_num_params]
            theta_flat = torch.cat(all_theta)

            g_norm = torch.linalg.norm(g_flat)         # sqrt(sum_j g_j^2)
            theta_norm = torch.linalg.norm(theta_flat) # sqrt(sum_j theta_j^2)
            rgn_batch = (g_norm / (theta_norm + eps)).item()
            rgn_vals.append(rgn_batch)
            
        # average over batches to get the global RGN for this layer's probe
        rgn_layer = float(torch.tensor(rgn_vals).mean().item())

        model.zero_grad(set_to_none=True)
        return rgn_layer

    def _compute_snr_for_layer(
        self,
        model: nn.Module,
        criterion: nn.Module,
        val_X: torch.Tensor,
        val_y: torch.Tensor,
        device: torch.device,
        batch_size: int,
        eps: float = 1e-12,
    ) -> float:
        """
        Runs backward on the validation set loss of the trained probe
        to estimate SNR for this layer: (sum_j g_j)^2 / sum_j g_j^2.
        """
        model.eval()

        val_dataset = TensorDataset(val_X, val_y)
        val_loader  = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        snr_vals = []

        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            model.zero_grad(set_to_none=True)

            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()

            all_g = []
            for p in model.parameters():
                all_g.append(p.grad.detach().float().view(-1))

            g_flat = torch.cat(all_g)

            sum_g = g_flat.sum()
            sum_g2 = (g_flat * g_flat).sum()
            snr_batch = ((sum_g * sum_g) / (sum_g2 + eps)).item()
            snr_vals.append(snr_batch)

        snr_layer = float(torch.tensor(snr_vals).mean().item())

        model.zero_grad(set_to_none=True)
        return snr_layer

    def _twonn_id_from_last_features(
        self,
        model: _LogReg,
        X: torch.Tensor,
        device: torch.device,
        batch_size: int = 4096,
        max_points: int = 20000,
        random_state: int = 0,
    ) -> float:
        """
        Estimates the intrinsic dimension of the probe's last-layer features using skdim TwoNN.
        - X: [N, D] (val_X or train_X_all recommended)
        - max_points: cap on points to prevent TwoNN from being too slow/memory-heavy (default 20k)
        """
        model.eval()

        # ---- optional downsampling ----
        N = X.size(0)
        if N > max_points:
            g = torch.Generator(device=X.device)
            g.manual_seed(random_state)
            idx = torch.randperm(N, generator=g, device=X.device)[:max_points]
            X_use = X[idx]
        else:
            X_use = X

        feats_list = []
        loader = DataLoader(TensorDataset(X_use), batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(device)
                f = model.last_hidden_features(bx)   # [b, 64]
                feats_list.append(f.detach().float().cpu().numpy())

        feats = np.concatenate(feats_list, axis=0)  # [N_use, 64]

        # ---- fit skdim TwoNN ----
        estimator = TwoNN()
        estimator.fit(feats)
        id_val = float(estimator.dimension_)
        return id_val

    def _plot_layer_metrics(self, df: pd.DataFrame, hidden_dim1: int) -> None:
        """
        df must contain columns: ["layer", "auroc_saplma", "best_val_loss", "rgn", "snr"]

        Left axis: AUROC
        Right axis: best_val_loss / rgn / snr each min-max normalized to [0, 1],
                    plotted on the same axis to compare their shapes across layers.
        """
        layers     = df["layer"].values
        aurocs     = df["auroc_saplma"].values
        val_losses = df["best_val_loss"].values
        rgns       = df["rgn"].values
        snrs       = df["snr"].values
        ID=df['id_lastfeat_twonn'].values
        def minmax_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            x_min = float(x.min())
            x_max = float(x.max())
            denom = max(x_max - x_min, eps)
            return (x - x_min) / denom
        
        val_losses_norm = minmax_norm(val_losses)
        rgns_norm       = minmax_norm(rgns)
        snrs_norm       = minmax_norm(snrs)

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # ---------- left axis: AUROC ----------
        ln1 = ax1.plot(
            layers,
            aurocs,
            marker="o",
            linestyle="-",
            label="AUROC (SAPLMA)",
        )
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("AUROC (SAPLMA)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # ---------- right axis: normalized ValLoss / RGN / SNR ----------
        ax2 = ax1.twinx()

        # ln2 = ax2.plot(
        #     layers,
        #     val_losses_norm,
        #     marker="s",
        #     linestyle="-",
        #     color="tab:red",
        #     label="Norm. Val Loss",
        # )
        # ln3 = ax2.plot(
        #     layers,
        #     rgns_norm,
        #     marker="^",
        #     linestyle="-",
        #     color="tab:green",
        #     label="Norm. RGN",
        # )
        # ln4 = ax2.plot(
        #     layers,
        #     snrs_norm,
        #     marker="d",
        #     linestyle="-",
        #     color="tab:purple",
        #     label="Norm. SNR",
        # )
        ln4 = ax2.plot(
            layers,
            ID,
            marker="s",
            linestyle="-",
            color="tab:red",
            label="ID",
        )
        ax2.set_ylabel("Normalized Val Loss / RGN / SNR (0–1)")

        # ---------- merge legend ----------
        # lines  = ln1 + ln2 + ln3 + ln4
        lines=ln1+ln4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        fig.tight_layout()
        fig_path = os.path.join(
            self.results_dir,
            f"auroc_valLoss_rgn_snr_lr{self.lr}_epochs{self.epochs}"
            f"_h1={hidden_dim1}_layer{self.layer_num}.png",
        )
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()

        print(f"Saved SAPLMA AUROC + ValLoss + RGN + SNR figure to {fig_path}")

if __name__ == "__main__":
    args = build_saplma_parser().parse_args()
    if args.all_data:
        for dataset_name in ['coqa','hotpotqa','squad','triviaqa','psiloqa','math']:
            args.dataset=dataset_name
            sap = SAPLMA(args=args)
            sap.run_saplma()
    else:
        sap = SAPLMA(args=args)
        sap.run_saplma()
