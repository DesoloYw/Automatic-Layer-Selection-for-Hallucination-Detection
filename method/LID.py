import os
import sys
import argparse
from typing import List, Dict, Optional
from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import faiss
from pathlib import Path
_PROJECT_ROOT = str(Path.cwd())
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.metrics import roc
from src.utils import last_token_stack,MODEL2HF,DATA2HF,MODEL2LAYER
# Defaults aligned with xp_prj1/get_activation.py



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute LID metrics over saved activations")
    parser.add_argument(
        "-m",
        "--model",
        default="llama_instruct",
        choices=MODEL2HF.keys(),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="hotpotqa",
        choices=DATA2HF.keys(),
    )
    parser.add_argument(
        "--basepath2prepared_data",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "prepared_data"),
        help="Base path where get_activation.py wrote activations and labels",
    )
    parser.add_argument(
        "-k",
        "--num-neighbors",
        type=int,
        default=50,
        dest="num_neighbors",
        help="Number of nearest neighbors used for LID computation.",
    )
    # Output directory for metrics CSVs
    parser.add_argument(
        "--results-base",
        dest="results_base",
        type=str,
        default=_RESULTS_DIR,
        help="Base directory to write results/metrics.csv",
    )
    parser.add_argument("--zero-shot", type=bool, default=True)
    parser.add_argument("--all_data",action="store_true",help='if set, run LID over all datasets')
    return parser


class LID:
    """
    Local Intrinsic Dimensionality (LID) scorer.

    Mirrors the computation in LID-HallucinationDetection/calculate_lid.py, but
    packaged as a reusable class and aligned with paths produced by
    xp_prj1/get_activation.py.

    Usage example:
        lid = LID(basepath2prepared_data=..., model="llama2-7b", dataset="triviaqa", num_neighbors=500)
        df = lid.compute_per_layer(write_csv=True)
    """

    def __init__(
        self,
        args: Optional[argparse.Namespace] = None,
    ) -> None:
        """Initialize LID configuration.

        You can either pass parameters directly, or provide an argparse
        Namespace via `args`, or omit both and this will call build_parser().parse_args().
        Explicit kwargs, when provided, override values from `args`.
        """


        # Prefer explicit kwargs, otherwise fall back to args
        self.args=args
        self.basepath2prepared_data = getattr(args, "basepath2prepared_data")
        self.model = getattr(args, "model")
        self.dataset = getattr(args, "dataset")
        self.num_neighbors = int(getattr(args, "num_neighbors"))
        self.zero_shot = bool(getattr(args, "zero_shot"))
        self.results_base = getattr(args, "results_base")

        # Resolve paths
        self.base = os.path.join(self.basepath2prepared_data, self.model, self.dataset)
        self.results_dir = os.path.join(self.results_base, self.model, self.dataset,f'{self.__class__.__name__}')

        self.train_set_dir=os.path.join(self.base, "train",'activations')
        self.train_acts_dir=os.path.join(self.train_set_dir,'predicted')
        self.train_labels_dir=os.path.join(self.train_set_dir,'labels')

        self.test_set_dir=os.path.join(self.base, "test",'activations')
        self.test_acts_dir=os.path.join(self.test_set_dir,'predicted')
        self.test_labels_dir=os.path.join(self.test_set_dir,'labels')

        self.log_name=f"lid_metrics_{self.args.num_neighbors}"
        torch.manual_seed(42)
        np.random.seed(42)
    # ------------------------
    # Public API
    # ------------------------

    def _load_layer_data(self, layer: int):
        train_acts_path=os.path.join(
            self.train_acts_dir, f"{self.model}_{self.dataset}_layer_{layer}_pred.pt"
        )
        train_label_path=os.path.join(
            self.train_labels_dir, f"{self.model}_{self.dataset}_all_layer_label.pt"
        )
        test_acts_path=os.path.join(        
            self.test_acts_dir, f"{self.model}_{self.dataset}_layer_{layer}_pred.pt"
        )
        test_label_path=os.path.join(
            self.test_labels_dir, f"{self.model}_{self.dataset}_all_layer_label.pt"
        )
        train_acts= torch.load(train_acts_path,weights_only=True)
        test_acts= torch.load(test_acts_path,weights_only=True) 
        train_labels= torch.load(train_label_path,weights_only=True)
        test_labels= torch.load(test_label_path,weights_only=True)
        return train_acts, train_labels, test_acts, test_labels
    
    def _compute_for_layer(self, layer: int) -> Optional[Dict[str, float]]:
        """
        Computes LID-related metrics for a single layer and returns a row dict containing:
          - 'layer'
          - 'auroc'
          - 'lid_mean'
          - 'lid_std'
        Ready to be appended directly in compute_per_layer.
        """
        train_acts, train_labels, test_acts, test_labels = self._load_layer_data(layer)

        # use only label=1 train_acts as the reference set
        pos_mask = (train_labels == 1)
        pos_train_acts = train_acts[pos_mask]

        # convert to numpy
        pos_train_acts = pos_train_acts.to(dtype=torch.float32, device="cpu").numpy()
        test_acts      = test_acts.to(dtype=torch.float32, device="cpu").numpy()
        test_labels    = test_labels.numpy()

        if pos_train_acts.shape[0] == 0:
            # no positive samples in edge case — skip this layer
            return None

        k = int(min(pos_train_acts.shape[0], self.num_neighbors))

        lids = self._lid_new(
            feats=test_acts,
            num_neighbors=[k],
            ref_feats=[pos_train_acts],
        )
        lids = np.asarray(lids)

        auroc    = roc(test_labels, -lids)
        lid_mean = float(np.mean(lids))
        lid_std  = float(np.std(lids))

        # return the row dict directly (including the layer field)
        return {
            "layer":   int(layer),
            "auroc":   float(auroc),
            "lid_mean": lid_mean,
            "lid_std":  lid_std,
        }


    def _plot_per_layer_metrics(self, df) -> None:
        """
        Given a DataFrame with 'layer', 'lid_mean', 'auroc' columns, plots a dual-axis figure and saves it.

        Subclasses can override this method to plot different metrics:
            def _plot_per_layer_metrics(self, df, basename):
                ...
        """
        if df.empty:
            return

        layers   = df["layer"].values
        lid_mean = df["lid_mean"].values
        auroc    = df["auroc"].values

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # left axis: LID mean
        ax1.plot(layers, lid_mean, marker="o", linestyle="-", label="LID mean")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("LID mean")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # right axis: AUROC
        ax2 = ax1.twinx()
        ax2.plot(layers, auroc, marker="s", linestyle="-", label="AUROC", color="tab:orange")
        ax2.set_ylabel("AUROC")

        # merge legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        fig.tight_layout()
        fig_path = os.path.join(self.results_dir, f"{self.log_name}.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved LID metrics figure to {fig_path}")

    # ------------------------
    # Public API
    # ------------------------
    def compute_per_layer(self) -> pd.DataFrame:
        """
        Compute LID scores for each saved layer and return a DataFrame with
        per-layer AUROC and LID statistics.
        """
        layer_ids = list(range(MODEL2LAYER[self.model]))

        metrics_rows = []
        for layer in tqdm(layer_ids):
            row = self._compute_for_layer(layer)
            if row is not None:
                metrics_rows.append(row)

        df = pd.DataFrame(metrics_rows)
        self.save_data(df)


        print(f"{self.__class__.__name__} computation complete for model={self.model}, dataset={self.dataset}")
        return df
    
    def save_data(self,df):
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = os.path.join(self.results_dir, f"{self.log_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {self.__class__.__name__} metrics CSV to {csv_path}")
        self._plot_per_layer_metrics(df)


    # ------------------------
    # Core LID kernels (ported)
    # ------------------------
    @staticmethod
    def _lid_new(
        feats: np.ndarray,
        num_neighbors: List[int],
        faiss_instances: Optional[List[faiss.IndexFlatL2]] = None,
        ref_feats: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Compute LID of `feats` using `ref_feats` as the reference points.
        Returns a numpy array of shape (batch,).
        Ported from LID-HallucinationDetection/calculate_lid.py:_lid_new
        """
        if faiss_instances is None:
            if ref_feats is None:
                raise ValueError("Provide either faiss_instances or ref_feats.")
            faiss_instances = []
            for rf in ref_feats:
                inst = faiss.IndexFlatL2(rf.shape[1])
                inst.add(np.ascontiguousarray(rf))
                faiss_instances.append(inst)
        else:
            if ref_feats is not None:
                # Prefer the provided FAISS indices; ignore ref_feats
                pass

        k_avg_lids = []  # (num_ks, batch)
        for k in num_neighbors:
            bootstrap_avg_lids = []  # (num_bootstrap, batch)
            for faiss_instance in faiss_instances:
                b, _ = faiss_instance.search(feats, k)
                b = np.sqrt(b)

                # Remove self-distance if present
                filtered = []
                for row in b:
                    if row[0] < 1e-6:
                        filtered.append(np.expand_dims(row[1:], axis=0))
                    else:
                        filtered.append(np.expand_dims(row[:-1], axis=0))
                D = np.concatenate(filtered, axis=0)

                rk = np.max(D, axis=1)
                rk[rk == 0] = 1e-8
                lids = D / rk[:, None]
                lids = np.clip(lids, 1e-6, 1.0)

                logs = np.log(lids)
                denom = np.mean(logs, axis=1)
                denom = np.where(np.abs(denom) < 1e-5, 1e-5 * np.sign(denom), denom)
                lids = -1.0 / denom

                lids[np.isinf(lids)] = feats.shape[1]
                lids = np.nan_to_num(lids, nan=feats.shape[1])

                bootstrap_avg_lids.append(lids.tolist())
            bootstrap_avg_lids = np.array(bootstrap_avg_lids).mean(axis=0)
            k_avg_lids.append(bootstrap_avg_lids.tolist())
        k_avg_lids = np.array(k_avg_lids).mean(axis=0)
        return k_avg_lids

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.all_data:
        for dataset_name in ['coqa','hotpotqa','squad','triviaqa','psiloqa','math']:
            args.dataset=dataset_name
            lid=LID(args=args)
            df=lid.compute_per_layer()
    else:
        lid = LID(args=args)
        df = lid.compute_per_layer()
    # print(df)
