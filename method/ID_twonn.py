
import argparse
import os
import warnings
from typing import Dict, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skdim.id import TwoNN
from tqdm import tqdm

from LID import LID, build_parser
from src.utils import MODEL2LAYER, get_least_used_gpu


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def _faiss_gpu_supported() -> bool:
    return hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")


def estimate_id_twonn_faiss_skdim(
    X: np.ndarray,
    use_gpu: bool = False,
    gpu_device: Optional[int] = None,
    gpu_resources=None,
) -> float:
    """Estimate intrinsic dimension with TwoNN (skdim) using FAISS for neighbour search.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        use_gpu: Whether to use a FAISS GPU index.
        gpu_device: GPU device ID (ignored when use_gpu=False).
        gpu_resources: Pre-allocated faiss.StandardGpuResources (optional).

    Returns:
        Estimated intrinsic dimension as a float.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    _, dim = X.shape

    index = faiss.IndexFlatL2(dim)
    if use_gpu:
        device_id = 0 if gpu_device is None else gpu_device
        res = gpu_resources if gpu_resources is not None else faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, device_id, index)

    index.add(X)
    dists, _ = index.search(X, 3)          # columns: self, 1st neighbour, 2nd neighbour
    r1 = np.sqrt(np.maximum(dists[:, 1], 0.0))
    r2 = np.sqrt(np.maximum(dists[:, 2], 0.0))

    twonn_input = np.column_stack([r1, r2]).astype(np.float64, copy=False)
    est = TwoNN(dist=True)
    est.fit(twonn_input)
    return float(est.dimension_)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class IntrinsicDimension(LID):
    """Per-layer intrinsic dimension estimator using TwoNN + FAISS."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self.log_name = "id_twonn_faiss_skdim"
        self.results_dir = os.path.join(self.results_dir, "twonn_faiss_skdim")
        os.makedirs(self.results_dir, exist_ok=True)

        self._use_gpu, self._gpu_device = self._resolve_faiss_device()
        self._gpu_resources = faiss.StandardGpuResources() if self._use_gpu else None

    # ------------------------------------------------------------------
    def _resolve_faiss_device(self) -> tuple[bool, Optional[int]]:
        use_gpu = torch.cuda.is_available() and _faiss_gpu_supported()
        gpu_device = None
        if use_gpu:
            try:
                gpu_device = get_least_used_gpu()
            except Exception as exc:
                warnings.warn(
                    f"Could not select GPU for FAISS ({exc}); using CPU.",
                    RuntimeWarning,
                )
                use_gpu = False
        return use_gpu, gpu_device

    def _load_train_acts(self, layer: int) -> torch.Tensor:
        path = os.path.join(
            self.train_acts_dir,
            f"{self.model}_{self.dataset}_layer_{layer}_pred.pt",
        )
        return torch.load(path, weights_only=True)

    def _prepare_features(self, acts: torch.Tensor) -> np.ndarray:
        return acts.detach().cpu().float().numpy()

    def _compute_for_layer(self, layer: int) -> Dict[str, float]:
        X = self._prepare_features(self._load_train_acts(layer))
        id_val = estimate_id_twonn_faiss_skdim(
            X,
            use_gpu=self._use_gpu,
            gpu_device=self._gpu_device,
            gpu_resources=self._gpu_resources,
        )
        return {"layer": int(layer), "ID_twonn_faiss_skdim": id_val}

    def compute_per_layer(self) -> pd.DataFrame:
        rows = [self._compute_for_layer(layer) for layer in tqdm(range(MODEL2LAYER[self.model]))]
        df = pd.DataFrame(rows)
        self.save_data(df)
        print(f"Done  model={self.model}  dataset={self.dataset}")
        return df

    def _plot_per_layer_metrics(self, df: pd.DataFrame) -> None:
        plt.figure(figsize=(9, 5))
        plt.plot(df["layer"].values, df["ID_twonn_faiss_skdim"].values, marker="o", linewidth=1.4)
        plt.xlabel("Layer")
        plt.ylabel("Intrinsic Dimension (TwoNN-FAISS-skdim)")
        plt.grid(True, linestyle="--", alpha=0.4)
        fig_path = os.path.join(self.results_dir, f"{self.log_name}.png")
        plt.savefig(fig_path, bbox_inches="tight", dpi=200)
        print(f"Saved figure to {fig_path}")
        plt.close()

    def save_data(self, df: pd.DataFrame, **_) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = os.path.join(self.results_dir, f"{self.log_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
        self._plot_per_layer_metrics(df)


def build_parser_id() -> argparse.ArgumentParser:
    return build_parser()


if __name__ == "__main__":
    args = build_parser_id().parse_args()
    datasets = ["coqa", "triviaqa", "hotpotqa", "squad", "psiloqa"]
    if args.all_data:
        for ds in datasets:
            args.dataset = ds
            IntrinsicDimension(args).compute_per_layer()
    else:
        IntrinsicDimension(args).compute_per_layer()
