from matplotlib import pyplot as plt
import torch
import os 
import pandas as pd
import sys
import time
from typing import List, Dict
import numpy as np
from LID import LID,build_parser
from src.utils import get_least_used_gpu, last_token_stack
from src.metrics import roc

EPSILON = 1e-7  # suitable for float32

class RankME(LID):
    def __init__(self, args):
        super().__init__(args)
        self.log_name=f"rank_me"

    def _compute_for_layer(self, layer: int,) -> Dict[str, float]:
        train_acts, _, test_acts, _ = self._load_layer_data(layer)
        compute_start = time.perf_counter()
        device=get_least_used_gpu()
        train_acts=train_acts.to(device=device, dtype=torch.float32)
        rank_me = calc_rankme(train_acts)
        rank_me_time_sec = time.perf_counter() - compute_start
        return {
            'layer': int(layer),
            'rank_me': rank_me,
            'rank_me_time_sec': float(rank_me_time_sec),
        }
    
    def _plot_per_layer_metrics(self, df):
        layers=df['layer'].values
        rank_me=df['rank_me'].values
        plt.figure(figsize=(8, 5))
        plt.plot(layers, rank_me, marker="o")
        plt.xlabel("Layer")
        plt.ylabel("rank_me")
        plt.grid(True, linestyle="--", alpha=0.5)
        fig_path = os.path.join(self.results_dir, f"{self.log_name}.png")
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved rankme metrics figure to {fig_path}")
        plt.close()

def calc_rankme(embeddings, epsilon: float = EPSILON) -> float:

    embeddings = embeddings / torch.norm(
                    embeddings, dim=1, keepdim=True
    ) 

    _u, s, _vh = torch.linalg.svd(
        embeddings, full_matrices=False
    )  # s.shape = (min(N, K),)

    p = (s / torch.sum(s, axis=0)) + epsilon
    entropy = -torch.sum(p * torch.log(p))
    rankme = torch.exp(entropy).item()

    return rankme

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.all_data:
        for dataset_name in ['coqa','hotpotqa','squad','triviaqa','math','psiloqa']:
            args.dataset=dataset_name
            rkm = RankME(args=args)
            df = rkm.compute_per_layer()
    else:
        rkm = RankME(args=args)
        df = rkm.compute_per_layer()
