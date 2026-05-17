from tqdm import tqdm
import torch
import os 
import pandas as pd
import sys
from typing import List, Dict
import numpy as np
from multiprocessing import Pool, cpu_count


def curvature(embeddings, epsilon=1e-6):
    # embeddings=embeddings.to(torch.float32)
    e_roll = torch.roll(embeddings, -1, 0)
    v = e_roll - embeddings
    v_diff = v[:-1]
    v_diff_roll = torch.roll(v_diff, -1, 0)
    cos = torch.nn.CosineSimilarity(dim=1, eps=epsilon)
    output = cos(v_diff[:-1], v_diff_roll[:-1])
    output = torch.clamp(output, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.mean(torch.acos(output)).item()

