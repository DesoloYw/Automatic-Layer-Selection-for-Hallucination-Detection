import numpy as np
import faiss
from typing import Dict, Optional
from skdim.id import TwoNN
def estimate_id_twonn_faiss_skdim(
    X: np.ndarray,
    use_gpu: bool,
    gpu_device: Optional[int],
    gpu_resources=None,
) -> float:
    # X is the batch of representations, with the shape of (n_samples, n_features)
    X = np.asarray(X, dtype=np.float32)
    _, dim = X.shape

    index = faiss.IndexFlatL2(dim)
    if use_gpu:
        device_id = 0 if gpu_device is None else gpu_device
        index = faiss.index_cpu_to_gpu(gpu_resources, device_id, index)

    index.add(X)
    dists, _ = index.search(X, 3)
    r1 = np.sqrt(np.maximum(dists[:, 1], 0.0))
    r2 = np.sqrt(np.maximum(dists[:, 2], 0.0))

    twonn_input = np.column_stack([r1, r2]).astype(np.float64, copy=False)
    est = TwoNN(dist=True)
    est.fit(twonn_input)
    return float(est.dimension_)