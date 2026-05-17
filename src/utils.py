import pynvml
import pandas as pd
from typing import Any, Dict, List, Optional
import torch
from tqdm import tqdm
import ast
import os
import tempfile
import numpy as np
from src.metrics import rouge_L
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL2HF={
    "llama2-7b":"meta-llama/Llama-2-7b-hf", 
    "mistral-7b-0.3":"mistralai/Mistral-7B-v0.3",
    "llama2-13b":"meta-llama/Llama-2-13b-hf",
    "llama3.1-8b":"meta-llama/Llama-3.1-8B",
    "llama3.2_1b_instruct":"/bigtemp/hqr4gx/Model/Llama-3.2-1B-Instruct",
    "llama3.2_3b_instruct":"/bigtemp/hqr4gx/Model/Llama-3.2-3B-Instruct",
    "mistral_instruct":"mistralai/Mistral-7B-Instruct-v0.3",
    "llama_instruct":"meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-1b":"/bigtemp/hqr4gx/Model/Llama-3.2-1B",
    "llama3.2-3b":"/bigtemp/hqr4gx/Model/Llama-3.2-3B",
    "qwen2.5_14b_instruct":"/bigtemp/hqr4gx/Model/Qwen2.5-14B-Instruct",
    "qwen2.5_14b":"/bigtemp/hqr4gx/Model/Qwen2.5-14B"
}
MODEL2LAYER = {
    "llama2-7b": 32,
    "mistral-7b": 32,
    "llama2-13b": 40,
    "llama3.1-8b":32,
    "llama3.2-3b":28,
    "llama3.2-1b":16,
    "llama_instruct":32,
    "llama3.2_3b_instruct":28,
    "llama3.2_1b_instruct":16,
    "mistral_instruct":32,
    "qwen2.5_14b_instruct":48
}

DATA2HF={
    "coqa":"stanfordnlp/coqa",
    "triviaqa":None, 
    "hotpotqa":None,

    "squad":"rajpurkar/squad",
    "hotpotqa_c":None,
    'psiloqa':None,
    "halueval_summary":None,
    "cnn_dailymail":"abisee/cnn_dailymail",
    "odd_man_out":None,
    "trec":"trec",
    "dbpedia_14":"dbpedia_14",
    "ag_news":"ag_news"
}
def get_least_used_gpu() -> int:
    """
    Returns the local GPU index (0-based) with the least used memory among visible GPUs.
    Raises RuntimeError if no GPU is available.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_physical_ids: Optional[List[int]] = None
    if visible_devices is not None and visible_devices.strip():
        visible_physical_ids = []
        for token in visible_devices.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.lstrip("-").isdigit():
                visible_physical_ids = None
                break
            physical_id = int(token)
            if physical_id < 0:
                visible_physical_ids = None
                break
            visible_physical_ids.append(physical_id)
        if visible_physical_ids == []:
            visible_physical_ids = None

    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            raise RuntimeError("No GPU devices found.")

        if visible_physical_ids is None:
            candidate_ids = list(range(device_count))
            local_to_physical = {idx: idx for idx in candidate_ids}
        else:
            candidate_ids = [idx for idx in visible_physical_ids if idx < device_count]
            if not candidate_ids:
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES does not reference any valid GPU."
                )
            local_to_physical = {
                local_idx: physical_idx
                for local_idx, physical_idx in enumerate(candidate_ids)
            }

        best_local_idx = 0
        best_used = None

        for local_idx, physical_idx in local_to_physical.items():
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # bytes
            used = mem_info.used

            if best_used is None or used < best_used:
                best_used = used
                best_local_idx = local_idx

        return best_local_idx
    finally:
        pynvml.nvmlShutdown()


def expand_df_to_data(df: pd.DataFrame, save_csv_path: Optional[str] = None) -> pd.DataFrame:
    """Expand a prepared QA DataFrame into model-ready records.

    """
    eval_expanded: Dict[str, List[Any]] = {"context_pred": [], "label": [], "context_gt": [], "context": []}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Expanding DataFrame"):
        context = row['context']
        prediction=row['sampled_answers']
        gold_answer: List[str] = row['gold_answer']
        # Score the first pair for label using local rouge_L helper
        score,matched_gold = rouge_L([prediction], gold_answer)
        label = score > 0.5
        context_pred = construct_input(context, prediction)
        context_gt = construct_input(context, matched_gold)
        eval_expanded["context_pred"].append(context_pred)
        eval_expanded["context"].append(context)
        eval_expanded["label"].append(label)
        eval_expanded["context_gt"].append(context_gt)

    eval_df = pd.DataFrame(eval_expanded)

    # Optionally persist labels (and minimal context) to CSV here
    if save_csv_path is not None:
        try:
            pd.DataFrame({
                "context": eval_df["context"].tolist(),
                "context_pred": eval_df["context_pred"].tolist(),
                "context_gt": eval_df["context_gt"].tolist(),
                "label": eval_df["label"].astype(int).tolist(),
            }).to_csv(save_csv_path, index=False)
        except Exception as e:
            print(f"Warning: failed to save labels CSV at {save_csv_path}: {e}")

    return eval_df



def process_row(row,args):
    context = row['context']
    prediction = row['sampled_answers']
    if args.model=='mistral-7b':
        prediction=ast.literal_eval(row['sampled_answers'])
        prediction=prediction[0]
    gold_answer = row['gold_answer']
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["HF_DATASETS_CACHE"] = tmpdirname
        score, matched_gold = rouge_L([prediction], gold_answer)
    label = score > 0.4
    context_pred = construct_input(context, prediction)
    context_gt = construct_input(context, matched_gold)
    return pd.Series({
        "context_pred": context_pred,
        "context": context,
        "label": label,
        "context_gt": context_gt
    })



def construct_input(context: str, answer: str) -> str:
    return f"{context}{answer}"

def last_token_stack(acts):
    """
    acts
    - List[Tensor]: each of shape [L_i, D]

    Returns:
    Tensor: [N, D]
    """
    last_vecs = []
    for i, t in enumerate(acts):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"acts[{i}] is not a Tensor, got {type(t)}")
        if t.ndim != 2:
            raise ValueError(f"acts[{i}] should have shape [L_i, D], got {t.shape}")
        if t.size(0) == 0:
            raise ValueError(f"acts[{i}] has zero length in seq dimension")
        last_vecs.append(t[-1])   # [D]
    return torch.stack(last_vecs, dim=0)  # [N, D]


def load_model_and_validate_gpu(model_path,cache_dir=None,single_gpu=False,args=None):
    tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,cache_dir=cache_dir)
    print("Started loading model")
    if single_gpu:
        device_id = get_least_used_gpu()
        device = torch.device(f"cuda:{device_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                    dtype=torch.bfloat16, low_cpu_mem_usage=True,cache_dir=cache_dir,)
        assert ('cpu' not in model.hf_device_map.values())

    return model, tokenizer

def ensure_dir(path: List[str] | str) -> None:
    if isinstance(path, list):
        for p in path:
            ensure_dir(p)
    elif not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def final_id_by_plateau(ids_k, k1, window=5):
    """
    ids_k: shape (k2-k1+1,)  ID curve corresponding to k1..k2
    returns:
      best_id: mean within the flattest window
      best_k_range: (k_start, k_end) actual k range of the window
      best_var: variance within the window (smaller means flatter)
    """
    ids_k = np.asarray(ids_k)
    if window < 2 or window > len(ids_k):
        raise ValueError(f"window must be in [2, {len(ids_k)}], got {window}")

    best_i, best_var = 0, np.inf
    for i in range(0, len(ids_k) - window + 1):
        v = np.var(ids_k[i:i+window])
        if v < best_var:
            best_var = v
            best_i = i

    best_id = float(ids_k[best_i:best_i+window].mean())
    best_k_range = (k1 + best_i, k1 + best_i + window - 1)
    return best_id, best_k_range, best_var
