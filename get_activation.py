from collections import defaultdict
from typing import Optional, List, Dict, Any
import os
import sys
import ast
import json
from matplotlib import pyplot as plt
import numpy as np
import torch
import tempfile
import pandas as pd
from tqdm import tqdm
import argparse
import glob
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import warnings
import copy
import traceback
import gc
from collections import defaultdict
from typing import Dict, DefaultDict, List, Optional
from transformers import AutoModelForCausalLM
from datasets import load_dataset, Dataset
from src.metrics import rouge_L
from src.utils import load_model_and_validate_gpu,MODEL2HF,DATA2HF,ensure_dir
from src.construct_dataset_utils import build_prompt_answer
from method.curvature import curvature
from pathlib import Path
PROJECT_PATH = str(Path.cwd())
MODEL_CACHE_DIR = "path2model"
Data_CACHE_DIR = "path2dataset"
# ----------------------
# Args
# ----------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract layer activations for QA datasets")
    parser.add_argument(
        "-m",
        "--model",
        default="llama_instruct",
        choices=MODEL2HF.keys(),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="halueval_summary",
        choices=DATA2HF.keys(),
    )
    parser.add_argument(
        "--basepath2prepared_data",
        type=str,
        default=f"{PROJECT_PATH}/prepared_data",
        help="base path to prepared data, which include context, gold answer and predicted answer",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=MODEL_CACHE_DIR,
        dest="model_cache_dir",
        help="HF cache dir",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device to place inputs on (and for dtype validation). "
            "Use 'cuda', 'cuda:0', 'cpu', etc. 'auto' prefers CUDA."
        ),
    )
    parser.add_argument("--split", type=str, default="train",choices=["train","test"])
    parser.add_argument("--zero-shot", type=bool, default=True)
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential", "none"],
        dest="device_map",
        help=(
            "Device map strategy for loading the model. Use 'none' to keep the model on a single device "
            "(controlled by --device)."
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for activation extraction.",
    )
    parser.add_argument(
        "--curvature",
        action="store_true",
        help="If set, compute per-layer curvature (using all non-padding tokens).",
    )
    parser.add_argument(
        "--act",
        action="store_true",
        help="If set, save last-token hidden states (original activations).",
    )
    parser.add_argument(
        "--dont-save",
        action="store_true",
        dest="dont_save",
        help="Run the computation but do not save activation or label tensors.",
    )
    parser.add_argument(
        "--all_data",
        action="store_true",
    )
    parser.add_argument(
        "--all_split",
        action="store_true",
    )
    parser.add_argument("--all_model",action='store_true')
    return parser


def get_acts(
    prompt_answers_list,
    answer_column,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    batch_size: int,
    compute_acts: bool,
    compute_curvature: bool,
):

    num_layers = len(model.model.layers)
    enc_all = tokenizer(
        prompt_answers_list,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        return_offsets_mapping=True, 
    )
    input_ids   = enc_all["input_ids"]          # [B, L]
    attn_mask   = enc_all["attention_mask"]     # [B, L]
    offsets_all = enc_all["offset_mapping"]     # [B, L, 2]
    B = input_ids.size(0)
    # tokenize answers once (list of list[int])
    ans_start_chars=torch.tensor([len(c)-len(f" {a}") for c, a in zip(prompt_answers_list, answer_column)],dtype=torch.long,device=input_ids.device)
    acts: Optional[Dict[int, List[torch.Tensor]]] = (
        {layer: [] for layer in range(num_layers)} if compute_acts else None
    )
    curvs: Optional[Dict[int, List[float]]] = (
        {layer: [] for layer in range(num_layers)} if compute_curvature else None
    )

    for start in tqdm(range(0, B, batch_size), desc="Collect hidden states / curvature / fisher"):
        end = min(B, start + batch_size)
        sl = slice(start, end)

        ids = input_ids[sl].to(device)       # [b, L]
        mask = attn_mask[sl].to(device)      # [b, L]
        off = offsets_all[sl].to(device)      # [b, L, 2]
        mask_bool = mask.bool()
        bsz = ids.size(0)
        a_sch = ans_start_chars[sl].to(device) # [b]

        # ---------- Forward ----------
        with torch.inference_mode():
            out = model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
            )


        hidden_states = out.hidden_states     # tuple, len = num_layers + 1
        batch_index = torch.arange(bsz, device=ids.device)

        # ---------- 1) activations & curvature ----------
        for layer in range(num_layers):
            h = hidden_states[layer + 1]      # [b, L, D], skip the embedding layer

            # last-token activations
            if compute_acts:
                last_h = h[batch_index, -1]   # [b, D]
                for i in range(bsz):
                    acts[layer].append(last_h[i].detach().cpu())

            # curvature using all non-padding tokens
            if compute_curvature:
                for i in range(bsz):
                    valid_h = h[i][mask_bool[i]]   # [T_i, D]
                    if valid_h.numel() == 0:
                        continue
                    curv_val = curvature(valid_h)
                    if curv_val is None or (isinstance(curv_val, float) and np.isnan(curv_val)):
                        print(f"[WARN] curvature is {curv_val} at layer {layer}, sample {start+i}")
                        continue
                    curvs[layer].append(float(curv_val))


    return acts, curvs


def get_acts_save(
    args,
    dataset: Dataset,
    acts_dir: Optional[str],
    curv_dir: Optional[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    gt_flag: bool = False,
):
    """
    Save activations / curvature / gradient-based metrics
    depending on args.act / args.curvature / args.gradient flags.
    """

    prompt_list = build_prompt_answer(args, dataset, gt_flag)
    acts_batch, curv_batch = get_acts(
        prompt_list,
        dataset['matched_ground_truth'] if gt_flag else dataset['best_answer'],
        tokenizer,
        model,
        device,
        args.batch_size,
        compute_acts=args.act,
        compute_curvature=args.curvature,
    )
    tag = "gt" if gt_flag else "pred"

    # ====== save activations ======
    if args.act and (not args.dont_save) and acts_dir is not None and acts_batch is not None:
        os.makedirs(acts_dir, exist_ok=True)
        for layer_id, acts_list in acts_batch.items():
            layer_acts = torch.stack(acts_list, dim=0)   # [N, D]
            save_path = os.path.join(
                acts_dir,
                f"{args.model}_{args.dataset}_layer_{layer_id}_{tag}.pt",
            )
            torch.save(layer_acts, save_path)
        print(f"[{tag}] activations saved to {acts_dir}")
    elif args.act and args.dont_save:
        print(f"[{tag}] activations were computed but not saved because --dont-save is enabled")

    # ====== save curvature ======
    if args.curvature and curv_dir is not None and curv_batch is not None:
        os.makedirs(curv_dir, exist_ok=True)
        layer_mean_curv = {
            layer_id: float(np.mean(vals))   # vals is the list of per-sample curvature values
            for layer_id, vals in curv_batch.items()
            if len(vals) > 0
        }

        df_curv = pd.DataFrame(
            [
                {"layer": int(layer), "curvature_mean": mean_c}
                for layer, mean_c in layer_mean_curv.items()
            ]
        )
        save_path = os.path.join(curv_dir, f"{args.model}_{args.dataset}_curvature_mean.csv")
        df_curv.to_csv(save_path, index=False)
        print(f"Saved mean curvature per layer to {save_path}")

        # plot and save the curvature curve
        layers = df_curv["layer"].values
        curv_means = df_curv["curvature_mean"].values

        plt.figure(figsize=(8, 5))
        plt.plot(layers, curv_means, marker="o")
        plt.xlabel("Layer")
        plt.ylabel("Mean curvature")
        plt.title(f"Curvature vs Layer ({args.model} on {args.dataset})")
        plt.grid(True, linestyle="--", alpha=0.5)

        fig_path = os.path.join(
            curv_dir,
            f"{args.model}_{args.dataset}_curvature_mean.png"
        )
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved curvature plot to {fig_path}")



def main(args):
    gpu_name = torch.cuda.get_device_name(0)
    # if args.all_data:
    if args.dataset in ['coqa','squad',]:
        args.batch_size=16 if (gpu_name=='NVIDIA RTX A6000' or 'NVIDIA A40' )else 64
    elif args.dataset in ['psiloqa','halueval_summary','cnn_dailymail']:
        args.batch_size=16 if  (gpu_name=='NVIDIA RTX A6000' or 'NVIDIA A40' ) else 4
    else:
        args.batch_size=32 if  (gpu_name=='NVIDIA RTX A6000' or 'NVIDIA A40' ) else 64
    if args.split == 'train':
        if not (args.act and args.curvature):
            args.act = True
            args.curvature=True
            warnings.warn("'act' and 'curvature' are set to True on split='train'.", UserWarning)

    else:
        if not args.act :
            args.act = True
            warnings.warn("'act' is set to True since on split='test'.", UserWarning)
        args.curvature = False
        

    model, tokenizer=load_model_and_validate_gpu(
        MODEL2HF[args.model],
        cache_dir=args.model_cache_dir,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left'
    device = model.device
    base_path = os.path.join(
        args.basepath2prepared_data,
        args.model,
        args.dataset,
    )
    dataset_path = os.path.join(base_path, f"{args.split}_data.jsonl")
    base_save_path = os.path.join(base_path,f"{args.split}", "activations")
    gt_acts_path   = os.path.join(base_save_path, "ground_truth")
    pred_acts_path = os.path.join(base_save_path, "predicted")
    label_path     = os.path.join(base_save_path, "labels")
    curv_base_path = os.path.join(PROJECT_PATH, "results", f"{args.model}", f"{args.dataset}","curvature")

    dirs_to_create = [curv_base_path] if args.curvature else []
    if not args.dont_save:
        dirs_to_create.extend([base_save_path, gt_acts_path, pred_acts_path, label_path])
    if dirs_to_create:
        ensure_dir(dirs_to_create)

    df=pd.read_json(dataset_path,orient="records",lines=True)
    dataset=Dataset.from_pandas(df)
    # predicted
    get_acts_save(
        args,
        dataset,
        acts_dir=pred_acts_path if args.act else None,
        curv_dir=curv_base_path if args.curvature else None,
        tokenizer=tokenizer,
        model=model,
        device=device,
        gt_flag=False,
    )

    if not args.dont_save:
        labels = dataset["label"]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        labels_path = os.path.join(
            label_path,
            f"{args.model}_{args.dataset}_all_layer_label.pt"
        )
        torch.save(labels_tensor, labels_path)
    else:
        print("Label tensor was not saved because --dont-save is enabled")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    def run_one(a):
        try:
            print(f"\n=== Running model={a.model} dataset={a.dataset} split={a.split} ===")
            main(a)
        except Exception:
            traceback.print_exc()

    def run_all_splits(a):
        if a.all_split:
            for split in ["train", "test"]:
                aa = copy.deepcopy(a)
                aa.split = split
                run_one(aa)
        else:
            run_one(a)

    def run_all_datasets(a):
        if a.all_data:
            for dataset_name in ["coqa", "squad", "hotpotqa", "triviaqa", "psiloqa"]:
                aa = copy.deepcopy(a)
                aa.dataset = dataset_name
                run_all_splits(aa)
        else:
            run_all_splits(a)

    def run_all_models(a):
        if a.all_model:
            for model_name in ["llama_instruct", "mistral_instruct"]:
                aa = copy.deepcopy(a)
                aa.model = model_name
                run_all_datasets(aa)
        else:
            run_all_datasets(a)

    run_all_models(args)

