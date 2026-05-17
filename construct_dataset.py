#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import string
import random
import gc
from datetime import datetime
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple,  Optional
import traceback
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import sys
import os
from src.utils import load_model_and_validate_gpu,MODEL2HF,DATA2HF
from src.construct_dataset_utils import load_hotpotqa,load_triviaqa,load_coqa,load_nq,build_prompt,postprocess_answers,measure_correctness,load_math,load_squad,reevaluate_label,remove_NAN,re_post_process, load_psiloqa, load_halueval_summary, load_cnn_dailymail
# Get current file path
from pathlib import Path
PROJECT_PATH = str(Path.cwd())
MODEL_CACHE_DIR = "path2model"
Data_CACHE_DIR = "path2dataset"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        default="llama_instruct",
        choices=MODEL2HF.keys(),
        help="model name.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="halueval_summary",
        choices=DATA2HF.keys(),
        help="dataset name. ",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        dest="max_new_tokens",
        help="Maximum number of tokens to generate per answer.",
    )
    parser.add_argument(
        "-zs",
        "--zero-shot",
        type=bool,
        default=True,
        help="Use zero-shot (True) or 5-shot (False) prompting. only for triviaqa and hotpotqa",
    )
    parser.add_argument(
        "--questions-per-story",
        type=int,
        default=5,
        help="Number of questions to answer per story (default: 4). This argument is for CoQA only.",
    )
    parser.add_argument(
        "--split",
        choices=['train','test'],
        default="train",
        help="Dataset split.",
    )
    parser.add_argument(
        "--basepath_2_save",
        default=f"{PROJECT_PATH}/prepared_data",
        help="Optional path to write CSV with columns: context,gold_answer,sampled_answers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for sampling-based decoding.",
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Whether to use only a single GPU (if multiple are available).",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=MODEL_CACHE_DIR,
        help="Cache directory for loading/storing model weights.",
    )
    parser.add_argument(
        "--data-cache-dir",
        default=Data_CACHE_DIR,
        help="Cache directory for downloading the CoQA dataset.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for question prompts.",
    )
    parser.add_argument(
        "--reevaluate",        
        action="store_true",   
    )
    parser.add_argument(                                                                                                                                                                                          
        "--repost",        
        action="store_true",   
    )
    parser.add_argument("--num_answers",default=10,type=int,help="number of answers to sample per question")
    parser.add_argument("--all_data",action='store_true')
    parser.add_argument("--all_split",action='store_true')
    parser.add_argument("--fs",action='store_true',help="use first sentence truncation")
    parser.add_argument("--all_model",action='store_true')
    return parser.parse_args()


def generate_answers_batch(
    args,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device,
) -> Tuple[List[List[str]], List[str]]:
    num_answers = args.num_answers
    stop_token_id=[
        tokenizer.encode('\n', add_special_tokens=False)[-1], 
        ]
    bad_tokens = ['Context','Question', 'Answer','Question:', 'Answer:', 'Q:','//','://','.Forms','_REF_','_REF','php','https','\\']

    question_framing_ids = [[tokenizer(bad_token)['input_ids'][-1]] for bad_token in bad_tokens]
    assert num_answers >= 1, "num_answers must be >= 1"
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    # sort prompts by length for efficiency
    tok_all = tokenizer(
        prompts,
        padding=False,
        truncation=False,
        return_length=True,
        add_special_tokens=True,
    )
    lengths = tok_all["length"]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i], reverse=True)
    prompts_sorted = [prompts[i] for i in order]
    
    bucketed_samples: Dict[int, List[str]] = {}
    bucketed_best: Dict[int, str] = {}

    for start in tqdm(range(0, len(prompts_sorted), batch_size),desc="Generating Responses"):
        batch_prompts = prompts_sorted[start:start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_device = model.get_input_embeddings().weight.device
        enc = {k: v.to(input_device) for k, v in enc.items()}

        # ---------- 1) multi-sample generation for uncertainty estimation ----------
        gen_kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=True,
            num_return_sequences=num_answers,
            # eos_token_id=stop_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            temperature=1.0,
            top_p=0.9,
            top_k=30,
            bad_words_ids=question_framing_ids,
        )

        with torch.inference_mode():
            out = model.generate(**gen_kwargs)

        sequences = out if isinstance(out, torch.Tensor) else out.sequences
        sequences = sequences.to("cpu")
        del out

        Lmax = enc["input_ids"].shape[1]
        gen_only = sequences[:, Lmax:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        B_actual = len(batch_prompts)
        assert len(texts) == B_actual * num_answers

        grouped_samples: List[List[str]] = []
        for i in range(B_actual):
            cur = texts[i * num_answers:(i + 1) * num_answers]
            if args.fs:
                grouped_samples.append(postprocess_answers(cur,args.model))
            else:
                if args.dataset in ['halueval_summary', 'cnn_dailymail']:
                    #truncate to the first line
                    cur=[s.split('\n')[0].strip() for s in cur]
                grouped_samples.append([s.strip() for s in cur])
                


        # ---------- 2) “best answer” (one per prompt) ----------
        best_kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=True,          # low-temperature sampling; set False for greedy decoding
            num_return_sequences=1,
            # eos_token_id=stop_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            temperature=0.1,
            # top_p=0.9,
            # top_k=30,
            bad_words_ids=question_framing_ids,
        )

        with torch.inference_mode():
            out_best = model.generate(**best_kwargs)

        seq_best = out_best if isinstance(out_best, torch.Tensor) else out_best.sequences
        seq_best = seq_best.to("cpu")
        del out_best

        gen_only_best = seq_best[:, Lmax:]
        best_sampled = tokenizer.batch_decode(gen_only_best, skip_special_tokens=True)
        assert len(best_sampled) == B_actual
        if args.fs:
            best_texts = postprocess_answers(best_sampled,args.model)
        else:
            if args.dataset in ['halueval_summary', 'cnn_dailymail']:
                #truncate to the first line
                best_sampled=[s.split('\n')[0].strip() for s in best_sampled]
            best_texts=[s.strip() for s in best_sampled]

        # ---------- 3) write into buckets ----------
        for i in range(B_actual):
            sorted_idx = start + i
            bucketed_samples[sorted_idx] = grouped_samples[i]
            bucketed_best[sorted_idx] = best_texts[i]

    # ---------- 4) restore original order ----------
    inv = [0] * len(order)
    for new_idx, old_idx in enumerate(order):
        inv[old_idx] = new_idx

    samples: List[List[str]] = []
    best_answers: List[str] = []
    for orig_i in range(len(prompts)):
        sorted_pos = inv[orig_i]
        samples.append(bucketed_samples[sorted_pos])
        best_answers.append(bucketed_best[sorted_pos])

    return samples, best_answers


def load_data(args):
    """Load and shuffle the requested dataset split."""
    context=None
    if args.dataset=="triviaqa":
        dataset=load_triviaqa(args)
    elif args.dataset=="hotpotqa":
        dataset= load_hotpotqa(args)
    elif args.dataset=="coqa":
        dataset = load_coqa(args)
    elif args.dataset=='squad':
        dataset= load_squad(args)
    elif args.dataset=='psiloqa':
        dataset= load_psiloqa(args)
    elif args.dataset=='halueval_summary':
        dataset= load_halueval_summary(args)
    elif args.dataset=='cnn_dailymail':
        dataset = load_cnn_dailymail(args)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")

    return dataset


def main(args) -> None:
    # init_wandb(args)
    set_seed(args.seed)
    random.seed(args.seed)
    # load data and model 
    dataset_iter= load_data(args)
    model, tokenizer = load_model_and_validate_gpu(MODEL2HF[args.model],cache_dir=args.model_cache_dir, single_gpu=args.single_gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    prompts_all= build_prompt(args, dataset_iter)

    # 2) Batched generation across all prompts  
    sampled_answer,best_answer = generate_answers_batch(
        args,
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_all,
        device=device,
    )
    assert len(sampled_answer) == len(best_answer)
    dataset_iter = dataset_iter.add_column("candidate_answers", sampled_answer)
    dataset_iter = dataset_iter.add_column("best_answer", best_answer)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    labels= measure_correctness(dataset_iter, args)
    dataset_iter = dataset_iter.add_column("label", labels)

    df_clean=remove_NAN(dataset_iter)

    df_clean['candidate_answers']=df_clean['candidate_answers'].apply(lambda x: x.tolist())
    df_clean['answers']=df_clean['answers'].apply(lambda x: x.tolist())
    json_output_dir = os.path.join(args.basepath_2_save, args.model, args.dataset)
    os.makedirs(json_output_dir, exist_ok=True)
    json_output = os.path.join(json_output_dir, f"{args.split}_data.jsonl")

    df_clean.to_json(
        json_output,
        orient="records",   # one object per record
        lines=True,         # jsonl format: one json per line
        # force_ascii=False   # preserve non-ASCII characters
    )
    print(f"Saved to {json_output}")
    

if __name__ == "__main__":
    args = parse_args()
    def process_task(args):
        gpu_name = torch.cuda.get_device_name(0)
        # if args.dataset in ['coqa','squad','psiloqa','halueval_summary']:
        if args.dataset in ['coqa','squad',]:
            args.batch_size=16 if gpu_name=='NVIDIA RTX A6000' else 16
        elif args.dataset in['psiloqa','halueval_summary','cnn_dailymail']:
            args.batch_size=8 if gpu_name=='NVIDIA RTX A6000' else 8
            if '14b' in args.model:
                args.batch_size=4 
        else:
            args.batch_size=32 if gpu_name=='NVIDIA RTX A6000' else 32
        if args.dataset in ['halueval_summary', 'cnn_dailymail']:
            args.max_new_tokens=130
        try: 
            if args.reevaluate:
                reevaluate_label(args)
            elif args.repost:
                re_post_process(args)
                reevaluate_label(args)
            else:
                main(args)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            traceback.print_exc()

    def split_judge(args):
        if args.all_split:
            for split in ['test','train']:
                args.split=split
                process_task(args)
        else:
            process_task(args)
            
    def all_data_judge(args):
        if args.all_data:
            for dataset_name in ['squad','coqa','hotpotqa','triviaqa','psiloqa',]:

                args.dataset=dataset_name
                split_judge(args)
        else:
            split_judge(args)

    if args.all_model:
        for model_name in ['llama_instruct','mistral_instruct']:
            args.model=model_name
            all_data_judge(args)
    else:
        all_data_judge(args)
            
