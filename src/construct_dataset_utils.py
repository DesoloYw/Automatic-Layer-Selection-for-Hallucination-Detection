import os
import json
import random
import glob
import shutil
import tempfile
from urllib.error import URLError
from urllib.request import urlopen
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from src.metrics import rouge_L
from src.utils import load_model_and_validate_gpu
from tqdm import tqdm
import ast
from typing import Any, Dict, List, Tuple
DATA2HF={
    "coqa":"stanfordnlp/coqa",
    "triviaqa":None, 
    "hotpotqa":None,
    "nq": None,
    "math":None,
    "squad":"rajpurkar/squad",
    "hotpotqa_c":None,
    'psiloqa':"s-nlp/PsiloQA",
    "halueval_summary":None,
    "cnn_dailymail":"abisee/cnn_dailymail",
}

TRUETEACHER_MODEL_ID = "google/t5_11b_trueteacher_and_anli"
TRUETEACHER_MAX_LENGTH = 2048
TRUETEACHER_DEFAULT_BATCH_SIZE = 4
TRUETEACHER_A6000_BATCH_SIZE = 2

HALUEVAL_SUMMARY_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"
)
HALUEVAL_SUMMARY_SUBDIRS = ["halueval", "HaluEval"]
HALUEVAL_SUMMARY_FILENAME = "summarization_data.json"


def _load_json_records(path: str) -> Any:
    def _load_as_jsonl() -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.endswith(".jsonl"):
        return _load_as_jsonl()

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        # Some local files use a .json suffix but are actually JSONL.
        try:
            return _load_as_jsonl()
        except json.JSONDecodeError:
            raise exc


def _pick_first_nonempty(example: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        return value
    return None


def _normalize_summary_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _normalize_summary_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        return _normalize_summary_text(
            _pick_first_nonempty(
                value,
                [
                    "text",
                    "summary",
                    "reference",
                    "reference_summary",
                    "gold_summary",
                    "right_summary",
                    "faithful_summary",
                ],
            )
        )
    return str(value).strip()


def _normalize_halueval_summary_record(example: Dict[str, Any]) -> Dict[str, Any]:
    context = _normalize_summary_text(
        _pick_first_nonempty(
            example,
            ["document", "source", "article", "context", "passage", "text", "input"],
        )
    )
    reference_summary = _normalize_summary_text(
        _pick_first_nonempty(
            example,
            [
                "reference_summary",
                "gold_summary",
                "summary",
                "reference",
                "right_summary",
                "faithful_summary",
            ],
        )
    )

    if not context:
        raise ValueError(f"Cannot find source document field in example keys: {list(example.keys())}")

    normalized = {
        "context": context,
        "answers": [reference_summary] if reference_summary else [""],
    }

    provided_label = _pick_first_nonempty(
        example,
        ["label", "hallucination_label", "faithfulness_label", "provided_label"],
    )
    if provided_label is not None:
        normalized["provided_label"] = provided_label

    hallucinated_summary = _normalize_summary_text(
        _pick_first_nonempty(example, ["hallucinated_summary", "hallucinated", "fake_summary"])
    )
    if hallucinated_summary:
        normalized["hallucinated_summary"] = hallucinated_summary

    return normalized


def _find_halueval_summary_file(data_cache_dir: str, split: str) -> Tuple[List[str], bool]:
    split_patterns = [
        os.path.join(data_cache_dir, "halueval", f"*summary*{split}*.json*"),
        os.path.join(data_cache_dir, "HaluEval", f"*summary*{split}*.json*"),
        os.path.join(data_cache_dir, "halueval", f"*summarization*{split}*.json*"),
        os.path.join(data_cache_dir, "HaluEval", f"*summarization*{split}*.json*"),
    ]
    fallback_patterns = [
        os.path.join(data_cache_dir, "halueval", "*summary*.json*"),
        os.path.join(data_cache_dir, "HaluEval", "*summary*.json*"),
        os.path.join(data_cache_dir, "halueval", "*summarization*.json*"),
        os.path.join(data_cache_dir, "HaluEval", "*summarization*.json*"),
    ]

    matches = []
    for pattern in split_patterns:
        matches.extend(sorted(glob.glob(pattern)))

    used_explicit_split_file = len(matches) > 0

    if not matches:
        for pattern in fallback_patterns:
            matches.extend(sorted(glob.glob(pattern)))

    return matches, used_explicit_split_file


def _download_halueval_summary(data_cache_dir: str) -> str:
    target_dir = os.path.join(data_cache_dir, HALUEVAL_SUMMARY_SUBDIRS[0])
    target_path = os.path.join(target_dir, HALUEVAL_SUMMARY_FILENAME)
    os.makedirs(target_dir, exist_ok=True)

    tmp_path = None
    try:
        with urlopen(HALUEVAL_SUMMARY_URL) as response:
            with tempfile.NamedTemporaryFile(delete=False, dir=target_dir) as tmp_file:
                tmp_path = tmp_file.name
                shutil.copyfileobj(response, tmp_file)
        os.replace(tmp_path, target_path)
        return target_path
    except Exception as exc:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if isinstance(exc, URLError):
            raise FileNotFoundError(
                "Failed to download HaluEval summarization data from official GitHub. "
                f"URL: {HALUEVAL_SUMMARY_URL}. "
                f"Intended local path: {target_path}. "
                f"Original error: {exc}"
            ) from exc
        raise


def load_halueval_summary(args) -> Dataset:
    matches, used_explicit_split_file = _find_halueval_summary_file(
        args.data_cache_dir,
        args.split,
    )

    if not matches:
        print("HaluEval summarization dataset not found locally. Downloading from official GitHub...")
        downloaded_path = _download_halueval_summary(args.data_cache_dir)
        print(f"Downloaded HaluEval summarization dataset to: {downloaded_path}")
        matches, used_explicit_split_file = _find_halueval_summary_file(
            args.data_cache_dir,
            args.split,
        )

    if not matches:
        searched = [
            os.path.join(args.data_cache_dir, subdir, f"*summary*{args.split}*.json*")
            for subdir in HALUEVAL_SUMMARY_SUBDIRS
        ] + [
            os.path.join(args.data_cache_dir, subdir, f"*summarization*{args.split}*.json*")
            for subdir in HALUEVAL_SUMMARY_SUBDIRS
        ] + [
            os.path.join(args.data_cache_dir, subdir, "*summary*.json*")
            for subdir in HALUEVAL_SUMMARY_SUBDIRS
        ] + [
            os.path.join(args.data_cache_dir, subdir, "*summarization*.json*")
            for subdir in HALUEVAL_SUMMARY_SUBDIRS
        ]
        raise FileNotFoundError(
            "HaluEval summarization data is still not discoverable after auto-download. "
            f"Searched patterns:\n{chr(10).join(searched)}"
        )

    raw = _load_json_records(matches[0])
    if isinstance(raw, dict):
        if args.split in raw and isinstance(raw[args.split], list):
            records = raw[args.split]
            used_explicit_split_file = True
        elif "data" in raw and isinstance(raw["data"], list):
            records = raw["data"]
        elif "examples" in raw and isinstance(raw["examples"], list):
            records = raw["examples"]
        elif "records" in raw and isinstance(raw["records"], list):
            records = raw["records"]
        else:
            raise ValueError(f"Unsupported HaluEval summary file format: {matches[0]}")
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unsupported HaluEval summary file format: {matches[0]}")
    # total size is 10000
    if not used_explicit_split_file:
        train_records, test_records = train_test_split(
            records,
            # records[:100],

            test_size=0.2,
            random_state=getattr(args, "seed", 42),
        )
        records = train_records if args.split == "train" else test_records

    normalized_records = [_normalize_halueval_summary_record(record) for record in records]

    dataset_dict = {
        "context": [record["context"] for record in normalized_records],
        "answers": [record["answers"] for record in normalized_records],
    }

    if any("provided_label" in record for record in normalized_records):
        dataset_dict["provided_label"] = [
            record.get("provided_label") for record in normalized_records
        ]
    if any("hallucinated_summary" in record for record in normalized_records):
        dataset_dict["hallucinated_summary"] = [
            record.get("hallucinated_summary", "") for record in normalized_records
        ]

    return Dataset.from_dict(dataset_dict)


def load_cnn_dailymail(args) -> Dataset:
    local_dir = os.path.join(args.data_cache_dir, "cnn_dailymail")
    train_file = os.path.join(local_dir, "train.jsonl")
    test_file = os.path.join(local_dir, "test.jsonl")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        os.makedirs(local_dir, exist_ok=True)
        ds = load_dataset(DATA2HF["cnn_dailymail"], "3.0.0", cache_dir=args.data_cache_dir)

        if len(ds["train"]) < 10000:
            raise ValueError(
                f"CNN/DailyMail train split has only {len(ds['train'])} samples; need at least 10000."
            )
        if len(ds["test"]) < 10000:
            raise ValueError(
                f"CNN/DailyMail test split has only {len(ds['test'])} samples; need at least 10000."
            )

        train_ds = ds["train"].shuffle(seed=getattr(args, "seed", 42)).select(range(10000))
        test_ds = ds["test"].shuffle(seed=getattr(args, "seed", 42)).select(range(10000))

        train_ds.to_json(train_file, orient="records", lines=True)
        test_ds.to_json(test_file, orient="records", lines=True)
        print(f"CNN/DailyMail train set size: {len(train_ds)}")
        print(f"CNN/DailyMail test set size: {len(test_ds)}")

    file_path = train_file if args.split == "train" else test_file
    ds = load_dataset("json", data_files=file_path, cache_dir=args.data_cache_dir)["train"]

    return Dataset.from_dict(
        {
            "context": ds["article"],
            "answers": [[summary] for summary in ds["highlights"]],
        }
    )

def load_triviaqa(args, legacy=False) -> Dataset:
    test=True if args.split=='test' else False
    if legacy:
        with open('../data/verified-web-dev.json') as f:
            data_verified = json.load(f)['Data']
        with open('../data/web-dev.json') as f:
            data = json.load(f)['Data']

        questions_from_verified = {x['Question'] for x in data_verified}
        data_not_verified = [
            x for x in data if x['Question'] not in questions_from_verified
        ]

        print("Length of not verified data: ", len(data_not_verified))
        print("Length of verified data: ", len(data_verified))

        if test:
            selected = data_verified
        else:
            selected = data_not_verified

        questions = [ex['Question'] for ex in selected]
        aliases = [ex['Answer']['Aliases'] for ex in selected]

        dataset = Dataset.from_dict({
            "question": questions,
            "answer_aliases": aliases,
        })
        return dataset

    else:
        if test:
            file_path = os.path.join(
                args.data_cache_dir,
                "triviaqa-unfiltered",
                "unfiltered-web-dev.json",
            )
        else:
            file_path = os.path.join(
                args.data_cache_dir,
                "triviaqa-unfiltered",
                "unfiltered-web-train.json",
            )

        with open(file_path) as f:
            data = json.load(f)['Data']

        
        data, _ = train_test_split(data, train_size=10000, random_state=42)

        questions = [ex['Question'] for ex in data]
        aliases = [ex['Answer']['Aliases'] for ex in data]

        dataset = Dataset.from_dict({
            "question": questions,
            "answers": aliases,
        })
        return dataset
    


def load_psiloqa(args) -> Dataset:
    """
    Load PsiloQA directly from Hugging Face Hub and convert to the field schema used by this project.
    """
    local_dir = os.path.join(args.data_cache_dir, "psiloqa")
    train_file = os.path.join(local_dir, "train.jsonl")
    test_file = os.path.join(local_dir, "test.jsonl")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        os.makedirs(local_dir, exist_ok=True)
        ds = load_dataset(DATA2HF["psiloqa"], cache_dir=args.data_cache_dir)
        dataset_split = {}
        for split in ds.keys():
            split_ds = ds[split]
            if "lang" in split_ds.column_names:
                split_ds = split_ds.filter(lambda example: example["lang"] == "en")
            dataset_split[split] = split_ds

        if len(dataset_split["train"]) < 10000:
            raise ValueError(
                f"PsiloQA English train split has only {len(dataset_split['train'])} samples; need at least 10000."
            )

        shuffled_train = dataset_split["train"].shuffle(seed=42)
        train_ds = shuffled_train.select(range(10000))
        rest_ds = shuffled_train.select(range(10000, len(shuffled_train)))
        combined_test_ds = concatenate_datasets(
            [rest_ds, dataset_split["validation"], dataset_split["test"]]
        )
        combined_test_ds = combined_test_ds.shuffle(seed=42)

        train_ds.to_json(train_file, orient="records", lines=True)
        combined_test_ds.to_json(test_file, orient="records", lines=True)
        print(f"PsiloQA train set size: {len(train_ds)}")
        print(f"PsiloQA test set size: {len(combined_test_ds)}")
        if args.split == 'train':
            return Dataset.from_dict(
                {
                    "context": train_ds["wiki_passage"],
                    "question": train_ds["question"],
                    "answers": [[answer] for answer in train_ds["golden_answer"]],
                }
            )
        else:
            return Dataset.from_dict(
                {
                    "context": combined_test_ds["wiki_passage"],
                    "question": combined_test_ds["question"],
                    "answers": [[answer] for answer in combined_test_ds["golden_answer"]],
                }
            )
    else:
        # directly load local dataset
        file_path = os.path.join(local_dir, f"{args.split}.jsonl")
        ds = load_dataset("json", data_files=file_path, cache_dir=args.data_cache_dir)["train"]

        return Dataset.from_dict(
            {
                "context": ds["wiki_passage"],
                "question": ds["question"],
                "answers": [[answer] for answer in ds["golden_answer"]],
            }
        )
    
    
def load_hotpotqa(args,context=False):
    if args.split=='train':
        file_path = os.path.join(
                    args.data_cache_dir,
                    "hotpotqa",
                    "train",
                    "hotpot_train_v1.1.json",
                )
    else:
        file_path = os.path.join(
                    args.data_cache_dir,
                    "hotpotqa",
                    "distractor_validation",
                    "hotpot_dev_distractor_v1.json",
        )
    with open(file_path) as f:
            data = json.load(f)
    if args.split=='train':
        data, _ = train_test_split(data, train_size=10000, random_state=42)
    all_questions = [item['question'] for item in data]
    labels = [ [item['answer']] for item in data]
    if context:
        contexts = []
        for item in data:
            title=item['supporting_facts'][0][0]
            title_list=[t[0] for t in item['context']]
            title_index=title_list.index(title)
            sentences_list=item['context'][title_index][1]
            context="".join(sentences_list)
            contexts.append(context)
        dataset= Dataset.from_dict({
            "context": contexts,
            "question": all_questions,
            "answers": labels,
        })
        return dataset
    dataset= Dataset.from_dict({
        "question": all_questions,
        "answers": labels,
    })
    return dataset

def load_coqa(args):
    data = load_dataset(
            DATA2HF[args.dataset],
            split="train" if args.split=="train" else "validation",
            cache_dir=args.data_cache_dir,
        )
    question_num=0
    dataset = {}
    dataset['story'] = []
    dataset['question'] = []
    dataset['answers'] = []
    for sample_id, sample in enumerate(data):
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']['input_text']
            for question,answer in zip(questions,answers):
                dataset['story'].append(story)
                dataset['question'].append(question)
                dataset['answers'].append([answer])
                if args.split=='train':
                    question_num+=1
                    if question_num>=10000:
                        return Dataset.from_dict({
                            'context':dataset['story'],
                            "question": dataset['question'],
                            "answers": dataset['answers'],
                        })
                    
    return Dataset.from_dict({
        'context':dataset['story'],
        "question": dataset['question'],
        "answers": dataset['answers'],
    })



def load_squad(args):
    """
    Manually load a local SQuAD v1.1 JSON and return:
      Dataset({
        'question': List[str],
        'answers':  List[List[str]],  # multiple answers per sample
      })

    Behavior:
      - train: shuffle and take up to 10000 samples
      - non-train: same random 10000
    """
    data_dir = os.path.join(args.data_cache_dir, "squad")

    # assumes the downloaded file is SQuAD v1.1
    if args.split == "train":
        json_path = os.path.join(data_dir, "train-v1.1.json")
    else:
        # dev serves as validation/test
        json_path = os.path.join(data_dir, "dev-v1.1.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"SQuAD json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    contexts  = []
    questions = []
    answers   = []   # List[List[str]]

    # raw structure: data -> [article] -> paragraphs -> context + qas
    for article in raw["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                q_text    = qa["question"]
                ans_texts = [a["text"] for a in qa.get("answers", [])]

                if not ans_texts:
                    ans_texts = [""]   # fallback to avoid empty list

                contexts.append(ctx)
                questions.append(q_text)
                answers.append(ans_texts)

    n = len(questions)
    indices = list(range(n))

    rng = random.Random(getattr(args, "seed", 42))
    rng.shuffle(indices)
    indices = indices[: min(10000, n)]

    # build the final Dataset from the selected index subset
    sel_questions = [questions[i] for i in indices]
    sel_answers   = [answers[i]   for i in indices]
    sel_contexts=[contexts[i] for i in indices]
    return Dataset.from_dict(
        {
            "context":sel_contexts,
            "question": sel_questions,
            "answers":  sel_answers,
        }
    )


def uses_completion_prompt(model_name: str) -> bool:
    return model_name == "llama3.1-8b"


def build_prompt(args,dataset):
    if uses_completion_prompt(args.model):
        if args.dataset=='triviaqa':
            return triviaqa_completion_prompt(dataset)
        elif args.dataset=='hotpotqa':
            return triviaqa_completion_prompt(dataset)
        elif args.dataset=='coqa':
            return coqa_completion_prompt(dataset)
        elif args.dataset=='nq':
            return triviaqa_completion_prompt(dataset)
        elif args.dataset=='squad':
            return coqa_completion_prompt(dataset)
        elif args.dataset=='math':
            return triviaqa_completion_prompt(dataset)
        elif args.dataset=='hotpotqa_c':
            return coqa_completion_prompt(dataset)
        elif args.dataset=='psiloqa':
            return coqa_completion_prompt(dataset)
        elif args.dataset in ['halueval_summary', 'cnn_dailymail']:
            return summarization_completion_prompt(dataset)
        else:
            raise NotImplementedError(f"Prompt for dataset {args.dataset} not implemented.")

    if args.dataset=='triviaqa':
        return triviaqa_prompt(dataset)
    elif args.dataset=='hotpotqa':
        return triviaqa_prompt(dataset)
    elif args.dataset=='coqa':
        return coqa_prompt(dataset)
    elif args.dataset=='nq':
        return triviaqa_prompt(dataset)
    elif args.dataset=='squad':
        return coqa_prompt(dataset)
    elif args.dataset=='math':
        return triviaqa_prompt(dataset)
    elif args.dataset=='hotpotqa_c':
        return coqa_prompt(dataset)
    elif args.dataset=='psiloqa':
        return coqa_prompt(dataset)
    elif args.dataset in ['halueval_summary', 'cnn_dailymail']:
        return summarization_prompt(dataset)
    else:
        raise NotImplementedError(f"Prompt for dataset {args.dataset} not implemented.")
   
    
def build_prompt_answer(args,dataset,gt_flag=False):
    prompts=build_prompt(args,dataset)
    if gt_flag:
        answers_list=dataset['answers'][0]
    else:
        answers_list=dataset['best_answer']
    prompt_answers=[ f"{prompt} {ans}" for prompt, ans in zip(prompts,answers_list)]
    return prompt_answers

def build_prompt_candidate_answer(args,dataset):
    prompts=build_prompt(args,dataset)
    candidate_answers_list=dataset['candidate_answers']
    prompt_candidate_answer=[]
    sample_idx=[]
    candidate_answers_flat=[]
    for idx,(prompt, candidate_answers) in enumerate(zip(prompts,candidate_answers_list)):
        for ans in candidate_answers:
            prompt_candidate_answer.append(f"{prompt} {ans}")
            sample_idx.append(idx)
            candidate_answers_flat.append(ans)
    return prompt_candidate_answer,sample_idx,candidate_answers_flat

def triviaqa_prompt(dataset:Dataset):
    prompts = []
    for q in dataset['question']:
        prompts.append(f"Answer the question as briefly as possible, using plain text only:\n Question:{q}\n Answer:")
        # prompts.append(f"""
        #                 Answer the question in English as briefly as possible:
        #                 Question:{q}
        #                 Answer:
        #                 """)

    return prompts


def triviaqa_completion_prompt(dataset: Dataset):
    prompts = []
    for q in dataset["question"]:
        prompts.append(f"Question: {q}\nAnswer:")
    return prompts


def coqa_prompt(dataset: Dataset):
    prompts = []
    for sample in dataset:
        context= sample['context']
        q= sample['question']
        # prompts.append(f'''Context: {context} \n Question: {q} \n Answer: ''')
        # prompts.append(f"Answer the question based on the context as briefly as possible:\n Context:{context.strip()}\n Question:{q.strip()}\n Answer:")
        prompts.append(f"Answer the question as briefly as possible, based only on the context:\n Context:{context.strip()}\n Question:{q.strip()}\n Answer:")
    return prompts


def coqa_completion_prompt(dataset: Dataset):
    prompts = []
    for sample in dataset:
        context = sample["context"]
        q = sample["question"]
        prompts.append(f"Context: {context.strip()}\nQuestion: {q.strip()}\nAnswer:")
    return prompts


def summarization_prompt(dataset: Dataset):
    prompts = []
    for sample in dataset:
        context = sample["context"]
        prompts.append(
            "Summarize the following document in one or two concise sentences.\n"
            # "Use only information supported by the document.\n"
            f"Document:{context.strip()}\n"
            "Summary:"
        )
    return prompts


def summarization_completion_prompt(dataset: Dataset):
    prompts = []
    for sample in dataset:
        context = sample["context"]
        prompts.append(
            f"Document: {context.strip()}\n"
            "Summary:"
        )
    return prompts


import re

strings_to_filter_on = [
    '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:',
    'Questions:', 'questions:', 'QUESTION:', 'ANSWER:', 'REF',
    '.Forms', 'http', 'php','Question','Answer'
]

ALLOWED_CHARS = r"A-Za-z0-9 ,.'\"!?;:-"  # characters to keep


def clean_english_answer(s: str) -> str:
    original_s = s  # save the original string first

    # 1. only look at the first line
    s = s.split('\n')[0].strip()

    # 2. remove all disallowed characters (e.g. Cyrillic, Chinese, odd symbols)
    # s = re.sub(fr"[^{ALLOWED_CHARS}]", " ", s)

    # 3. collapse extra whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 4. shorten long repeated chars like "kkkkkkkkk" (>=3 collapsed to 1)
    s = re.sub(r"(.)\1{2,}", r"\1", s)

    # ==== handle periods ====
    # 1) normalize ". . . ." to "...."
    s = re.sub(r"\.\s+(?=\.)", ".", s)
    # 2) collapse 3+ consecutive periods into a single "."
    s = re.sub(r"\.{3,}", ".", s)

    # ==== handle commas ====
    # 1) normalize ", , , ," to ",,,,"
    s = re.sub(r",\s+(?=,)", ",", s)
    # 2) collapse 3+ consecutive commas into a single ","
    s = re.sub(r",{3,}", ",", s)

    # 7. strip trailing punctuation and whitespace
    s = s.strip()

    # if the result is empty after cleaning, fall back to the original text
    if not s:
        return original_s.strip()

    return s

import re

# 1) word-level abbreviations (the dot is usually followed by a name/word)
WORD_ABBRS = {
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
    "Gen", "Brig", "Adm", "Rear", "Lt", "Col", "Maj", "Capt",
    "St",  # St. George
    "vs", "etc", "Fig", "Eq", "No",
}

# 2) multi-dot abbreviation patterns: U.S. / U.K. / e.g. / i.e. / G. / J.R.R. etc.
MULTI_DOT_ABBR_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")   # e.g. U.S.  i.e.  J.R.R.
SINGLE_INITIAL_RE = re.compile(r"^[A-Za-z]$")           # G.  J.

def extract_first_sentence(text: str) -> str:
    s = text.strip()
    n = len(s)
    if n == 0:
        return s

    i = 0
    while i < n:
        ch = s[i]
        if ch not in ".!?":
            i += 1
            continue

        # A) ellipsis ...
        if ch == "." and s[i:i+3] == "...":
            i += 3
            continue

        # B) decimal point 3.14
        if ch == "." and 0 < i < n-1 and s[i-1].isdigit() and s[i+1].isdigit():
            i += 1
            continue

        # C) token
        j = i - 1
        while j >= 0 and (s[j].isalpha() or s[j] == "."):
            j -= 1
        token_norm = s[j+1:i].strip()

        # D) multi-dot abbreviation: X.Y.
        if ch == ".":
            right_is_letter_dot = (i+2 < n and s[i+1].isalpha() and s[i+2] == ".")
            left_is_letter = (i-1 >= 0 and s[i-1].isalpha())
            if left_is_letter and right_is_letter_dot:
                i += 1
                continue
        if "." in token_norm and MULTI_DOT_ABBR_RE.match(token_norm + "."):
            i += 1
            continue

        # E) word abbreviation
        if token_norm in WORD_ABBRS:
            if token_norm == "No":
                k = i + 1
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k].isdigit():
                    i += 1
                    continue
            else:
                i += 1
                continue

        # F) personal name initial
        if SINGLE_INITIAL_RE.match(token_norm):
            k = i + 1
            while k < n and s[k].isspace():
                k += 1
            if k < n and s[k].isupper():
                i += 1
                continue

        # reached here: treat this punctuation as the end of the first sentence
        return s[:i+1].strip()

    return s

def postprocess_answers(answers, model_name, filters=strings_to_filter_on):
    cleaned = []
    for ans in answers:
        if ans is None:
            ans = ""
        if not isinstance(ans, str):
            ans = str(ans)

        original_ans = ans  # fallback: revert to original if result is empty after truncation

        # 1) truncate at keyword markers first
        cut_pos = len(ans)
        for f in filters:
            idx = ans.find(f)
            if 0 <= idx < cut_pos:
                cut_pos = idx
        truncated = ans[:cut_pos].strip()

        # if truncation by keyword markers yields an empty string, revert to original
        if not truncated:
            truncated = original_ans.strip()

        truncated=extract_first_sentence(truncated)
        cleaned.append(truncated)
    return cleaned


def measure_correctness(dataset:Dataset,args):
    labels=[]
    # matched_ground_truths=[]
    # d_name=args.dataset
    if args.dataset=="math":
        for sample in tqdm(dataset,desc="Measuring correctness"):
            best_answer=sample['best_answer']
            answers= sample['answers']
            label=0
            for ans in answers:
                if ans.strip().lower() in best_answer.strip().lower():
                    label=1
                    labels.append(label)
                    break
                ans_float=float(ans)
                if ans_float.is_integer():
                    ans_int_str=str(int(ans_float))
                    if ans_int_str.strip().lower() in best_answer.strip().lower():
                        label=1
                        labels.append(label)
                        break
            if label==0:
                labels.append(label)
                # matched_ground_truths.append(answers[0])
    elif args.dataset in ["halueval_summary", "cnn_dailymail"]:
        labels = trueteacher_judge_batch(dataset, args)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        if args.dataset in ['coqa','squad','psiloqa']:
            bs=16 if gpu_name=='NVIDIA RTX A6000' else 32
        else:
            bs=32 if gpu_name=='NVIDIA RTX A6000' else 64
        labels=LLM_judge_batch(dataset,args,batch_size=bs)
    # matched_ground_truths=[sample['answers'][0] for sample in dataset]
    # return labels,matched_ground_truths
    return labels



def reevaluate_label(args,remove_NAN_flag=False):
    # load dataset
    print(f"reevaluate {args.model} {args.dataset} {args.split}")
    data_dir=os.path.join(args.basepath_2_save,args.model,args.dataset)
    json_path=os.path.join(data_dir,f"{args.split}_data.jsonl")
    df = pd.read_json(
        json_path,
        orient="records",
        lines=True
    )
    if remove_NAN_flag:
        df=remove_NAN(df)
    dataset_iter= Dataset.from_pandas(df)
    # compute correctness
    labels= measure_correctness(dataset_iter,args)
    df["label"] = labels
    # save to json
    df.to_json(
        json_path,
        orient="records",
        lines=True,
    )
    print(f"Re-evaluated labels saved to {json_path}")

def remove_NAN(dataset):
    if not isinstance(dataset, pd.DataFrame):
        df = dataset.to_pandas()
    else:
        df = dataset
    orig_rows = len(df)

    invalid_stats = {
        "best_answer_missing": 0,
        "candidate_answers_missing": 0,
        "candidate_answers_empty_list": 0,
        "candidate_answers_contains_empty": 0,
    }

    def row_valid(row):
        # 1) check best_answer
        ba = row.get("best_answer", None)
        # NaN / None / empty string are all treated as invalid
        if ba is None or (isinstance(ba, float) and pd.isna(ba)):
            invalid_stats["best_answer_missing"] += 1
            return False
        if isinstance(ba, str) and ba.strip() == "":
            invalid_stats["best_answer_missing"] += 1
            return False

        # 2) check candidate_answers
        ca = row.get("candidate_answers", None)

        if ca is None or (isinstance(ca, float) and pd.isna(ca)):
            invalid_stats["candidate_answers_missing"] += 1
            return False

        if len(ca) == 0:
            invalid_stats["candidate_answers_empty_list"] += 1
            return False

        # list must not contain None / NaN / empty string
        for x in ca:
            if x is None:
                invalid_stats["candidate_answers_contains_empty"] += 1
                return False
            if isinstance(x, float) and pd.isna(x):
                invalid_stats["candidate_answers_contains_empty"] += 1
                return False
            if isinstance(x, str) and x.strip() == "":
                invalid_stats["candidate_answers_contains_empty"] += 1
                return False
        
        return True

    mask = df.apply(row_valid, axis=1)
    df_clean = df[mask].reset_index(drop=True)

    clean_rows = len(df_clean)
    removed_rows = orig_rows - clean_rows

    print(f"original lines: {orig_rows}")
    print(f"deleted: {removed_rows}")
    print("deletion breakdown:")
    print(f"  best_answer missing/empty: {invalid_stats['best_answer_missing']}")
    print(f"  candidate_answers missing: {invalid_stats['candidate_answers_missing']}")
    print(f"  candidate_answers empty list: {invalid_stats['candidate_answers_empty_list']}")
    print(f"  candidate_answers contains empty item: {invalid_stats['candidate_answers_contains_empty']}")
    print(f"left: {clean_rows}")

    return df_clean


def _is_summary_dataset(dataset_name: str) -> bool:
    return dataset_name in {"halueval_summary", "cnn_dailymail"}


def get_trueteacher_input(context: str, model_summary: str) -> str:
    context_text = "" if context is None else str(context).strip()
    if not context_text:
        raise ValueError("TrueTeacher requires a non-empty 'context' field for summary datasets.")

    summary_text = "" if model_summary is None else str(model_summary).strip()
    return f"premise: {context_text} hypothesis: {summary_text}"


def _get_trueteacher_batch_size() -> int:
    if not torch.cuda.is_available():
        return TRUETEACHER_DEFAULT_BATCH_SIZE
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name == "NVIDIA RTX A6000":
        return TRUETEACHER_A6000_BATCH_SIZE
    return TRUETEACHER_DEFAULT_BATCH_SIZE


def _get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for device in model.hf_device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    return next(model.parameters()).device


def trueteacher_judge_batch(dataset, args, batch_size=None):
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "TrueTeacher evaluation requires transformers with T5 support installed."
        ) from exc

    if batch_size is None:
        batch_size = _get_trueteacher_batch_size()

    model_kwargs = {
        "cache_dir": args.model_cache_dir,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        if torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16

    try:
        tokenizer = T5Tokenizer.from_pretrained(
            TRUETEACHER_MODEL_ID,
            cache_dir=args.model_cache_dir,
        )
        model = T5ForConditionalGeneration.from_pretrained(
            TRUETEACHER_MODEL_ID,
            **model_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TrueTeacher model '{TRUETEACHER_MODEL_ID}' "
            f"from cache_dir={args.model_cache_dir}. Original error: {exc}"
        ) from exc

    input_device = _get_model_input_device(model)
    labels = []

    for start in tqdm(range(0, len(dataset), batch_size), desc="TrueTeacher judging"):
        batch = dataset[start:start + batch_size]
        batch_inputs = []
        empty_summary_flags = []

        for context, best_answer in zip(batch["context"], batch["best_answer"]):
            summary_text = "" if best_answer is None else str(best_answer).strip()
            if not summary_text:
                batch_inputs.append(None)
                empty_summary_flags.append(True)
                continue

            batch_inputs.append(get_trueteacher_input(context, summary_text))
            empty_summary_flags.append(False)

        valid_inputs = [text for text in batch_inputs if text is not None]
        decoded_outputs = []

        if valid_inputs:
            input_id = tokenizer(
                valid_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=TRUETEACHER_MAX_LENGTH,
            ).input_ids.to(input_device)
            

            try:
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_id,
                    )
            except RuntimeError as exc:
                raise RuntimeError(
                    "TrueTeacher inference failed. This may be caused by insufficient GPU memory. "
                    "Try reducing the batch size or running on a larger GPU."
                ) from exc

            decoded_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

        decoded_iter = iter(decoded_outputs)
        for is_empty in empty_summary_flags:
            if is_empty:
                labels.append(0)
                continue

            text = next(decoded_iter).strip()
            if text.startswith("1"):
                labels.append(1)
            elif text.startswith("0"):
                labels.append(0)
            else:
                print(f"Invalid TrueTeacher output: {text!r}, defaulting to 0")
                labels.append(0)

    return labels



def LLM_judge_batch(dataset, args, batch_size=32):
    """
    dataset: HF Dataset with fields:
        - 'question'
        - 'best_answer'
        - 'answers'  (list; only the 0th element is used as reference)
        - 'context'  (optional, used for prompt)
    """
    model, tokenizer = load_model_and_validate_gpu(
        # 'mistralai/Mistral-7B-Instruct-v0.2',
        'mistralai/Ministral-8B-Instruct-2410',
        cache_dir=args.model_cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = 'left'
    device = model.device

    n = len(dataset)
    correctness = [None] * n          # placeholder
    indices_to_judge = []             # indices of samples that need LLM judging
    prompts_to_judge = []             # corresponding prompts

    # ---------- stage 1: string match ----------
    for idx, sample in enumerate(tqdm(dataset, desc="String match pre-filter")):
        model_answer = sample["best_answer"]

        if _is_summary_dataset(args.dataset):
            context = sample.get("context", None)
            prompt = get_summary_prompt(context, sample.get("answers", []), model_answer)
            indices_to_judge.append(idx)
            prompts_to_judge.append(prompt)
            continue

        question = sample["question"]

        # simple string match (lowercased + stripped)
        if any(
            answ.strip().lower() in model_answer.strip().lower()
            for answ in sample["answers"]
        ):
            correctness[idx] = 1
            continue

        # unmatched samples: build prompt for batched LLM judging
        context = sample.get("context", None)
        prompt = get_prompt(context, question, sample["answers"], model_answer)

        indices_to_judge.append(idx)
        prompts_to_judge.append(prompt)

    # stats: total samples that failed string match (i.e. no hit)
    total_to_judge = len(indices_to_judge)
    num_llm_1 = 0   # of those, how many the LLM judged as 1

    # ---------- stage 2: batched LLM-as-judge ----------
    for start in tqdm(range(0, len(prompts_to_judge), batch_size), desc="LLM judging"):
        end = min(start + batch_size, len(prompts_to_judge))
        batch_prompts = prompts_to_judge[start:end]
        batch_indices = indices_to_judge[start:end]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                top_p=1.0,
                return_dict_in_generate=True,
                eos_token_id=None,
            )

        # newly generated tokens: everything after the prompt length
        seqs = outputs["sequences"]
        prompt_len = enc["input_ids"].shape[1]  # same for all samples due to padding
        gen_tokens = seqs[:, prompt_len:]

        for i, seq in enumerate(gen_tokens):
            text_decoded = tokenizer.decode(seq, skip_special_tokens=True)
            # light cleanup
            text = (
                text_decoded.replace(".</s>", "")
                .replace("</s>", "")
                .split("\n")[0]
                .strip()
                .strip(".")
            )

            idx1 = text.find("1")
            idx0 = text.find("0")

            if idx1 != -1 and (idx0 == -1 or idx1 < idx0):
                lab = 1
            elif idx0 != -1 and (idx1 == -1 or idx0 < idx1):
                lab = 0
            else:
                print(f"Invalid judge output: {text!r}, default to 0")
                lab = 0

            # count: among string-match=0 samples, how many the LLM judged as 1
            if lab == 1:
                num_llm_1 += 1

            # write back to the corresponding global index
            global_idx = batch_indices[i]
            correctness[global_idx] = lab

    # print summary statistics
    if total_to_judge > 0:
        ratio = num_llm_1 / total_to_judge
    else:
        ratio = 0.0

    print(
        f"[LLM-judge stats] "
        f"Number of string-match=0: {total_to_judge}, "
        f"Numer of LLM judge 1: {num_llm_1} "
        f"({ratio:.2%})"
    )

    return correctness

def get_prompt(context,question,reference_answers,model_answer):
    reference_answer="; ".join(reference_answers)
    if context is None:
        prompt = f"""
        Evaluate the following answers to questions. For each question you are given a model answer and the correct answer.
        You must determine if the model answer is correct or not. If the model answer is correct, write '1' and if it is not correct, write '0'.
        For example:

        Question: who is the young guitarist who played with buddy guy?
        Ground Truth: Quinn Sullivan
        Model Answer: Ronnie Earl Explanation: Ronnie Earl is an American blues guitarist and singer who has played with many famous blues musicians, including Buddy Guy. He is known for his soulful and melodic playing style, and has released many albums that blend blues, jazz, and rock music. Earl has also been a member of the Buddy Guy Blues Band and has played with other notable blues musicians such as B.B. King, Eric Clapton, and Stevie Ray Vaughan. He is considered one of the most
        Correctness: 0

        Question: name of the first episode of stranger things 
        Ground Truth: Chapter One : The Vanishing of Will Byers
        Model Answer:  The disappearance of Will Byers. Explanation: The first episode of the first season of Stranger Things is titled "The Vanishing of Will Byers". The episode introduces the main characters and sets the tone for the rest of the series. It follows the story of Will Byers, a young boy who goes missing in the fictional town of Hawkins, Indiana, and the subsequent search for him by his mother Joyce and his friends Mike, Dustin, and Lucas. The episode sets the stage for the supernatural
        Correctness: 1

        Question: {question}
        Ground Truth: {reference_answer}
        Model Answer: {model_answer}
        Correctness:
        """.strip()
    else:
        prompt = f"""
        Evaluate the following answers to questions. For each question you are given a model answer and the correct answer.
        You must determine if the model answer is correct or not. If the model answer is correct, write '1' and if it is not correct, write '0'.
        For example:

        Question: who is the young guitarist who played with buddy guy?
        Ground Truth: Quinn Sullivan
        Model Answer: Ronnie Earl Explanation: Ronnie Earl is an American blues guitarist and singer who has played with many famous blues musicians, including Buddy Guy. He is known for his soulful and melodic playing style, and has released many albums that blend blues, jazz, and rock music. Earl has also been a member of the Buddy Guy Blues Band and has played with other notable blues musicians such as B.B. King, Eric Clapton, and Stevie Ray Vaughan. He is considered one of the most
        Correctness: 0

        Question: name of the first episode of stranger things 
        Ground Truth: Chapter One : The Vanishing of Will Byers
        Model Answer:  The disappearance of Will Byers. Explanation: The first episode of the first season of Stranger Things is titled "The Vanishing of Will Byers". The episode introduces the main characters and sets the tone for the rest of the series. It follows the story of Will Byers, a young boy who goes missing in the fictional town of Hawkins, Indiana, and the subsequent search for him by his mother Joyce and his friends Mike, Dustin, and Lucas. The episode sets the stage for the supernatural
        Correctness: 1

        Context: {context}
        Question: {question}
        Ground Truth: {reference_answer}
        Model Answer: {model_answer}
        Correctness:
        """.strip()
    return prompt


def get_summary_prompt(context, reference_answers, model_summary):
    reference_summary = "; ".join(
        [str(ans).strip() for ans in reference_answers if str(ans).strip()]
    ) or "N/A"
    prompt = f"""
    Evaluate whether the model summary is faithful to the source document.
    Write '1' if the summary is fully supported by the document and does not introduce materially unsupported information.
    Write '0' if the summary contains hallucinated, contradictory, or unsupported content.
    If the summary is incomplete but still faithful, prefer '1'.

    Source Document: {context}
    Reference Summary: {reference_summary}
    Model Summary: {model_summary}
    Faithfulness:
    """.strip()
    return prompt


def _clean_best_answer_cell(x,args):
    """Process a single best_answer cell (string)."""
    if x is None or not isinstance(x, str):
        raise ValueError(
            f"best_answer cell must be a non-empty string, got {type(x)}: {x!r}"
        )
    # postprocess_answers expects list[str]; pass a single-element list
    return postprocess_answers([x],args.model)[0]

def _clean_candidate_answers_cell(x,args):
    """Process a single candidate_answers cell."""
    if x is None:
        raise ValueError("candidate_answers cell must be a list of strings, got None")

    # neither a list nor a bare string is a valid type here
    if not isinstance(x, list) or isinstance(x, str):
        raise ValueError(
            f"candidate_answers cell must be a list of strings, got {type(x)}: {x!r}"
        )

    return postprocess_answers(x,args.model)


def re_post_process(args):
    """
    Re-post-process an already generated dataset:
      - clean 'best_answer'
      - clean 'candidate_answers' (list of strings)
    Overwrites the existing {split}_data.jsonl in place.
    """
    print(f"repost {args.model} {args.dataset} {args.split}")

    data_dir  = os.path.join(args.basepath_2_save, args.model, args.dataset)
    json_path = os.path.join(data_dir, f"{args.split}_data.jsonl")

    print(f"Loading dataset from: {json_path}")
    df = pd.read_json(json_path, orient="records", lines=True)
    assert "best_answer" in df.columns
    assert "candidate_answers" in df.columns

    print("Post-processing 'best_answer' column...")
    df["best_answer"] = df["best_answer"].apply(_clean_best_answer_cell,args=(args,),)

    print("Post-processing 'candidate_answers' column...")
    df["candidate_answers"] = df["candidate_answers"].apply(_clean_candidate_answers_cell,args=(args,),)
    df=remove_NAN(df)
    # overwrite the original file
    df.to_json(
        json_path,
        orient="records",
        lines=True,
    )
    print(f"Re-post-processed dataset saved to: {json_path}")
