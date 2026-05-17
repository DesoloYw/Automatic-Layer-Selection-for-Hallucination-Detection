import re
import string
import nltk
import evaluate
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor
import os


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(preds, golds):
    scores = []

    for pred in preds:
        ps = 0.0
        for gold in golds:
            ret = nltk.f_measure(set(normalize_text(pred).split()), set(normalize_text(gold).split()))
            if ret is None:
                ret = 0.0
            if ret > ps:
                ps = ret
        scores.append(ps)
    return max(scores)

rouge = evaluate.load('rouge', cache_dir=None)


def rouge_L(pred, golds):
    # deduplicate and normalize gold answers, preserving original text
    gold_map = {}
    for g in golds:
        gn = normalize_text(g)
        gold_map.setdefault(gn, g)  # keep the original gold text
    gold_norms = list(gold_map.keys())

    if not gold_norms:
        return 0.0, None

    # normalize the single pred and replicate it to match each gold
    pn = normalize_text(pred)
    pred_list = [pn] * len(gold_norms)

    out = rouge.compute(
        predictions=pred_list,
        references=gold_norms,
        use_aggregator=False,
    )
    scores = out["rougeL"]
    if not scores:
        return 0.0, None

    # pick the gold with the highest rougeL score
    i = max(range(len(scores)), key=scores.__getitem__)
    best_overall_score = scores[i]
    best_overall_gold = gold_map[gold_norms[i]]

    return best_overall_score, best_overall_gold


def roc(labels, scores):
    auroc = metrics.roc_auc_score(labels, scores)
    return auroc
