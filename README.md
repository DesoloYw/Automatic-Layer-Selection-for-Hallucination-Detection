# Automatic Layer Selection for Hallucination Detection

## Requirements

- Python 3.10
- CUDA 12.8

## 1. Install dependencies

```bash
pip install torch torchvision 
pip install transformers datasets evaluate
pip install faiss-gpu          # GPU build (recommended)
# pip install faiss-cpu        # CPU-only fallback
pip install scikit-learn scipy numpy pandas matplotlib tqdm
pip install skdim pynvml nltk wandb
```


## 2. Workflow

`construct_dataset.py` -> `get_activation.py` -> `method/ID_twonn.py`


