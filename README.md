## Architecture & Design Decisions
### Dataset Versioning
Each run derives a content-based version tag for its data slice:

```
ag_news_{YYYYMMDD}_{HHMMSS}_v{SHA256_hash[:10]}
```

The hash is computed over the text of both the train and test splits.\
The timestamp records when a data was first loaded and versioned,
hashing ensures content of datset version and prevents that two different versions do not overwrite even if they are created in the same second.\
Before creating new version loader scans `data/` for an existing folder ending in `_v{hash}`. If hash match, uses that snapshot instead of saving a duplicate even if timestamp differs.

### Determinism and Reproducibility
Seeds are set across sources of randomness, and cudnn is pinned toeterministic kernels:
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
The same seed is also passed to `TrainingArguments`.

### Promotion Gate
Model promotion has to clear both an accuracy and a F1 threshold before it is written to `models/production_candidate`.\
Prevents a model that suffers from class imbalance from being promoted on accuracy alone.\
Thresholds can be set in args;

```bash
python src/main_pipeline.py --min-accuracy 0.9 --min-f1 0.9
```

### Logging Strategy
Logging written to JSON to `./logs/run_seed_{seed}_{timestamp}.json` locally as environment is assumed to have no direct internet acess, so no reporting to external trackers such as W&B.\
Each record contains the seed, the data version tag, the full metric set, the promotion verdict, the resolved arguments, and the step training history(i.e loss).\
Additionally, structured JSON could be load into a dashboard later if needed.

---

## Repository Structure

```
TASK_B/
├── ckpts/                          # Training checkpoints
├── data/                           # Raw CSVs + versioned dataset
├── logs/                           # Metrics output, one JSON per run
├── models/                         # Model weights 
│   ├── distilbert-base-uncased/    # Pre-trained base model
│   └── production_candidate/       # Promoted model, written on a passing requirements
├── src/
│   ├── main_pipeline.py            # Main execution pipeline
│   └── utils.py                    # Data loading/versioning, evaluation, promotion
├── tests/
│   └── test_pipe.py                # Unit tests for promotion logic and versioning
├── README.md
├── requirements.txt
└── Written_answers.pdf             # PDF file containing written answers for part A and C
```
---

## Setup
Built and tested with **Python 3.10** and **PyTorch 2.13** on CUDA 13.2.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### Downloading pre-trained model

The pipeline loads the base model locally again with assumption of no direct network access, 
so it must be present on disk:

```bash
hf download distilbert-base-uncased --local-dir {LOCAL_PRETRAINED_MODEL_DIR}
```

### Provide the dataset

Dataset is fetched from [Here](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)\
Place the original dataset at `data/ag_news/train.csv` and `data/ag_news/test.csv`. 

---

### Training and promotion

```bash
python src/main_pipeline.py --model-path {LOCAL_PRETRAINED_MODEL_DIR}
```

### Unit tests
```bash
python -m pytest tests/
```

### Args references
```
`--data-dir'        Directory containing the raw CSVs
`--ckpt-dir`        output directory for training checkpoints 
`--model-path`      Pre-trained checkpoint to fine-tune 
`--seed`            Seed for reproducibility 
`--epochs`          Number of training epochs 
`--batch-size`      Per-device train batch size 
`--lr`              Learning rate 
`--max-length`      Max tokenized sequence length 
`--train-samples`   Subset the train split (`-1` = full) 
`--test-samples`    Subset the test split (`-1` = full) 
`--min-accuracy`    Accuracy required for promotion 
`--min-f1`          F1 required for promotion 
`--visualize-preds` Print n samples of predictions vs. ground truth 
```
---
