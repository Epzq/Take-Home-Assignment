import hashlib
import json
import os
import logging
import random
import numpy as np
import torch
from datetime import datetime
from datasets import load_dataset, load_from_disk
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

def set_seed(seed: int = 42) -> None:
    """Ensure determinism"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_row(example):
    title = example["Title"] if example["Title"] else ""
    desc = example["Description"] if example["Description"] else ""
    # Convert 1..4 -> 0..3
    return {
        "text": f"{title} - {desc}".strip(),
        "label": int(example["Class Index"]) - 1,  
    }

def load_and_version_data(
    raw_csv_dir: str = "data/raw_csv",
    save_dir: str = "data",
    train_samples: int = -1,
    test_samples: int = -1,
) -> tuple:
    train_path = os.path.join(raw_csv_dir, "train.csv")
    test_path = os.path.join(raw_csv_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Missing CSV files in {raw_csv_dir}. Ensure train.csv and test.csv exist."
        )

    data_files = {"train": train_path, "test": test_path}
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.map(
        process_row, remove_columns=["Class Index", "Title", "Description"]
    )

    train_subset = (
        dataset["train"].select(range(train_samples))
        if train_samples > 0
        else dataset["train"]
    )
    test_subset = (
        dataset["test"].select(range(test_samples))
        if test_samples > 0
        else dataset["test"]
    )

    serialized = json.dumps(
        {
            "train": list(train_subset["text"]),
            "test": list(test_subset["text"]),
        },
        sort_keys=True,
    ).encode("utf-8")
    data_hash = hashlib.sha256(serialized).hexdigest()[:10]

    # Check if a dataset with this hash already exists (ignoring timestamp)   
    existing_version_dir = None
    if os.path.exists(save_dir):
        for folder in os.listdir(save_dir):
            if folder.startswith("ag_news_") and folder.endswith(f"_v{data_hash}"):
                existing_version_dir = os.path.join(save_dir, folder)
                break

    if existing_version_dir:
        # Load from existing directory to prevent duplicating identical data across multiple timestamps
        version_path = existing_version_dir
    else:
        # Create a new version directory with current timestamp and hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"ag_news_{timestamp}_v{data_hash}"
        version_path = os.path.join(save_dir, version_name)

        version_train = os.path.join(version_path, "train")
        version_test = os.path.join(version_path, "test")

        os.makedirs(version_path, exist_ok=True)
        train_subset.save_to_disk(version_train)
        test_subset.save_to_disk(version_test)

    train_data = load_from_disk(os.path.join(version_path, "train"))
    test_data = load_from_disk(os.path.join(version_path, "test"))

    version_tag = os.path.basename(version_path)
    return train_data, test_data, version_tag


def evaluate_model(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Calculate accuracy, precision, recall, and f1-score (weighted & macro)."""
    acc = accuracy_score(labels, preds)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision_w), 4),
        "recall": round(float(recall_w), 4),
        "f1_score": round(float(f1_w), 4),
        "precision_macro": round(float(precision_m), 4),
        "recall_macro": round(float(recall_m), 4),
        "f1_macro": round(float(f1_m), 4),
    }
    return metrics


def check_model_promotion(
    metrics: dict, min_accuracy: float = 0.85, min_f1: float = 0.8
) -> bool:
    """Check if model meets performance thresholds"""
    passed_acc = metrics.get("accuracy", 0.0) >= min_accuracy
    passed_f1 = metrics.get("f1_score", 0.0) >= min_f1
    return passed_acc and passed_f1

def display_sample_predictions(
    test_dataset, predictions: np.ndarray, labels: np.ndarray, num_samples: int = 10
):
    """Print a side-by-side view of text, ground truth label, and predicted label."""
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    print("\n" + "=" * 50)
    print("=" * 80)

    for i in range(min(num_samples, len(labels))):
        text_sample = test_dataset[i]["text"]
        true_label_idx = int(labels[i])
        pred_label_idx = int(predictions[i])

        true_label_str = label_map.get(true_label_idx, str(true_label_idx))
        pred_label_str = label_map.get(pred_label_idx, str(pred_label_idx))

        status = "Correct" if true_label_idx == pred_label_idx else "Wrong"
        display_text = (f"{text_sample[:100]}..." if len(text_sample) > 115 else text_sample)

        print(f"[{i+1}] {status}")
        print(f"Text         : {display_text}")
        print(f"Ground Truth : {true_label_str} (Class {true_label_idx})")
        print(f"Predicted    : {pred_label_str} (Class {pred_label_idx})")
        print("-" * 50)