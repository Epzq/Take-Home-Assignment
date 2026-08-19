import argparse
import json
import os
import shutil
import numpy as np
import torch
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils import (
    check_model_promotion,
    evaluate_model,
    load_and_version_data,
    set_seed,
    display_sample_predictions
)

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        "--data-dir",type=str, default="data/ag_news", help="Path to directory containing local raw CSV files",
    )
    parser.add_argument(
        "--ckpt-dir",type=str, default="ckpts", help="Directory for training output i.e checkpointing",
    )
    parser.add_argument(
        "--model-path",type=str, default="models/distilbert-base-uncased", help="Path of pre-trained model checkpoint",
    )

    # Hyperparameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size per GPU/CPU"
    )
    parser.add_argument(
        "--lr",type=float,default=5e-5,help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--max-length",type=int,default=128,help="Maximum sequence length for tokenization",
    )

    # Dataset Slicing (for quick testing and debugging)
    parser.add_argument(
        "--train-samples", type=int, default=-1, help="Number of training samples to subset (-1 for full set)",
    )
    parser.add_argument( 
        "--test-samples", type=int, default=-1, help="Number of testing samples to subset (-1 for full set)",
    )

    # Promotion Requirements Thresholds
    parser.add_argument( 
        "--min-accuracy", type=float, default=0.85, help="Minimum accuracy required to pass model promotion",
    )
    parser.add_argument( 
        "--min-f1", type=float, default=0.8, help="Minimum F1 score required to pass model promotion",
    )
    parser.add_argument( 
        '--visualize-preds', action='store_true', help="Print and visualize some prediction samples of model"
    )
    return parser.parse_args()


def run_pipeline(args):
    set_seed(args.seed)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Load and version data
    train_data, test_data, data_version = load_and_version_data(
        raw_csv_dir=args.data_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=4, local_files_only=True)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_tokens = train_data.map(tokenize_fn, batched=True)
    test_tokens = test_data.map(tokenize_fn, batched=True)

    # Trainer args

    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        seed=args.seed,
        data_seed=args.seed,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="no",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
    )
    trainer.train()

    # Evaluation
    predictions = trainer.predict(test_tokens)
    preds = np.argmax(predictions.predictions, axis=-1)
    metrics = evaluate_model(preds, predictions.label_ids)

    #Model Promotion Check
    is_promoted = check_model_promotion(
        metrics, min_accuracy=args.min_accuracy, min_f1=args.min_f1
    )

    log_info = {
        "seed": args.seed,
        "data_version": data_version,
        "metrics": metrics,
        "is_promoted": is_promoted,
        "args": vars(args),
        "training_step_history": trainer.state.log_history,
    }
    log_file = os.path.join("logs", f"run_seed_{args.seed}_{timestamp}.json")
    with open(log_file, "w") as f:
        json.dump(log_info, f, indent=2)

    print("=" * 50)
    for k, v in metrics.items():
        print(f" - {k:<18}: {v}")
    print("=" * 50 + "\n")

    if args.visualize_preds:
        display_sample_predictions(
            test_dataset=test_data, 
            predictions=preds, 
            labels=predictions.label_ids, 
            num_samples=5
        )

    if is_promoted:
        promotion_path = os.path.join("models", "production_candidate")
        if os.path.exists(promotion_path):
            shutil.rmtree(promotion_path)
        trainer.save_model(promotion_path)
        tokenizer.save_pretrained(promotion_path)

    return metrics, is_promoted


if __name__ == "__main__":
    args = parse_args()
    metrics, promoted = run_pipeline(args)
    print(
        f"Pipeline complete. Metrics: {metrics} \nPromoted to Production Candidate: {promoted}"
    )