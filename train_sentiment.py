# train_sentiment.py — stable on transformers 4.57: eval/save at STEPS, no best-at-end constraint

import argparse
import os
import numpy as np

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_w": prec_w, "recall_w": rec_w, "f1_w": f1_w,
        "precision_m": prec_m, "recall_m": rec_m, "f1_m": f1_m
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on Amazon reviews")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--dataset", default="mteb/amazon_polarity",
                        help='Use "mteb/amazon_polarity" (text,label) or "amazon_polarity" (title,content,label)')
    parser.add_argument("--train_samples", type=int, default=20000)
    parser.add_argument("--eval_samples", type=int, default=5000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="./sentiment_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load + subset
    ds = load_dataset(args.dataset)
    train_ds = ds["train"].shuffle(seed=args.seed).select(
        range(min(args.train_samples, len(ds["train"]))))
    eval_ds = ds["test"].shuffle(seed=args.seed).select(
        range(min(args.eval_samples,  len(ds["test"]))))

    # Detect schema
    cols = set(train_ds.column_names)
    if "text" in cols:
        text_mode = "text_only"              # mteb/amazon_polarity
    elif "content" in cols and "title" in cols:
        text_mode = "title_plus_content"     # amazon_polarity
    elif "content" in cols:
        text_mode = "content_only"
    else:
        raise ValueError(
            f"No usable text field in columns: {train_ds.column_names}")

    # Model + tokenizer
    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    # Tokenize
    def build_texts(ex):
        if text_mode == "text_only":
            return ex["text"]
        elif text_mode == "content_only":
            return ex["content"]
        else:
            titles = ex.get("title", [""] * len(ex["content"]))
            contents = ex["content"]
            return [((t or "") + " " + (c or "")).strip() for t, c in zip(titles, contents)]

    def tokenize_batch(ex):
        return tokenizer(build_texts(ex), truncation=True, max_length=256)

    drop_train = [c for c in train_ds.column_names if c != "label"]
    drop_eval = [c for c in eval_ds.column_names if c != "label"]
    train_tok = train_ds.map(
        tokenize_batch, batched=True, remove_columns=drop_train)
    eval_tok = eval_ds.map(tokenize_batch,  batched=True,
                           remove_columns=drop_eval)

    # Strategies: use STEPS for BOTH, disable best-at-end to avoid mismatch requirements
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        load_best_model_at_end=False,  # <- avoids the eval/save equality constraint
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    preds = trainer.predict(eval_tok)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=[
          id2label[0], id2label[1]], zero_division=0))
    print("=== Confusion Matrix (rows: true, cols: pred) ===")
    print(confusion_matrix(y_true, y_pred))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
