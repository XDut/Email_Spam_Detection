#!/usr/bin/env python3
"""
train_save.py

Train a simple text classifier (CountVectorizer + MultinomialNB) and save the fitted Pipeline.
Detects a few common column names for spam datasets (Category/Message, v1/v2, label/text).
Prints evaluation metrics and saves the model to disk.

Usage:
    python train_save.py --input data/spam.csv --output models/spam_classifier.pkl
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Accept multiple common formats
    if "category" in (k := {c.lower(): c for c in df.columns}):
        cat_col = cols["category"]
        msg_col = cols.get("message") or cols.get("v2") or cols.get("text") or cols.get("message_text")
    elif "v1" in (k := {c.lower(): c for c in df.columns}):
        cat_col = cols["v1"]
        msg_col = cols.get("v2")
    elif "label" in (k := {c.lower(): c for c in df.columns}):
        cat_col = cols["label"]
        msg_col = cols.get("text") or cols.get("message")
    else:
        # fallback to first two columns (guess)
        if len(df.columns) >= 2:
            cat_col = df.columns[0]
            msg_col = df.columns[1]
        else:
            raise ValueError("Couldn't infer category/message columns from CSV.")

    if msg_col is None:
        raise ValueError("Couldn't find a message/text column in the input CSV.")

    df = df[[cat_col, msg_col]].rename(columns={cat_col: "Category", msg_col: "Message"})
    return df


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Create binary IsSpam column. Map common spam labels.
    def label_to_binary(x):
        if isinstance(x, str):
            x_lower = x.strip().lower()
            if x_lower in {"spam", "s", "1", "true", "yes", "y"}:
                return 1
            if x_lower in {"ham", "not spam", "0", "false", "no", "n"}:
                return 0
        # fallback: try numeric
        try:
            return 1 if int(x) == 1 else 0
        except Exception:
            # if unknown, treat as ham (0)
            return 0

    df = df.copy()
    df["Message"] = df["Message"].astype(str)
    df["IsSpam"] = df["Category"].apply(label_to_binary)
    df = df.dropna(subset=["Message"])
    return df


def build_and_train(X_train, y_train):
    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("nb", MultinomialNB()),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    clf_report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()
    return {"accuracy": acc, "classification_report": clf_report, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser(description="Train and save spam classifier pipeline.")
    parser.add_argument("--input", "-i", type=str, default="data/spam.csv", help="Path to CSV dataset")
    parser.add_argument("--output", "-o", type=str, default="models/spam_classifier.pkl", help="Path to save model")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--metrics-out", type=str, default="models/metrics.json", help="Where to save metrics JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading dataset from %s", input_path)
    df = load_dataset(input_path)
    df = prepare_labels(df)

    X = df["Message"]
    y = df["IsSpam"]

    logging.info("Splitting dataset (test_size=%s, random_state=%s)", args.test_size, args.random_state)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
    except ValueError:
        # If stratify fails due to single class in y, fallback to no stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

    logging.info("Building and training pipeline")
    pipeline = build_and_train(X_train, y_train)

    logging.info("Evaluating on test set")
    metrics = evaluate(pipeline, X_test, y_test)
    logging.info("Accuracy: %.4f", metrics["accuracy"])
    logging.info("Classification report (summary):\n%s", json.dumps(metrics["classification_report"]["weighted avg"], indent=2))

    # Save pipeline
    logging.info("Saving model to %s", output_path)
    joblib.dump(pipeline, output_path)

    # Save metrics JSON
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
