import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

from config import (
    BATCH_SIZE, DATASET_NAME, DATASET_SPLIT, EVAL_SAMPLES, EXPERIMENT_NAME,
    MAX_LATENCY_RATIO, MAX_LENGTH, MINIMUM_MARGIN, MODELS, PRIMARY_METRIC,
    SEED, STUDENT_USN, TASK, TRACKING_URI,
)
from hf_wrapper import HFSentimentModel, normalise_label


def disagreement_comment(text):
    lower = f" {text.casefold()} "
    comments = []
    if any(x in lower for x in (" not ", " never ", " no ", "n't", " hardly ", " barely ")):
        comments.append("Negation may change which part of the review carries the sentiment.")
    if any(x in lower for x in (" but ", " although ", " however ", " though ", " yet ")):
        comments.append("The contrast between clauses may lead the models to weigh different parts.")
    if any(x in lower for x in (" terribly ", " really ", " very ", " awfully ", " quite ")):
        comments.append("An intensifier may be read differently in its surrounding context.")
    return " ".join(comments) or (
        "The wording is brief or mixed; compare which sentiment cue each model may have weighted."
    )


def replace_section(text, start_marker, end_marker, replacement):
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + "\n\n" + replacement.strip() + "\n\n" + text[end:]


def format_results_table(results):
    by_name = {row["model"]: row for row in results}
    metrics = [
        ("Accuracy", "accuracy", 4),
        ("Precision", "precision", 4),
        ("Recall", "recall", 4),
        ("Macro-F1", "macro_f1", 4),
        ("Average latency (ms)", "avg_latency_ms", 2),
        ("Total inference time (s)", "total_inference_s", 2),
    ]
    lines = [
        "| Metric | DistilBERT | RoBERTa | Difference (RoBERTa − DistilBERT) |",
        "|---|---:|---:|---:|",
    ]
    for label, key, digits in metrics:
        a = by_name["DistilBERT"][key]
        b = by_name["RoBERTa"][key]
        lines.append(f"| {label} | {a:.{digits}f} | {b:.{digits}f} | {b-a:+.{digits}f} |")
    return "\n".join(lines), by_name


def make_decision_paragraph(by_name, winner, delta, latency_ratio, sample_count):
    a, b = by_name["DistilBERT"], by_name["RoBERTa"]
    if winner == "RoBERTa":
        outcome = (
            "RoBERTa passed both parts of the prewritten rule and becomes the Champion: "
            "its macro-F1 gain met the margin and its latency stayed inside the guardrail."
        )
    else:
        outcome = (
            "RoBERTa did not pass both parts of the prewritten rule, so DistilBERT remains "
            "the Champion. The rule retains the faster incumbent when the gain is too small "
            "or the latency guardrail fails."
        )
    return (
        f"On the fixed {sample_count}-review Rotten Tomatoes validation sample, RoBERTa "
        f"(the Challenger) reached macro-F1 {b['macro_f1']:.4f}, while DistilBERT (the "
        f"initial Champion) reached {a['macro_f1']:.4f}. The difference was {delta:+.4f}, "
        f"against the precommitted minimum improvement of {MINIMUM_MARGIN:.2f}. {outcome} "
        f"Average latency was {a['avg_latency_ms']:.2f} ms per review for DistilBERT and "
        f"{b['avg_latency_ms']:.2f} ms for RoBERTa, a ratio of {latency_ratio:.2f} times. "
        "That speed cost matters when processing many reviews, and RoBERTa also has a much "
        "larger model download and storage footprint. Both models received the same saved "
        "inputs in the same order, so this is a fair comparison for this run. The sample has "
        "only 50 examples in each class; one changed prediction is 1% of the sample, and "
        "macro-F1 is not a direct example count. The result can therefore change on a larger "
        "or different dataset. The disagreement file contains examples for discussion, but a "
        "confidence score does not explain a model's reasoning. This project covers one short "
        "review dataset and two pretrained classifiers; it does not train a model or establish "
        "which model is best for every sentiment task."
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs("outputs/plots", exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = pd.read_csv("data/eval_inputs.csv")
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data["label"] = data["label"].fillna("").astype(str).str.lower().str.strip()
    data = data[data["label"].isin(["positive", "negative"])].copy()
    if len(data) != EVAL_SAMPLES:
        raise ValueError(
            f"Expected {EVAL_SAMPLES} evaluation rows; found {len(data)}. "
            "Run python data/prepare_dataset.py first."
        )

    texts, actual = data["text"].tolist(), data["label"].tolist()
    data.to_csv("outputs/eval_inputs.csv", index=False)
    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"
    library_versions = (
        f"transformers={transformers.__version__}; torch={torch.__version__}; "
        f"mlflow={mlflow.__version__}; scikit-learn={sklearn.__version__}"
    )
    results, predictions_by_model = [], {}

    print(f"Evaluation samples: {len(data)}; device: {device_name}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    with mlflow.start_run(run_name=f"compare_{TASK}"):
        for model_info in MODELS:
            name, model_id = model_info["display_name"], model_info["model_id"]
            print(f"\nEvaluating {name}: {model_id}")

            with mlflow.start_run(run_name=model_info["run_name"], nested=True):
                classifier = pipeline("text-classification", model=model_id, device=device)
                classifier(texts[:1], truncation=True, max_length=MAX_LENGTH, batch_size=1)

                started = time.perf_counter()
                raw = classifier(
                    texts, truncation=True, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
                )
                total_s = time.perf_counter() - started
                predicted = [normalise_label(row["label"]) for row in raw]
                scores = [float(row["score"]) for row in raw]
                latency = total_s * 1000 / len(texts)

                metrics = {
                    "accuracy": accuracy_score(actual, predicted),
                    "precision": precision_score(
                        actual, predicted, pos_label="positive", zero_division=0
                    ),
                    "recall": recall_score(
                        actual, predicted, pos_label="positive", zero_division=0
                    ),
                    "macro_f1": f1_score(
                        actual, predicted, average="macro", zero_division=0
                    ),
                    "avg_latency_ms": latency,
                    "total_inference_s": total_s,
                }

                pred_file = f"outputs/predictions_{model_info['short_name']}.csv"
                pred_frame = pd.DataFrame(
                    {
                        "text": texts,
                        "reference": actual,
                        "prediction": predicted,
                        "score": scores,
                        "correct": [a == p for a, p in zip(actual, predicted)],
                    }
                )
                pred_frame.to_csv(pred_file, index=False)
                predictions_by_model[name] = pred_frame

                cm_file = f"outputs/plots/{model_info['short_name']}_confusion_matrix.png"
                cm = confusion_matrix(actual, predicted, labels=["negative", "positive"])
                ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=["Negative", "Positive"],
                ).plot(cmap="Blues", values_format="d")
                plt.title(f"{name} Confusion Matrix")
                plt.tight_layout()
                plt.savefig(cm_file)
                plt.close()

                mlflow.log_params(
                    {
                        "hf_model_id": model_id,
                        "task": TASK,
                        "role": model_info["role"],
                        "max_length": MAX_LENGTH,
                        "batch_size": BATCH_SIZE,
                        "eval_samples": len(data),
                        "seed": SEED,
                        "device": device_name,
                    }
                )
                mlflow.set_tags(
                    {
                        "student_usn": STUDENT_USN,
                        "dataset_name": f"{DATASET_NAME}:{DATASET_SPLIT}",
                        "library_versions": library_versions,
                    }
                )
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(pred_file)
                mlflow.log_artifact("data/eval_inputs.csv")
                mlflow.log_artifact(cm_file)

                logged_model = mlflow.pyfunc.log_model(
                    name=f"{model_info['short_name']}_model",
                    python_model=HFSentimentModel(
                        model_id, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
                    ),
                    code_paths=["hf_wrapper.py"],
                    pip_requirements=[
                        "mlflow==3.18.0",
                        "transformers==5.23.0",
                        "torch==2.15.0",
                        "pandas==3.1.1",
                    ],
                )
                mlflow.set_tag("logged_model_uri", logged_model.model_uri)

                results.append(
                    {
                        "model": name,
                        "hf_model_id": model_id,
                        "run_name": model_info["run_name"],
                        **metrics,
                    }
                )
                print(metrics)

        by_name = {row["model"]: row for row in results}
        delta = by_name["RoBERTa"][PRIMARY_METRIC] - by_name["DistilBERT"][PRIMARY_METRIC]
        latency_ratio = by_name["RoBERTa"]["avg_latency_ms"] / max(
            by_name["DistilBERT"]["avg_latency_ms"], 1e-9
        )
        winner = (
            "RoBERTa"
            if delta >= MINIMUM_MARGIN and latency_ratio <= MAX_LATENCY_RATIO
            else "DistilBERT"
        )

        disagreements = data[["text", "label"]].rename(
            columns={"label": "reference"}
        ).copy()
        disagreements["distilbert_prediction"] = predictions_by_model[
            "DistilBERT"
        ]["prediction"].values
        disagreements["roberta_prediction"] = predictions_by_model[
            "RoBERTa"
        ]["prediction"].values
        disagreements = disagreements[
            disagreements["distilbert_prediction"]
            != disagreements["roberta_prediction"]
        ].head(5).copy()
        disagreements["comment"] = disagreements["text"].map(disagreement_comment)
        disagreement_file = "outputs/model_disagreements.csv"
        disagreements.to_csv(disagreement_file, index=False)

        summary_frame = pd.DataFrame(results)
        summary_frame.to_csv("outputs/metrics_summary.csv", index=False)
        summary = {
            "task": TASK,
            "dataset_name": DATASET_NAME,
            "dataset_split": DATASET_SPLIT,
            "eval_samples": len(data),
            "primary_metric": PRIMARY_METRIC,
            "minimum_margin": MINIMUM_MARGIN,
            "maximum_latency_ratio": MAX_LATENCY_RATIO,
            "delta_primary_metric": delta,
            "latency_ratio": latency_ratio,
            "winner": winner,
            "models": results,
        }
        with open("outputs/metrics_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        quality_plot = "outputs/plots/model_quality_comparison.png"
        summary_frame.set_index("model")[
            ["accuracy", "precision", "recall", "macro_f1"]
        ].plot(kind="bar", figsize=(9, 5))
        plt.title("DistilBERT vs RoBERTa: Classification Metrics")
        plt.ylabel("Score (higher is better)")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(quality_plot)
        plt.close()

        latency_plot = "outputs/plots/latency_comparison.png"
        summary_frame.set_index("model")["avg_latency_ms"].plot(
            kind="bar", color=["steelblue", "darkorange"], figsize=(7, 5)
        )
        plt.title("Average Inference Latency")
        plt.ylabel("Milliseconds per review (lower is better)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(latency_plot)
        plt.close()

        table, by_name = format_results_table(results)
        readme_path = Path("README.md")
        readme = readme_path.read_text(encoding="utf-8")
        readme = replace_section(
            readme, "<!-- RESULTS_START -->", "<!-- RESULTS_END -->", table
        )
        paragraph = make_decision_paragraph(
            by_name, winner, delta, latency_ratio, len(data)
        )
        readme = replace_section(
            readme, "<!-- DECISION_START -->", "<!-- DECISION_END -->", paragraph
        )
        readme_path.write_text(readme, encoding="utf-8")

        mlflow.log_params(
            {
                "primary_metric": PRIMARY_METRIC,
                "margin": MINIMUM_MARGIN,
                "max_latency_ratio": MAX_LATENCY_RATIO,
                "eval_samples": len(data),
                "seed": SEED,
            }
        )
        mlflow.log_metrics(
            {"delta_primary_metric": delta, "latency_ratio": latency_ratio}
        )
        mlflow.set_tags(
            {
                "winner": winner,
                "student_usn": STUDENT_USN,
                "dataset_name": f"{DATASET_NAME}:{DATASET_SPLIT}",
            }
        )
        for artifact in (
            "data/eval_inputs.csv",
            "outputs/eval_inputs.csv",
            "outputs/metrics_summary.csv",
            "outputs/metrics_summary.json",
            disagreement_file,
            "README.md",
            "requirements.txt",
        ):
            mlflow.log_artifact(artifact)
        mlflow.log_artifacts("outputs/plots", artifact_path="plots")

    print("\nFinal comparison")
    print(summary_frame.to_string(index=False))
    print(f"\nSelected Champion: {winner}")
    print(f"Challenger macro-F1 delta: {delta:+.4f}")
    print(f"Latency ratio: {latency_ratio:.2f}x")
    print("README results and decision paragraph updated.")


if __name__ == "__main__":
    main()
