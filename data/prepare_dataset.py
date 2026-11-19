from pathlib import Path
import sys

import pandas as pd
from datasets import load_dataset

# When this file is run directly, Python starts its import search in data/.
# Add the project folder so that config.py can be imported from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASET_NAME, DATASET_SPLIT, EVAL_SAMPLES, SEED


def main():
    print(f"Loading {DATASET_NAME}, split={DATASET_SPLIT}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    frame = pd.DataFrame(dataset)

    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    frame["label"] = frame["label"].map({0: "negative", 1: "positive"})
    frame = frame.dropna(subset=["text", "label"]).copy()

    per_class = EVAL_SAMPLES // 2
    negative = frame[frame["label"] == "negative"].sample(
        n=per_class, random_state=SEED
    )
    positive = frame[frame["label"] == "positive"].sample(
        n=per_class, random_state=SEED
    )
    evaluation = (
        pd.concat([negative, positive])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    Path("data").mkdir(exist_ok=True)
    evaluation.to_csv("data/eval_inputs.csv", index=False)

    new_path = Path("data/new_data.csv")
    if not new_path.exists():
        raise FileNotFoundError("Add 5–10 examples to data/new_data.csv first.")
    new_data = pd.read_csv(new_path)
    eval_texts = set(evaluation["text"].astype(str).str.strip().str.casefold())
    new_texts = set(new_data["text"].fillna("").astype(str).str.strip().str.casefold())
    if eval_texts.intersection(new_texts):
        raise ValueError("Some new examples also appear in the evaluation set.")

    print(f"Saved {len(evaluation)} validation examples to data/eval_inputs.csv")
    print(evaluation["label"].value_counts().to_string())
    print("New-data examples are separate from the evaluation examples.")


if __name__ == "__main__":
    main()
