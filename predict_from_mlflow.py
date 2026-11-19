import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from config import REGISTERED_MODEL_NAME, TRACKING_URI


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    alias = "champion"
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    model_version = str(version.version)
    hf_model_id = version.tags.get("hf_model_id", "Not available")

    model = mlflow.pyfunc.load_model(
        f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    )

    data = pd.read_csv("data/new_data.csv")
    if "text" not in data.columns:
        raise ValueError("data/new_data.csv must have a 'text' column.")
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["text"] != ""].copy()
    if not 5 <= len(data) <= 10:
        raise ValueError(f"Expected 5–10 new examples; found {len(data)}.")

    evaluation = pd.read_csv("data/eval_inputs.csv")
    eval_texts = set(
        evaluation["text"].fillna("").astype(str).str.strip().str.casefold()
    )
    new_texts = set(data["text"].str.casefold())
    if eval_texts.intersection(new_texts):
        raise ValueError("New examples overlap with the evaluation examples.")

    results = model.predict(data[["text"]])
    results["champion_model_version"] = model_version
    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/new_data_predictions.csv"
    results.to_csv(output_file, index=False)

    print("Champion loaded from the MLflow Model Registry")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")
    print(f"Alias: {alias}")
    print(f"Model version: {model_version}")
    print(f"Hugging Face model ID: {hf_model_id}")
    print("\nPredictions on new examples")
    print(results.to_string(index=False))
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
