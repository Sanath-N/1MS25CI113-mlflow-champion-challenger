import pandas as pd
import mlflow


def normalise_label(label):
    """Convert the model label formats to plain positive/negative labels."""
    label = str(label).strip().upper()
    if label in {"NEGATIVE", "LABEL_0"}:
        return "negative"
    if label in {"POSITIVE", "LABEL_1"}:
        return "positive"
    raise ValueError(f"Unexpected sentiment label: {label}")


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Load the Hugging Face pipeline only when MLflow loads the model."""

    def __init__(self, model_id, max_length=256, batch_size=16):
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size

    def load_context(self, context):
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model=self.model_id,
            device=device,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            frame = model_input.copy()
        else:
            frame = pd.DataFrame(model_input)

        if "text" not in frame.columns:
            raise ValueError("Input must include a 'text' column.")

        texts = frame["text"].fillna("").astype(str).str.strip().tolist()
        pipeline_texts = [text if text else " " for text in texts]
        raw = self.classifier(
            pipeline_texts,
            truncation=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": [normalise_label(item["label"]) for item in raw],
                "score": [float(item["score"]) for item in raw],
            }
        )
