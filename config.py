STUDENT_USN = "1MS25CI113"
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

TASK = "sentiment"
DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
DATASET_SPLIT = "validation"
EVAL_SAMPLES = 100
SEED = 42
BATCH_SIZE = 16
MAX_LENGTH = 256

PRIMARY_METRIC = "macro_f1"
MINIMUM_MARGIN = 0.01
MAX_LATENCY_RATIO = 5.0

MODELS = [
    {
        "display_name": "DistilBERT",
        "short_name": "distilbert",
        "run_name": "A_sentiment_distilbert-sst2",
        "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion_candidate",
    },
    {
        "display_name": "RoBERTa",
        "short_name": "roberta",
        "run_name": "B_sentiment_roberta-large",
        "model_id": "siebert/sentiment-roberta-large-english",
        "role": "challenger_candidate",
    },
]
