# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student:** Sanath  
**USN:** 1MS25CI113

## 1. Task and Models

**Task:** Binary sentiment classification.

**Dataset:** [Rotten Tomatoes movie reviews](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes), validation split. The project selects 100 examples with a fixed seed: 50 positive and 50 negative.

**Model A — initial Champion:**
- DistilBERT: \`distilbert/distilbert-base-uncased-finetuned-sst-2-english\`

**Model B — Challenger:**
- RoBERTa: \`siebert/sentiment-roberta-large-english\`

Both models receive the same saved rows from \`data/eval_inputs.csv\`. The six examples in \`data/new_data.csv\` are reserved for registry prediction and are checked to ensure they do not overlap with the evaluation set.

## 2. How to Run

Run these commands in the VS Code terminal from the project folder. Keep the MLflow server running in its own terminal.

### Create the environment and install packages

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

### Prepare evaluation inputs

~~~powershell
python data\prepare_dataset.py
~~~

This samples 100 balanced examples from the Rotten Tomatoes validation split using seed 42.

### Commit the rule and project before the final evaluation

This first commit records the Champion rule before the final run. Do not run evaluate.py until this commit is made.

~~~powershell
git init
git add .gitignore README.md requirements.txt config.py hf_wrapper.py evaluate.py register_champion.py predict_from_mlflow.py main.py data
git commit -m "Add project and pre-run champion rule"
~~~

### Start the MLflow server

Open a second VS Code terminal at the project folder and leave this running:

~~~powershell
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
~~~

Open \`http://127.0.0.1:5000\` in a browser.

### Evaluate, register, and predict

In the first terminal:

~~~powershell
python evaluate.py
python register_champion.py
python predict_from_mlflow.py
~~~

The evaluation updates the results and decision sections below. It writes the shared-input predictions, metrics, plots, and disagreements to \`outputs/\`.

Make the second commit:

~~~powershell
git add README.md outputs
git commit -m "Add final evaluation and prediction results"
~~~

Capture the required screenshots in \`screenshots/\`, then make the third commit:

~~~powershell
git add screenshots
git commit -m "Add MLflow submission screenshots"
~~~

### Push to GitHub

Create a GitHub repository named \`1MS25CI113-mlflow-champion-challenger\`. Do not initialize it with another README or .gitignore. Replace \`YOUR_GITHUB_USERNAME\` with your GitHub username:

~~~powershell
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/1MS25CI113-mlflow-champion-challenger.git
git push -u origin main
~~~

The link to submit will be \`https://github.com/YOUR_GITHUB_USERNAME/1MS25CI113-mlflow-champion-challenger\`.

## 3. Champion–Challenger Decision Rule

This rule is committed before the final full evaluation. DistilBERT starts as the Champion and RoBERTa is the Challenger.

- **Primary metric:** Macro-F1. Higher is better.
- **Minimum improvement:** Promote RoBERTa only if its macro-F1 is at least 0.01 higher than DistilBERT. With 100 examples, one changed prediction is 1% of the sample, so a smaller gain is too small to justify replacing the current Champion.
- **Latency guardrail and tie-breaker:** RoBERTa must take no more than five times DistilBERT's average latency. If the improvement is below 0.01 or the guardrail fails, retain DistilBERT, the faster incumbent.
- **Metric directions:** Accuracy, precision, recall, macro-F1, and Challenger-minus-Champion macro-F1 delta are higher-is-better. Average latency and total inference time are lower-is-better.

The code applies this same rule and records the winner on the MLflow comparison run.

## 4. Results Table

<!-- RESULTS_START -->
The final results table will be filled automatically by evaluate.py.
<!-- RESULTS_END -->

## 5. Decision and Conclusion

<!-- DECISION_START -->
The final decision paragraph will be filled automatically by evaluate.py after the final run.
<!-- DECISION_END -->

## 6. Limitations and Declaration

### Limitations

- The evaluation contains 100 short reviews from one dataset and may not represent other domains or longer text.
- Latency depends on the computer, processor or GPU, background activity, and package versions.
- Only two pretrained models were compared. No model was trained.
- The result applies to this fixed evaluation and rule, not every sentiment task.

### AI-tool declaration

ChatGPT was used for step-by-step guidance, code review, debugging assistance, and documentation. I reviewed the project and can explain the dataset split, shared inputs, MLflow parameters, metrics, tags, artifacts, model wrapper, registry aliases, and new-data prediction flow.
