import argparse
import time

import mlflow
from mlflow.tracking import MlflowClient

from config import (
    EXPERIMENT_NAME,
    MODELS,
    REGISTERED_MODEL_NAME,
    STUDENT_USN,
    TRACKING_URI,
)


def main():
    parser = argparse.ArgumentParser(
        description="Register both models and assign the champion/challenger aliases."
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Point champion to the other version for the stretch demo.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Experiment not found. Run evaluate.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    parent = next(
        (
            run for run in runs
            if run.data.tags.get("mlflow.runName") == "compare_sentiment"
            and run.info.status == "FINISHED"
        ),
        None,
    )
    if parent is None:
        raise RuntimeError("No completed compare_sentiment run was found.")

    children = {
        run.data.tags.get("mlflow.runName"): run
        for run in runs
        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
    }
    winner = parent.data.tags.get("winner")
    names = {model["display_name"] for model in MODELS}
    if winner not in names:
        raise RuntimeError(f"Unexpected winner tag: {winner!r}")

    versions = {}
    for model in MODELS:
        name = model["display_name"]
        run = children.get(model["run_name"])
        if run is None:
            raise RuntimeError(f"Missing nested run {model['run_name']}.")
        model_uri = run.data.tags.get("logged_model_uri")
        if not model_uri:
            raise RuntimeError(f"No logged_model_uri tag on {model['run_name']}.")

        print(f"Registering {name} from {model_uri}")
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )
        version = str(registered.version)
        for _ in range(120):
            current = client.get_model_version(REGISTERED_MODEL_NAME, version)
            if current.status == "READY":
                break
            if current.status == "FAILED":
                raise RuntimeError(f"Registration failed for {name}.")
            time.sleep(2)
        else:
            raise TimeoutError(f"Registration timed out for {name}.")

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, version, "hf_model_id", model["model_id"]
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, version, "run_name", model["run_name"]
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, version, "student_usn", STUDENT_USN
        )
        versions[name] = version

    champion = next(name for name in names if name != winner) if args.flip else winner
    challenger = next(name for name in names if name != champion)
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "champion", versions[champion]
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "challenger", versions[challenger]
    )

    print("\nRegistry aliases updated")
    print(f"champion: {champion}, version {versions[champion]}")
    print(f"challenger: {challenger}, version {versions[challenger]}")
    if args.flip:
        print("Flip mode active. Run without --flip to restore the rule-based winner.")


if __name__ == "__main__":
    main()
