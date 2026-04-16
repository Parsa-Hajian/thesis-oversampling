"""
Master experiment runner.

Runs the full experiment matrix as defined in configs/experiment.yaml:
  - For each seed selection strategy (custom, gaussian)
    - For each dataset
      - For each proposed method + baseline
        - For each classifier
          - 5-fold CV with oversampling inside folds
          - Report median metrics

Results are saved to results/tables/ as CSV files.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --config configs/experiment.yaml
    python experiments/run_all.py --dataset ecoli1 --method gravity_vonmises_km_A
"""

import argparse
import os
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.circular_smote import CircularSMOTE
from src.core.gravity_vonmises import GravityVonMises
from src.core.local_regions import LocalRegions
from src.core.layered_segmental import LayeredSegmentalOversampler
from src.seed_selection.selector import SeedSelector
from src.seed_selection.gaussian import GaussianSeedSelector
from src.comparison.baselines import get_baseline
from src.evaluation.classifiers import get_classifier
from src.evaluation.cross_validation import cross_validate_with_oversampling
from src.datasets.loader import load_dataset


# Registry: method name -> class
METHOD_CLASSES = {
    "circular_smote": CircularSMOTE,
    "gravity_vonmises": GravityVonMises,
    "local_regions": LocalRegions,
    "layered_segmental": LayeredSegmentalOversampler,
}


def create_oversampler(method_config, random_state=42):
    """Create an oversampler instance from config."""
    cls_name = method_config["class"]
    params = method_config.get("params", {})
    params["random_state"] = random_state
    params["sampling_strategy"] = params.get("sampling_strategy", 1.0)
    return METHOD_CLASSES[cls_name](**params)


def run_single_experiment(dataset_name, method_name, method_config, classifier_name,
                          seed_strategy=None, denoise_method=None,
                          cv_folds=5, random_state=42):
    """
    Run a single experiment: one dataset, one method, one classifier.

    Returns dict with all metric results.
    """
    # Load dataset
    X, y = load_dataset(dataset_name)

    # Create oversampler
    if method_config.get("is_baseline", False):
        oversampler = get_baseline(method_name, random_state=random_state)
    else:
        oversampler = create_oversampler(method_config, random_state=random_state)

    # Create classifier
    clf = get_classifier(classifier_name)

    # Run cross-validation
    results = cross_validate_with_oversampling(
        X, y, oversampler, clf,
        n_folds=cv_folds,
        random_state=random_state,
    )

    return {
        "dataset": dataset_name,
        "method": method_name,
        "classifier": classifier_name,
        "seed_strategy": seed_strategy or "none",
        "denoise": denoise_method or "none",
        **{f"{k}_median": v for k, v in results["median"].items()},
        **{f"{k}_std": v for k, v in results["std"].items()},
    }


def run_full_experiment(config_path="configs/experiment.yaml"):
    """
    Run the complete experiment matrix from config.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    random_state = config.get("random_state", 42)
    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    datasets = config["datasets"]["keel_uci"]
    seed_strategies = config["seed_strategies"]
    proposed_methods = config["proposed_methods"]
    baselines = config["baselines"]
    classifiers = config["classifiers"]
    cv_folds = config["evaluation"]["cv_folds"]

    all_results = []

    # Total experiments for progress bar
    n_methods = len(proposed_methods) + len(baselines)
    total = len(seed_strategies) * len(datasets) * n_methods * len(classifiers)

    pbar = tqdm(total=total, desc="Running experiments")

    for seed_config in seed_strategies:
        seed_name = seed_config["name"]

        for dataset_name in datasets:
            # Run baselines (seed-independent)
            for baseline_name in baselines:
                for clf_name in classifiers:
                    try:
                        result = run_single_experiment(
                            dataset_name=dataset_name,
                            method_name=baseline_name,
                            method_config={"is_baseline": True},
                            classifier_name=clf_name,
                            seed_strategy=seed_name,
                            cv_folds=cv_folds,
                            random_state=random_state,
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"ERROR: {dataset_name}/{baseline_name}/{clf_name}: {e}")
                    pbar.update(1)

            # Run proposed methods
            for method_config in proposed_methods:
                method_name = method_config["name"]
                for clf_name in classifiers:
                    try:
                        result = run_single_experiment(
                            dataset_name=dataset_name,
                            method_name=method_name,
                            method_config=method_config,
                            classifier_name=clf_name,
                            seed_strategy=seed_name,
                            cv_folds=cv_folds,
                            random_state=random_state,
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"ERROR: {dataset_name}/{method_name}/{clf_name}: {e}")
                    pbar.update(1)

    pbar.close()

    # Save results
    df = pd.DataFrame(all_results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / "tables" / f"full_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total experiments: {len(all_results)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run oversampling experiments")
    parser.add_argument("--config", default="configs/experiment.yaml",
                       help="Path to experiment config YAML")
    parser.add_argument("--dataset", default=None,
                       help="Run only this dataset (optional)")
    parser.add_argument("--method", default=None,
                       help="Run only this method (optional)")
    args = parser.parse_args()

    run_full_experiment(args.config)


if __name__ == "__main__":
    main()
