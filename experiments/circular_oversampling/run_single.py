"""
Run a single experiment for quick testing/debugging.

Usage:
    python experiments/run_single.py --dataset ecoli1 --method smote --classifier knn
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.loader import load_dataset
from src.comparison.baselines import get_baseline, BASELINES
from src.core.circular_smote import CircularSMOTE
from src.core.gravity_vonmises import GravityVonMises
from src.core.local_regions import LocalRegions
from src.core.layered_segmental import LayeredSegmentalOversampler
from src.evaluation.classifiers import get_classifier
from src.evaluation.cross_validation import cross_validate_with_oversampling


PROPOSED_METHODS = {
    "circular_smote": CircularSMOTE,
    "gravity_vonmises": GravityVonMises,
    "local_regions": LocalRegions,
    "layered_segmental": LayeredSegmentalOversampler,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--classifier", default="knn")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Classifier: {args.classifier}")
    print(f"Folds: {args.folds}")
    print()

    # Load dataset
    X, y = load_dataset(args.dataset)
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    classes, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(classes, counts))}")
    print(f"IR: {max(counts)/min(counts):.2f}")
    print()

    # Create oversampler
    if args.method in BASELINES:
        oversampler = get_baseline(args.method, random_state=args.seed)
    elif args.method in PROPOSED_METHODS:
        oversampler = PROPOSED_METHODS[args.method](random_state=args.seed)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Create classifier
    clf = get_classifier(args.classifier)

    # Run CV
    results = cross_validate_with_oversampling(
        X, y, oversampler, clf, n_folds=args.folds, random_state=args.seed,
    )

    print("Results (median across folds):")
    for metric, value in results["median"].items():
        print(f"  {metric}: {value:.4f}")

    print("\nPer-fold values:")
    for fold_idx, fold in enumerate(results["folds"]):
        print(f"  Fold {fold_idx}: {', '.join(f'{k}={v:.4f}' for k, v in fold.items())}")


if __name__ == "__main__":
    main()
