"""
pso_feature_selection.py

Member 2 - PSO Feature Selection

Responsibilities:
1. Build the PSO fitness function.
2. Perform Binary Particle Swarm Optimization (PSO).
3. Select a subset of useful features.
4. Compare PSO against baseline feature-selection methods:
      - All Features
      - Mutual Information
      - RFE
      - Genetic Algorithm
5. Return PSO history and runtime for downstream visualization.
6. Use TRAINING data only for feature selection.

Important:
The test set must NEVER be used inside this module.
"""

import hashlib
import time
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


SEED = 42

# Cache for previously completed PSO runs
_PSO_CACHE = {}


# ============================================================
# Helper functions
# ============================================================

def _as_dataframe(X, feature_cols=None):
    """
    Convert X into a pandas DataFrame if it is not already one.
    """

    if isinstance(X, pd.DataFrame):
        return X.copy()

    return pd.DataFrame(X, columns=feature_cols)


def _as_array(y):
    """
    Convert target values to a NumPy array.
    """

    if isinstance(y, pd.Series):
        return y.to_numpy()

    return np.asarray(y)


def _safe_feature_count(mask):
    """
    Return number of selected features.
    """

    return int(np.sum(mask))


def _dataset_hash(X, y):
    """
    Create a reproducible hash for caching.
    """

    X_bytes = pd.util.hash_pandas_object(
        X,
        index=True
    ).values.tobytes()

    y_bytes = pd.util.hash_pandas_object(
        pd.Series(y),
        index=True
    ).values.tobytes()

    return hashlib.md5(X_bytes + y_bytes).hexdigest()


# ============================================================
# Fast classifier used by PSO
# ============================================================

def _create_fitness_model():
    """
    Create the lightweight classifier used during PSO.

    Logistic Regression is used because the PSO fitness function
    is evaluated many times.
    """

    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED
    )


# ============================================================
# Fitness function
# ============================================================

def fitness_function(
    mask,
    X_train,
    y_train,
    lambda_=0.05,
    cv=3,
    seed=SEED
):
    """
    Calculate the fitness of one binary feature mask.

    Fitness:

        (1 - mean CV AUC)
        + lambda * (selected_features / total_features)

    Lower fitness is better.

    AUC is used instead of accuracy because this project
    focuses on loan-default prediction where the classes
    may be imbalanced.

    Parameters
    ----------
    mask : array-like
        Binary vector:
            1 = feature selected
            0 = feature rejected

    X_train : pandas DataFrame
        Training features only.

    y_train : array-like
        Training target only.

    lambda_ : float
        Penalty for selecting too many features.

    cv : int
        Number of stratified CV folds.

    Returns
    -------
    float
        Fitness value.
    """

    X_train = _as_dataframe(X_train)
    y_train = _as_array(y_train)

    mask = np.asarray(mask).astype(int)

    selected_indices = np.where(mask == 1)[0]

    # Do not allow an empty feature set.
    if len(selected_indices) == 0:
        return 1e6

    X_selected = X_train.iloc[:, selected_indices]

    # Need at least two classes for AUC.
    if len(np.unique(y_train)) < 2:
        return 1e6

    model = _create_fitness_model()

    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=seed
    )

    try:

        scores = cross_val_score(
            model,
            X_selected,
            y_train,
            cv=cv_splitter,
            scoring="roc_auc",
            n_jobs=1
        )

        mean_auc = float(np.mean(scores))

    except Exception:
        return 1e6

    total_features = X_train.shape[1]

    feature_penalty = (
        lambda_ * len(selected_indices) / total_features
    )

    fitness = (
        1.0 - mean_auc
    ) + feature_penalty

    return float(fitness)


# ============================================================
# Binary PSO
# ============================================================

def run_pso(
    X_train,
    y_train,
    n_particles=20,
    iters=30,
    lambda_=0.05,
    cv=3,
    seed=42,
    progress_callback=None,
    max_rows=20000
):
    """
    Run Binary Particle Swarm Optimization.

    Parameters
    ----------
    X_train : pandas DataFrame
        Training feature matrix.

    y_train : array-like
        Training target.

    n_particles : int
        Number of particles.

    iters : int
        Number of PSO iterations.

    lambda_ : float
        Feature-count penalty.

    cv : int
        Number of CV folds.

    seed : int
        Random seed.

    progress_callback : callable, optional
        Called with:
            progress_callback(current_iteration, total_iterations)

    max_rows : int
        Maximum number of training rows used during PSO search.

    Returns
    -------
    dict
        selected_features
        best_fitness
        pso_history
        pso_runtime
    """

    start_time = time.time()

    X_train = _as_dataframe(X_train)
    y_train = _as_array(y_train)

    feature_cols = list(X_train.columns)

# --------------------------------------------------------
# Cache check
# --------------------------------------------------------

    dataset_hash = _dataset_hash(
        X_train,
        y_train
    )

    cache_key = (
          dataset_hash,
          n_particles,
          iters,
          lambda_,
          cv,
          seed,
          max_rows
    )

    if cache_key in _PSO_CACHE:

       print("[PSO] Returning cached result.")

       cached_result = _PSO_CACHE[cache_key].copy()

    # Return a copy so the cached object is not modified
       cached_result["selected_features"] = list(
       cached_result["selected_features"]
       )

       cached_result["pso_history"] = list(
       cached_result["pso_history"]
       )

       return cached_result

   

    # --------------------------------------------------------
    # Safety check and input validation
    # --------------------------------------------------------

    if X_train.shape[1] < 2:
         raise ValueError(
               "PSO feature selection requires at least 2 features."
         )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
               "Target must contain at least two classes."
         )

    if n_particles < 2:
        raise ValueError(
               "n_particles must be at least 2."
        )

    if iters < 1:
        raise ValueError(
              "iters must be at least 1."
        )

    if cv < 2:
        raise ValueError(
              "cv must be at least 2."
        )

    if lambda_ < 0:
        raise ValueError(
              "lambda_ cannot be negative."
        )

    if max_rows < 1:
        raise ValueError(
               "max_rows must be at least 1."
        )

# Prevent accidentally requesting an extremely large
# PSO search through the UI.
    estimated_evaluations = n_particles * iters

    MAX_EVALUATIONS = 2000

    if estimated_evaluations > MAX_EVALUATIONS:

        raise ValueError(
              f"PSO configuration is too large: "
              f"{n_particles} particles × {iters} iterations "
              f"= {estimated_evaluations} evaluations. "
              f"Maximum allowed is {MAX_EVALUATIONS}. "
              f"Please reduce particles or iterations."
        )

    # --------------------------------------------------------
    # Row cap
    # --------------------------------------------------------

    search_X = X_train
    search_y = y_train

    if len(X_train) > max_rows:

        rng = np.random.default_rng(seed)

        indices = np.arange(len(X_train))

        # Try to preserve class distribution.
        selected_indices = []

        classes = np.unique(y_train)

        for cls in classes:

            cls_indices = indices[y_train == cls]

            class_limit = int(
                max_rows * len(cls_indices) / len(X_train)
            )

            class_limit = max(1, class_limit)

            chosen = rng.choice(
                cls_indices,
                size=min(class_limit, len(cls_indices)),
                replace=False
            )

            selected_indices.extend(chosen)

        selected_indices = np.array(selected_indices)

        if len(selected_indices) > max_rows:
            selected_indices = rng.choice(
                selected_indices,
                size=max_rows,
                replace=False
            )

        search_X = X_train.iloc[selected_indices]
        search_y = y_train[selected_indices]

        print(
            f"[PSO] Row cap applied: "
            f"{len(X_train)} -> {len(search_X)} rows"
        )

    # --------------------------------------------------------
    # Import PySwarms
    # --------------------------------------------------------

    try:

        from pyswarms.discrete import BinaryPSO

    except ImportError:

        raise ImportError(
            "PySwarms is required. Install it using:\n"
            "pip install pyswarms"
        )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    np.random.seed(seed)

    dimensions = X_train.shape[1]

    options = {
    "c1": 2.0,
    "c2": 2.0,
    "w": 0.9,
    "k": 5,
    "p": 2
}

    optimizer = BinaryPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options=options
    )

    # --------------------------------------------------------
    # Fitness history
    # --------------------------------------------------------

    iteration_history = []

    def objective_function(particles):

        costs = []

        for particle in particles:

            cost = fitness_function(
                particle,
                search_X,
                search_y,
                lambda_=lambda_,
                cv=cv,
                seed=seed
            )

            costs.append(cost)

        return np.asarray(costs)

    # --------------------------------------------------------
    # Run optimization
    # --------------------------------------------------------

    print(
        f"[PSO] Starting optimization: "
        f"{n_particles} particles × {iters} iterations"
    )

    best_cost, best_position = optimizer.optimize(
        objective_function,
        iters=iters,
        verbose=False
    )

    # PySwarms stores best cost after every iteration.
    if hasattr(optimizer, "cost_history"):

        iteration_history = [
            float(value)
            for value in optimizer.cost_history
        ]

    # --------------------------------------------------------
    # Progress callback
    #
    # PySwarms does not expose a native callback for every
    # iteration through optimize(). Therefore we replay the
    # recorded history after optimization.
    # --------------------------------------------------------

    if progress_callback is not None:

        total_iterations = len(iteration_history)

        for i in range(total_iterations):

            progress_callback(
                i + 1,
                total_iterations
            )

    # --------------------------------------------------------
    # Ensure at least two features are selected.
    # --------------------------------------------------------

    best_position = np.asarray(
        best_position
    ).astype(int)

    selected_indices = np.where(
        best_position == 1
    )[0]

    if len(selected_indices) < 2:

        # Calculate individual feature usefulness.
        feature_scores = []

        for i in range(dimensions):

            mask = np.zeros(dimensions, dtype=int)
            mask[i] = 1

            score = fitness_function(
                mask,
                search_X,
                search_y,
                lambda_=lambda_,
                cv=cv,
                seed=seed
            )

            feature_scores.append(score)

        best_two = np.argsort(feature_scores)[:2]

        best_position[:] = 0
        best_position[best_two] = 1

        selected_indices = best_two

        best_cost = fitness_function(
            best_position,
            search_X,
            search_y,
            lambda_=lambda_,
            cv=cv,
            seed=seed
        )

    selected_features = [
        feature_cols[i]
        for i in selected_indices
    ]

    runtime = time.time() - start_time

    print(
        f"[PSO] Finished in {runtime:.2f} seconds"
    )

    print(
        f"[PSO] Selected "
        f"{len(selected_features)} / "
        f"{len(feature_cols)} features"
    )

    result = {
    "selected_features": selected_features,
    "best_fitness": float(best_cost),
    "pso_history": iteration_history,
    "pso_runtime": round(runtime, 2)
    }

# Save completed result in cache
    _PSO_CACHE[cache_key] = result.copy()

    return result      

# ============================================================
# PSO Experiments
# ============================================================

def run_pso_experiments(
    X_train,
    y_train,
    seeds=None,
    lambda_values=None,
    n_particles=20,
    iters=30,
    cv=3,
    max_rows=20000
):
    """
    Run PSO experiments across multiple random seeds
    and lambda values.

    This is intended for the project paper/experimental
    analysis.

    Parameters
    ----------
    X_train : pandas DataFrame
        Training features only.

    y_train : array-like
        Training target only.

    seeds : list
        Random seeds to test.

    lambda_values : list
        Feature-selection penalty values to test.

    n_particles : int
        Number of PSO particles.

    iters : int
        Number of PSO iterations.

    cv : int
        Number of cross-validation folds.

    max_rows : int
        Maximum number of rows used during PSO.

    Returns
    -------
    dict
        Experiment results and summary statistics.
    """

    X_train = _as_dataframe(X_train)
    y_train = _as_array(y_train)

    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    if lambda_values is None:
        lambda_values = [0.01, 0.05, 0.10]

    experiment_results = []

    total_experiments = (
        len(seeds) * len(lambda_values)
    )

    completed = 0

    print("\n" + "=" * 60)
    print("PSO EXPERIMENTS")
    print("=" * 60)

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"Lambda values: {lambda_values}"
    )

    print(
        f"Total experiments: {total_experiments}"
    )

    # --------------------------------------------------------
    # Run every seed × lambda combination
    # --------------------------------------------------------

    for lambda_ in lambda_values:

        for seed in seeds:

            completed += 1

            print(
                f"\n[Experiment "
                f"{completed}/{total_experiments}] "
                f"lambda={lambda_}, seed={seed}"
            )

            result = run_pso(
                X_train=X_train,
                y_train=y_train,
                n_particles=n_particles,
                iters=iters,
                lambda_=lambda_,
                cv=cv,
                seed=seed,
                progress_callback=None,
                max_rows=max_rows
            )

            experiment_results.append({
                "seed": seed,
                "lambda": lambda_,
                "selected_features": result[
                    "selected_features"
                ],
                "n_features": len(
                    result["selected_features"]
                ),
                "fitness": result[
                    "best_fitness"
                ],
                "runtime": result[
                    "pso_runtime"
                ],
                "pso_history": result[
                    "pso_history"
                ]
            })

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary = {}

    for lambda_ in lambda_values:

        lambda_results = [
            result
            for result in experiment_results
            if result["lambda"] == lambda_
        ]

        feature_counts = np.array([
            result["n_features"]
            for result in lambda_results
        ])

        fitness_values = np.array([
            result["fitness"]
            for result in lambda_results
        ])

        runtimes = np.array([
            result["runtime"]
            for result in lambda_results
        ])

        summary[lambda_] = {
            "mean_features": float(
                np.mean(feature_counts)
            ),

            "std_features": float(
                np.std(
                    feature_counts,
                    ddof=1
                )
            ) if len(feature_counts) > 1 else 0.0,

            "mean_fitness": float(
                np.mean(fitness_values)
            ),

            "std_fitness": float(
                np.std(
                    fitness_values,
                    ddof=1
                )
            ) if len(fitness_values) > 1 else 0.0,

            "mean_runtime": float(
                np.mean(runtimes)
            ),

            "std_runtime": float(
                np.std(
                    runtimes,
                    ddof=1
                )
            ) if len(runtimes) > 1 else 0.0
        }

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PSO EXPERIMENT SUMMARY")
    print("=" * 60)

    for lambda_, stats in summary.items():

        print(
            f"\nLambda = {lambda_}"
        )

        print(
            f"Mean features: "
            f"{stats['mean_features']:.2f}"
        )

        print(
            f"Std features: "
            f"{stats['std_features']:.2f}"
        )

        print(
            f"Mean fitness: "
            f"{stats['mean_fitness']:.6f}"
        )

        print(
            f"Std fitness: "
            f"{stats['std_fitness']:.6f}"
        )

        print(
            f"Mean runtime: "
            f"{stats['mean_runtime']:.2f} seconds"
        )

    # Save experiment results
    output_dir = Path("pso_experiments")
    output_dir.mkdir(exist_ok=True)

    experiment_rows = []

    for result in experiment_results:
        experiment_rows.append({
            "seed": result["seed"],
            "lambda": result["lambda"],
            "n_features": result["n_features"],
            "fitness": result["fitness"],
            "runtime": result["runtime"]
        })

    experiment_df = pd.DataFrame(experiment_rows)

    csv_path = output_dir / "pso_experiment_results.csv"

    experiment_df.to_csv(
        csv_path,
        index=False
    )

# Save convergence histories
    history_path = output_dir / "pso_convergence_history.json"

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
             experiment_results,
             f,
             indent=4
        )

    print("\n============================================================")
    print("PSO EXPERIMENT FILES SAVED")
    print("============================================================")

    print(f"Experiment results: {csv_path}")
    print(f"Convergence histories: {history_path}")

    return {
         "experiments": experiment_results,
         "summary": summary,
         "csv_path": str(csv_path),
         "history_path": str(history_path)
    }



# ============================================================
# Mutual Information
# ============================================================

def mutual_information_selection(
    X_train,
    y_train,
    k,
    seed=SEED
):
    """
    Select the top-k features using Mutual Information.
    """

    X_train = _as_dataframe(X_train)
    y_train = _as_array(y_train)

    k = max(
        1,
        min(k, X_train.shape[1])
    )

    scores = mutual_info_classif(
        X_train,
        y_train,
        random_state=seed
    )

    order = np.argsort(scores)[::-1]

    selected_indices = order[:k]

    return [
        X_train.columns[i]
        for i in selected_indices
    ]


# ============================================================
# RFE
# ============================================================

def rfe_selection(
    X_train,
    y_train,
    k,
    seed=SEED
):
    """
    Select top-k features using Recursive Feature Elimination.
    """

    X_train = _as_dataframe(X_train)
    y_train = _as_array(y_train)

    k = max(
        1,
        min(k, X_train.shape[1])
    )

    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed
    )

    selector = RFE(
        estimator=estimator,
        n_features_to_select=k,
        step=1
    )

    selector.fit(
        X_train,
        y_train
    )

    return list(
        X_train.columns[
            selector.support_
        ]
    )


# ============================================================
# Genetic Algorithm
# ============================================================

def genetic_algorithm_selection(
    X_train,
    y_train,
    k,
    population_size=20,
    generations=30,
    lambda_=0.05,
    cv=3,
    seed=SEED
):
    """
    Lightweight Genetic Algorithm for feature selection.

    The GA uses the same fitness function as PSO so that the
    comparison is fair.

    The returned solution is constrained to approximately k
    features so it can be compared fairly with PSO.
    """

    X_train = _as_dataframe(X_train)
    y_train = _as_array(y_train)

    rng = np.random.default_rng(seed)

    n_features = X_train.shape[1]

    k = max(
        2,
        min(k, n_features)
    )

    def create_individual():

        individual = np.zeros(
            n_features,
            dtype=int
        )

        selected = rng.choice(
            n_features,
            size=k,
            replace=False
        )

        individual[selected] = 1

        return individual

    def evaluate(individual):

        return fitness_function(
            individual,
            X_train,
            y_train,
            lambda_=lambda_,
            cv=cv,
            seed=seed
        )

    population = [
        create_individual()
        for _ in range(population_size)
    ]

    best_individual = None
    best_score = float("inf")

    for generation in range(generations):

        scores = [
            evaluate(individual)
            for individual in population
        ]

        order = np.argsort(scores)

        if scores[order[0]] < best_score:

            best_score = scores[order[0]]

            best_individual = (
                population[order[0]].copy()
            )

        # Keep the best 25%.
        elite_count = max(
            2,
            population_size // 4
        )

        elites = [
            population[i].copy()
            for i in order[:elite_count]
        ]

        new_population = elites.copy()

        while len(new_population) < population_size:

            parent1 = elites[
                rng.integers(len(elites))
            ]

            parent2 = elites[
                rng.integers(len(elites))
            ]

            # Single-point crossover.
            point = rng.integers(
                1,
                n_features
            )

            child = np.concatenate(
                [
                    parent1[:point],
                    parent2[point:]
                ]
            )

            # Mutation.
            mutation_rate = 1.0 / n_features

            for j in range(n_features):

                if rng.random() < mutation_rate:

                    child[j] = 1 - child[j]

            # Maintain approximately k selected features.
            selected = np.where(
                child == 1
            )[0]

            if len(selected) > k:

                remove_count = len(selected) - k

                remove = rng.choice(
                    selected,
                    size=remove_count,
                    replace=False
                )

                child[remove] = 0

            elif len(selected) < k:

                missing = k - len(selected)

                available = np.where(
                    child == 0
                )[0]

                add = rng.choice(
                    available,
                    size=min(
                        missing,
                        len(available)
                    ),
                    replace=False
                )

                child[add] = 1

            new_population.append(child)

        population = new_population

    selected_indices = np.where(
        best_individual == 1
    )[0]

    return [
        X_train.columns[i]
        for i in selected_indices
    ]


# ============================================================
# Main Member 2 function
# ============================================================

def run_feature_selection(
    X_train,
    y_train,
    feature_cols=None,
    n_particles=20,
    iters=30,
    lambda_=0.05,
    cv=3,
    seed=42,
    progress_callback=None,
    max_rows=20000,
    run_ga=True
):
    """
    Main entry point for Member 2.

    Receives:
        X_train
        y_train
        feature_cols

    Returns:
        selected_features
        best_fitness
        pso_history
        pso_runtime
        baseline_feature_sets
    """

    X_train = _as_dataframe(
        X_train,
        feature_cols
    )

    y_train = _as_array(y_train)

    if feature_cols is not None:

        missing = [
            col
            for col in feature_cols
            if col not in X_train.columns
        ]

        if missing:

            raise ValueError(
                f"Features missing from X_train: {missing}"
            )

        X_train = X_train[
            list(feature_cols)
        ]

    print("\n" + "=" * 60)
    print("MEMBER 2 - FEATURE SELECTION")
    print("=" * 60)

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Total features: {X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # 1. PSO
    # --------------------------------------------------------

    pso_result = run_pso(
        X_train=X_train,
        y_train=y_train,
        n_particles=n_particles,
        iters=iters,
        lambda_=lambda_,
        cv=cv,
        seed=seed,
        progress_callback=progress_callback,
        max_rows=max_rows
    )

    selected_features = pso_result[
        "selected_features"
    ]

    k = len(selected_features)

    # --------------------------------------------------------
    # 2. Baselines
    # --------------------------------------------------------

    print(
        "\n[Baseline] Mutual Information..."
    )

    mi_features = mutual_information_selection(
        X_train,
        y_train,
        k=k,
        seed=seed
    )

    print(
        "\n[Baseline] RFE..."
    )

    rfe_features = rfe_selection(
        X_train,
        y_train,
        k=k,
        seed=seed
    )

    baseline_feature_sets = {

        "All Features": list(
            X_train.columns
        ),

        "Mutual Information": mi_features,

        "RFE": rfe_features
    }

    # --------------------------------------------------------
    # 3. Genetic Algorithm
    # --------------------------------------------------------

    if run_ga:

        print(
            "\n[Baseline] Genetic Algorithm..."
        )

        ga_features = genetic_algorithm_selection(
            X_train,
            y_train,
            k=k,
            population_size=n_particles,
            generations=iters,
            lambda_=lambda_,
            cv=cv,
            seed=seed
        )

        baseline_feature_sets[
            "Genetic Algorithm"
        ] = ga_features

    print("\n" + "=" * 60)
    print("MEMBER 2 COMPLETE")
    print("=" * 60)

    print(
        f"PSO selected: {k} / "
        f"{X_train.shape[1]} features"
    )

    print(
        f"Best PSO fitness: "
        f"{pso_result['best_fitness']:.6f}"
    )

    print(
        f"PSO runtime: "
        f"{pso_result['pso_runtime']:.2f} seconds"
    )

    return {
        "selected_features": selected_features,
        "best_fitness": pso_result[
            "best_fitness"
        ],
        "pso_history": pso_result[
            "pso_history"
        ],
        "pso_runtime": pso_result[
            "pso_runtime"
        ],
        "baseline_feature_sets":
            baseline_feature_sets
    }


# ============================================================
# Simple standalone test
# ============================================================

if __name__ == "__main__":

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print(
        "\nRunning standalone Member 2 test..."
    )

    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=6,
        n_redundant=4,
        n_classes=2,
        weights=[0.75, 0.25],
        random_state=SEED
    )

    feature_names = [
        f"feature_{i}"
        for i in range(X.shape[1])
    ]

    X = pd.DataFrame(
        X,
        columns=feature_names
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    def progress(current, total):

        print(
            f"PSO progress: "
            f"{current}/{total}"
        )

    result = run_feature_selection(
        X_train=X_train,
        y_train=y_train,
        feature_cols=feature_names,
        n_particles=5,
        iters=5,
        lambda_=0.05,
        cv=3,
        seed=SEED,
        progress_callback=progress,
        max_rows=20000,
        run_ga=True
    )

    print("\nSelected features:")
    print(result["selected_features"])

    print("\nBest fitness:")
    print(result["best_fitness"])

    print("\nPSO history:")
    print(result["pso_history"])

    print("\nRuntime:")
    print(result["pso_runtime"])

    print("\nBaseline feature sets:")

    for method, features in result[
        "baseline_feature_sets"
    ].items():

        print(
            f"{method}: "
            f"{len(features)} features"
        )

    print("\n\nRunning full PSO experiment test...")

    experiment_result = run_pso_experiments(
         X_train=X_train,
         y_train=y_train,
         seeds=[42, 43, 44, 45, 46],
         lambda_values=[0.01, 0.05, 0.10],
         n_particles=20,
         iters=30,
         cv=3,
         max_rows=20000
    )