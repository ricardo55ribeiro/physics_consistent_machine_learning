"""
Robustness Experiment for the Physics-Consistent Low-Temperature Plasma model.

Train/Load NN models for several architectures, then test how robust the
predictions are when relative Gaussian measurement noise is added to the test
inputs P, I, and R.

Experiment Setup:
    - Architectures: [30, 30], [50, 50], [30, 30, 30]
    - Models Compared: NN and NN + Projection
    - Noise Cases: P only, I only, R only, P + I + R
    - Noise Type: relative Gaussian noise [ x_noisy = x_clean * (1 + eps) ]
    - Impossible Physical Values: resample if P <= 0, I <= 0, or R <= 0
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.ltp_system.data_prep import (
    DataPreprocessor,
    LoadDataset,
    setup_dataset_with_preproprocessing_info,
)
from src.ltp_system.pinn_nn import (
    NeuralNetwork,
    get_trained_bootstraped_models,
    load_checkpoints,
)
from src.ltp_system.projection import constraint_p_i_ne, project_output
from src.ltp_system.utils import (
    load_config,
    load_dataset,
    sample_dataset,
    select_random_rows,
    set_seed,
)


# =============================================================================
# USER SETTINGS
# =============================================================================

CONFIG_PATH = "configs/ltp_system_config.yaml"
LARGE_DATASET_PATH = "data/ltp_system/data_3000_points.txt"

OUTPUT_DIR = Path("src/ltp_system/figures/Noise_Robustness")
CHECKPOINT_ROOT = Path("output/ltp_system/checkpoints/noise_robustness")

# Use False to load existing checkpoints if available.
# If checkpoints for an architecture are missing, the script trains them anyway.
RETRAIN_MODELS = False
TRAIN_ONLY = RETRAIN_MODELS

GLOBAL_SEED = 42
DATASET_SIZE = 1000
N_TESTING_POINTS = 300

ARCHITECTURES = [
    [30, 30],
    [50, 50],
    [30, 30, 30],
]

# Relative noise standard deviations.
# Example: 0.01 = 1% Gaussian noise.
NOISE_STDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]

# Noise cases: indices correspond to input_features = ['P', 'I', 'R'].
NOISE_CASES = {
    "P_only": [0],
    "I_only": [1],
    "R_only": [2],
    "P_I_R": [0, 1, 2],
}

# Number of random noise realizations per non-zero noise std.
N_NOISE_SEEDS = 20
NOISE_SEED_START = 1000

# W matrix for projection. For this experiment we use the identity matrix.
W_MATRIX = torch.eye(17, dtype=torch.float64)

# Outputs to highlight in the final tables.
IMPORTANT_OUTPUTS = {
    "O2X": 0,        # O2(X)
    "O2plus": 4,    # O2(+,X)
    "ne": 16,       # electron density
}

# Device
DEVICE = torch.device("cpu")


# =============================================================================
# SMALL UTILITIES
# =============================================================================


def architecture_name(hidden_sizes: Sequence[int]) -> str:
    return "_".join(str(v) for v in hidden_sizes)


def architecture_label(hidden_sizes: Sequence[int]) -> str:
    return "[" + ", ".join(str(v) for v in hidden_sizes) + "]"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def transform_physical_matrix(
    physical_values: np.ndarray,
    scalers: Sequence,
    skewed_indices: Iterable[int],
) -> np.ndarray:
    """Apply the same log1p + MinMax scaling used in training."""
    values = np.asarray(physical_values, dtype=np.float64).copy()
    skewed_set = set(int(i) for i in list(skewed_indices))

    for idx in skewed_set:
        values[:, idx] = np.log1p(values[:, idx])

    scaled_columns = []
    for idx, scaler in enumerate(scalers):
        scaled = scaler.transform(values[:, idx : idx + 1])
        scaled_columns.append(scaled)

    return np.hstack(scaled_columns).astype(np.float64)


def transform_inputs_physical(inputs_physical: np.ndarray, preprocessor: DataPreprocessor) -> np.ndarray:
    return transform_physical_matrix(
        inputs_physical,
        preprocessor.scalers_input,
        preprocessor.skewed_features_in,
    )


def transform_outputs_physical(outputs_physical: np.ndarray, preprocessor: DataPreprocessor) -> np.ndarray:
    return transform_physical_matrix(
        outputs_physical,
        preprocessor.scalers_output,
        preprocessor.skewed_features_out,
    )


def normalize_single_feature(
    values_physical: np.ndarray,
    scaler,
    feature_index: int,
    skewed_indices: Iterable[int],
) -> np.ndarray:
    """Normalize one physical feature using a fitted scaler."""
    values = np.asarray(values_physical, dtype=np.float64).reshape(-1, 1).copy()
    if feature_index in set(int(i) for i in list(skewed_indices)):
        values = np.log1p(values)
    return scaler.transform(values).reshape(-1)


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_data(config: Dict) -> Tuple[DataPreprocessor, object, torch.utils.data.DataLoader, str, Dict[str, np.ndarray]]:
    """Replicates the main LowTemperaturePlasma.py data preparation."""
    print("\nPreparing data...")
    set_seed(GLOBAL_SEED)

    _, large_dataset = load_dataset(config, LARGE_DATASET_PATH)

    preprocessor = DataPreprocessor(config)
    preprocessor.setup_dataset(large_dataset.x, large_dataset.y)

    testing_file, training_file = sample_dataset(
        LARGE_DATASET_PATH,
        n_testing_points=N_TESTING_POINTS,
    )
    if testing_file is None or training_file is None:
        raise RuntimeError("Could not create testing/training split files.")

    sampled_training_file = select_random_rows(
        training_file,
        DATASET_SIZE,
        seed=GLOBAL_SEED,
    )

    _, sampled_training_dataset = load_dataset(config, sampled_training_file)
    train_data, _, val_data = setup_dataset_with_preproprocessing_info(
        sampled_training_dataset.x,
        sampled_training_dataset.y,
        preprocessor,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config["nn_model"]["batch_size"],
        shuffle=True,
    )

    domain_info = {
        "input_min": np.min(large_dataset.x, axis=0),
        "input_max": np.max(large_dataset.x, axis=0),
    }

    print(f"   Full dataset: {len(large_dataset.x)} points")
    print(f"   Testing set:  {N_TESTING_POINTS} points")
    print(f"   Training subset before internal split: {DATASET_SIZE} points")
    print(f"   Actual train_data after internal split: {len(train_data)} points")
    print(f"   Actual val_data after internal split:   {len(val_data)} points")

    return preprocessor, train_data, val_loader, testing_file, domain_info


def load_clean_test_set(testing_file: str, preprocessor: DataPreprocessor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return clean physical and normalized test inputs/targets."""
    test_dataset = LoadDataset(testing_file)

    clean_inputs_physical = np.asarray(test_dataset.x, dtype=np.float64)
    clean_targets_physical = np.asarray(test_dataset.y, dtype=np.float64)

    clean_inputs_norm = transform_inputs_physical(clean_inputs_physical, preprocessor)
    clean_targets_norm = transform_outputs_physical(clean_targets_physical, preprocessor)

    return clean_inputs_physical, clean_targets_physical, clean_inputs_norm, clean_targets_norm


# =============================================================================
# MODEL TRAINING / LOADING
# =============================================================================


def build_model_config(base_config: Dict, hidden_sizes: Sequence[int], retrain: bool) -> Dict:
    """Create a config_model for one architecture."""
    config_model = copy.deepcopy(base_config["nn_model"])
    config_model["hidden_sizes"] = list(hidden_sizes)
    config_model["activation_fns"] = ["leaky_relu"] * len(hidden_sizes)
    config_model["lambda_physics"] = [0, 0, 0]
    config_model["RETRAIN_MODEL"] = retrain
    return config_model


def get_models_for_architecture(
    config: Dict,
    hidden_sizes: Sequence[int],
    preprocessor: DataPreprocessor,
    train_data,
    val_loader,
) -> Tuple[List[NeuralNetwork], Dict, Dict]:
    """Load existing checkpoints or train the bootstrap ensemble."""
    arch_name = architecture_name(hidden_sizes)
    checkpoint_dir = CHECKPOINT_ROOT / f"arch_{arch_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_model = build_model_config(config, hidden_sizes, RETRAIN_MODELS)
    config_plotting = copy.deepcopy(config["plotting"])
    config_plotting["PRINT_LOSS_VALUES"] = False

    should_train = RETRAIN_MODELS
    models = None
    losses = None

    if not should_train:
        try:
            models, losses, loaded_hidden_sizes, loaded_activation_fns, training_time = load_checkpoints(
                config_model,
                NeuralNetwork,
                str(checkpoint_dir),
                print_messages=True,
            )

            expected = config_model["n_bootstrap_models"]
            if len(models) != expected:
                print(f"   Incomplete checkpoints for {architecture_label(hidden_sizes)}: "
                      f"loaded {len(models)}/{expected}. Retraining.")
                should_train = True
            else:
                print(f"   Using existing checkpoints for architecture {architecture_label(hidden_sizes)}")
        except Exception as exc:
            print(f"   Could not load checkpoints for {architecture_label(hidden_sizes)}: {exc}")
            print("   Retraining this architecture.")
            should_train = True

    if should_train:
        print(f"\nTraining architecture {architecture_label(hidden_sizes)}...")
        models, losses, training_time = get_trained_bootstraped_models(
            config_model,
            config_plotting,
            preprocessor,
            nn.MSELoss(),
            str(checkpoint_dir),
            DEVICE,
            val_loader,
            train_data,
            seed="default",
            print_messages=True,
        )

    for model in models:
        model.to(DEVICE)
        model.to(torch.double)
        model.eval()

    return models, losses, config_model


# =============================================================================
# NOISE GENERATION
# =============================================================================


def add_relative_gaussian_noise(
    clean_inputs: np.ndarray,
    noisy_columns: Sequence[int],
    noise_std: float,
    seed: int,
    min_positive: float = 1e-300,
    max_resample_rounds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Add relative Gaussian noise to selected input columns.

    Impossible physical values are not clipped. They are resampled until positive.
    Returns:
        noisy_inputs, relative_noise_matrix, number_of_resampled_values
    """
    clean_inputs = np.asarray(clean_inputs, dtype=np.float64)
    noisy_inputs = clean_inputs.copy()
    relative_noise = np.zeros_like(clean_inputs, dtype=np.float64)
    resample_count = 0

    if noise_std == 0.0 or len(noisy_columns) == 0:
        return noisy_inputs, relative_noise, resample_count

    rng = np.random.default_rng(seed)
    n_points = clean_inputs.shape[0]

    for col in noisy_columns:
        eps = rng.normal(loc=0.0, scale=noise_std, size=n_points)
        proposed = clean_inputs[:, col] * (1.0 + eps)
        invalid = proposed <= min_positive

        n_rounds = 0
        while np.any(invalid):
            n_bad = int(np.sum(invalid))
            resample_count += n_bad
            eps_new = rng.normal(loc=0.0, scale=noise_std, size=n_bad)
            proposed[invalid] = clean_inputs[invalid, col] * (1.0 + eps_new)
            eps[invalid] = eps_new
            invalid = proposed <= min_positive
            n_rounds += 1

            if n_rounds > max_resample_rounds:
                raise RuntimeError(
                    f"Could not resample physically valid values for input column {col}. "
                    f"noise_std={noise_std} may be too large."
                )

        noisy_inputs[:, col] = proposed
        relative_noise[:, col] = eps

    return noisy_inputs, relative_noise, resample_count


def outside_domain_stats(noisy_inputs: np.ndarray, domain_info: Dict[str, np.ndarray]) -> Dict[str, float]:
    input_min = domain_info["input_min"]
    input_max = domain_info["input_max"]

    below = noisy_inputs < input_min
    above = noisy_inputs > input_max
    outside = below | above

    stats = {
        "outside_domain_fraction_any": float(np.mean(np.any(outside, axis=1))),
        "outside_domain_fraction_P": float(np.mean(outside[:, 0])),
        "outside_domain_fraction_I": float(np.mean(outside[:, 1])),
        "outside_domain_fraction_R": float(np.mean(outside[:, 2])),
    }
    return stats


# =============================================================================
# PREDICTION AND PROJECTION
# =============================================================================


def ensemble_predict(models: Sequence[NeuralNetwork], inputs_norm: np.ndarray) -> np.ndarray:
    """Average predictions of the bootstrapped NN ensemble."""
    inputs_tensor = torch.as_tensor(inputs_norm, dtype=torch.float64, device=DEVICE)
    n_points = inputs_tensor.shape[0]
    n_outputs = 17

    avg = torch.zeros((n_points, n_outputs), dtype=torch.float64, device=DEVICE)

    with torch.no_grad():
        for model in models:
            model.to(DEVICE)
            model.to(torch.double)
            model.eval()
            avg += model(inputs_tensor)

    avg /= len(models)
    return avg.detach().cpu().numpy()


def project_predictions_safely(
    inputs_norm: np.ndarray,
    predictions_norm: np.ndarray,
    preprocessor: DataPreprocessor,
    W: torch.Tensor,
) -> Tuple[np.ndarray, int]:
    """Project each prediction with full P+I+ne constraints. Count failures."""
    projected = []
    failures = 0

    inputs_tensor = torch.as_tensor(inputs_norm, dtype=torch.float64)
    preds_tensor = torch.as_tensor(predictions_norm, dtype=torch.float64)

    for x_i, y_i in tqdm(
        list(zip(inputs_tensor, preds_tensor)),
        desc="Projecting test predictions",
        leave=False,
    ):
        try:
            p_opt = project_output(
                x_i.unsqueeze(0),
                y_i.unsqueeze(0),
                constraint_p_i_ne,
                preprocessor,
                W,
            )
            projected.append(p_opt)
        except Exception:
            failures += 1
            projected.append(np.full(17, np.nan, dtype=np.float64))

    return np.vstack(projected), failures


# =============================================================================
# METRICS
# =============================================================================


def prediction_metrics(
    predictions_norm: np.ndarray,
    targets_norm: np.ndarray,
    important_outputs: Dict[str, int],
) -> Dict[str, float]:
    """Compute MSE/RMSE in normalized output space."""
    predictions_norm = np.asarray(predictions_norm, dtype=np.float64)
    targets_norm = np.asarray(targets_norm, dtype=np.float64)

    valid_rows = np.isfinite(predictions_norm).all(axis=1)
    n_valid = int(np.sum(valid_rows))
    n_total = int(len(valid_rows))

    metrics = {
        "n_total": n_total,
        "n_valid": n_valid,
    }

    if n_valid == 0:
        metrics["mse_all"] = np.nan
        metrics["rmse_all"] = np.nan
        for name in important_outputs:
            metrics[f"mse_{name}"] = np.nan
            metrics[f"rmse_{name}"] = np.nan
        return metrics

    pred = predictions_norm[valid_rows]
    targ = targets_norm[valid_rows]
    sq = (pred - targ) ** 2

    metrics["mse_all"] = float(np.mean(sq))
    metrics["rmse_all"] = float(math.sqrt(metrics["mse_all"]))

    for name, idx in important_outputs.items():
        mse = float(np.mean(sq[:, idx]))
        metrics[f"mse_{name}"] = mse
        metrics[f"rmse_{name}"] = float(math.sqrt(mse))

    return metrics


def physical_law_metrics(
    inputs_norm: np.ndarray,
    predictions_norm: np.ndarray,
    preprocessor: DataPreprocessor,
) -> Dict[str, float]:
    """
    Compute physical-law RMSEs in normalized units.

    Laws:
        P  = sum(species densities) * kB * Tg
        I  = e * ne * vd * pi * R^2
        ne = O2+ + O+ - O-
    """
    predictions_norm = np.asarray(predictions_norm, dtype=np.float64)
    inputs_norm = np.asarray(inputs_norm, dtype=np.float64)

    valid_rows = np.isfinite(predictions_norm).all(axis=1)
    if int(np.sum(valid_rows)) == 0:
        return {
            "phys_rmse_P": np.nan,
            "phys_rmse_I": np.nan,
            "phys_rmse_ne": np.nan,
        }

    inputs_norm_valid = torch.as_tensor(inputs_norm[valid_rows], dtype=torch.float64)
    preds_norm_valid = torch.as_tensor(predictions_norm[valid_rows], dtype=torch.float64)

    inputs_phys, preds_phys = preprocessor.inverse_transform(inputs_norm_valid, preds_norm_valid)
    inputs_phys = to_numpy(inputs_phys).astype(np.float64)
    preds_phys = to_numpy(preds_phys).astype(np.float64)

    P_model = inputs_phys[:, 0]
    I_model = inputs_phys[:, 1]
    R = inputs_phys[:, 2]

    Tg = preds_phys[:, 11]
    ne_model = preds_phys[:, 16]
    vd = preds_phys[:, 14]

    species_density_sum = np.sum(preds_phys[:, :11], axis=1)
    k_b = 1.380649e-23
    e_charge = 1.602176634e-19

    P_calc = species_density_sum * k_b * Tg
    I_calc = e_charge * ne_model * vd * np.pi * R * R
    ne_calc = preds_phys[:, 4] + preds_phys[:, 7] - preds_phys[:, 8]

    P_model_norm = normalize_single_feature(P_model, preprocessor.scalers_input[0], 0, preprocessor.skewed_features_in)
    P_calc_norm = normalize_single_feature(P_calc, preprocessor.scalers_input[0], 0, preprocessor.skewed_features_in)

    I_model_norm = normalize_single_feature(I_model, preprocessor.scalers_input[1], 1, preprocessor.skewed_features_in)
    I_calc_norm = normalize_single_feature(I_calc, preprocessor.scalers_input[1], 1, preprocessor.skewed_features_in)

    ne_model_norm = normalize_single_feature(ne_model, preprocessor.scalers_output[16], 16, preprocessor.skewed_features_out)
    ne_calc_norm = normalize_single_feature(ne_calc, preprocessor.scalers_output[16], 16, preprocessor.skewed_features_out)

    return {
        "phys_rmse_P": float(np.sqrt(np.mean((P_calc_norm - P_model_norm) ** 2))),
        "phys_rmse_I": float(np.sqrt(np.mean((I_calc_norm - I_model_norm) ** 2))),
        "phys_rmse_ne": float(np.sqrt(np.mean((ne_calc_norm - ne_model_norm) ** 2))),
    }


def build_result_row(
    architecture: Sequence[int],
    noise_case: str,
    noise_std: float,
    noise_seed: int,
    model_type: str,
    predictions_norm: np.ndarray,
    targets_norm: np.ndarray,
    inputs_norm: np.ndarray,
    preprocessor: DataPreprocessor,
    projection_failures: int,
    resample_count: int,
    relative_noise: np.ndarray,
    outside_stats: Dict[str, float],
) -> Dict[str, float]:
    row = {
        "architecture": architecture_label(architecture),
        "architecture_id": architecture_name(architecture),
        "noise_case": noise_case,
        "noise_std": noise_std,
        "noise_std_percent": 100.0 * noise_std,
        "noise_seed": noise_seed,
        "model_type": model_type,
        "projection_failures": projection_failures,
        "resampled_impossible_values": resample_count,
        "actual_abs_noise_mean_P_percent": float(100.0 * np.mean(np.abs(relative_noise[:, 0]))),
        "actual_abs_noise_mean_I_percent": float(100.0 * np.mean(np.abs(relative_noise[:, 1]))),
        "actual_abs_noise_mean_R_percent": float(100.0 * np.mean(np.abs(relative_noise[:, 2]))),
    }

    row.update(outside_stats)
    row.update(prediction_metrics(predictions_norm, targets_norm, IMPORTANT_OUTPUTS))
    row.update(physical_law_metrics(inputs_norm, predictions_norm, preprocessor))

    return row


# =============================================================================
# RESULTS SAVING
# =============================================================================


def save_results(detailed_rows: List[Dict]) -> None:
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = OUTPUT_DIR / "noise_robustness_detailed_results.csv"
    detailed_df.to_csv(detailed_path, index=False)

    metric_cols = [
        "mse_all",
        "rmse_all",
        "mse_O2X",
        "rmse_O2X",
        "mse_O2plus",
        "rmse_O2plus",
        "mse_ne",
        "rmse_ne",
        "phys_rmse_P",
        "phys_rmse_I",
        "phys_rmse_ne",
        "projection_failures",
        "resampled_impossible_values",
        "outside_domain_fraction_any",
        "outside_domain_fraction_P",
        "outside_domain_fraction_I",
        "outside_domain_fraction_R",
        "actual_abs_noise_mean_P_percent",
        "actual_abs_noise_mean_I_percent",
        "actual_abs_noise_mean_R_percent",
        "n_valid",
    ]

    group_cols = [
        "architecture",
        "architecture_id",
        "noise_case",
        "noise_std",
        "noise_std_percent",
        "model_type",
    ]

    available_metric_cols = [col for col in metric_cols if col in detailed_df.columns]

    summary_df = detailed_df.groupby(group_cols, dropna=False)[available_metric_cols].agg(["mean", "std"])
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    summary_df = summary_df.reset_index()

    summary_path = OUTPUT_DIR / "noise_robustness_summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved results:")
    print(f"   Detailed: {detailed_path}")
    print(f"   Summary:  {summary_path}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def main() -> None:
    ensure_dirs()

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║          Physics-Consistent LTP: Input Noise Robustness Study             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    set_seed(GLOBAL_SEED)
    config = load_config(None, CONFIG_PATH)

    # Keep the experiment focused on NN, not PINN.
    config["nn_model"]["lambda_physics"] = [0, 0, 0]
    config["plotting"]["PRINT_LOSS_VALUES"] = False

    preprocessor, train_data, val_loader, testing_file, domain_info = prepare_data(config)

    clean_inputs_phys, clean_targets_phys, clean_inputs_norm, clean_targets_norm = load_clean_test_set(
        testing_file,
        preprocessor,
    )

    experiment_settings = {
        "architectures": ARCHITECTURES,
        "noise_stds": NOISE_STDS,
        "noise_cases": NOISE_CASES,
        "n_noise_seeds": N_NOISE_SEEDS,
        "global_seed": GLOBAL_SEED,
        "dataset_size": DATASET_SIZE,
        "n_testing_points": N_TESTING_POINTS,
        "projection_W": "torch.eye(17)",
        "projection_constraints": "P + I + ne",
        "target": "clean LoKI outputs",
        "noise": "relative Gaussian measurement noise on selected input columns",
    }
    with open(OUTPUT_DIR / "noise_robustness_settings.json", "w") as f:
        json.dump(experiment_settings, f, indent=2)

    detailed_rows: List[Dict] = []

    for architecture in ARCHITECTURES:
        print("\n" + "=" * 80)
        print(f"Architecture: {architecture_label(architecture)}")
        print("=" * 80)

        models, losses, config_model = get_models_for_architecture(
            config,
            architecture,
            preprocessor,
            train_data,
            val_loader,
        )

        if TRAIN_ONLY:
            print("Training-only mode: skipping clean/noisy testing and projection.")
            continue

        # Clean baseline: compute once per architecture, not once per noise case.
        print("\nEvaluating clean baseline...")
        nn_clean_pred_norm = ensemble_predict(models, clean_inputs_norm)
        proj_clean_pred_norm, clean_projection_failures = project_predictions_safely(
            clean_inputs_norm,
            nn_clean_pred_norm,
            preprocessor,
            W_MATRIX,
        )

        zero_noise = np.zeros_like(clean_inputs_phys, dtype=np.float64)
        clean_outside_stats = outside_domain_stats(clean_inputs_phys, domain_info)

        detailed_rows.append(
            build_result_row(
                architecture=architecture,
                noise_case="clean",
                noise_std=0.0,
                noise_seed=0,
                model_type="NN",
                predictions_norm=nn_clean_pred_norm,
                targets_norm=clean_targets_norm,
                inputs_norm=clean_inputs_norm,
                preprocessor=preprocessor,
                projection_failures=0,
                resample_count=0,
                relative_noise=zero_noise,
                outside_stats=clean_outside_stats,
            )
        )
        detailed_rows.append(
            build_result_row(
                architecture=architecture,
                noise_case="clean",
                noise_std=0.0,
                noise_seed=0,
                model_type="NN_projection",
                predictions_norm=proj_clean_pred_norm,
                targets_norm=clean_targets_norm,
                inputs_norm=clean_inputs_norm,
                preprocessor=preprocessor,
                projection_failures=clean_projection_failures,
                resample_count=0,
                relative_noise=zero_noise,
                outside_stats=clean_outside_stats,
            )
        )

        # Noisy cases.
        for noise_std in NOISE_STDS:
            if noise_std == 0.0:
                continue

            for noise_case, noisy_columns in NOISE_CASES.items():
                print(f"\nNoise case={noise_case}, std={100.0 * noise_std:.2f}%")

                for seed_idx in range(N_NOISE_SEEDS):
                    noise_seed = NOISE_SEED_START + seed_idx

                    noisy_inputs_phys, relative_noise, resample_count = add_relative_gaussian_noise(
                        clean_inputs_phys,
                        noisy_columns,
                        noise_std,
                        seed=noise_seed,
                    )
                    noisy_inputs_norm = transform_inputs_physical(noisy_inputs_phys, preprocessor)
                    outside_stats = outside_domain_stats(noisy_inputs_phys, domain_info)

                    nn_pred_norm = ensemble_predict(models, noisy_inputs_norm)
                    proj_pred_norm, projection_failures = project_predictions_safely(
                        noisy_inputs_norm,
                        nn_pred_norm,
                        preprocessor,
                        W_MATRIX,
                    )

                    detailed_rows.append(
                        build_result_row(
                            architecture=architecture,
                            noise_case=noise_case,
                            noise_std=noise_std,
                            noise_seed=noise_seed,
                            model_type="NN",
                            predictions_norm=nn_pred_norm,
                            targets_norm=clean_targets_norm,
                            inputs_norm=noisy_inputs_norm,
                            preprocessor=preprocessor,
                            projection_failures=0,
                            resample_count=resample_count,
                            relative_noise=relative_noise,
                            outside_stats=outside_stats,
                        )
                    )
                    detailed_rows.append(
                        build_result_row(
                            architecture=architecture,
                            noise_case=noise_case,
                            noise_std=noise_std,
                            noise_seed=noise_seed,
                            model_type="NN_projection",
                            predictions_norm=proj_pred_norm,
                            targets_norm=clean_targets_norm,
                            inputs_norm=noisy_inputs_norm,
                            preprocessor=preprocessor,
                            projection_failures=projection_failures,
                            resample_count=resample_count,
                            relative_noise=relative_noise,
                            outside_stats=outside_stats,
                        )
                    )

                # Save incrementally so partial results are not lost if the run is interrupted.
                save_results(detailed_rows)

    if detailed_rows:
        save_results(detailed_rows)
        print("\nNoise robustness study complete.")
    else:
        print("\nTraining-only run complete. Checkpoints saved.")


if __name__ == "__main__":
    main()
