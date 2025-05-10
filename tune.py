import logging
import time
import random
from pathlib import Path
import optuna
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf, MISSING
from hydra import main
from torch.utils.data import DataLoader
from optuna.samplers import GPSampler

from model import SDnCNN
from utils import (
    SpectrogramPatchDataset,
    AudioMetricDataset,
    _evaluate_audio_metrics_bare,
)
from train import train_model


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | HPO_SCRIPT | %(levelname)s | %(message)s"
)


@main(config_path="conf", config_name="primary", version_base=None)
def run_hpo_experiment(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Full Hydra Config Loaded.")

    hpo_cfg_key = "variations"
    hpo_cfg = cfg[hpo_cfg_key]
    logger.info(f"Successfully loaded HPO parameters: {hpo_cfg.name}")

    num_optuna_trials = hpo_cfg.study.trials
    epochs_per_trial = hpo_cfg.study.epochs_per_trial
    fixed_hpo_seed = hpo_cfg.study.seed
    optuna_study_name = hpo_cfg.study.name
    optuna_db_output = hpo_cfg.study.output_db

    activation_options = list(hpo_cfg.search_space.activations)
    window_options = list(hpo_cfg.search_space.window_types)
    cyclic_dilation_patterns = OmegaConf.to_container(
        hpo_cfg.search_space.dilation_patterns, resolve=True
    )

    logger.info(
        f"Optuna Trials: {num_optuna_trials}, Epochs/Trial: {epochs_per_trial}, HPO Seed: {fixed_hpo_seed}"
    )
    logger.info(f"Optuna Study: {optuna_study_name}, DB: {optuna_db_output}")
    logger.info(f"Search Space - Activations: {activation_options}")
    logger.info(f"Search Space - Windows: {window_options}")
    logger.info(f"Search Space - Dilations: {cyclic_dilation_patterns}")

    base_config_for_trial = OmegaConf.to_container(cfg, resolve=True)
    if "variations" in base_config_for_trial:
        del base_config_for_trial["variations"]

    global_best_pesq_this_run = -float("inf")

    def objective(trial: optuna.trial.Trial) -> float:
        nonlocal global_best_pesq_this_run
        trial_start_time = time.time()
        logger.info(f"--- Starting Optuna Trial {trial.number} ---")

        cfg_trial = OmegaConf.create(base_config_for_trial.copy())

        activation = trial.suggest_categorical("activation", activation_options)
        window = trial.suggest_categorical("window_type", window_options)
        dilation_pattern_names = list(cyclic_dilation_patterns.keys())
        cycle_name = trial.suggest_categorical(
            "dilation_cycle_name", dilation_pattern_names
        )
        dilation_list = cyclic_dilation_patterns[cycle_name]
        current_seed_for_trial = fixed_hpo_seed

        trial.set_user_attr("dilation_list", dilation_list)
        trial.set_user_attr("fixed_hpo_seed_used", current_seed_for_trial)

        cfg_trial.model.activation = activation
        cfg_trial.stft.window_type = window
        cfg_trial.model.dilation_rates = dilation_list
        cfg_trial.train.epochs = epochs_per_trial
        cfg_trial.seed = current_seed_for_trial

        logger.info(
            f"Trial {trial.number} Effective Config → model.activation={cfg_trial.model.activation}, "
            f"stft.window_type={cfg_trial.stft.window_type}, "
            f"model.dilation_rates={OmegaConf.to_container(cfg_trial.model.dilation_rates)}, "
            f"train.epochs={cfg_trial.train.epochs}, seed={cfg_trial.seed}"
        )

        random.seed(current_seed_for_trial)
        np.random.seed(current_seed_for_trial)
        torch.manual_seed(current_seed_for_trial)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed_for_trial)

        common_args = {
            "fft_size": cfg_trial.stft.n_fft,
            "hop_length": cfg_trial.stft.hop_length,
            "window_length": cfg_trial.stft.win_length,
            "window_type": cfg_trial.stft.window_type,
            "patch_height": cfg_trial.patch.height,
            "patch_width": cfg_trial.patch.width,
            "patch_stride": cfg_trial.patch.stride,
            "sample_rate": cfg_trial.data.sample_rate,
        }

        train_ds = SpectrogramPatchDataset(
            cfg_trial.data.samples.train, **common_args, purpose="training_trial"
        )
        val_patch_ds = SpectrogramPatchDataset(
            cfg_trial.data.samples.valid,
            **common_args,
            purpose="validation_patches_trial",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg_trial.train.batch_size,
            shuffle=True,
            num_workers=cfg_trial.train.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(cfg_trial.train.num_workers > 0),
        )
        val_patch_loader = DataLoader(
            val_patch_ds,
            batch_size=getattr(
                cfg_trial.train, "val_batch_size", cfg_trial.train.batch_size
            ),
            shuffle=False,
            num_workers=cfg_trial.train.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(cfg_trial.train.num_workers > 0),
        )

        model = SDnCNN(
            cfg_trial.model.depth,
            cfg_trial.model.channels,
            cfg_trial.model.activation,
            dilation_rates=cfg_trial.model.dilation_rates,
        ).to(device)

        train_model(model, train_loader, val_patch_loader, cfg_trial, device)

        trial_best_model_path = Path(f"best_model_trial_{trial.number}.pt")
        default_best_model_path = Path("best_model.pt")
        if default_best_model_path.exists():
            default_best_model_path.rename(trial_best_model_path)

        if trial_best_model_path.exists():
            model.load_state_dict(
                torch.load(trial_best_model_path, map_location=device)
            )
            logger.info(
                f"Trial {trial.number}: Loaded best model from this trial for final metric evaluation."
            )
        else:
            logger.warning(
                f"Trial {trial.number}: Best model file '{trial_best_model_path}' not found. Using model's current state."
            )

        metric_ds = AudioMetricDataset(
            cfg_trial.data.samples.valid,
            cfg_trial.data.sample_rate,
            purpose="hpo_trial_metrics",
        )
        if len(metric_ds) == 0:
            logger.error(
                f"Trial {trial.number}: No validation samples for metrics at {cfg_trial.data.samples.valid}"
            )
            return -float("inf")

        metric_loader = DataLoader(
            metric_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cfg_trial.train.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(cfg_trial.train.num_workers > 0),
        )
        stft_params_eval = {
            "fft_size": cfg_trial.stft.n_fft,
            "hop_length": cfg_trial.stft.hop_length,
            "window_length": cfg_trial.stft.win_length,
            "window_type": cfg_trial.stft.window_type,
        }
        eval_results = _evaluate_audio_metrics_bare(
            model, metric_loader, cfg_trial, device, stft_params_eval
        )

        pesq_score = eval_results.get("pesq", -float("inf"))
        if pesq_score is None or not np.isfinite(pesq_score):
            pesq_score = -float("inf")

        logger.info(f"Trial {trial.number} -> Final PESQ: {pesq_score:.4f}")

        if pesq_score > global_best_pesq_this_run:
            global_best_pesq_this_run = pesq_score
            logger.info(
                f"*** Trial {trial.number}: New global best PESQ in this HPO run: {pesq_score:.4f}"
            )

        trial_duration = time.time() - trial_start_time
        logger.info(f"--- Trial {trial.number} finished in {trial_duration:.1f}s ---")
        return pesq_score

    sampler = GPSampler(seed=fixed_hpo_seed)

    study = optuna.create_study(
        direction="maximize",
        study_name=optuna_study_name,
        sampler=sampler,
        storage=f"sqlite:///{optuna_db_output}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=num_optuna_trials)

    logger.info(
        f"======== HPO Optimization Finished for Study: {optuna_study_name} ========"
    )
    if study.best_trial:
        logger.info(
            f"Best Trial Overall: #{study.best_trial.number} "
            f"with PESQ = {study.best_trial.value:.4f}"
        )
        logger.info(f"Best Hyperparameters: {study.best_trial.params}")
        if "dilation_list" in study.best_trial.user_attrs:
            logger.info(
                f"Corresponding Dilation Rates: {study.best_trial.user_attrs['dilation_list']}"
            )
        if "fixed_hpo_seed_used" in study.best_trial.user_attrs:
            logger.info(
                f"Fixed HPO Seed Used for Best Trial: {study.best_trial.user_attrs['fixed_hpo_seed_used']}"
            )
    else:
        logger.info("No trials were completed successfully in this study.")


if __name__ == "__main__":
    run_hpo_experiment()
