import time
import logging
import random
from pathlib import Path
import csv
import math
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from utils import (
    SpectrogramPatchDataset,
    AudioMetricDataset,
    _evaluate_audio_metrics_bare,
)
from model import SDnCNN


logging_format = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def train_model(
    model,
    train_loader,
    val_patch_loader,
    config: DictConfig,
    device,
):
    model.to(device)
    epochs = config.train.epochs

    sgd_config = config.train.sgd
    optimizer = optim.SGD(
        model.parameters(),
        lr=sgd_config.lr_start,
        momentum=sgd_config.momentum,
        weight_decay=sgd_config.weight_decay,
    )
    logger.info(f"Using SGD optimizer with initial LR: {sgd_config.lr_start}")

    gamma = None
    if sgd_config.lr_decay_epochs > 1 and sgd_config.lr_end < sgd_config.lr_start:
        gamma = (sgd_config.lr_end / sgd_config.lr_start) ** (
            1.0 / max(1, sgd_config.lr_decay_epochs - 1)
        )

    criterion = nn.MSELoss()

    log_file_path = Path("training_log.csv")
    log_header = [
        "epoch",
        "avg_train_loss",
        "avg_val_patch_loss",
        "val_pesq",
        "val_stoi",
        "val_si_sdr",
        "learning_rate",
        "epoch_time_s",
        "best_model_saved",
    ]
    write_header = not log_file_path.exists() or log_file_path.stat().st_size == 0
    with open(log_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(log_header)

    early_stop_config = config.train.early_stopping
    patience = early_stop_config.get("patience", 10)
    min_delta = early_stop_config.get("min_delta", 0.0)
    early_stop_enabled = early_stop_config.get("enabled", True)
    monitor_metric = early_stop_config.get("monitor_metric", "val_loss")
    higher_is_better = monitor_metric in ["val_pesq", "val_stoi", "val_si_sdr"]

    epochs_no_improve = 0
    best_metric_value = float("-inf") if higher_is_better else float("inf")
    best_epoch = 0
    early_stopped = False
    checkpoint_path = Path("best_model.pt")

    logger.info(f"Starting training for up to {epochs} epochs on {device}")
    logger.info(f" - Train batches/epoch: {len(train_loader)}")
    logger.info(f" - Val patch batches/epoch: {len(val_patch_loader)}")
    logger.info(f"Early stopping: {'Enabled' if early_stop_enabled else 'Disabled'}")
    if early_stop_enabled:
        logger.info(f" - Patience: {patience}, Min Delta: {min_delta}")

    val_metric_config_path = config.data.samples.valid
    val_sample_rate = config.data.sample_rate
    eval_metric_freq = config.train.get("eval_metric_freq", 1)

    val_metric_dataset = AudioMetricDataset(
        samples_directory=val_metric_config_path,
        sample_rate=val_sample_rate,
        purpose="validation metrics",
    )
    val_metric_loader = None
    if len(val_metric_dataset) > 0:
        val_metric_loader = DataLoader(
            val_metric_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(config.train.num_workers > 0),
        )
        logger.info(
            f"Validation metrics will be evaluated every {eval_metric_freq} epochs on {len(val_metric_dataset)} files."
        )
    else:
        logger.warning(
            "Validation metric dataset is empty. Metric evaluation will be skipped."
        )
        eval_metric_freq = float("inf")

    stft_params = {
        "fft_size": config.stft.n_fft,
        "hop_length": config.stft.hop_length,
        "window_length": config.stft.win_length,
        "window_type": config.stft.window_type.lower(),
    }

    for e in range(1, epochs + 1):
        epoch_start_time = time.time()
        saved_best_this_epoch = False

        model.train()
        total_train_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        if gamma is not None and gamma != 1.0:
            effective_epoch_for_decay = max(1, min(e, config.train.sgd.lr_decay_epochs))
            current_lr = config.train.sgd.lr_start * (
                gamma ** (effective_epoch_for_decay - 1)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        for batch_idx, (n_batch, c_batch) in enumerate(train_loader):
            n_batch, c_batch = n_batch.to(device), c_batch.to(device)
            target = n_batch - c_batch

            optimizer.zero_grad()
            pred_noise = model(n_batch)
            loss = criterion(pred_noise, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.train.get("grad_clip_norm", 1.0)
            )
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = (
            total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        )

        model.eval()
        total_val_patch_loss = 0.0
        val_batches_processed = 0
        with torch.no_grad():
            for n_batch_val, c_batch_val in val_patch_loader:
                n_batch_val, c_batch_val = (
                    n_batch_val.to(device),
                    c_batch_val.to(device),
                )
                target_val = n_batch_val - c_batch_val
                pred_val = model(n_batch_val)
                val_loss = criterion(pred_val, target_val)
                total_val_patch_loss += val_loss.item()
                val_batches_processed += 1

        avg_val_patch_loss = (
            total_val_patch_loss / val_batches_processed
            if val_batches_processed > 0
            else float("inf")
        )

        val_metric_results = {}
        if e % eval_metric_freq == 0 or e == epochs:
            if val_metric_loader:
                logger.info(f"Epoch {e}: Evaluating metrics on validation set...")
                eval_start_time = time.time()
                val_metric_results = _evaluate_audio_metrics_bare(
                    model, val_metric_loader, config, device, stft_params
                )
                eval_time = time.time() - eval_start_time
                pesq_val = val_metric_results.get("pesq", float("nan"))
                stoi_val = val_metric_results.get("stoi", float("nan"))
                sisdr_val = val_metric_results.get("si_sdr", float("nan"))
                logger.info(
                    f"Epoch {e}: Val Metrics | PESQ: {pesq_val:.3f} | STOI: {stoi_val:.3f} | SI-SDR: {sisdr_val:.3f} | Eval Time: {eval_time:.1f}s"
                )

        epoch_time = time.time() - epoch_start_time
        log_data = [
            e,
            avg_train_loss,
            avg_val_patch_loss,
            val_metric_results.get("pesq", float("nan")),
            val_metric_results.get("stoi", float("nan")),
            val_metric_results.get("si_sdr", float("nan")),
            current_lr,
            epoch_time,
        ]

        logger.info(
            f"Epoch {e}/{epochs} | LR: {current_lr:.2e} | Tr Loss: {avg_train_loss:.4f} | Vl Loss(p): {avg_val_patch_loss:.4f} | Time: {epoch_time:.1f}s"
        )

        current_metric_value_for_monitor = float("nan")

        if monitor_metric == "val_loss":
            current_metric_value_for_monitor = avg_val_patch_loss
        elif monitor_metric in val_metric_results:
            metric_key_for_results = monitor_metric
            if monitor_metric.startswith("val_"):
                metric_key_for_results = monitor_metric.split("val_")[-1]

            current_metric_value_for_monitor = val_metric_results.get(
                metric_key_for_results, float("nan")
            )

        improved = False
        if math.isfinite(current_metric_value_for_monitor):
            if higher_is_better:
                if current_metric_value_for_monitor > best_metric_value + min_delta:
                    improved = True
            else:
                if current_metric_value_for_monitor < best_metric_value - min_delta:
                    improved = True

        if improved:
            best_metric_value = current_metric_value_for_monitor
            epochs_no_improve = 0
            best_epoch = e
            torch.save(model.state_dict(), checkpoint_path)
            saved_best_this_epoch = True
            logger.info(
                f"Metric {monitor_metric} improved ({best_metric_value:.4f} --> {current_metric_value_for_monitor:.4f}). Saved best model."
            )
        else:
            if math.isfinite(current_metric_value_for_monitor):
                epochs_no_improve += 1

        log_data.append(saved_best_this_epoch)
        with open(log_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log_data)

        if early_stop_enabled and epochs_no_improve >= patience:
            logger.info(f"--- Early stopping triggered at epoch {e} ---")
            early_stopped = True
            break

    logger.info("=" * 30)
    logger.info("Training finished.")
    status = (
        f"Stopped early at epoch {e}" if early_stopped else f"Completed {epochs} epochs"
    )
    if best_epoch > 0:
        logger.info(
            f"{status}. Best {monitor_metric}: {best_metric_value:.4f} achieved at epoch {best_epoch}."
        )
        logger.info(f"Best model weights saved to: {checkpoint_path.resolve()}")
    else:
        logger.warning(f"{status}, but no valid best metric recorded or model saved.")
    logger.info("=" * 30)


@hydra.main(config_path="conf", config_name="primary", version_base=None)
def main(config: DictConfig):
    seed = config.get("seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if config.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    common_dataset_args = {
        "fft_size": config.stft.n_fft,
        "hop_length": config.stft.hop_length,
        "window_length": config.stft.win_length,
        "window_type": config.stft.window_type,
        "patch_height": config.patch.height,
        "patch_width": config.patch.width,
        "patch_stride": config.patch.stride,
        "sample_rate": config.data.sample_rate,
    }

    train_dataset = SpectrogramPatchDataset(
        config.data.samples.train, **common_dataset_args, purpose="training"
    )
    val_patch_dataset = SpectrogramPatchDataset(
        config.data.samples.valid, **common_dataset_args, purpose="validation"
    )

    train_batch_size = config.train.batch_size
    val_batch_size = config.train.get("val_batch_size", train_batch_size)

    use_persistent_workers = config.train.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=config.train.get("drop_last", True),
        num_workers=config.train.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent_workers,
    )
    val_patch_loader = DataLoader(
        val_patch_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.train.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent_workers,
    )

    raw_dilation_rates_from_cfg = config.model.get("dilation_rates")
    parsed_dilation_rates_for_model_constructor = None

    if raw_dilation_rates_from_cfg is not None:
        candidate_list = list(raw_dilation_rates_from_cfg)
        if (
            isinstance(candidate_list, list)
            and len(candidate_list) > 0
            and all(isinstance(item, int) for item in candidate_list)
            and all(item > 0 for item in candidate_list)
        ):
            parsed_dilation_rates_for_model_constructor = candidate_list
        else:
            logger.warning(
                f"Invalid 'dilation_rates' in config: {raw_dilation_rates_from_cfg}. "
                f"Expected a non-empty list of positive integers. Model will use its internal default."
            )

    else:
        logger.info(
            "'dilation_rates' not specified in config.model. Model will use its internal default."
        )

    model_instance = SDnCNN(
        config.model.depth,
        config.model.channels,
        config.model.get("activation", "relu"),
        dilation_rates=parsed_dilation_rates_for_model_constructor,
    )

    num_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_params:,}")

    train_model(model_instance, train_loader, val_patch_loader, config, device)


if __name__ == "__main__":
    main()
