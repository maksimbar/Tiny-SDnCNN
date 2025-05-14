import logging
from pathlib import Path
import csv
import random
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import torchaudio

from model import SDnCNN
from utils import (
    AudioMetricDataset,
    average_metrics,
    compute_stft_spectrogram,
    normalize_spectrogram_paper,
    denormalize_spectrogram_paper,
    reconstruct_waveform,
    calculate_metrics_bare,
)

logging_format = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def run_evaluation_for_test(
    model, metric_loader, config: DictConfig, device, stft_params
):
    model.eval()
    all_individual_metrics = []
    files_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(metric_loader):
            if batch[0] is None or batch[1] is None:
                continue

            clean_wav, noisy_wav, noisy_fname_tuple, snr_val_tensor = batch
            clean_wav = clean_wav.squeeze(0).to(device)
            noisy_wav = noisy_wav.squeeze(0).to(device)
            noisy_fname = noisy_fname_tuple[0]
            snr_val = snr_val_tensor.item()

            orig_length = noisy_wav.shape[-1]
            if orig_length == 0:
                continue

            clean_db, _ = compute_stft_spectrogram(clean_wav, **stft_params)
            noisy_db, noisy_phase = compute_stft_spectrogram(noisy_wav, **stft_params)

            if noisy_db.shape[1] == 0 or clean_db.shape[1] == 0:
                continue

            T = min(clean_db.shape[1], noisy_db.shape[1])
            clean_db = clean_db[:, :T]
            noisy_db = noisy_db[:, :T]
            noisy_phase = noisy_phase[:, :T]

            n_tilde, x_hat_min, x_bar_max = normalize_spectrogram_paper(noisy_db)

            n_tilde_batch = n_tilde.unsqueeze(0).unsqueeze(0)
            pred_noise_tilde_batch = model(n_tilde_batch)
            pred_noise_tilde = pred_noise_tilde_batch.squeeze(0).squeeze(0)

            if pred_noise_tilde.shape != n_tilde.shape:
                resize_transform = torchaudio.transforms.Resize(
                    n_tilde.shape, interpolation="bilinear", antialias=True
                )
                pred_noise_tilde = resize_transform(
                    pred_noise_tilde.unsqueeze(0)
                ).squeeze(0)

            est_clean_tilde = n_tilde - pred_noise_tilde
            est_clean_db = denormalize_spectrogram_paper(
                est_clean_tilde, x_hat_min, x_bar_max
            )
            enhanced_wav = reconstruct_waveform(
                est_clean_db, noisy_phase, **stft_params, orig_length=orig_length
            )

            clean_wav_np = clean_wav.cpu().numpy()
            enhanced_wav_np = enhanced_wav.cpu().numpy()

            metrics = calculate_metrics_bare(
                clean_wav_np, enhanced_wav_np, config.data.sample_rate, noisy_fname
            )
            metrics["snr"] = snr_val
            all_individual_metrics.append(metrics)
            files_processed += 1

            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.info(f"Processed {files_processed} files for testing...")

    logger.info(f"Finished evaluating. Total files processed: {files_processed}.")
    return all_individual_metrics


def report_training_log(training_log_path: Path):
    logger.info("\n--- Training Log Summary ---")
    with open(training_log_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required_cols = ["epoch", "avg_train_loss", "epoch_time_s"]

        has_val_loss = "avg_val_patch_loss" in reader.fieldnames

        header_parts = [f"{'Epoch':>5}", f"{'Avg Train Loss':>15}"]
        if has_val_loss:
            header_parts.append(f"{'Avg Val Loss':>15}")
        header_parts.append(f"{'Epoch Time (s)':>15}")

        header_str = " | ".join(header_parts)
        print(header_str)
        print("-" * len(header_str))

        for row in reader:
            epoch = int(row["epoch"])
            avg_train_loss = float(row["avg_train_loss"])
            epoch_time_s = float(row["epoch_time_s"])

            row_parts_data = [f"{epoch:>5}", f"{avg_train_loss:>15.4f}"]
            if has_val_loss:
                avg_val_loss_str = row.get("avg_val_patch_loss", "nan")
                avg_val_loss = (
                    float(avg_val_loss_str)
                    if avg_val_loss_str.lower() != "nan"
                    else float("nan")
                )
                row_parts_data.append(f"{avg_val_loss:>15.4f}")
            row_parts_data.append(f"{epoch_time_s:>15.1f}")

            print(" | ".join(row_parts_data))
        print("-" * len(header_str))
    logger.info("--- End of Training Log Summary ---\n")


@hydra.main(config_path="conf", config_name="primary", version_base=None)
def main(config: DictConfig):
    seed = config.get("seed")
    logger.info(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if config.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    model_dilation_rates = config.model.get("dilation_rates")
    model_dilation_rates_parsed = None
    if model_dilation_rates is not None:
        model_dilation_rates_parsed = [
            int(d) for d in OmegaConf.to_container(model_dilation_rates, resolve=True)
        ]
        if not all(isinstance(d, int) and d > 0 for d in model_dilation_rates_parsed):
            model_dilation_rates_parsed = None

    model_instance = SDnCNN(
        config.model.depth,
        config.model.channels,
        config.model.get("activation", "relu"),
        dilation_rates=model_dilation_rates_parsed,
    )

    model_checkpoint_path_str = config.eval.model_checkpoint
    model_checkpoint_path = Path(model_checkpoint_path_str)
    if not model_checkpoint_path.is_absolute():
        original_cwd = Path(hydra.utils.get_original_cwd())
        model_checkpoint_path = original_cwd / model_checkpoint_path
        logger.info(
            f"Resolved relative model checkpoint path to: {model_checkpoint_path}"
        )

    logger.info(f"Loading model checkpoint from: {model_checkpoint_path}")
    model_instance.load_state_dict(
        torch.load(model_checkpoint_path, map_location=device)
    )
    model_instance.to(device)
    model_instance.eval()

    stft_params = {
        "fft_size": config.stft.n_fft,
        "hop_length": config.stft.hop_length,
        "window_length": config.stft.win_length,
        "window_type": config.stft.window_type.lower(),
    }

    test_data_dir_str = config.data.samples.test
    test_data_dir = Path(test_data_dir_str)
    if not test_data_dir.is_absolute():
        original_cwd = Path(hydra.utils.get_original_cwd())
        test_data_dir = original_cwd / test_data_dir
        logger.info(f"Resolved relative test data directory to: {test_data_dir}")

    target_snr_levels = sorted(list(range(0, 33 + 1, 3)))
    logger.info(f"Target SNR levels for evaluation: {target_snr_levels}")

    test_metric_dataset = AudioMetricDataset(
        samples_directory=str(test_data_dir),
        sample_rate=config.data.sample_rate,
        snr_levels=None,
        purpose="testing_metrics",
    )

    test_metric_loader = DataLoader(
        test_metric_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.eval.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.eval.get("num_workers", 0) > 0),
    )
    logger.info(
        f"Evaluating on {len(test_metric_dataset)} test file pairs from {test_data_dir}."
    )

    all_individual_metrics = run_evaluation_for_test(
        model_instance, test_metric_loader, config, device, stft_params
    )

    if not all_individual_metrics:
        logger.warning("No metrics were collected during evaluation. Exiting.")
        return

    metrics_by_snr = defaultdict(list)
    for item in all_individual_metrics:
        if item.get("error"):
            continue
        snr_key = int(round(item["snr"]))
        if snr_key in target_snr_levels:
            metrics_by_snr[snr_key].append(item)

    output_dir_str = config.eval.output_results_dir
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_summary_output_path = output_dir / "test_metrics_by_snr.csv"
    logger.info(f"Writing SNR-specific metrics to: {snr_summary_output_path}")

    with open(snr_summary_output_path, "w", newline="") as csvfile:
        fieldnames = [
            "snr",
            "avg_pesq",
            "avg_stoi",
            "avg_si_sdr",
            "pesq_count",
            "stoi_count",
            "sisdr_count",
            "total_files_at_snr",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        sorted_snrs_present_and_targeted = sorted(
            [snr for snr in metrics_by_snr.keys() if snr in target_snr_levels]
        )

        for snr_val in sorted_snrs_present_and_targeted:
            snr_metrics_list = metrics_by_snr[snr_val]
            if not snr_metrics_list:
                continue

            avg_snr_results = average_metrics(snr_metrics_list)
            writer.writerow(
                {
                    "snr": snr_val,
                    "avg_pesq": f"{avg_snr_results.get('pesq', float('nan')):.4f}",
                    "avg_stoi": f"{avg_snr_results.get('stoi', float('nan')):.4f}",
                    "avg_si_sdr": f"{avg_snr_results.get('si_sdr', float('nan')):.4f}",
                    "pesq_count": avg_snr_results.get("valid_pesq_count", 0),
                    "stoi_count": avg_snr_results.get("valid_stoi_count", 0),
                    "sisdr_count": avg_snr_results.get("valid_si_sdr_count", 0),
                    "total_files_at_snr": avg_snr_results.get("count", 0),
                }
            )
            logger.info(
                f"SNR {snr_val:2d} dB | PESQ: {avg_snr_results.get('pesq', float('nan')):.3f} ({avg_snr_results.get('valid_pesq_count', 0)}/{avg_snr_results.get('count', 0)}) | "
                f"STOI: {avg_snr_results.get('stoi', float('nan')):.3f} ({avg_snr_results.get('valid_stoi_count', 0)}/{avg_snr_results.get('count', 0)}) | "
                f"SI-SDR: {avg_snr_results.get('si_sdr', float('nan')):.3f} ({avg_snr_results.get('valid_si_sdr_count', 0)}/{avg_snr_results.get('count', 0)})"
            )

    metrics_for_overall_average = []
    for snr_val in target_snr_levels:
        metrics_for_overall_average.extend(metrics_by_snr.get(snr_val, []))

    if metrics_for_overall_average:
        overall_avg_metrics = average_metrics(metrics_for_overall_average)
        logger.info("-" * 30)
        logger.info("Overall Average Metrics (for target SNRs):")
        logger.info(
            f"  PESQ:   {overall_avg_metrics.get('pesq', float('nan')):.4f} (over {overall_avg_metrics.get('valid_pesq_count', 0)} of {len(metrics_for_overall_average)} files)"
        )
        logger.info(
            f"  STOI:   {overall_avg_metrics.get('stoi', float('nan')):.4f} (over {overall_avg_metrics.get('valid_stoi_count', 0)} of {len(metrics_for_overall_average)} files)"
        )
        logger.info(
            f"  SI-SDR: {overall_avg_metrics.get('si_sdr', float('nan')):.4f} (over {overall_avg_metrics.get('valid_si_sdr_count', 0)} of {len(metrics_for_overall_average)} files)"
        )
        logger.info(
            f"  Total files considered for overall average: {len(metrics_for_overall_average)}"
        )
        logger.info("-" * 30)

        overall_summary_path = output_dir / "test_metrics_overall.txt"
        with open(overall_summary_path, "w") as f:
            f.write("Overall Average Metrics (for target SNRs):\n")
            f.write(
                f"  PESQ:   {overall_avg_metrics.get('pesq', float('nan')):.4f} (over {overall_avg_metrics.get('valid_pesq_count', 0)} of {len(metrics_for_overall_average)} files)\n"
            )
            f.write(
                f"  STOI:   {overall_avg_metrics.get('stoi', float('nan')):.4f} (over {overall_avg_metrics.get('valid_stoi_count', 0)} of {len(metrics_for_overall_average)} files)\n"
            )
            f.write(
                f"  SI-SDR: {overall_avg_metrics.get('si_sdr', float('nan')):.4f} (over {overall_avg_metrics.get('valid_si_sdr_count', 0)} of {len(metrics_for_overall_average)} files)\n"
            )
            f.write(
                f"  Total files considered for overall average: {len(metrics_for_overall_average)}\n"
            )
        logger.info(f"Overall average metrics saved to: {overall_summary_path}")
    else:
        logger.warning(
            "No metrics available for target SNRs to calculate overall average."
        )

    training_log_file_path_str = config.eval.training_log_path
    training_log_file = Path(training_log_file_path_str)

    original_cwd = Path(hydra.utils.get_original_cwd())

    actual_log_path = training_log_file
    if not training_log_file.is_absolute():
        if (original_cwd / training_log_file).exists():
            actual_log_path = original_cwd / training_log_file
        elif (Path.cwd() / training_log_file).exists():
            actual_log_path = Path.cwd() / training_log_file
        else:
            actual_log_path = original_cwd / training_log_file

    report_training_log(actual_log_path)


if __name__ == "__main__":
    main()
