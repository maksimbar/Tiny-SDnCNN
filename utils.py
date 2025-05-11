import logging
from pathlib import Path
import re
from collections import defaultdict
import math

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from pesq import pesq
from pystoi import stoi

logger = logging.getLogger(__name__)


def si_sdr(reference, estimate, eps=1e-8):
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference)
    if isinstance(estimate, np.ndarray):
        estimate = torch.from_numpy(estimate)

    reference = reference.to(estimate.device)
    if reference.dim() > 1:
        reference = reference.squeeze()
    if estimate.dim() > 1:
        estimate = estimate.squeeze()

    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    alpha = torch.dot(estimate, reference) / (reference.norm(p=2) ** 2 + eps)
    proj = alpha * reference
    noise = estimate - proj

    ratio = proj.norm(p=2) ** 2 / (noise.norm(p=2) ** 2 + eps)
    sdr_val = 10 * torch.log10(ratio + eps)
    return sdr_val.item()


def compute_stft_spectrogram(
    audio_signal, fft_size, hop_length, window_length, window_type="hann"
):
    window_fn_map = {
        "hann": torch.hann_window,
        "hamming": torch.hamming_window,
        "blackman": torch.blackman_window,
    }

    window_fn = window_fn_map[window_type]
    window = window_fn(window_length, device=audio_signal.device)

    complex_spec = torch.stft(
        audio_signal,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    if complex_spec.dim() == 3 and complex_spec.shape[0] == 1:
        complex_spec = complex_spec.squeeze(0)

    mag = complex_spec.abs()
    phase = torch.angle(complex_spec)
    log_mag_db = 10 * torch.log10(mag.pow(2) + 1e-9)
    return log_mag_db, phase


def normalize_spectrogram_paper(
    spectrogram_db,
):
    x_max = torch.max(spectrogram_db)
    x_min_thresh = x_max - 255.0
    spectrogram_hat = torch.maximum(spectrogram_db, x_min_thresh)

    x_hat_min = torch.min(spectrogram_hat)
    spectrogram_bar = spectrogram_hat - x_hat_min
    x_bar_max = torch.max(spectrogram_bar)
    if x_bar_max < 1e-9:
        spectrogram_tilde = torch.zeros_like(spectrogram_bar)
        x_bar_max_adjusted = 1.0
    else:
        spectrogram_tilde = spectrogram_bar / x_bar_max
        x_bar_max_adjusted = x_bar_max.item()
    return spectrogram_tilde, x_hat_min.item(), x_bar_max_adjusted


def denormalize_spectrogram_paper(spectrogram_tilde, x_hat_min, x_bar_max):
    spectrogram_bar = spectrogram_tilde * x_bar_max
    spectrogram_hat = spectrogram_bar + x_hat_min
    return spectrogram_hat


def load_and_preprocess_audio(file_path, target_sample_rate):
    waveform, orig_sr = torchaudio.load(str(file_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_sr, target_sample_rate, dtype=waveform.dtype
        )
        waveform = resampler(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    return waveform


class SpectrogramPatchDataset(Dataset):
    def __init__(
        self,
        samples_directory,
        fft_size,
        hop_length,
        window_length,
        window_type,
        patch_height,
        patch_width,
        patch_stride,
        sample_rate,
        purpose="training",
    ):
        self.purpose = purpose
        self.sample_rate = sample_rate
        self.stft_params = {
            "fft_size": fft_size,
            "hop_length": hop_length,
            "window_length": window_length,
            "window_type": window_type,
        }
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_stride = patch_stride

        clean_dir = Path(samples_directory) / "clean"
        noisy_dir = Path(samples_directory) / "noisy"

        self.noisy_patches = []
        self.clean_patches = []
        self.metadata = []

        processed_files = 0

        clean_files = sorted(list(clean_dir.glob("*.wav")))

        for clean_path in clean_files:
            stem = clean_path.stem
            clean_wav = load_and_preprocess_audio(clean_path, self.sample_rate)

            noisy_candidates = sorted(list(noisy_dir.glob(f"{stem}_w*.wav")))
            file_processed_flag = False
            for noisy_path in noisy_candidates:
                noisy_wav = load_and_preprocess_audio(noisy_path, self.sample_rate)

                min_len_samples = (
                    self.stft_params["hop_length"] * (self.patch_width - 1)
                    + self.stft_params["window_length"]
                )
                L = min(clean_wav.shape[-1], noisy_wav.shape[-1])

                clean_wav_trunc = clean_wav[:L]
                noisy_wav_trunc = noisy_wav[:L]

                clean_db, _ = compute_stft_spectrogram(
                    clean_wav_trunc, **self.stft_params
                )
                noisy_db, _ = compute_stft_spectrogram(
                    noisy_wav_trunc, **self.stft_params
                )

                c_tilde, _, _ = normalize_spectrogram_paper(clean_db)
                n_tilde, _, _ = normalize_spectrogram_paper(noisy_db)

                F, T = n_tilde.shape
                h = min(self.patch_height, F)
                w, s = self.patch_width, self.patch_stride

                if T < w:
                    pad_n_target_shape = torch.zeros(
                        F, w, device=n_tilde.device, dtype=n_tilde.dtype
                    )
                    pad_c_target_shape = torch.zeros(
                        F, w, device=c_tilde.device, dtype=c_tilde.dtype
                    )

                    pad_n_target_shape[:, :T] = n_tilde
                    pad_c_target_shape[:, :T] = c_tilde

                    self.noisy_patches.append(pad_n_target_shape[:h, :w].unsqueeze(0))
                    self.clean_patches.append(pad_c_target_shape[:h, :w].unsqueeze(0))
                    self.metadata.append(
                        {
                            "clean_file": clean_path.name,
                            "noisy_file": noisy_path.name,
                            "patch_idx": 0,
                            "total_patches": 1,
                            "padded": True,
                        }
                    )
                else:
                    patch_count = 0
                    for i in range(0, T - w + 1, s):
                        self.noisy_patches.append(n_tilde[:h, i : i + w].unsqueeze(0))
                        self.clean_patches.append(c_tilde[:h, i : i + w].unsqueeze(0))
                        self.metadata.append(
                            {
                                "clean_file": clean_path.name,
                                "noisy_file": noisy_path.name,
                                "patch_idx": patch_count,
                                "padded": False,
                            }
                        )
                        patch_count += 1
                    for meta_idx in range(
                        len(self.metadata) - patch_count, len(self.metadata)
                    ):
                        self.metadata[meta_idx]["total_patches"] = patch_count

                file_processed_flag = True

            if file_processed_flag:
                processed_files += 1

        logger.info(
            f"Created {self.purpose} dataset / Clean files: {processed_files} / Total patches: {len(self.noisy_patches)}"
        )

    def __len__(self):
        return len(self.noisy_patches)

    def __getitem__(self, idx):
        return self.noisy_patches[idx], self.clean_patches[idx]


class AudioMetricDataset(Dataset):
    def __init__(
        self,
        samples_directory,
        sample_rate,
        snr_levels=None,
        purpose="validation metrics",
    ):
        self.sample_rate = sample_rate
        self.purpose = purpose
        clean_dir = Path(samples_directory) / "clean"
        noisy_dir = Path(samples_directory) / "noisy"
        self.file_pairs = []
        snr_pattern = re.compile(r"_w(-?\d+)\.wav$")

        logger.info(
            f"Creating {self.purpose} AudioMetricDataset from {samples_directory}..."
        )

        clean_files = {p.stem: p for p in clean_dir.glob("*.wav")}
        processed_pairs = 0

        for noisy_path in sorted(list(noisy_dir.glob("*.wav"))):
            match = snr_pattern.search(noisy_path.name)
            if not match:
                continue

            snr = int(match.group(1))
            if snr_levels is not None and snr not in snr_levels:
                continue

            base_stem = noisy_path.stem.split("_w")[0]

            if base_stem in clean_files:
                self.file_pairs.append(
                    {
                        "clean": clean_files[base_stem],
                        "noisy": noisy_path,
                        "snr": snr,
                    }
                )
                processed_pairs += 1

        logger.info(f"Finished creating {self.purpose} dataset.")
        logger.info(f"Found {processed_pairs} clean/noisy pairs.")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        clean_wav = load_and_preprocess_audio(pair["clean"], self.sample_rate)
        noisy_wav = load_and_preprocess_audio(pair["noisy"], self.sample_rate)
        return (clean_wav, noisy_wav, pair["noisy"].name, pair["snr"])


def average_metrics(metric_list):
    avg = defaultdict(lambda: {"sum": 0.0, "count": 0})
    keys = ["pesq", "stoi", "si_sdr"]
    for metrics_dict in metric_list:
        for key in keys:
            value = metrics_dict.get(key)
            if value is not None and math.isfinite(value):
                avg[key]["sum"] += value
                avg[key]["count"] += 1

    results = {
        key: (
            avg[key]["sum"] / avg[key]["count"]
            if avg[key]["count"] > 0
            else float("nan")
        )
        for key in keys
    }
    results["count"] = len(metric_list)
    results["valid_pesq_count"] = avg["pesq"]["count"]
    results["valid_stoi_count"] = avg["stoi"]["count"]
    results["valid_si_sdr_count"] = avg["si_sdr"]["count"]
    return results


def calculate_metrics_bare(ref_wav_np, enh_wav_np, sr, filename):
    metrics = {
        "pesq": float("nan"),
        "stoi": float("nan"),
        "si_sdr": float("nan"),
        "filename": filename,
    }
    min_len = min(len(ref_wav_np), len(enh_wav_np))
    if min_len == 0:
        return metrics

    ref = ref_wav_np[:min_len]
    enh = enh_wav_np[:min_len]

    mode = "wb" if sr > 8000 else "nb"

    metrics["pesq"] = pesq(sr, ref, enh, mode)
    metrics["stoi"] = stoi(
        ref.astype(np.float64), enh.astype(np.float64), sr, extended=False
    )
    metrics["si_sdr"] = si_sdr(ref, enh)

    return metrics


def reconstruct_waveform(
    log_mag_spec_db,
    noisy_phase,
    fft_size,
    hop_length,
    window_length,
    window_type="hamming",
    orig_length=None,
):
    expected_freq_bins = fft_size // 2 + 1
    current_freq_bins = log_mag_spec_db.shape[0]

    if current_freq_bins != expected_freq_bins:
        diff = expected_freq_bins - current_freq_bins
        if diff > 0:
            pad_val = (
                torch.min(log_mag_spec_db) if log_mag_spec_db.numel() > 0 else -100.0
            )
            padding = torch.full(
                (diff, log_mag_spec_db.shape[1]), pad_val, device=log_mag_spec_db.device
            )
            log_mag_spec_db = torch.cat((log_mag_spec_db, padding), dim=0)
        else:
            log_mag_spec_db = log_mag_spec_db[:expected_freq_bins, :]

    current_phase_bins = noisy_phase.shape[0]
    if current_phase_bins != expected_freq_bins:
        diff_phase = expected_freq_bins - current_phase_bins
        if diff_phase > 0:
            padding_phase = torch.zeros(
                (diff_phase, noisy_phase.shape[1]), device=noisy_phase.device
            )
            noisy_phase = torch.cat((noisy_phase, padding_phase), dim=0)
        else:
            noisy_phase = noisy_phase[:expected_freq_bins, :]

    min_time = min(log_mag_spec_db.shape[1], noisy_phase.shape[1])
    log_mag_spec_db = log_mag_spec_db[:, :min_time]
    noisy_phase = noisy_phase[:, :min_time]

    mag = torch.pow(10.0, log_mag_spec_db / 20.0).clamp(min=0.0)
    complex_spec = torch.polar(mag, noisy_phase)

    window_fn_map = {
        "hann": torch.hann_window,
        "hamming": torch.hamming_window,
        "blackman": torch.blackman_window,
    }
    window_fn = window_fn_map[window_type]
    window = window_fn(window_length, device=complex_spec.device)

    waveform = torch.istft(
        complex_spec,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        length=orig_length,
        center=True,
    )
    return waveform


def _evaluate_audio_metrics_bare(model, metric_loader, config, device, stft_params):
    model.eval()
    all_metrics = []
    files_processed = 0

    with torch.no_grad():
        for batch in metric_loader:
            if batch[0] is None or batch[1] is None:
                continue

            clean_wav, noisy_wav, noisy_fname, snr_val = batch
            clean_wav = clean_wav.squeeze(0).to(device)
            noisy_wav = noisy_wav.squeeze(0).to(device)
            noisy_fname = noisy_fname[0]

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
                    n_tilde.shape, interpolation="bilinear"
                )
                pred_noise_tilde = (
                    resize_transform(pred_noise_tilde_batch).squeeze(0).squeeze(0)
                )

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
            all_metrics.append(metrics)
            files_processed += 1

    avg_metric_results = average_metrics(all_metrics)
    return avg_metric_results


def compute_log_db(magnitude, eps):
    return 20 * np.log10(magnitude + eps)
