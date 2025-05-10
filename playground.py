import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
import hydra
import sys

from model import SDnCNN
from utils import (
    compute_stft_spectrogram,
    normalize_spectrogram_paper,
    denormalize_spectrogram_paper,
    load_and_preprocess_audio,
    compute_log_db,
)


@hydra.main(
    config_path=str("conf"),
    config_name="primary",
    version_base=None,
)
def main(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Paths relative to this script's location (e.g., scripts/playground.py) ---
    script_dir = Path(__file__).parent.resolve()
    samples_dir = script_dir / "samples"  # WAV files directly in samples/
    output_dir_plots = script_dir / "plots"
    # --- End Local Path Definitions ---

    model_path_str = "models/1.pt"  # Assuming model is in "scripts/playground/models/"
    model_path = script_dir / model_path_str

    samples_dir.mkdir(parents=True, exist_ok=True)  # Ensure samples dir exists
    output_dir_plots.mkdir(parents=True, exist_ok=True)

    target_sample_rate = cfg.data.sample_rate
    fft_size = cfg.stft.n_fft
    hop_length = cfg.stft.hop_length
    window_length = cfg.stft.win_length
    window_type = cfg.stft.window_type.lower()
    model_depth = cfg.model.depth
    model_channels = cfg.model.channels
    model_activation = cfg.model.activation
    model_dilation_rates = (
        list(cfg.model.dilation_rates) if cfg.model.dilation_rates is not None else None
    )
    eps = cfg.get("constants.eps", 1e-9)

    print(f"Loading model from {model_path}")
    model_instance = SDnCNN(
        model_depth, model_channels, model_activation, model_dilation_rates
    ).to(device)
    try:
        sd = torch.load(str(model_path), map_location=device)
    except FileNotFoundError:
        print(
            f"Model file not found at {model_path}. "
            f"Please ensure '{model_path_str}' exists relative to the script directory ({script_dir})."
        )
        # Attempting relative to CWD as a fallback
        cwd_model_path = Path.cwd() / "models" / "best_model.pt"
        print(f"Trying relative to CWD: {cwd_model_path}")
        try:
            sd = torch.load(str(cwd_model_path), map_location=device)
        except FileNotFoundError:
            print(f"Model file also not found at {cwd_model_path}. Exiting.")
            sys.exit(1)

    model_instance.load_state_dict(sd)
    model_instance.eval()

    # --- Specific file processing ---
    noisy_file_name = "F_BG014_02-a0221_w00.wav"
    # Per your previous instruction: "matching clean samples/FF_BG014_02-a0221.wav"
    # If clean file has FF prefix and noisy does not, reflect that here:
    clean_file_name = "F_BG014_02-a0221.wav"

    noisy_file_path = samples_dir / noisy_file_name
    clean_file_path = samples_dir / clean_file_name

    # Check if files exist
    if not noisy_file_path.exists():
        print(f"ERROR: Noisy file not found at {noisy_file_path}")
        sys.exit(1)
    if not clean_file_path.exists():
        print(f"ERROR: Clean file not found at {clean_file_path}")
        sys.exit(1)

    # Restored WAV will be saved directly in samples_dir
    output_wav_path = samples_dir / f"{Path(noisy_file_name).stem}_restored.wav"
    output_plot_path = output_dir_plots / f"{Path(noisy_file_name).stem}_comparison.png"

    print(f"Processing: {noisy_file_path}")
    print(f"Clean reference: {clean_file_path}")

    clean_wav_orig = load_and_preprocess_audio(clean_file_path, target_sample_rate)
    noisy_wav_orig = load_and_preprocess_audio(noisy_file_path, target_sample_rate)

    L = min(clean_wav_orig.shape[-1], noisy_wav_orig.shape[-1])
    clean_wav = clean_wav_orig[:L]
    noisy_wav = noisy_wav_orig[:L]

    noisy_gpu = noisy_wav.to(device)
    clean_gpu = clean_wav.to(device)

    log_power_clean_db, _ = compute_stft_spectrogram(
        clean_gpu, fft_size, hop_length, window_length, window_type
    )
    log_power_noisy_db, phase_noisy = compute_stft_spectrogram(
        noisy_gpu, fft_size, hop_length, window_length, window_type
    )

    norm_noisy, x_hat_min, x_bar_max = normalize_spectrogram_paper(log_power_noisy_db)

    with torch.no_grad():
        inp = norm_noisy.unsqueeze(0).unsqueeze(0)
        resid = model_instance(inp).squeeze(0).squeeze(0)

    clean_norm_est = norm_noisy - resid
    log_power_clean_est_db = denormalize_spectrogram_paper(
        clean_norm_est, x_hat_min, x_bar_max
    )
    mag_est_amp = torch.sqrt(
        torch.clamp(10 ** (log_power_clean_est_db / 10.0), min=0.0)
    )

    if window_type == "blackman":
        window_fn_torch = torch.blackman_window
    elif window_type == "hann":
        window_fn_torch = torch.hann_window
    elif window_type == "hamming":
        window_fn_torch = torch.hamming_window
    else:
        window_fn_torch = torch.hann_window

    window_gpu = window_fn_torch(window_length, device=device)

    spec_est_complex = mag_est_amp * torch.exp(1j * phase_noisy)
    wave_est = torch.istft(
        spec_est_complex,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=window_gpu,
        length=L,
    ).cpu()

    print(f"Saving reconstructed waveform to {output_wav_path}")
    torchaudio.save(str(output_wav_path), wave_est.unsqueeze(0), target_sample_rate)

    # --- Plotting Preparation ---
    clean_wav_np = clean_wav.cpu().numpy()
    noisy_wav_np = noisy_wav.cpu().numpy()
    wave_est_np = wave_est.numpy()

    mag_clean_amp_np = (
        torch.sqrt(torch.clamp(10 ** (log_power_clean_db / 10.0), min=0.0, max=1e12))
        .cpu()
        .numpy()
    )
    mag_noisy_amp_np = (
        torch.sqrt(torch.clamp(10 ** (log_power_noisy_db / 10.0), min=0.0, max=1e12))
        .cpu()
        .numpy()
    )
    mag_est_amp_np = mag_est_amp.cpu().numpy()

    db_clean_plot = compute_log_db(mag_clean_amp_np, eps)
    db_noisy_plot = compute_log_db(mag_noisy_amp_np, eps)
    db_est_plot = compute_log_db(mag_est_amp_np, eps)

    vmin = np.percentile(db_noisy_plot, 5)
    vmax = np.percentile(db_noisy_plot, 99)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(clean_wav_np)
    plt.title(f"Clean Waveform ({Path(clean_file_name).name})")
    plt.ylim(np.min(noisy_wav_np) * 1.1, np.max(noisy_wav_np) * 1.1)

    plt.subplot(3, 2, 2)
    plt.imshow(
        db_clean_plot,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"Clean Spec (dB from Amplitude)")
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(3, 2, 3)
    plt.plot(noisy_wav_np)
    plt.title(f"Noisy Waveform ({Path(noisy_file_name).name})")
    plt.ylim(np.min(noisy_wav_np) * 1.1, np.max(noisy_wav_np) * 1.1)

    plt.subplot(3, 2, 4)
    plt.imshow(
        db_noisy_plot,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"Noisy Spec (dB from Amplitude)")
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(3, 2, 5)
    plt.plot(wave_est_np)
    plt.title(f"Restored Waveform")
    plt.ylim(np.min(noisy_wav_np) * 1.1, np.max(noisy_wav_np) * 1.1)

    plt.subplot(3, 2, 6)
    plt.imshow(
        db_est_plot,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"Restored Spec (dB from Amplitude)")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved comparison plot to {output_plot_path}")

    print("Single file processed.")


if __name__ == "__main__":
    main()
