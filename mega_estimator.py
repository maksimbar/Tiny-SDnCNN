import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn


from utils import (
    compute_stft_spectrogram,
    normalize_spectrogram_paper,
    denormalize_spectrogram_paper,
    load_and_preprocess_audio,
    compute_log_db,
)

# --- Constants ---
CLEAN_FILE = Path("data_small/test/clean/F_BG014_02-a0221.wav")
NOISY_FILE = Path("data_small/test/noisy/F_BG014_02-a0221_w15.wav")
MODEL_PATH = Path("models/best_model.pt")
OUT_WAV = Path("reconstructed_test.wav")

TARGET_SAMPLE_RATE = 16000
FFT_SIZE = 512
HOP_LENGTH = 128
WINDOW_LENGTH = 512
MODEL_DEPTH = 17
MODEL_CHANNELS = 64
EPS = 1e-9  # Epsilon for log calculation

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SDnCNN(nn.Module):
    def __init__(self, num_layers: int = 17, num_channels: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(1, num_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_channels, 1, 3, padding=1, bias=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    # --- sanity checks ---
    if not CLEAN_FILE.exists() or not NOISY_FILE.exists():
        print(
            f"ERROR: Audio file(s) not found. Ensure {CLEAN_FILE} and {NOISY_FILE} exist."
        )
        exit(1)
    if not MODEL_PATH.exists():
        print(
            f"ERROR: Model file {MODEL_PATH} not found. Please provide a valid model."
        )
        exit(1)

    print(f"Loading model from {MODEL_PATH}")
    model = SDnCNN(MODEL_DEPTH, MODEL_CHANNELS).to(device)
    try:
        sd = torch.load(str(MODEL_PATH), map_location=device)
        model.load_state_dict(sd)
    except Exception as e:
        print(f"ERROR: Could not load model from {MODEL_PATH}. Error: {e}")
        exit(1)
    model.eval()

    # --- load & report length ---
    print("Loading audio…")
    try:
        clean_wav = load_and_preprocess_audio(CLEAN_FILE, TARGET_SAMPLE_RATE)
        noisy_wav = load_and_preprocess_audio(NOISY_FILE, TARGET_SAMPLE_RATE)
        L = min(clean_wav.shape[-1], noisy_wav.shape[-1])
        print(f"Duration (s): {L / TARGET_SAMPLE_RATE:.3f}")
        print(f"Sample length (in samples): {L}")
    except Exception as e:
        print(f"ERROR: Could not load audio files. Error: {e}")
        exit(1)

    if L == 0:
        print("ERROR: Loaded audio has zero length. Cannot proceed.")
        exit(1)

    clean_wav, noisy_wav = clean_wav[:L], noisy_wav[:L]
    noisy_gpu, clean_gpu = noisy_wav.to(device), clean_wav.to(device)

    # --- STFT + inference ---
    print("Computing STFTs…")
    mag_clean, _ = compute_stft_spectrogram(
        clean_gpu, FFT_SIZE, HOP_LENGTH, WINDOW_LENGTH
    )
    mag_noisy, phase_noisy = compute_stft_spectrogram(
        noisy_gpu, FFT_SIZE, HOP_LENGTH, WINDOW_LENGTH
    )
    log_noisy_db = 10 * torch.log10(mag_noisy.pow(2) + EPS**2)

    print("Normalizing…")
    norm_noisy, x_hat_min, x_bar_max = normalize_spectrogram_paper(log_noisy_db)

    print("Running model…")
    with torch.no_grad():
        inp = norm_noisy.unsqueeze(0).unsqueeze(0)
        resid = model(inp).squeeze(0).squeeze(0)
    clean_norm_est = norm_noisy - resid

    print("Denormalizing…")
    log_clean_est_db = denormalize_spectrogram_paper(
        clean_norm_est, x_hat_min, x_bar_max
    )
    mag_est = torch.sqrt(torch.clamp(10 ** (log_clean_est_db / 10.0), min=0.0) + EPS**2)

    print("Reconstructing waveform…")
    spec_est = mag_est * torch.exp(1j * phase_noisy)
    window_gpu = torch.hann_window(WINDOW_LENGTH, device=device)
    wave_est = (
        torch.istft(
            spec_est,
            n_fft=FFT_SIZE,
            hop_length=HOP_LENGTH,
            win_length=WINDOW_LENGTH,
            window=window_gpu,
            length=L,
        )
        .cpu()
        .numpy()
    )

    print(f"Saving reconstructed waveform to {OUT_WAV}")
    torchaudio.save(
        str(OUT_WAV), torch.from_numpy(wave_est).unsqueeze(0), TARGET_SAMPLE_RATE
    )

    # --- prepare for plotting ---
    clean_wav_np = clean_wav.cpu().numpy()
    noisy_wav_np = noisy_wav.cpu().numpy()
    mag_clean_np = mag_clean.cpu().numpy()
    mag_noisy_np = mag_noisy.cpu().numpy()
    mag_est_np = mag_est.cpu().numpy()
    db_clean = compute_log_db(mag_clean_np, EPS)
    db_noisy = compute_log_db(mag_noisy_np, EPS)
    db_est = compute_log_db(mag_est_np, EPS)

    # spectrogram vmin/vmax
    valid_db_noisy = db_noisy[np.isfinite(db_noisy)]
    if valid_db_noisy.size > 0:
        vmin_spec, vmax_spec = valid_db_noisy.min(), valid_db_noisy.max()
        if vmin_spec == vmax_spec:
            vmin_spec -= 10
            vmax_spec += 10
    else:
        print(
            "Warning: Noisy spectrogram data is non-finite. Using default [-80,0] dB."
        )
        vmin_spec, vmax_spec = -80, 0

    # waveform y-limits
    if noisy_wav_np.size > 0:
        mn, mx = noisy_wav_np.min(), noisy_wav_np.max()
        rng = mx - mn
        if abs(rng) < EPS:
            y_min_wave, y_max_wave = mn - 0.1, mx + 0.1
        else:
            pad = rng * 0.05
            y_min_wave, y_max_wave = mn - pad, mx + pad
    else:
        y_min_wave, y_max_wave = -1.0, 1.0
    if abs(y_min_wave - y_max_wave) < EPS:
        y_min_wave -= 0.1
        y_max_wave += 0.1

    appendix_dir = Path("appendix")
    appendix_dir.mkdir(exist_ok=True)

    # spectrogram plots
    spectrogram_plots = [
        (db_clean, "clean_spectrogram.png"),
        (db_noisy, "noisy_spectrogram.png"),
        (db_est, "estimated_spectrogram.png"),
    ]
    for data, fn in spectrogram_plots:
        fig, ax = plt.subplots(figsize=(12, 4))
        if data.size == 0 or not np.any(np.isfinite(data)):
            plt.close(fig)
            continue
        ax.imshow(data, origin="lower", aspect="auto", vmin=vmin_spec, vmax=vmax_spec)
        ax.set_axis_off()
        fig.savefig(appendix_dir / fn, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)

    # waveform plots with blue line
    waveform_plots = [
        (clean_wav_np, "clean_waveform.png"),
        (noisy_wav_np, "noisy_waveform.png"),
        (wave_est, "estimated_waveform.png"),
    ]
    for data, fn in waveform_plots:
        fig, ax = plt.subplots(figsize=(12, 2.5))
        if data.size == 0:
            plt.close(fig)
            continue
        ax.plot(data, color="#0000FF")  # ← explicit blue
        ax.set_ylim(y_min_wave, y_max_wave)
        ax.set_axis_off()
        ax.margins(x=0, y=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(
            appendix_dir / fn,
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
            transparent=True,
        )
        plt.close(fig)

    print("Done.")
