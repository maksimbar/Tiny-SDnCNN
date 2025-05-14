import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from model import SDnCNN
from utils import (
    load_and_preprocess_audio,
    compute_stft_spectrogram,
    normalize_spectrogram_paper,
    denormalize_spectrogram_paper,
    reconstruct_waveform,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).parent.resolve()

CLEAN_FILE = REPO_ROOT / "data_small/test/clean/F_BG014_02-a0221.wav"
NOISY_FILE = REPO_ROOT / "data_small/test/noisy/F_BG014_02-a0221_w15.wav"
MODEL_PATH = REPO_ROOT / "models/best_model.pt"

OUTPUT_IMAGE = SCRIPT_DIR / "visual.png"
OUT_WAV = SCRIPT_DIR / "reconstructed.wav"


TARGET_SAMPLE_RATE = 16000
FFT_SIZE = 512
HOP_LENGTH = 128
WINDOW_LENGTH = 512
STFT_WINDOW_TYPE = "blackman"

MODEL_DEPTH = 17
MODEL_CHANNELS = 64
# Example: [1, 2, 4, 2, 1] will cycle through these for intermediate layers
# Set to None or [] or [1] for standard convolutions (dilation=1) for intermediate layers
MODEL_DILATION_RATES = [
    2
]  # Ensure this matches the trained model if MODEL_PATH is used

EPS = 1e-9  # Epsilon for log calculation and avoiding division by zero

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Helper Functions for Plotting ---
def get_magnitude_from_log_power_db(log_power_db_tensor):
    """Converts log-power spectrogram (10*log10(mag^2)) to magnitude spectrogram."""
    # log_power_db = 10 * log10(mag^2)  (ignoring epsilon for simplicity here)
    # log_power_db / 20 = 0.5 * log10(mag^2) = log10(mag)
    # So, 10^(log_power_db / 20) = mag
    magnitude = torch.pow(10.0, log_power_db_tensor / 20.0)
    return magnitude


def compute_log_magnitude_db_for_plotting(magnitude_tensor, eps=EPS):
    """Computes log-magnitude spectrogram (20*log10(mag)) for plotting."""
    return 20 * np.log10(magnitude_tensor.cpu().numpy() + eps)


if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}")
    # Pass MODEL_DILATION_RATES to the constructor
    # Ensure model activation in constructor matches training if not default relu
    model = SDnCNN(
        MODEL_DEPTH,
        MODEL_CHANNELS,
        activation="prelu",
        dilation_rates=MODEL_DILATION_RATES,
    ).to(device)

    try:
        sd = torch.load(str(MODEL_PATH), map_location=device)
        model.load_state_dict(sd)
        print(f"Successfully loaded model weights from {MODEL_PATH}")
    except FileNotFoundError:
        print(
            f"WARNING: Model file {MODEL_PATH} not found. Using randomly initialized weights."
        )
    except RuntimeError as e:
        print(f"WARNING: Error loading model weights from {MODEL_PATH}: {e}")
        print(
            "This might be due to a mismatch in model architecture (e.g., depth, channels, dilation rates). "
            "Using randomly initialized weights."
        )
    model.eval()

    stft_params = {
        "fft_size": FFT_SIZE,
        "hop_length": HOP_LENGTH,
        "window_length": WINDOW_LENGTH,
        "window_type": STFT_WINDOW_TYPE.lower(),
    }

    print("Loading audio…")
    if not CLEAN_FILE.exists() or not NOISY_FILE.exists():
        print(
            f"Warning: {CLEAN_FILE} or {NOISY_FILE} not found. Using dummy noise data."
        )
        L_dummy = TARGET_SAMPLE_RATE * 3  # 3 seconds of dummy audio
        clean_wav_cpu = torch.zeros(L_dummy)
        noisy_wav_cpu = torch.rand(L_dummy) * 0.5 - 0.25
    else:
        clean_wav_cpu = load_and_preprocess_audio(CLEAN_FILE, TARGET_SAMPLE_RATE)
        noisy_wav_cpu = load_and_preprocess_audio(NOISY_FILE, TARGET_SAMPLE_RATE)

    L = min(clean_wav_cpu.shape[-1], noisy_wav_cpu.shape[-1])
    if L == 0:
        print("Error: Audio length is zero after loading/preprocessing. Exiting.")
        exit()

    clean_wav_cpu, noisy_wav_cpu = clean_wav_cpu[:L], noisy_wav_cpu[:L]

    noisy_wav_gpu = noisy_wav_cpu.to(device)
    clean_wav_gpu = clean_wav_cpu.to(device)

    print("Processing noisy audio for model input…")
    # This returns log-power spectrogram (10*log10(mag^2+eps)) and phase
    log_noisy_db_model, noisy_phase_model = compute_stft_spectrogram(
        noisy_wav_gpu, **stft_params
    )

    print("Normalizing noisy spectrogram…")
    norm_noisy_model, x_hat_min, x_bar_max = normalize_spectrogram_paper(
        log_noisy_db_model
    )

    print("Running model…")
    with torch.no_grad():
        inp = norm_noisy_model.unsqueeze(0).unsqueeze(0)
        pred_noise_norm_model = model(inp)

        if pred_noise_norm_model.shape[-2:] != inp.shape[-2:]:
            print(
                f"Warning: Model output shape {pred_noise_norm_model.shape} differs from input feature map shape {inp.shape}. "
                "This might lead to errors or unexpected behavior if not handled by subsequent processing. "
                "It could indicate an issue with model architecture or loaded weights if padding is meant to preserve dimensions."
            )
            # If using the provided SDnCNN with padding=dilation, this should not happen unless model loading failed
            # or there's a mismatch. Forcing a resize here is a patch, not a fix.
            # from torchvision.transforms import Resize
            # resize_transform = Resize(inp.shape[-2:])
            # pred_noise_norm_model = resize_transform(pred_noise_norm_model)

        pred_noise_norm_model = pred_noise_norm_model.squeeze(0).squeeze(0)

    est_clean_norm_model = norm_noisy_model - pred_noise_norm_model

    print("Denormalizing estimated clean spectrogram…")
    log_est_clean_db_model = denormalize_spectrogram_paper(
        est_clean_norm_model, x_hat_min, x_bar_max
    )

    print("Reconstructing waveform…")
    # utils.reconstruct_waveform handles windowing internally based on stft_params
    wave_est = reconstruct_waveform(
        log_est_clean_db_model,
        noisy_phase_model,
        **stft_params,
        orig_length=L,
    )
    wave_est_np = wave_est.cpu().numpy()

    print(f"Saving reconstructed waveform to {OUT_WAV}")
    torchaudio.save(
        str(OUT_WAV), torch.from_numpy(wave_est_np).unsqueeze(0), TARGET_SAMPLE_RATE
    )

    print("Preparing data for plotting…")
    # For clean signal (for plotting)
    log_clean_db_plot, _ = compute_stft_spectrogram(clean_wav_gpu, **stft_params)
    mag_clean_plot = get_magnitude_from_log_power_db(log_clean_db_plot)
    db_clean_to_plot = compute_log_magnitude_db_for_plotting(mag_clean_plot)

    # For noisy signal (for plotting) - can reuse log_noisy_db_model
    mag_noisy_plot = get_magnitude_from_log_power_db(log_noisy_db_model)
    db_noisy_to_plot = compute_log_magnitude_db_for_plotting(mag_noisy_plot)

    # For estimated signal (for plotting) - can reuse log_est_clean_db_model
    mag_est_plot = get_magnitude_from_log_power_db(log_est_clean_db_model)
    db_est_to_plot = compute_log_magnitude_db_for_plotting(mag_est_plot)

    vmin = np.min(db_noisy_to_plot)
    vmax = np.max(db_noisy_to_plot)
    print(
        f"Using common color scale for spectrograms: vmin={vmin:.2f} dB, vmax={vmax:.2f} dB"
    )

    print("Plotting…")
    plt.figure(figsize=(12, 9))

    # Waveforms
    clean_wav_np = clean_wav_cpu.numpy()
    noisy_wav_np = noisy_wav_cpu.numpy()
    # wave_est_np is already available

    waveform_data = [
        (clean_wav_np, "Clean", 1),
        (noisy_wav_np, "Noisy", 3),
        (wave_est_np, "Estimated Clean", 5),
    ]
    # Determine common y-limits for waveforms for better comparison
    min_y_wav = (
        min(np.min(clean_wav_np), np.min(noisy_wav_np), np.min(wave_est_np)) * 1.1
    )
    max_y_wav = (
        max(np.max(clean_wav_np), np.max(noisy_wav_np), np.max(wave_est_np)) * 1.1
    )

    for sig, title, pos in waveform_data:
        ax = plt.subplot(3, 2, pos)
        ax.plot(sig)
        ax.set_title(f"{title} Waveform")
        ax.set_ylim(min_y_wav, max_y_wav)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")

    # Spectrograms
    spectrogram_data = [
        (db_clean_to_plot, "Clean Spec (dB)", 2),
        (db_noisy_to_plot, "Noisy Spec (dB)", 4),
        (db_est_to_plot, "Estimated Clean Spec (dB)", 6),
    ]
    for spec_data, title, pos in spectrogram_data:
        ax = plt.subplot(3, 2, pos)
        im = ax.imshow(
            spec_data,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Frequency (bins)")
        plt.colorbar(im, ax=ax, format="%+2.0f dB", label="Log Magnitude (dB)")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Saved comparison plot to {OUTPUT_IMAGE}")
    print("Done.")
