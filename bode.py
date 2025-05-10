"""This script compares Butterworth forward-backward and Savitzky-Golay filters on a Bode plot."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz, freqz_sos, savgol_coeffs

from helpers import savgol_window_length

N = 4096  # Number of frequencies to examine on the Bode plot


def freqz_butter_forward_back(order: int, cutoff: float):
    # Design Butterworth filter in second-order sections (sos)
    sos = butter(order, cutoff, fs=1.0, btype="low", output="sos")
    w, h = freqz_sos(sos, worN=N)
    # The backwards filter is the complex conjugate of h
    # See C.-Y. Tan, "Non-causal Zero Phase FIR Filter With Examples,"
    # Fermilab Accelerator Division/Tevatron.
    # https://larpdocs.fnal.gov/LARP/DocDB/0003/000359/001/zerophase20Mar2006.pdf
    h_eff = h * np.conj(h)
    return w, h_eff, f"butter forward-back order={order:d}, cutoff={cutoff:.2f} $f_s$"


def freqz_savgol(window_length: int, polyorder: int):
    coeffs = savgol_coeffs(window_length, polyorder, deriv=0)
    w, h = freqz(coeffs, worN=N)
    # `freqz` assumes the filter is applied causally, but we are applying it as a non-causal
    # filter that is symmetric about the present sample time.
    # We know this should have zero phase.
    h = np.abs(h)
    return w, h, f"savgol window_length={window_length:d}, polyorder={polyorder:d}"


# Plot Bode magnitude and phase response
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
ax1, ax2 = axes

cutoff = 0.01  # cutoff frequency / sample frequency
for (w, h, label), style in [
    (
        freqz_butter_forward_back(5, cutoff),
        dict(color="tab:blue", linewidth=3.0),
    ),
    (
        freqz_savgol(savgol_window_length(5, cutoff), 5),
        dict(color="tan"),
    ),
]:
    freqs = w / (2.0 * np.pi)
    # Magnitude response
    ax1.semilogx(freqs, 20 * np.log10(np.abs(h)), label=label, **style)
    # Phase response
    ax2.semilogx(
        freqs, np.unwrap(np.angle(h, deg=True), period=180), label=label, **style
    )

ax1.axvline(cutoff, color="black")
ax1.set_title("Bode Plot of Zero-Phase Butterworth vs. Savitzky-Golay Filters")
ax1.set_ylim((-80, 3))
ax1.set_ylabel("Magnitude [dB]")
ax1.legend(loc="lower left")

ax1.text(0.2, 0.7, "Both have gain = 1\nin pass band", transform=ax1.transAxes)
ax1.text(
    0.6,
    0.8,
    "butter suppresses high frequencies\nbetter than savgol",
    transform=ax1.transAxes,
)

ax2.axvline(cutoff, color="black")
ax2.set_ylim((-180, 180))
ax2.set_xlabel("Frequency / sample frequency [dimensionless]")
ax2.set_ylabel("Phase [degrees]")

ax2.text(0.1, 0.6, "Both have zero phase delay", transform=ax2.transAxes)

for ax in axes:
    ax.set_xlim((1.0 / N, 0.5))
    ax.grid(which="both", linestyle="--", linewidth=0.5)
    # The figure created by this script is licensed under Creative Commons CC BY 4.0
    # https://creativecommons.org/licenses/by/4.0/
    ax.text(
        0.98,
        0.02,
        "M. Vernacchia\n2025, CC BY 4.0",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        color="gray",
    )

plt.tight_layout()
fig.savefig("bode.png", dpi=200)
plt.show()
